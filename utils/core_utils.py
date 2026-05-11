"""Per-fold training loop.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize
from sksurv.metrics import concordance_index_censored

from datasets.dataset_generic import save_splits
from models.par import PaR
from utils.utils import calculate_error, get_optim, get_split_loader


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y) + eps)
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    ce_l = (
        - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps))
        - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps, max=1 - eps))
    )
    loss = (1 - alpha) * ce_l + alpha * reg
    return loss.mean()


class CrossEntropySurvLoss:
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        return ce_loss(hazards, S, Y, c, alpha=self.alpha if alpha is None else alpha)


class EarlyStopping:
    """Standard early stopping with checkpointing of the best model."""

    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):
        # NaN losses immediately stop training to avoid corrupting the checkpoint.
        if val_loss != val_loss:
            print('NaN loss')
            self.early_stop = True
            return

        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(val_loss, model, ckpt_name)
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience or epoch > self.stop_epoch:
                self.early_stop = True

    def _save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def _set_trainable(model):
    """Freeze the FMs; only prompt parameters and the classifier are trained."""
    for name, param in model.named_parameters():
        if name.startswith('radiology_model') or name.startswith('pathology_model'):
            param.requires_grad_('ctx' in name)
        elif name.startswith('mpmoe'):
            param.requires_grad_(
                'ctx' in name or 'W_s' in name or 'W_shared' in name or 'MoE' in name
            )
        else:
            param.requires_grad_(True)


def _print_param_count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total:,}')
    print(f'Trainable parameters: {trainable:,}')
    print(f'Non-trainable parameters: {total - trainable:,}')


def train(datasets, cur, args):
    """Train + evaluate a single CV fold."""
    print(f'\n------------- Training Fold {cur}! -------------')
    writer_dir = os.path.join(args.results_dir, str(cur))
    os.makedirs(writer_dir, exist_ok=True)

    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, f'splits_{cur}.csv'))
    print(f'Training on {len(train_split)} samples')
    print(f'Validating on {len(val_split)} samples')
    print(f'Testing on {len(test_split)} samples')

    if args.survival:
        loss_fn = CrossEntropySurvLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PaR(n_classes=args.n_classes, args=args).to(device)

    _set_trainable(model)
    _print_param_count(model)

    optimizer = get_optim(model, args)
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)

    early_stopping = EarlyStopping(patience=args.patience, stop_epoch=args.max_epochs, verbose=True) \
        if args.early_stopping else None

    for epoch in range(args.max_epochs):
        print(f'\nProcessing Epoch {epoch} ...', end=' ')
        time_start = time.perf_counter()

        if args.survival:
            _train_loop_surv(epoch, model, train_loader, optimizer, loss_fn)
            stop = _validate_surv(cur, epoch, model, val_loader, loss_fn,
                                  early_stopping, args.results_dir)
        else:
            _train_loop_cls(epoch, model, train_loader, optimizer, loss_fn)
            stop = _validate_cls(cur, epoch, model, val_loader, loss_fn,
                                 early_stopping, args.results_dir)

        print(f'Epoch time: {time.perf_counter() - time_start:.2f}s')
        if stop:
            break

    if args.early_stopping:
        ckpt_path = os.path.join(args.results_dir, f's_{cur}_checkpoint.pt')
        model.load_state_dict(torch.load(ckpt_path))
        if not args.save_model:
            os.remove(ckpt_path)
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, f's_{cur}_checkpoint.pt'))

    if args.survival:
        _, val_error, val_cindex = _summary_surv(model, val_loader)
        print(f'Val C-Index: {val_cindex:.4f}')
        results_dict, test_error, test_cindex = _summary_surv(model, test_loader)
        print(f'Test C-Index: {test_cindex:.4f}')
        return results_dict, test_cindex, val_cindex, 1 - test_error, 1 - val_error

    _, val_error, val_auc = _summary_cls(model, val_loader, args.n_classes)
    print(f'Val error: {val_error:.4f}, ROC AUC: {val_auc:.4f}')
    results_dict, test_error, test_auc = _summary_cls(model, test_loader, args.n_classes)
    print(f'Test error: {test_error:.4f}, ROC AUC: {test_auc:.4f}')
    return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------

def _train_loop_surv(epoch, model, loader, optimizer, loss_fn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    _set_trainable(model)

    risks = np.zeros(len(loader))
    censorships = np.zeros(len(loader))
    event_times = np.zeros(len(loader))
    losses = []

    for i, (data, data2, _, censor, survival_days, label, _, _) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        data2 = data2.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        censor = censor.to(device, non_blocking=True)

        logits, _, _, aux_loss = model(data, data2, None)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        Y = torch.LongTensor([label]).to(device)
        c_t = torch.LongTensor([censor]).to(device)
        loss = loss_fn(hazards=hazards, S=S, Y=Y, c=c_t) + aux_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        risks[i] = -torch.sum(S, dim=1).cpu().detach().numpy()
        censorships[i] = censor
        event_times[i] = survival_days
        print(f'\rEpoch {epoch}, Progress {i / len(loader):.2f}, loss {np.mean(losses):.4f}',
              end='', flush=True)

    train_loss = float(np.mean(losses))
    c_index = concordance_index_censored(
        (1 - censorships).astype(bool), event_times, risks, tied_tol=1e-08,
    )[0]
    wandb.log({'train_loss': train_loss, 'train_c_index': c_index})
    print(f'\nEpoch {epoch}, train_loss {train_loss:.4f}, c_index {c_index:.4f}')


def _validate_surv(cur, epoch, model, loader, loss_fn, early_stopping, results_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    risks = np.zeros(len(loader))
    censorships = np.zeros(len(loader))
    event_times = np.zeros(len(loader))
    val_loss = 0.0

    with torch.no_grad():
        for i, (data, data2, _, censor, survival_days, label, _, _) in enumerate(loader):
            data = data.to(device); data2 = data2.to(device); label = label.to(device); censor = censor.to(device)
            logits, _, _, _ = model(data, data2, None)
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            Y = torch.LongTensor([label]).to(device)
            c_t = torch.LongTensor([censor]).to(device)
            val_loss += loss_fn(hazards=hazards, S=S, Y=Y, c=c_t).item()
            risks[i] = -torch.sum(S, dim=1).cpu().detach().numpy()
            censorships[i] = censor
            event_times[i] = survival_days

    val_loss /= len(loader)
    c_index = concordance_index_censored(
        (1 - censorships).astype(bool), event_times, risks, tied_tol=1e-08,
    )[0]
    wandb.log({'val_loss': val_loss, 'val_c_index': c_index})
    print(f'\nVal Set, val_loss {val_loss:.4f}, c_index {c_index:.4f}')

    if early_stopping is not None:
        ckpt = os.path.join(results_dir, f's_{cur}_checkpoint.pt')
        early_stopping(epoch, val_loss, model, ckpt_name=ckpt)
        if early_stopping.early_stop:
            print('Early stopping')
            return True
    return False


def _summary_surv(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    risks = np.zeros(len(loader))
    censorships = np.zeros(len(loader))
    event_times = np.zeros(len(loader))
    patient_results = {}
    slide_ids = loader.dataset.slide_data['slide_id']

    with torch.no_grad():
        for i, (data, data2, _, censor, survival_days, label, _, _) in enumerate(loader):
            data = data.to(device); data2 = data2.to(device); label = label.to(device)
            logits, _, _, _ = model(data, data2, None)
            S = torch.cumprod(1 - torch.sigmoid(logits), dim=1)
            risks[i] = -torch.sum(S, dim=1).cpu().detach().numpy()
            censorships[i] = censor
            event_times[i] = survival_days
            patient_results[slide_ids.iloc[i]] = {
                'slide_id': np.array(slide_ids.iloc[i]),
                'risk': risks[i],
                'label': label.item(),
            }

    c_index = concordance_index_censored(
        (1 - censorships).astype(bool), event_times, risks, tied_tol=1e-08,
    )[0]
    return patient_results, 0.0, c_index


# ---------------------------------------------------------------------------
# Classification (grading)
# ---------------------------------------------------------------------------

def _train_loop_cls(epoch, model, loader, optimizer, loss_fn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    _set_trainable(model)

    losses = []
    correct = 0
    total = 0
    for i, (data, data2, label, _, _) in enumerate(loader):
        data = data.to(device); data2 = data2.to(device); label = label.to(device)
        logits, Y_hat, _, aux_loss = model(data, data2, None)
        loss = loss_fn(logits, label) + aux_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        correct += int(Y_hat.item() == label.item())
        total += 1
        print(f'\rEpoch {epoch}, Progress {i / len(loader):.2f}, loss {np.mean(losses):.4f}',
              end='', flush=True)

    wandb.log({'train_loss': float(np.mean(losses)), 'train_acc': correct / max(total, 1)})
    print(f'\nEpoch {epoch}, train_loss {np.mean(losses):.4f}, acc {correct / max(total, 1):.4f}')


def _validate_cls(cur, epoch, model, loader, loss_fn, early_stopping, results_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    val_loss = 0.0
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    with torch.no_grad():
        for i, (data, data2, label, _, _) in enumerate(loader):
            data = data.to(device); data2 = data2.to(device); label = label.to(device)
            logits, Y_hat, _, _ = model(data, data2, None)
            val_loss += loss_fn(logits, label).item()
            all_labels[i] = label.item()
            all_preds[i] = Y_hat.item()

    val_loss /= len(loader)
    acc = float(np.mean(all_labels == all_preds))
    wandb.log({'val_loss': val_loss, 'val_acc': acc})
    print(f'\nVal Set, val_loss {val_loss:.4f}, acc {acc:.4f}')

    if early_stopping is not None:
        ckpt = os.path.join(results_dir, f's_{cur}_checkpoint.pt')
        early_stopping(epoch, val_loss, model, ckpt_name=ckpt)
        if early_stopping.early_stop:
            print('Early stopping')
            return True
    return False


def _summary_cls(model, loader, n_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    test_error = 0.0
    patient_results = {}
    slide_ids = loader.dataset.slide_data['slide_id']

    with torch.no_grad():
        for i, (data, data2, label, _, _) in enumerate(loader):
            data = data.to(device); data2 = data2.to(device); label = label.to(device)
            logits, Y_hat, Y_prob, _ = model(data, data2, None)
            probs = Y_prob.detach().cpu().numpy()
            all_probs[i] = probs
            all_labels[i] = label.item()
            test_error += calculate_error(Y_hat, label)
            patient_results[slide_ids.iloc[i]] = {
                'slide_id': np.array(slide_ids.iloc[i]),
                'prob': probs,
                'label': label.item(),
            }

    test_error /= len(loader)

    if n_classes == 2:
        auc_score = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        binary_labels = label_binarize(all_labels, classes=list(range(n_classes)))
        aucs = []
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))
        auc_score = float(np.nanmean(aucs))

    return patient_results, test_error, auc_score
