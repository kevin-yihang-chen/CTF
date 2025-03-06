import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
# from models.tcformer import tcformer
from models.tnt import HIT
from models.transmil import TransMIL
# from models.tnt import TNT
from models.model_dsmil import FCLayer, BClassifier, MILNet
from models.model_hierarchical_mil import HIPT_None_FC, HIPT_LGP_FC
from models.vmamba import VSSM
from models.dtfdmil import DTFD_MIL
from models.s4 import S4Model
from models.wikg import WiKG
# from models.mambamil import MambaMIL
from models.patchgcn import PatchGCN_Surv
from models.par import PaR
from tqdm import tqdm
import time
from sksurv.metrics import concordance_index_censored
from .loss_func import NLLSurvLoss
from .loss_func import l1_reg_modules

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps,max=1-eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss

class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if val_loss != val_loss:
            print('NaN loss')
            self.early_stop = True
            return

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience or epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\n------------- Training Fold {}! -------------'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        # from tensorboardX import SummaryWriter
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    if args.bag_loss == 'svm':
        # from topk.svm import SmoothTop1SVM
        # loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        # if device.type == 'cuda':
        #     loss_fn = loss_fn.cuda()
        raise NotImplementedError
    else:
        if args.survival:
            if args.fuse == 'PIBD':
                loss_fn = NLLSurvLoss(0.5)
            else:
                loss_fn = CrossEntropySurvLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    # Init model
    if args.model_type in ['clam_sb', 'clam_mb', 'hit', 'dsmil', 'mamba', 'transmil', 'dtfdmil', 's4', 'wikg', 'mambamil', 'patchgcn', 'moe']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        # if args.inst_loss == 'svm':
        #     from topk.svm import SmoothTop1SVM
        #     instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        #     if device.type == 'cuda':
        #         instance_loss_fn = instance_loss_fn.cuda()
        # else:
        instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'hit':
            model = HIT(n_classes=args.n_classes)
            # model = TransMIL()
        elif args.model_type == 'transmil':
            model = TransMIL(n_classes=args.n_classes)
        elif args.model_type == 'dsmil':
            i_classifier = FCLayer(in_size=768, out_size=model_dict['n_classes'])
            b_classifier = BClassifier(input_size=768, output_class=model_dict['n_classes'], dropout_v=0.0)
            model = MILNet(i_classifier, b_classifier)
        elif args.model_type == 'mamba':
            model = VSSM(**model_dict)
        elif args.model_type == 'dtfdmil':
            model = DTFD_MIL(n_classes=args.n_classes)
        elif args.model_type == 's4':
            model = S4Model(n_classes=args.n_classes)
        elif args.model_type == 'wikg':
            model = WiKG(n_classes=args.n_classes)
        elif args.model_type == 'mambamil':
            # model = MambaMIL(n_classes=args.n_classes)
            raise NotImplementedError
        elif args.model_type == 'patchgcn':
            model = PatchGCN_Surv(n_classes=args.n_classes)
        elif args.model_type == 'moe':
            model = PaR(n_classes=args.n_classes) 
        else:
            raise NotImplementedError

        # model.half()


    elif 'hipt' in args.model_type:
        if args.model_type == 'hipt_n':
            model = HIPT_None_FC(**model_dict)
        elif args.model_type == 'hipt_lgp':
            model = HIPT_LGP_FC(**model_dict, freeze_4k=True, pretrain_4k='vit4k_xs_dino', freeze_WSI=True, pretrain_WSI='None')



    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    # model.relocate()
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(device)
        
    # print_network(model)
    if args.model_type != 'dtfdmil':
        optimizer = get_optim(model, args)
    else:
        trainable_parameters = []
        trainable_parameters += list(model.classifier.parameters())
        trainable_parameters += list(model.attention.parameters())
        trainable_parameters += list(model.dimReduction.parameters())
        optimizer0 = torch.optim.Adam(trainable_parameters, lr=args.lr,  weight_decay=args.reg)
        optimizer1 = torch.optim.Adam(model.attCls.parameters(), lr=args.lr,  weight_decay=args.reg)
        optimizer = [optimizer0, optimizer1]

    
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=6, stop_epoch=20, verbose=True)

    else:
        early_stopping = None

    for epoch in range(args.max_epochs):
        print(f'\nProcessing Epoch {epoch} ...', end=' ')
        time_start = time.perf_counter()
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            if args.survival:
                train_loop_clam_surv(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
                stop = validate_clam_surv(cur, epoch, model, val_loader, args.n_classes, 
                    early_stopping, writer, loss_fn, args.results_dir)
            else:
                # train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
                # stop = validate_clam(cur, epoch, model, val_loader, args.n_classes,
                #     early_stopping, writer, loss_fn, args.results_dir)
                raise NotImplementedError
        elif args.model_type in ['hit', 'dsmil', 'hipt_lgp', 'hipt_n', 'mamba', 'transmil', 'dtfdmil', 's4', 'wikg', 'mambamil', 'patchgcn', 'moe'] and not args.no_inst_cluster:
            if args.survival:
                train_loop_hit_surv(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, args, writer, loss_fn, model_type=args.model_type)
                stop = validate_hit_surv(cur, epoch, model, val_loader, args.n_classes, 
                    args, early_stopping, writer, loss_fn, args.results_dir)
            else:
                # train_loop_hit(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, model_type=args.model_type)
                # stop = validate_hit(cur, epoch, model, val_loader, args.n_classes,
                #     early_stopping, writer, loss_fn, args.results_dir)
                raise NotImplementedError
        
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)

        time_end = time.perf_counter()
        print(f'Epoch time: {time_end - time_start}')
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    if args.model_type in ['hit', 'dsmil', 'hipt_lgp', 'hipt_n', 'mamba', 'transmil', 'dtfdmil', 's4', 'wikg', 'mambamil', 'patchgcn', 'moe']:
        if args.survival:
            _, val_error, val_cindex = summary_my_surv(model, val_loader, args.n_classes, args)
            print('Val C-Index: {:.4f}'.format(val_cindex))

            results_dict, test_error, test_cindex = summary_my_surv(model, test_loader, args.n_classes, args)
            print('Test C-Index: {:.4f}'.format(test_cindex))
        else:
            _, val_error, val_auc, _= summary_my(model, val_loader, args.n_classes)
            print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

            results_dict, test_error, test_auc, acc_logger = summary_my(model, test_loader, args.n_classes)
            print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    elif args.model_type in ['clam_sb', 'clam_mb']:
        if args.survival:
            _, val_error, val_cindex= summary_clam_surv(model, val_loader, args.n_classes)
            print('Val C-Index: {:.4f}'.format(val_cindex))

            results_dict, test_error, test_cindex = summary_clam_surv(model, test_loader, args.n_classes)
            print('Test C-Index: {:.4f}'.format(test_cindex))
        else:
            _, val_error, val_auc, _= summary_clam(model, val_loader, args.n_classes)
            print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

            results_dict, test_error, test_auc, acc_logger = summary_clam(model, test_loader, args.n_classes)
            print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    # acc_list = []
    # correct_list = []
    # count_list = []
    # for i in range(args.n_classes):
    #     acc, correct, count = acc_logger.get_summary(i)
    #     acc_list.append(acc)
    #     correct_list.append(correct)
    #     count_list.append(count)
    #     print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
    #     if writer:
    #         writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
    # print('Overall: acc {:.4f}, correct {}/{}'.format(np.sum(acc_list)/2, np.sum(correct_list), np.sum(count_list)))
    if args.survival:
        if writer:
            writer.add_scalar('final/val_cindex', val_cindex, 0)
            writer.add_scalar('final/test_cindex', test_cindex, 0)
            writer.close()
        return results_dict, test_cindex, val_cindex, 1-test_error, 1-val_error
    else:
        if writer:
            writer.add_scalar('final/val_error', val_error, 0)
            writer.add_scalar('final/val_auc', val_auc, 0)
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_auc', test_auc, 0)
            writer.close()
        return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 



def train_loop_clam_surv(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0
    losses = []

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    print('\n')
    for batch_idx, (data, data2, censor, survival_days, label, slide_id, coords) in enumerate(loader):
        data, label, censor = data.to(device), label.to(device), censor.to(device)
        # import ipdb;ipdb.set_trace()
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        survival_labels = torch.LongTensor(1)
        survival_labels[0] = label
        survival_labels = survival_labels.cuda()
        censoreds = torch.LongTensor(1)
        censoreds[0] = censor
        censoreds = censoreds.cuda()
        # acc_logger.log(Y_hat, label)
        # loss = loss_fn(logits, label)
        # import ipdb;ipdb.set_trace()
        loss = loss_fn(hazards=hazards, S=S, Y=survival_labels, c=censoreds)
        # loss = loss_fn(hazards=hazards, S=S, Y=label, c=censor)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        # total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 
        total_loss = loss

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        # if (batch_idx + 1) % 20 == 0:
        #     print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
        #         'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        # error = calculate_error(Y_hat, label)
        # train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss_value)
        print('\r' + f'Epoch: {epoch}, Progress: {(batch_idx/len(loader)):.2}, loss:{np.mean(losses):.2}', end='', flush=True)

        risk = -torch.sum(S, dim=1).cpu().detach().numpy()
        all_risk_scores[batch_idx] = risk
        all_event_times[batch_idx] = survival_days
        all_censorships[batch_idx] = censor

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]


    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    # if inst_count > 0:
    #     train_inst_loss /= inst_count
    #     print('\n')
    #     for i in range(2):
    #         acc, correct, count = inst_logger.get_summary(i)
    #         print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_cindex: {:.4f}'.format(epoch, train_loss, train_inst_loss, c_index))
    # acc_list = []
    # correct_list = []
    # count_list = []
    # for i in range(n_classes):
    #     acc, correct, count = acc_logger.get_summary(i)
    #     acc_list.append(acc)
    #     correct_list.append(correct)
    #     count_list.append(count)
    #     print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
    #     if writer and acc is not None:
    #         writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
    # print('Overall: acc {:.4f}, correct {}/{}'.format(np.sum(acc_list)/2, np.sum(correct_list), np.sum(count_list)))

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        # writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)


def calculate_risk(h):
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def train_loop_hit_surv(epoch, model, loader, optimizer, n_classes, bag_weight, args, writer = None, loss_fn = None, model_type='None'):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_loss2 = 0.
    train_error = 0.
    train_inst_loss = 0.
    losses = []

    print('\n')
    # for batch_idx, (data, label) in enumerate(loader):
    # pbar = tqdm(total=len(loader))
    # import ipdb;ipdb.set_trace()
    # num_sam = []
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for name, param in model.named_parameters():
        if name.startswith('radiology_model'):
            if 'ctx' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        elif name.startswith('pathology_model'):
            if 'ctx' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        elif name.startswith('mpmoe'):
            if 'ctx' in name:
                param.requires_grad_(True)
            elif 'W_l' in name:
                param.requires_grad_(True)
            elif 'MoE' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")
    print(f"Parameters count: {len(enabled)}")

    for i, (data, data2, data3, censor, survival_days, label, slide_id, coords) in enumerate(loader):
        # data, data2, censor, survival_days, label = data.to(device, non_blocking=True), data2.to(device, non_blocking=True), censor.to(device, non_blocking=True), survival_days.to(device, non_blocking=True), label.to(device, non_blocking=True)
        # import ipdb;ipdb.set_trace()
        try:
            data, data2, data3, censor, survival_days, label = data.to(device, non_blocking=True), data2.to(device, non_blocking=True), data3.to(device, non_blocking=True), censor.to(device, non_blocking=True), survival_days.to(device, non_blocking=True), label.to(device, non_blocking=True)
        except:
            data, censor, survival_days, label = data[0].to(device, non_blocking=True), censor.to(device, non_blocking=True), survival_days.to(device, non_blocking=True), label.to(device, non_blocking=True)
        if model_type != 'dtfdmil':

            if args.fuse == 'PIBD':
                logits, IB_loss_proxy, proxy_loss, mimin_total, mimin_loss_total = model(data, data2, label, censor)

                loss_surv = loss_fn(h=logits, y=label, t=survival_days, c=censor)

                loss = loss_surv + 1 * proxy_loss + 0.1*IB_loss_proxy + 0.1 * (
                            mimin_total + mimin_loss_total)

                h = logits

                risk, _ = calculate_risk(h)

            else:
                logits, Y_hat, Y_prob, _ = model(data, data2, data3)

                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                survival_labels = torch.LongTensor(1)
                survival_labels[0] = label
                survival_labels = survival_labels.cuda()
                censoreds = torch.LongTensor(1)
                censoreds[0] = censor
                censoreds = censoreds.cuda()

                # import ipdb;ipdb.set_trace()
                loss = loss_fn(hazards=hazards, S=S, Y=survival_labels, c=censoreds)
                if args.fuse == 'MOTCAT':
                    reg_loss = 0
                elif args.fuse == 'MPMoE':
                    cv_loss = _
                    loss = loss + cv_loss
                # import ipdb;ipdb.set_trace()
                risk = -torch.sum(S, dim=1).cpu().detach().numpy()

            all_risk_scores[i] = risk
            all_censorships[i] = censor
            all_event_times[i] = survival_days

            loss.backward()
            loss_value = loss.item()
            train_loss += loss_value

            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss_value)
            print('\r' + f'Epoch: {epoch}, Progress: {(i / len(loader)):.2}, loss:{np.mean(losses):.2}', end='',
                  flush=True)

        else:

            logits0, Y_hat, Y_prob, slide_sub_preds = model(data, data2)

            # acc_logger.log(Y_hat, label)

            # import ipdb;ipdb.set_trace()
            slide_sub_labels = label.repeat(slide_sub_preds.size(0))
            # slide_sub_labels = torch.repeat(label, slide_sub_preds.size(0))
            loss0 = loss_fn(slide_sub_preds, slide_sub_labels).mean()
            optimizer[0].zero_grad()
            loss0.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.dimReduction.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(model.attention.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 5)
            # optimizer[0].step()

            loss1 = loss_fn(logits0, label).mean()
            optimizer[1].zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(model.attCls.parameters(), 5)
            optimizer[0].step()
            optimizer[1].step()

            # loss = loss_fn(logits0, label)
            loss_value = loss0.item() + loss1.item()

            # total_loss = loss
            train_loss += loss_value
            error = calculate_error(Y_hat, label)
            train_error += error

    # import ipdb;ipdb.set_trace()
    #     pbar.update(1)
    # pbar.close()
    # calculate loss and error for epoch

    train_loss /= len(loader)
    c_index = concordance_index_censored((1 - all_censorships).astype(bool),
            all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, c_index: {:.4f}'.format(epoch, train_loss,  c_index))
    


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, data2, label, slide_id, coords) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    # if inst_count > 0:
    #     val_inst_loss /= inst_count
    #     for i in range(2):
    #         acc, correct, count = inst_logger.get_summary(i)
    #         print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        # writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    acc_list = []
    correct_list = []
    count_list = []
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        acc_list.append(acc)
        correct_list.append(correct)
        count_list.append(count)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
    print('Overall: acc {:.4f}, correct {}/{}'.format(np.sum(acc_list)/2, np.sum(correct_list), np.sum(count_list)))


    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam_surv(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    with torch.no_grad():
        for batch_idx, (data, data2, censor, survival_days, label, slide_id, coords) in enumerate(loader):
            data, label, censor = data.to(device), label.to(device), censor.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            # acc_logger.log(Y_hat, label)

            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            survival_labels = torch.LongTensor(1)
            survival_labels[0] = label
            survival_labels = survival_labels.cuda()
            censoreds = torch.LongTensor(1)
            censoreds[0] = censor
            censoreds = censoreds.cuda()
            # import ipdb;ipdb.set_trace()
            loss = loss_fn(hazards=hazards, S=S, Y=survival_labels, c=censoreds)

            val_loss += loss.item()

            # instance_loss = instance_dict['instance_loss']
            
            # inst_count+=1
            # instance_loss_value = instance_loss.item()
            # val_inst_loss += instance_loss_value

            # inst_preds = instance_dict['inst_preds']
            # inst_labels = instance_dict['inst_labels']
            # inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            # error = calculate_error(Y_hat, label)
            # val_error += error

            risk = -torch.sum(S, dim=1).cpu().detach().numpy()
            all_risk_scores[batch_idx] = risk
            all_event_times[batch_idx] = survival_days
            all_censorships[batch_idx] = censor

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    val_error /= len(loader)
    val_loss /= len(loader)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, c_index: {:.4f}'.format(val_loss, val_error, c_index))
    
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False 

def validate_hit(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    # sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, data2, label, slide_id, coords) in enumerate(loader):
            data, data2, label = data.to(device), data2.to(device), label.to(device)      
            logits, Y_hat, Y_prob, _ = model(data, data2)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)
            val_loss += loss.item()

            # import ipdb;ipdb.set_trace()
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    # if inst_count > 0:
    #     val_inst_loss /= inst_count
    #     for i in range(2):
    #         acc, correct, count = inst_logger.get_summary(i)
    #         print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    acc_list = []
    correct_list = []
    count_list = []
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        acc_list.append(acc)
        correct_list.append(correct)
        count_list.append(count)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
    print('Overall: acc {:.4f}, correct {}/{}'.format(np.sum(acc_list)/2, np.sum(correct_list), np.sum(count_list)))
     
    # import ipdb;ipdb.set_trace()
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_hit_surv(cur, epoch, model, loader, n_classes, args, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    # sample_size = model.k_sample
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    with torch.no_grad():
        for i, (data, data2, data3, censor, survival_months, label, slide_id, coords) in enumerate(loader):
            try:
                data, data2, data3, label = data.to(device), data2.to(device), data3.to(device), label.to(device)
            except:
                data, label = data[0].to(device), label.to(device)
            if args.fuse == 'PIBD':
                h, _, _, _, _ = model(data, data2)
                if len(h.shape) == 1:
                    h = h.unsqueeze(0)
                censor,label,survival_months = censor.to(device),label.to(device),survival_months.to(device)
                loss = loss_fn(h=h, y=label, t=survival_months, c=censor)
                val_loss += loss.item()
                risk = calculate_risk(h)
                all_risk_scores[i] = risk[0]
                all_censorships[i] = censor
                all_event_times[i] = survival_months
            else:
                logits, Y_hat, Y_prob, _ = model(data, data2, data3)
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                survival_labels = torch.LongTensor(1)
                survival_labels[0] = label
                survival_labels = survival_labels.cuda()
                censoreds = torch.LongTensor(1)
                censoreds[0] = censor
                censoreds = censoreds.cuda()
                # import ipdb;ipdb.set_trace()
                if cur==3:
                    loss = loss_fn(hazards=hazards, S=S, Y=survival_labels, c=censoreds)
                else:
                    loss = loss_fn(hazards=hazards, S=S, Y=survival_labels, c=censoreds)
                # import ipdb;ipdb.set_trace()
                risk = -torch.sum(S, dim=1).cpu().detach().numpy()
                all_risk_scores[i] = risk
                all_censorships[i] = censor
                all_event_times[i] = survival_months

                # loss.backward()

                # loss = loss_fn(logits, label)
                val_loss += loss.item()

                prob[i] = Y_prob.cpu().numpy()
                labels[i] = label.item()

    val_loss /= len(loader)
    c_index = concordance_index_censored((1 - all_censorships).astype(bool),
            all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('\nVal Set, val_loss: {:.4f}, c_index: {:.4f}'.format(val_loss, c_index))
     
    # import ipdb;ipdb.set_trace()
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary_my(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, data2, label, slide_id, coords) in enumerate(loader):
        data, data2, label = data.to(device), data2.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_hat, Y_prob, _ = model(data, data2)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger

def summary_my_surv(model, loader, n_classes, args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    # for batch_idx, (data, data2, label, slide_id, coords) in enumerate(loader):
    for i, (data, data2, data3, censor, survival_months, label, slide_id, coords) in enumerate(loader):
        try:
            data, data2, data3, label = data.to(device), data2.to(device), data3.to(device), label.to(device)
        except:
            data, label = data[0].to(device), label.to(device)
        slide_id = slide_ids.iloc[i]
        with torch.no_grad():
            if args.fuse == 'PIBD':
                h, _, _, _, _ = model(data, data2)
                if len(h.shape) == 1:
                    h = h.unsqueeze(0)
                censor,label,survival_months = censor.to(device),label.to(device),survival_months.to(device)
                risk = calculate_risk(h)
                risk = risk[0]
            else:
                logits, Y_hat, Y_prob, _ = model(data, data2, data3)
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(S, dim=1).cpu().detach().numpy()

        # acc_logger.log(Y_hat, label)
        # probs = Y_prob.cpu().numpy()
        # all_probs[batch_idx] = probs
        # all_labels[batch_idx] = label.item()
        
        # patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        # error = calculate_error(Y_hat, label)
        # test_error += error

        survival_labels = torch.LongTensor(1)
        survival_labels[0] = label
        survival_labels = survival_labels.cuda()
        censoreds = torch.LongTensor(1)
        censoreds[0] = censor
        censoreds = censoreds.cuda()
        # import ipdb;ipdb.set_trace()
        # import ipdb;ipdb.set_trace()
        all_risk_scores[i] = risk
        all_censorships[i] = censor
        all_event_times[i] = survival_months


    # test_error /= len(loader)
    c_index = concordance_index_censored((1 - all_censorships).astype(bool),
            all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    # if n_classes == 2:
    #     auc = roc_auc_score(all_labels, all_probs[:, 1])
    #     aucs = []
    # else:
    #     aucs = []
    #     binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
    #     for class_idx in range(n_classes):
    #         if class_idx in all_labels:
    #             fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
    #             aucs.append(calc_auc(fpr, tpr))
    #         else:
    #             aucs.append(float('nan'))

    #     auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, c_index

def summary_clam(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, data2, label, slide_id, coords) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger

def summary_clam_surv(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for i, (data, data2, censor, survival_months, label, slide_id, coords) in enumerate(loader):
        data, data2, label = data.to(device), data2.to(device), label.to(device)
        slide_id = slide_ids.iloc[i]
        with torch.no_grad():
            logits, Y_hat, Y_prob, _, _ = model(data)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        survival_labels = torch.LongTensor(1)
        survival_labels[0] = label
        survival_labels = survival_labels.cuda()
        censoreds = torch.LongTensor(1)
        censoreds[0] = censor
        censoreds = censoreds.cuda()

        risk = -torch.sum(S, dim=1).cpu().detach().numpy()
        all_risk_scores[i] = risk
        all_censorships[i] = censor
        all_event_times[i] = survival_months

    c_index = concordance_index_censored((1 - all_censorships).astype(bool),
            all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    
    return patient_results, test_error, c_index