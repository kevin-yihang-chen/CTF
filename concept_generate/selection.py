import argparse
import json
import os

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.feature_selection import mutual_info_classif
from torchvision import transforms

from open_clip import create_model_from_pretrained, get_tokenizer
from conch.open_clip_custom import create_model_from_pretrained as create_conch
from conch.open_clip_custom import get_tokenizer as get_tokenizer_conch
from conch.open_clip_custom import tokenize as tokenize_conch


STAGES = ['early', 'intermediate', 'advanced', 'metastatic']


def load_json(path):
    with open(path) as f:
        return json.load(f)


def topj_pooling(logits, topj=(1, 5, 10, 50, 100)):
    maxj = min(max(topj), logits.size(0))
    values, _ = logits.topk(maxj, 0, True, True)
    pooled = {j: values[:min(j, maxj)].mean(dim=0, keepdim=True) for j in topj}
    return pooled


def get_common_train_list(data_split_dir: str, num_folds: int):
    common = []
    for i in range(num_folds):
        fold = pd.read_csv(os.path.join(data_split_dir, f'splits_{i}.csv'))
        train_list = list(fold['train'])
        common = train_list if i == 0 else list(set(common) & set(train_list))
    fold0_labels = pd.read_csv(os.path.join(data_split_dir, 'splits_0_bool_label.csv'))
    labels = []
    for slide_id in common:
        match = fold0_labels[fold0_labels['slide_id'] == slide_id]
        labels.append(int(match['label'].values[0]))
    return common, labels


def encode_concepts(modal, concepts, conch_ckpt):
    if modal == 'path':
        model, _ = create_conch('conch_ViT-B-16', conch_ckpt)
        tokenizer = get_tokenizer_conch()
        tokens = tokenize_conch(texts=concepts, tokenizer=tokenizer)
        with torch.no_grad():
            embs = model.encode_text(tokens)
        return model, embs

    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    )
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokens = tokenizer(concepts, context_length=128)
    with torch.no_grad():
        _, embs, _ = model(None, tokens)
    return model, embs


def alignment_score_path(model, slide_id, data_dir, text_embs):
    h5 = os.path.join(data_dir, 'h5_files', f'{slide_id}.h5')
    with h5py.File(h5, 'r') as f:
        path_features = torch.from_numpy(f['features'][:])
    sim = path_features @ text_embs.T
    pooled = topj_pooling(sim)
    return torch.stack(list(pooled.values())).mean(dim=0).cpu().numpy().squeeze()


def alignment_score_rad(model, rad_id, data_dir, text_embs, cohort):
    rad_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    if cohort == 'dicom_2d':
        rad = pydicom.dcmread(os.path.join(data_dir, f'{rad_id}.dcm')).pixel_array.astype(np.float32)
        rad = rad / (rad.max() + 1e-5)
        rad = torch.from_numpy(rad).unsqueeze(0).repeat(3, 1, 1)
        mask = torch.from_numpy(nib.load(os.path.join(data_dir, f'{rad_id}_mask.nii')).get_fdata().astype(np.float32))
        rad = rad * mask.permute(2, 0, 1)
        rad = rad_transform(rad)
        with torch.no_grad():
            feat, _, _ = model(rad.unsqueeze(0))
        return (feat @ text_embs.t()).cpu().numpy().squeeze()

    raise NotImplementedError(f'cohort={cohort!r} not implemented for radiology alignment scoring.')


def find_max_similarity(i, sim_matrix, selected_idx):
    return max((abs(sim_matrix[i][j]) for j in selected_idx if j != i), default=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer', required=True)
    parser.add_argument('--modal', choices=['path', 'rad'], required=True)
    parser.add_argument('--candidate_json', required=True,
                        help='Output of generate.py.')
    parser.add_argument('--label_csv', required=True,
                        help='CSV with columns slide_id, radiology_id, label.')
    parser.add_argument('--data_split_dir', required=True,
                        help='10-fold splits directory containing splits_<i>.csv and splits_0_bool_label.csv.')
    parser.add_argument('--data_dir', required=True,
                        help='Directory of pathology h5 features (modal=path) or radiology images (modal=rad).')
    parser.add_argument('--cohort', default='dicom_2d', choices=['dicom_2d'])
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--num_concept_per_stage', type=int, default=32)
    parser.add_argument('--discriminative_ratio', type=float, default=0.8)
    parser.add_argument('--diversity_ratio', type=float, default=0.2)
    parser.add_argument('--conch_ckpt', default=os.environ.get('CONCH_CKPT'),
                        help='Path to the CONCH pytorch_model.bin (only required for modal=path).')
    parser.add_argument('--out_xlsx', required=True)
    args = parser.parse_args()

    all_concepts = load_json(args.candidate_json)
    label_df = pd.read_csv(args.label_csv)
    common_train_list, labels = get_common_train_list(args.data_split_dir, args.num_folds)

    selected_concepts = []
    for stage in STAGES:
        concepts = all_concepts[stage]
        if not concepts:
            continue

        model, text_embs = encode_concepts(args.modal, concepts, args.conch_ckpt)

        # Pre-compute pairwise similarity for diversity step.
        sim_within = (text_embs @ text_embs.T).cpu().numpy()

        # Per-image alignment vectors.
        rows = []
        for slide_id, label in zip(common_train_list, labels):
            if args.modal == 'path':
                scores = alignment_score_path(model, slide_id, args.data_dir, text_embs)
            else:
                rad_id = label_df.loc[label_df['slide_id'] == slide_id, 'radiology_id'].values[0]
                scores = alignment_score_rad(model, rad_id, args.data_dir, text_embs, args.cohort)
            rows.append((slide_id, label, scores))

        X = np.stack([r[2] for r in rows])
        y = np.array([r[1] for r in rows])

        # Stage 1: MI ranking.
        mi = mutual_info_classif(X, y)
        sorted_indices = np.argsort(-mi)
        keep_d = int(args.num_concept_per_stage * args.discriminative_ratio)
        discriminative = sorted_indices[:keep_d].tolist()

        # Stage 2: diversity selection from the remainder.
        not_selected = [i for i in range(len(concepts)) if i not in discriminative]
        max_sim = {i: find_max_similarity(i, sim_within, discriminative) for i in not_selected}
        sorted_max_sim = sorted(max_sim.items(), key=lambda kv: kv[1])
        keep_div = int(args.num_concept_per_stage * args.diversity_ratio)
        diverse = [i for i, _ in sorted_max_sim[:keep_div]]

        selected_concepts.extend(concepts[i] for i in discriminative + diverse)

    pd.DataFrame({'concepts': selected_concepts}).to_excel(args.out_xlsx, index=False)
    print(f'Wrote {len(selected_concepts)} concepts to {args.out_xlsx}')


if __name__ == '__main__':
    main()
