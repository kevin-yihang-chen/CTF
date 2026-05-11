"""Concept Tuning and Fusing (CTF) wrapper.

This module wires the frozen radiology / pathology vision-language foundation
models together with the GCSP prompt mechanism (``models/MP_MoE.py``) and a
small concept-score classifier.

Vision encoders are kept frozen at all times; only the prompt modules and the
classifier are trainable.

Class is named ``PaR`` for backward compatibility with the original research
code.  In the paper this corresponds to the full ``CTF`` model.
"""

import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_clip import create_model_from_pretrained
from conch.open_clip_custom import create_model_from_pretrained as create_conch

from models.abmil import DAttention
from models.MP_MoE import MPMoE, MLP


# Resolve the CONCH checkpoint via an environment variable so the repo is
# portable. Falls back to a sensible relative path under ``checkpoints/``.
CONCH_CKPT = os.environ.get(
    'CONCH_CKPT',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'conch', 'pytorch_model.bin'),
)

CONCEPTS_DIR = os.environ.get(
    'CTF_CONCEPTS_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'concepts'),
)


# Concept Excel files are looked up under ``CONCEPTS_DIR`` using the naming
# convention ``<TASK>_path.xlsx`` and ``<TASK>_rad.xlsx``.  An example for
# ``LGGGBM_Grading`` (TCGA-GBMLGG grading task) ships under ``concepts/``;
# generate the others yourself via ``concept_generate/``.
def _concept_files(task):
    return f'{task}_path.xlsx', f'{task}_rad.xlsx'


def _load_concepts(task, n_concept, n_concept_rad=None):
    path_file, rad_file = _concept_files(task)
    path_path = os.path.join(CONCEPTS_DIR, path_file)
    rad_path = os.path.join(CONCEPTS_DIR, rad_file)
    if not os.path.isfile(path_path) or not os.path.isfile(rad_path):
        raise FileNotFoundError(
            f'Could not find concept Excel files for task {task!r}.\n'
            f'  expected: {path_path}\n            {rad_path}\n'
            f'Generate them with concept_generate/ or set $CTF_CONCEPTS_DIR.'
        )
    path_df = pd.read_excel(path_path)
    rad_df = pd.read_excel(rad_path)
    concepts_path = list(path_df['concepts'].values)[:n_concept]
    concepts_rad = list(rad_df['concepts'].values)[:(n_concept_rad or n_concept)]
    return concepts_path, concepts_rad


class PaR(nn.Module):
    """CTF model: frozen FMs + GCSP prompts + concept-score classifier."""

    def __init__(self, n_classes=4, args=None):
        super().__init__()
        if args is None:
            raise ValueError('args is required')

        self.method = args.fuse  # kept for forward-pass dispatch; only "MPMoE" is supported.
        if self.method != 'MPMoE':
            raise ValueError(
                f'This release only supports the CTF (fuse=MPMoE) configuration; got {self.method!r}.'
            )

        # Pathology bag-level pooling on top of frozen patch features.
        self.mil_model = DAttention(input_dim=512)

        # Concept pool. Radiology uses 128 concepts by default to match the paper.
        self.concepts_path, self.concepts_rad = _load_concepts(
            args.task, n_concept=args.n_concept, n_concept_rad=128,
        )

        # Final classifier: input dim = |C_rad| + 128 (pathology MIL output).
        self.classifier = MLP(
            args.n_concept + 128,
            [args.cls_hidden_dim] * args.cls_layers,
            n_classes,
            dropout=0.1,
        )

        # Frozen radiology FM (BiomedCLIP).
        radiology_model, _ = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        )
        self.radiology_model = radiology_model.to('cuda')

        # Frozen pathology FM (CONCH).
        pathology_model, _ = create_conch('conch_ViT-B-16', CONCH_CKPT)
        pathology_model = pathology_model.to('cuda')

        # GCSP / cross-domain prompt fuser.
        self.mpmoe = MPMoE(
            pathology_model, self.radiology_model,
            self.concepts_path, self.concepts_rad, args,
        ).to('cuda')

    def forward(self, path, rad, pyrad=None, y=None, c=None, st=None):
        # Frozen radiology forward — accepts a single 2D slice or a stack.
        with torch.no_grad():
            if rad.dim() == 3:
                rad, _, _ = self.radiology_model(rad.unsqueeze(0))
            else:
                rad, _, _ = self.radiology_model(rad)
                rad = torch.mean(rad, dim=0, keepdim=True)

        # Pathology pooled embedding from the frozen patch features.
        path = self.mil_model(path)

        # GCSP-tuned per-domain concept scores + load-balance auxiliary loss.
        similarity, aux_loss = self.mpmoe(path, rad)
        z = torch.cat(similarity, dim=1)

        logits = self.classifier(z)
        Y_hat = torch.argmax(logits)
        Y_prob = F.softmax(logits, dim=-1)
        return logits, Y_hat, Y_prob, aux_loss
