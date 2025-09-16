from email.policy import strict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from models.abmil import DAttention
from open_clip import create_model_from_pretrained, get_tokenizer
from conch.open_clip_custom import create_model_from_pretrained as create_conch
from conch.open_clip_custom import get_tokenizer as get_tokenizer_conch
from conch.open_clip_custom import tokenize as tokenize_conch
import pandas as pd
from models.custom_clip import CustomCLIP
from timm import create_model
import merlin
from models.MP_MoE import MPMoE

from transformers import BertTokenizer, BertModel
from CT_CLIP import CTViT, CTCLIP
from models.PIBD import PIBD
from models.model_motcat import MOTCAT_Surv
from models.custom_conch import CustomCONCH
from models.MP_MoE import MLP


class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True) # Bias is common
    def forward(self, features):
        projected_features = self.linear(features)
        normalized_features = F.normalize(projected_features, p=2, dim=-1)
        return normalized_features


class PaR(nn.Module):
    def __init__(self, n_classes=4, backbone_rad='BioMed', domain_align=False, concept_risk_align=False, args=None):
        super(PaR, self).__init__()
        if args.fuse != 'rad':
            self.mil_model = DAttention(input_dim=512)
        self.method = args.fuse
        self.PT = args.PT
        self.backbone = backbone_rad

        self.classifier = MLP(384, [args.cls_hidden_dim] * args.cls_layers, n_classes, dropout=0.1)

        if args.task == 'SYSU_SV':
            concepts_path = pd.read_excel('/home/yhchen/Documents/CONCH/gastric tumor_path_selected_concepts_1.xlsx')
            concepts_rad = pd.read_excel('/home/yhchen/Documents/CONCH/gastric tumor_rad_selected_concepts_1.xlsx')
        elif args.task == 'LGG_SV':
            concepts_path = pd.read_excel('/home/yhchen/Documents/CONCH/concept_generate/Lower-Grade Glioma (Grade II and III)_path_selected_concepts_2.xlsx')
            concepts_rad = pd.read_excel('/home/yhchen/Documents/CONCH/concept_generate/Lower-Grade Glioma (Grade II and III)_rad_selected_concepts_2.xlsx')
        elif args.task == 'GBM_SV':
            concepts_path = pd.read_excel('/home/yhchen/Documents/CONCH/concept_generate/Glioblastoma Multiforme_path_selected_concepts_2.xlsx')
            concepts_rad = pd.read_excel('/home/yhchen/Documents/CONCH/concept_generate/Glioblastoma Multiforme_rad_selected_concepts_2.xlsx')

        self.concepts_path = list(concepts_path['concepts'].values)
        self.concepts_rad = list(concepts_rad['concepts'].values)[:128]
        self.domain_align = domain_align

        if domain_align:
            self.path_proj = Projector(256, 128)
            self.rad_proj = Projector(128, 128)

        if concept_risk_align:
            self.concept_path_proj = Projector(512, 256)
            self.concept_rad_proj = Projector(512, 256)

        if not args.PT:
            if args.fuse != 'path':
                if backbone_rad == 'BioMed':
                    self.radiology_model, _ = create_model_from_pretrained(
                        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                    self.radiology_model = self.radiology_model.to('cuda')
                    # radiology tokenization
                    with torch.no_grad():
                        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                        texts = tokenizer(self.concepts_rad, context_length=128).to('cuda')
                        _, self.text_embs_rad, _ = self.radiology_model(None, texts)


            if args.fuse != 'rad':
                self.pathology_model, _ = create_conch('conch_ViT-B-16',
                                                       "/home/yhchen/Documents/CONCH/checkpoints/conch/pytorch_model.bin")

                # pathology tokenization
                with torch.no_grad():
                    tokenizer = get_tokenizer_conch()  # load tokenizer
                    text_tokens = tokenize_conch(texts=self.concepts_path, tokenizer=tokenizer)
                    self.text_embs_path = self.pathology_model.encode_text(text_tokens)

            if args.fuse == 'PIBD':
                self.fuse_model = PIBD(n_classes=n_classes, ratio_patch=0.5, sample_num=50, seed=1)
                self.fuse_model = self.fuse_model.to('cuda')

            if args.fuse == 'MOTCAT':
                model_dict = {'ot_reg': 0.1,
                              'ot_tau': 0.5,
                              'ot_impl': 'pot-uot-l2',
                              'fusion': 'concat',
                              'omic_sizes': [512],
                              'n_classes': 4}
                self.motcat = MOTCAT_Surv(**model_dict)
                self.motcat = self.motcat.to('cuda')

            if args.fuse == 'concat':
                self.classifier = MLP(1024, [args.cls_hidden_dim] * args.cls_layers, n_classes)
            elif args.fuse == 'path' or args.fuse == 'rad':
                self.classifier = MLP(512, [args.cls_hidden_dim] * args.cls_layers, n_classes)

        else:
            if args.fuse == 'rad':
                if backbone_rad == 'BioMed':
                    radiology_model, _ = create_model_from_pretrained(
                        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                    radiology_model = radiology_model.to('cpu')
                    self.radiology_model = CustomCLIP(radiology_model, 8, self.concepts_rad).to('cuda')
            elif args.fuse == 'path':
                pathology_model, _ = create_conch('conch_ViT-B-16',
                                                  "/home/yhchen/Documents/CONCH/checkpoints/conch/pytorch_model.bin")
                self.pathology_model = CustomCONCH(pathology_model, 8, self.concepts_path).to('cuda')
            elif args.fuse == 'MPMoE':
                radiology_model, _ = create_model_from_pretrained(
                    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.radiology_model = radiology_model.to('cuda')
                pathology_model, _ = create_conch('conch_ViT-B-16',
                                                  "/home/yhchen/Documents/CONCH/checkpoints/conch/pytorch_model.bin")
                pathology_model = pathology_model.to('cuda')
                self.mpmoe = MPMoE(pathology_model, self.radiology_model, args.n_ctx, self.concepts_path,
                                   self.concepts_rad).to('cuda')

    def get_sim(self):
        return self.sim_path, self.sim_rad

    def forward(self, path, rad, pyrad, y=None, c=None, st=None):

        if self.method == 'path' and not self.PT:
            path = self.mil_model(path)
            x = path
            x = self.classifier(x)
            Y_hat = torch.argmax(x)
            Y_prob = F.softmax(x)

        elif self.method == 'rad' and not self.PT:
            with torch.no_grad():
                if len(rad.shape) == 3:
                    rad, _, _ = self.radiology_model(rad.unsqueeze(0))
                else:
                    rad, _, _ = self.radiology_model(rad)
                    rad = torch.mean(rad, dim=0, keepdim=True)

            x = rad
            x = self.classifier(x)
            Y_hat = torch.argmax(x)
            Y_prob = F.softmax(x)

        elif self.method == 'concat':
            with torch.no_grad():
                if len(rad.shape) == 3:
                    rad, _, _ = self.radiology_model(rad.unsqueeze(0))
                else:
                    rad, _, _ = self.radiology_model(rad)
                    rad = torch.mean(rad, dim=0, keepdim=True)

            path = self.mil_model(path)
            x = torch.cat((rad, path), dim=1)
            x = self.classifier(x)
            Y_hat = torch.argmax(x)
            Y_prob = F.softmax(x)

        elif self.method == 'concat_concept' and not self.PT:
            with torch.no_grad():
                rad, _, _ = self.radiology_model(rad.unsqueeze(0))
                similarity_rad = rad @ self.text_embs_rad.t()
                x_1 = similarity_rad

            similarity_path = path @ self.text_embs_path.t()
            x_2 = similarity_path
            x_2 = self.mil_model(x_2)
            x = torch.cat((x_1, x_2), dim=1)
            x = self.classifier(x)
            Y_hat = torch.argmax(x)
            Y_prob = F.softmax(x)

        elif self.method == 'rad' and self.PT and self.backbone == 'BioMed':
            similarity = self.radiology_model(rad.unsqueeze(0))
            x = similarity
            x = self.classifier(x)
            Y_hat = torch.argmax(x)
            Y_prob = F.softmax(x)

        elif self.method == 'rad' and not self.PT:
            if self.backbone == 'Merlin':
                with torch.no_grad():
                    rad = rad[0, :, :]
                    rad = rad.unsqueeze(0)
                    rad = rad.unsqueeze(0)
                    rad = rad.unsqueeze(-1)
                    rad, _ = self.radiology_model.model.encode_image(rad)
                x = self.classifier(rad)
                Y_hat = torch.argmax(x)
                Y_prob = F.softmax(x)
            elif self.backbone == 'BioMed':
                with torch.no_grad():
                    rad, _, _ = self.radiology_model(rad.unsqueeze(0))

                x = rad
                x = self.classifier(x)
                Y_hat = torch.argmax(x)
                Y_prob = F.softmax(x)
        elif self.method == 'path' and self.PT:
            similarity = self.pathology_model(path)
            x = similarity
            x = self.mil_model(x)
            x = self.classifier(x)
            Y_hat = torch.argmax(x)
            Y_prob = F.softmax(x)

        elif self.method == 'MPMoE':
            if self.backbone == 'BioMed':
                with torch.no_grad():
                    if len(rad.shape) == 3:
                        rad, _, _ = self.radiology_model(rad.unsqueeze(0))
                    else:
                        rad, _, _ = self.radiology_model(rad)
                        rad = torch.mean(rad, dim=0, keepdim=True)
            path = self.mil_model(path)
            similarity, aux_loss = self.mpmoe(path, rad)
            self.sim_path, self.sim_rad = similarity

            similarity = torch.cat(similarity, dim=1)
            x = similarity
            x = self.classifier(x)
            Y_hat = torch.argmax(x)
            Y_prob = F.softmax(x)

            return x, Y_hat, Y_prob, aux_loss


        elif self.method == 'PIBD' and not self.PT:
            if self.backbone == 'BioMed':
                with torch.no_grad():
                    if len(rad.shape) == 3:
                        rad, _, _ = self.radiology_model(rad.unsqueeze(0))
                    else:
                        rad, _, _ = self.radiology_model(rad)
                        rad = torch.mean(rad, dim=0, keepdim=True)
            rad = rad.unsqueeze(0)
            path = path.unsqueeze(0)
            logits, IB_loss_proxy, proxy_loss, mimin_total, mimin_loss_total = self.fuse_model(path, rad, y, c)

            return logits, IB_loss_proxy, proxy_loss, mimin_total, mimin_loss_total

        elif self.method == 'MOTCAT' and not self.PT:
            if self.backbone == 'BioMed':
                with torch.no_grad():
                    if len(rad.shape) == 3:
                        rad, _, _ = self.radiology_model(rad.unsqueeze(0))
                    else:
                        rad, _, _ = self.radiology_model(rad)
                        rad = torch.mean(rad, dim=0, keepdim=True)
            x, Y_hat, Y_prob, A = self.motcat(x_path=path, x_rad1=rad.squeeze())

        return x, Y_hat, Y_prob, None