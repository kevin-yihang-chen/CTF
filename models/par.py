from email.policy import strict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from einops import rearrange
from models.sparse_moe import MoE
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

class PaR(nn.Module):
    def __init__(self, n_classes=4, method='MPMoE', PT=True, backbone_rad='BioMed'):
        super(PaR, self).__init__()
        if method != 'rad':
            self.mil_model = DAttention(input_dim=256)
        self.method = method
        self.PT = PT
        self.backbone = backbone_rad
        # radiology_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        # radiology_model = radiology_model.to('cpu')
        # radiology_model = create_model('convnext_tiny', pretrained=True, num_classes=n_classes)
        # self.radiology_model = radiology_model.to('cuda')

        # pathology_model, _ = create_conch('conch_ViT-B-16', "/home/yhchen/Documents/CONCH/checkpoints/conch/pytorch_model.bin")
        # pathology_model = pathology_model.to('cuda')
        # self.pathology_model = pathology_model

        # self.MoE = MoE()
        # self.radiology_model = create_model('convnext_tiny', pretrained=True, num_classes=n_classes).to('cuda')
        # self.classifier = nn.Sequential(
        #     # nn.Linear(512, n_classes),
        #     # nn.Linear(1024, n_classes),
        #     nn.Linear(1000, n_classes),
        # )
        self.classifier = nn.Sequential(
            # nn.Linear(1270, 512),
            # nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )
        concepts_path = pd.read_excel('./path_concepts_sysu.xlsx')
        concepts_rad = pd.read_excel('./concept_rad.xlsx')
        self.concepts_path = list(concepts_path['concepts'].values)
        self.concepts_rad = list(concepts_rad['concepts'].values)[:128]

        if not PT:
            if method != 'path':
                if backbone_rad == 'BioMed':
                    self.radiology_model,_ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                    self.radiology_model = self.radiology_model.to('cuda')
                    # radiology tokenization
                    with torch.no_grad():
                        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                        texts = tokenizer(self.concepts_rad, context_length=128).to('cuda')
                        _, self.text_embs_rad, _ = self.radiology_model(None,texts)

                if backbone_rad == 'Merlin':
                    model = merlin.models.Merlin()
                    model.eval()
                    model.cuda()
                    self.radiology_model = model
                    with torch.no_grad():
                        self.text_embs = []
                        for i in range(int(128 / 8)):
                            self.text_embs.append(
                                self.radiology_model.model.encode_text(self.concepts_rad[i * 8:(i + 1) * 8]))
                        self.text_embs = torch.cat(self.text_embs, dim=0)

            if method != 'rad':

                self.pathology_model, _ = create_conch('conch_ViT-B-16',
                                                       "/home/yhchen/Documents/CONCH/checkpoints/conch/pytorch_model.bin")

                # pathology tokenization
                with torch.no_grad():
                    tokenizer = get_tokenizer_conch()  # load tokenizer
                    text_tokens = tokenize_conch(texts=self.concepts_path, tokenizer=tokenizer)
                    self.text_embs_path = self.pathology_model.encode_text(text_tokens)

            if method == 'PIBD':
                self.fuse_model = PIBD(n_classes=n_classes, ratio_patch=0.5, sample_num=50, seed=1)
                self.fuse_model = self.fuse_model.to('cuda')

            if method == 'MOTCAT':
                model_dict = {'ot_reg': 0.1,
                 'ot_tau': 0.5,
                 'ot_impl': 'pot-uot-l2',
                 'fusion': 'concat',
                 'omic_sizes': [107, 93, 372, 93, 93, 512],
                 'n_classes': 4}
                self.motcat = MOTCAT_Surv(**model_dict)
                self.motcat = self.motcat.to('cuda')

        else:
            if method == 'rad':
                if backbone_rad == 'CTCLIP':
                    self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
                    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
                    text_encoder.resize_token_embeddings(len(self.tokenizer))
                    image_encoder = CTViT(
                        dim=512,
                        codebook_size=8192,
                        image_size=480,
                        patch_size=20,
                        temporal_patch_size=10,
                        spatial_depth=4,
                        temporal_depth=4,
                        dim_head=32,
                        heads=8
                    )
                    self.clip = CTCLIP(
                        image_encoder=image_encoder,
                        text_encoder=text_encoder,
                        dim_image=294912,
                        dim_text=768,
                        dim_latent=512,
                        extra_latent_projection=False,
                        use_mlm=False,
                        downsample_image_embeds=False,
                        use_all_token_embeds=False
                    ).to('cuda')
                    self.clip.load("/home/yhchen/Documents/CONCH/CT_CLIP/CT-CLIP_v2.pt")
                    self.text_tokens = self.tokenizer(self.concepts_rad, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to('cuda')
                    self.text_embs, _, _ = self.clip(self.text_tokens, None, 'cuda', return_latents=True)

                elif backbone_rad == 'BioMed':
                    radiology_model, _ = create_model_from_pretrained(
                        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                    radiology_model = radiology_model.to('cpu')
                    self.radiology_model = CustomCLIP(radiology_model, 8, self.concepts_rad).to('cuda')
            elif method == 'path':
                pathology_model, _ = create_conch('conch_ViT-B-16',
                                                  "/home/yhchen/Documents/CONCH/checkpoints/conch/pytorch_model.bin")
                self.pathology_model = CustomCONCH(pathology_model, 8, self.concepts_path).to('cuda')
            elif method == 'MPMoE':
                self.radiology_model, _ = create_model_from_pretrained(
                    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.radiology_model = self.radiology_model.to('cuda')
                pathology_model, _ = create_conch('conch_ViT-B-16',
                                                  "/home/yhchen/Documents/CONCH/checkpoints/conch/pytorch_model.bin")
                self.mpmoe = MPMoE(pathology_model, 8, self.concepts_path).to('cuda')



    def forward(self, path, rad, pyrad, y=None, c=None, st=None):
        
        # pathology_output = self.mil_model(path)

        # import ipdb; ipdb.set_trace()
        # rad = rad.unsqueeze(0)
        # rad = rad.permute(2, 0, 1)
        # rad = self.preprocess(rad)
        # import ipdb; ipdb.set_trace()
        # tokenizer = get_tokenizer()  # load tokenizer
        # text_tokens = tokenize(texts=self.concepts, tokenizer=tokenizer).to('cuda')  # tokenize the text
        # self.text_embs = self.model.encode_text(text_tokens)

        # x_1 = self.radiology_model(rad.unsqueeze(0))
        if self.method=='concat' and not self.PT:
            with torch.no_grad():
                rad, _, _ = self.radiology_model(rad.unsqueeze(0))
                similarity_rad = rad @ self.text_embs_rad.t()
                x_1 = similarity_rad
            # with torch.no_grad():
            # similarity = self.pathology_model(pathology_output)
                # rad = rad[0,:,:]
                # rad = rad.unsqueeze(0).unsqueeze(1).repeat(1, 10, 1, 1)
                # text_tokens = self.tokenizer("", return_tensors="pt", padding="max_length", truncation=True, max_length=200).to(
                #     'cuda')
                # text_latents, image_latents, _ = self.clip(self.text_tokens, rad.unsqueeze(0), 'cuda', return_latents=True)

                # radiology_output, _, _ = self.radiology_model(rad.unsqueeze(0))

            # radiology_output = self.radiology_model(rad.unsqueeze(0))
            # output = self.MoE(pathology_output, radiology_output)
            # return pathology_output, radiology_output
            # x = radiology_output
            # x = torch.cat((pathology_output, radiology_output), dim=1)
            # x = pathology_output
            # st_bulk = st.mean(dim=0).unsqueeze(0)

            similarity_path = path @ self.text_embs_path.t()
            # similarity = torch.cosine_similarity(image_latents, text_latents, dim=1)
            x_2 = similarity_path
            x_2 = self.mil_model(x_2)
            # x = image_latents
            # x = st_bulk
            # import ipdb; ipdb.set_trace()
            x = torch.cat((x_1, x_2), dim=1)
            # x = x_1
            x = self.classifier(x)
            Y_hat = torch.argmax(x)
            Y_prob = F.softmax(x)

        elif self.method == 'rad' and self.PT and self.backbone=='BioMed':
            similarity, aux_loss = self.radiology_model(rad.unsqueeze(0))
            x = similarity
            x = self.classifier(x)
            Y_hat = torch.argmax(x)
            Y_prob = F.softmax(x)

        elif self.method == 'rad' and not self.PT:
            if self.backbone=='Merlin':
                with torch.no_grad():
                    rad = rad[0, :, :]
                    rad = rad.unsqueeze(0)
                    rad = rad.unsqueeze(0)
                    rad = rad.unsqueeze(-1)
                    rad, _ = self.radiology_model.model.encode_image(rad)
                    # similarity = rad @ self.text_embs.t()
                    # x = similarity
                x = self.classifier(rad)
                Y_hat = torch.argmax(x)
                Y_prob = F.softmax(x)
            elif self.backbone=='BioMed':
                with torch.no_grad():
                    rad, _, _ = self.radiology_model(rad.unsqueeze(0))

                # pyrad = pyrad.unsqueeze(0).to(torch.float32)
                # x = torch.cat([rad,pyrad],dim=1)
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
                    rad, _, _ = self.radiology_model(rad.unsqueeze(0))
            similarity, aux_loss = self.mpmoe(path,rad)
            x = similarity
            x = self.mil_model(x)
            x = self.classifier(x)
            Y_hat = torch.argmax(x)
            Y_prob = F.softmax(x)

            return x, Y_hat, Y_prob, aux_loss


        elif self.method == 'PIBD' and not self.PT:
            if self.backbone == 'BioMed':
                with torch.no_grad():
                    rad, _, _ = self.radiology_model(rad.unsqueeze(0))
            rad = rad.unsqueeze(0)
            path = path.unsqueeze(0)
            logits, IB_loss_proxy, proxy_loss, mimin_total, mimin_loss_total = self.fuse_model(path, rad, y, c)

            return logits, IB_loss_proxy, proxy_loss, mimin_total, mimin_loss_total

        elif self.method == 'MOTCAT' and not self.PT:
            pyrad = pyrad
            rad = rad
            pyrad1, pyrad2, pyrad3, pyrad4, pyrad5 = pyrad[:107].to(torch.float32), pyrad[107:200].to(torch.float32), pyrad[200:572].to(torch.float32), pyrad[572:665].to(torch.float32), pyrad[665:].to(torch.float32)
            # pyrad1, pyrad2, pyrad3, pyrad4, pyrad5 = pyrad1.unsqueeze(0), pyrad2.unsqueeze(0), pyrad3.unsqueeze(0), pyrad4.unsqueeze(0), pyrad5.unsqueeze(0)
            if self.backbone == 'BioMed':
                with torch.no_grad():
                    rad, _, _ = self.radiology_model(rad.unsqueeze(0))
            x,  Y_hat, Y_prob, A = self.motcat(x_path=path, x_rad1=pyrad1, x_rad2=pyrad2,
                                         x_rad3=pyrad3, x_rad4=pyrad4, x_rad5=pyrad5, x_rad6=rad.squeeze())

        return x, Y_hat, Y_prob, None