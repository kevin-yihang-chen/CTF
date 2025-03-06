import torch
import torch.nn as nn
import torch.nn.functional as F

from CT_CLIP import CTViT, CTCLIP
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
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
clip = CTCLIP(
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

clip.load("/home/yhchen/Documents/CONCH/CT_CLIP/CT-CLIP_v2.pt")


text_tokens = tokenizer(concepts_rad, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to('cuda')

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.text_transformer
        # self.positional_embedding = clip_model.positional_embedding
        # self.positional_embedding = clip_model.transformer.embeddings.position_embeddings
        # self.ln_final = clip_model.ln_final
        # self.ln_final = clip_model.pooler
        # self.text_projection = clip_model.text_projection
        self.text_projection = clip_model.to_text_latent
        # self.dtype = clip_model.ln_final.weight.dtype
        self.dtype = clip_model.dtype
        # self.cls_emb = clip_model.cls_emb
        # self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

    def forward(self, prompts, mask):

        x = prompts
        # x = prompts + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # mask = (tokenized_prompts != 0).long()
        mask = mask.to('cuda')
        x = self.transformer(inputs_embeds=x,attention_mask=mask)
        x = x[0]
        x = x[:, :] if x.ndim == 3 else x
        x = x[:, 0, :]

        # x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x,mask).type(self.dtype)
        x = self.text_projection(x)

        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class CustomCTCLIP(nn.Module):
    def __init__(self, clip_model, n_ctx, text):
        super().__init__()
        self.prompt_learner = LPromptLearner(clip_model,n_ctx,text)
        self.image_encoder = clip_model.visual_transformer
        self.image_proj = clip_model.to_image_latent
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.temperature
        self.dtype = clip_model.text.transformer.dtype
        self.text = text
        # self.prompts, self.tokenized_prompts = self.prompt_learner(text)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = torch.mean(image_features, dim=1)
        image_features = image_features.view(image_features.shape[0], -1)
        image_features = image_features[:, :] if image_features.ndim == 3 else image_features
        image_features = self.image_proj(image_features)

        prompts, mask = self.prompt_learner(self.text)

        text_features = self.text_encoder(prompts, mask)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

class LPromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx, texts):
        super().__init__()
        self.clip_model = clip_model.text_transformer
        dtype = self.clip_model.dtype
        # dtype = self.clip_model.ln_final.weight.dtype
        ctx_dim = 768
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial text context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        # self.register_buffer("ctx", ctx_vectors)
        self.ctx = nn.Parameter(ctx_vectors) # (n_ctx, ctx_dim)
        self.n_ctx = n_ctx
        self.texts = texts


        classnames = [name.replace("_", " ") for name in texts]
        prompts = [self.prompt_prefix + " " + name for name in classnames]
        tokenized_prompts = torch.cat([(tokenizer(p, return_tensors="pt", padding="max_length", truncation=True, max_length=128)).input_ids for p in prompts])

        with torch.no_grad():
            embedding = self.clip_model.embeddings.word_embeddings(tokenized_prompts).type(torch.float)

        self.mask = (tokenized_prompts != 0).long()
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def forward(self, texts=None):
        ctx = self.ctx
        n_cls = len(self.texts)


        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        # embedding = embedding.unsqueeze(0)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts, self.mask
