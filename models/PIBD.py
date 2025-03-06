import torch
import torch.nn as nn
import torch.nn.functional as F
from .disentangle_transformer import MITransformerLayer
from .club import MIEstimator
from utils.loss_func import NLLSurvLoss
import numpy as np
BAG_SIZE = 512

'''KL divergence between two normal distributions'''
def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5


'''Product of Experts: getting the joint distribution of multiple modalities.'''
class PoE(nn.Module):
    def __init__(self,modality_num=2, sample_num=50, seed = 1):
        super(PoE, self).__init__()

        self.sample_num = sample_num
        self.seed = seed

        phi = torch.ones(modality_num, requires_grad=True)
        self.phi = torch.nn.Parameter(phi)

    def forward(self, mu_list, var_list, eps=1e-8):
        t_sum = 0
        mu_t_sum = 0

        alpha = F.softmax(self.phi, dim=0)

        for idx, (mu, var) in enumerate(zip(mu_list, var_list)):
            T = 1 / (var + eps)

            t_sum += alpha[idx] * T
            mu_t_sum += mu * alpha[idx] * T

        mu = mu_t_sum / t_sum
        var = 1 / t_sum

        dim = mu.shape[1]
        batch_size = mu.shape[0]
        eps = self.gaussian_noise(samples=(batch_size, self.sample_num), k=dim, seed = self.seed)
        poe_features = torch.unsqueeze(mu, dim=1) + torch.unsqueeze(var, dim=1) * eps

        return poe_features

    def gaussian_noise(self, samples, k, seed):
        # works with integers as well as tuples
        if self.training:
            return torch.normal(torch.zeros(*samples, k), torch.ones(*samples, k)).cuda()
        else:
            return torch.normal(torch.zeros(*samples, k), torch.ones(*samples, k),
                                generator=torch.manual_seed(seed)).cuda() # must be the same seed as the training seed



'''Prototypical Information Bottleneck'''
class PIB(nn.Module):
    def __init__(self,
                 x_dim,
                 z_dim = 256,
                 beta = 1e-2,
                 sample_num = 50,
                 topk = 256,
                 num_classes = 4,
                 seed = 1):
        super(PIB, self).__init__()

        self.beta = beta
        self.sample_num = sample_num
        self.topk = topk
        self.num_classes = num_classes
        self.seed = seed
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, z_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim*2, z_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim*2, z_dim),
        )

        # decoder a simple logistic regression as in the paper
        self.decoder_logits = nn.Linear(z_dim, num_classes)

        # design proxies for histology images
        self.proxies = nn.Parameter(torch.empty([num_classes*2, z_dim*2]))
        torch.nn.init.xavier_uniform_(self.proxies, gain=1.0)
        # self.proxies_dict = {"0,0":0,"0,1":1,"0,2":2,"0,3":3,"1,0":4,"1,1":5,"1,2":6,"1,3":7} #"censor,laberl":index
        self.proxies_dict = {"0":0,"1":1,"2":2,"3":3}
    def gaussian_noise(self, samples, K, seed):
        # works with integers as well as tuples
        if self.training:
            return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K)).cuda()
        else:
            return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K),
                                generator=torch.manual_seed(seed)).cuda() # must be the same seed as the training seed

    def encoder_result(self, x):

        encoder_output = self.encoder(x)

        return encoder_output

    def encoder_proxies(self):
        mu_proxy = self.proxies[:, :self.z_dim]
        sigma_proxy = torch.nn.functional.softplus(self.proxies[:, self.z_dim:]) # Make sigma always positive

        return mu_proxy, sigma_proxy

    def forward(self,x, y=None, c=None):
        #feature number
        feature_num = x.shape[1]

        # get z from encoder
        z = self.encoder_result(x)

        # get mu and sigma from proxies
        mu_proxy, sigma_proxy = self.encoder_proxies()

        # sample
        eps_proxy = self.gaussian_noise(samples=([self.num_classes*2, self.sample_num]), K=self.z_dim,seed = self.seed)
        z_proxy_sample = mu_proxy.unsqueeze(dim=1) + sigma_proxy.unsqueeze(dim=1) * eps_proxy
        z_proxy = torch.mean(z_proxy_sample, dim=1)

        # get attention maps
        z_norm = F.normalize(z,dim=2) # normalize z in the feature dimension
        z_proxy_norm = torch.unsqueeze(F.normalize(z_proxy),dim=0)

        att = torch.matmul(z_norm, torch.transpose(z_proxy_norm, 1, 2))

        if y is None and c is None:
            '''validating and testing'''
            if self.topk != None:
                # get the proxy with the highest attention map
                att_unbind_proxy = torch.cat(torch.unbind(att, dim=1), dim=1)
                _, att_topk_proxy_idx = torch.topk(att_unbind_proxy, self.topk, dim=1)
                att_topk_proxy_idx = att_topk_proxy_idx % (self.num_classes*2)
            else:
                att_unbind_proxy = torch.cat(torch.unbind(att, dim=1), dim=1)
                _, att_topk_proxy_idx = torch.topk(att_unbind_proxy, 1, dim=1)

            # get the positive proxy index
            positive_proxy_idx,_ = torch.mode(att_topk_proxy_idx, dim=1)
            positive_proxy_idx = positive_proxy_idx.unsqueeze(1).repeat(1, self.z_dim).unsqueeze(dim=1)#batch_size, index,z_dim

            proxy_loss = None

        else:
            '''training'''
            # get proxy_index for each sample
            proxy_indices = [self.proxies_dict[str(int(y_item))] for c_item, y_item in
                             zip(c, y)]
            proxy_indices = torch.tensor(proxy_indices).long().cuda()
            mask = torch.zeros_like(att, dtype=torch.bool).cuda()
            mask[torch.arange(att.size(0)), :, proxy_indices] = True
            # get att_positive for each sample
            att_positive = torch.masked_select(att, mask).view(att.size(0), att.size(1), 1)
            # get att_negative for each sample
            att_negative = torch.masked_select(att, ~mask).view(att.size(0), att.size(1), -1)

            # calculate proxy loss
            if self.topk != None:
                att_topk_positive, att_topk_idx_positive = torch.topk(att_positive.squeeze(dim=2), self.topk, dim=1)
                att_topk_negative, _ = torch.topk(att_negative, self.topk, dim=1)
                att_positive_mean = torch.mean(att_topk_positive, dim=1)
                att_negative_mean = torch.mean(torch.mean(att_topk_negative, dim=1),dim=1)
                proxy_loss = -(att_positive_mean-att_negative_mean).mean()
            else:
                att_negative_mean = torch.mean(torch.mean(att_negative, dim=1), dim=1)
                proxy_loss = -(att_positive.squeeze(dim=2)-att_negative_mean).mean()

            positive_proxy_idx = proxy_indices.unsqueeze(1).repeat(1, self.z_dim).unsqueeze(dim=1)

        # Gather mu_proxy and sigma_proxy for each sample
        mu_proxy_repeat = mu_proxy.repeat(x.shape[0], 1, 1)  # batch_size, num_classes*2, z_dim
        sigma_proxy_repeat = sigma_proxy.repeat(x.shape[0], 1, 1)
        mu_topk = torch.gather(mu_proxy_repeat, 1, positive_proxy_idx).squeeze(dim=1)
        sigma_topk = torch.gather(sigma_proxy_repeat, 1, positive_proxy_idx).squeeze(dim=1)

        if self.topk != None:
            att_unbind = torch.cat(torch.unbind(att, dim=2), dim=1)
            # get topk z features from attention maps
            att_topk, att_topk_idx = torch.topk(att_unbind, self.topk, dim=1)
            att_topk_idx = att_topk_idx % feature_num
            # get topk z features from z
            z_topk = torch.gather(z, 1, att_topk_idx.unsqueeze(dim=2).repeat(1, 1, self.z_dim))
        else:
            z_topk = z
        decoder_logits_proxy = torch.mean(self.decoder_logits(z_proxy_sample), dim=1)

        return decoder_logits_proxy, mu_proxy, sigma_proxy, z_topk, mu_topk, sigma_topk, proxy_loss





'''Prototypical Information Bottleneck and Disentanglement'''
class PIBD(nn.Module):
    def __init__(
            self,
            n_classes,
            ratio_patch,
            sample_num,
            seed,
            rad_dim=512,
            path_dim=512,
            latent_dim=256,
    ):
        super(PIBD, self).__init__()

        # ---> general props
        self.num_classes = n_classes
        # self.args = args

        # ---> path props
        self.path_embedding_dim = path_dim

        # ---> prototype props
        self.prtp_dim = latent_dim

        # ---> rad props
        self.rad_embedding_dim = rad_dim

        self.bag_size = BAG_SIZE
        self.topk_patch = int(self.bag_size * ratio_patch)
        self.PIB_path = PIB(self.path_embedding_dim, self.prtp_dim, num_classes=self.num_classes,
                                     topk=self.topk_patch, sample_num=sample_num, seed=seed)
        self.PIB_rad = PIB(self.rad_embedding_dim,self.prtp_dim, num_classes=self.num_classes,
                                      topk=None,sample_num=sample_num, seed=seed)

        # ---> Disentanglement
        self.PID = MITransformerLayer(
            dim=self.prtp_dim,
            num_heads=4,
            mlp_ratio=1.,
            qkv_bias=True,
            attn_drop=0.1,
            proj_drop=0.1,
            drop_path=0.1,
        )

        # ---> classifier
        self.to_logits = nn.Sequential(
            nn.Linear(self.prtp_dim * 3, self.prtp_dim),
            nn.ReLU(),
            nn.Linear(self.prtp_dim, self.num_classes),
        )

        # ---> PoE
        self.PoE = PoE(modality_num=2, sample_num=sample_num, seed=seed)

        # --->CLUB
        self.CLUB = MIEstimator(self.prtp_dim)

        # SurvLoss
        self.loss_surv = NLLSurvLoss(alpha=0.5)

    def forward(self, path, rad, y=None, c=None, return_attn=False):
        # path: (batch_size, num_patches, path_embedding_dim)
        # rad: (batch_size, rad_embedding_dim)

        decoder_logits_proxy_rad, mu_proxy_rad, sigma_proxy_rad, h_rad_bag, \
        mu_topk_rad, sigma_topk_rad, proxy_loss_rad = self.PIB_rad(rad, y, c)

        num_patches = path.size(1)
        self.bag_size = min(num_patches, self.bag_size)
        iter_num = num_patches // self.bag_size # use subset of patches, and average the results

        logits = torch.empty((iter_num, path.size(0), self.num_classes)).cuda()
        mimin_total,mimin_loss_total,IB_loss_proxy,proxy_loss = 0.0,0.0,0.0,0.0

        for i in range(iter_num):
            # strategy 1 random sampling from patches for each bag, which can reduce the computation
            idx = torch.randperm(num_patches)[:self.bag_size]
            h_wsi_bag = path[:, idx, :]  #(batch_size, bag_num, path_embedding_dim)

            # ---> get wsi embeddings
            decoder_logits_proxy_path, mu_proxy_path, sigma_proxy_path, h_wsi_bag, mu_topk_path, sigma_topk_path,\
            proxy_loss_path = self.PIB_path(h_wsi_bag,y=y,c=c)

            # ---> PoE
            mu_list = [mu_topk_path, mu_topk_rad]
            var_list = [sigma_topk_path, sigma_topk_rad]
            poe_features = self.PoE(mu_list, var_list)
            poe_features = torch.mean(poe_features, dim=1).unsqueeze(dim=1)

            poe_embed = poe_features.expand(h_wsi_bag.shape[0], 1,-1)

            if return_attn:
                histology, pathways, global_embed, attns = self.PID(h_wsi_bag, h_rad_bag, poe_embed,return_attn)
            else:
                histology, pathways, global_embed, = self.PID(h_wsi_bag, h_rad_bag, poe_embed, return_attn)

            # ----> aggregate
            histology = torch.mean(histology, dim=1)
            pathways = torch.mean(pathways, dim=1)
            global_embed = torch.mean(global_embed, dim=1)

            mimin = self.CLUB(histology, pathways, global_embed)
            mimin_loss = self.CLUB.learning_loss(histology, pathways, global_embed)

            logits[i] = self.to_logits(torch.cat([histology, pathways, global_embed], dim=-1))

            if self.training:
                mimin_total += mimin
                mimin_loss_total += mimin_loss
                IB_loss_proxy += 0.05 * self.get_loss_proxy(decoder_logits_proxy_path, self.loss_surv) \
                                 + 0.05 * self.get_loss_proxy(decoder_logits_proxy_rad, self.loss_surv) \
                                 + 0.01 * self.get_KL_loss(mu_proxy_path, sigma_proxy_path) \
                                 + 0.01 * self.get_KL_loss(mu_proxy_rad, sigma_proxy_rad)
                proxy_loss += (proxy_loss_path + proxy_loss_rad)

        IB_loss_proxy /= iter_num
        proxy_loss /= iter_num
        mimin_total /= iter_num
        mimin_loss_total /= iter_num

        # if self.training:
        #     print('mimin:{:.4f},mimin_loss:{:.4f}'.format(mimin_total, mimin_loss_total))

        logits = torch.mean(logits, dim=0)

        if return_attn:
            return logits, attns
        else:
            return logits, IB_loss_proxy, proxy_loss, mimin_total, mimin_loss_total

    def get_loss_proxy(self, x, loss):
        censor = torch.empty([self.num_classes*2]).cuda()
        for i in range(self.num_classes):
            censor[i] = 0
            censor[i + self.num_classes] = 1
        y = torch.arange(0, self.num_classes).repeat(2).cuda()
        loss_proxy = loss(h=x, y=y, t=None, c=censor)

        return loss_proxy

    def get_KL_loss(self, mu, std):
        '''
        :param mu: [batch_size, dimZ]
        :param std: [batch_size, dimZ]
        :return:
        '''
        # KL divergence between prior and posterior
        prior_z_distr = torch.zeros_like(mu), torch.ones_like(std)
        encoder_z_distr = mu, std

        I_zx_bound = torch.mean(KL_between_normals(encoder_z_distr, prior_z_distr))

        return torch.mean(I_zx_bound)