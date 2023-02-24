import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from torch.autograd import Variable
import torchvision.models as models
import numpy as np

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from einops import rearrange
from src.hpenalty import hessian_penalty
from src.qkcodebook import QKCodebook
from src.utils import unique_sampling_fn, get_euclidian_distance, sorting_idxs, get_cosine_distance, get_cb_variance
from torch.distributions import Categorical


class BaseVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, 
                        embedding_dim,
                        nhidden,
                        codebook_dim = 8,
                        commitment_cost=0.25,
                        usage_threshold=1.0e-9,
                        qk=False,
                        cosine=False,
                        gumble=False,
                        temperature=1.0,
                        kld_scale=1.0):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._usage_threshold = usage_threshold
        self.codebook_dim = codebook_dim
        self.nhidden = nhidden
        self._cosine = cosine
        self.qk=qk


        requires_projection = codebook_dim != embedding_dim
        self.project_in = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(embedding_dim, codebook_dim)) if requires_projection else nn.Identity()
        self.project_out = nn.Sequential(nn.Linear(codebook_dim, embedding_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(embedding_dim, embedding_dim)) if requires_projection else nn.Identity()

        self.norm_in  = nn.LayerNorm(codebook_dim)
        self.norm_out  = nn.LayerNorm(embedding_dim)



        self._embedding = nn.Embedding(self._num_embeddings, codebook_dim)
        self._get_distance = get_euclidian_distance
        self.loss_fn = F.mse_loss
        


        if self._cosine:
            sphere = Hypersphere(dim=codebook_dim - 1)
            points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=self._num_embeddings))
            self._embedding.weight.data.copy_(points_in_manifold)
            self._get_distance = get_cosine_distance
            self.loss_fn = lambda x1,x2: 2.0*(1 - F.cosine_similarity(x1, x2).mean())


        self.data_mean = 0
        self.data_std = 0

        self.register_buffer('_usage', torch.ones(self._num_embeddings), persistent=False)


        self.kld_scale = kld_scale
        # ======================
        # QK codebook init
        if self.qk:
            self.qkclass = QKCodebook(self.nhidden, self._num_embeddings)

        # ======================
        # Gumble parameters
        self.gumble = gumble
        if self.gumble:
            self.temperature = temperature
            self.gumble_proj = nn.Sequential(nn.Linear(codebook_dim, num_embeddings))


    def update_usage(self, min_enc):
        self._usage[min_enc] = self._usage[min_enc] + 1  # if code is used add 1 to usage
        self._usage /= 2 # decay all codes usage

    def reset_usage(self):
        self._usage.zero_() #  reset usage between epochs

    def random_restart(self):
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self._usage < self._usage_threshold).squeeze(1)
        useful_codes = torch.nonzero(self._usage > self._usage_threshold).squeeze(1)
        N = self.data_std.shape[0]

        if len(dead_codes) > 0:
            eps = torch.randn((len(dead_codes), self._embedding_dim)).to(self._embedding.weight.device)
            rand_codes = eps*self.data_std.unsqueeze(0).repeat(len(dead_codes), 1) +\
                                self.data_mean.unsqueeze(0).repeat(len(dead_codes), 1)

            with torch.no_grad():
                self._embedding.weight[dead_codes] = rand_codes

            self._embedding.weight.requires_grad = True


    def vq_sample(self, features, 
                        hard = False, 
                        idxs=None, 
                        MCsamples = 1, 
                        final=False):
        input_shape = features.shape
        
        def _min_encoding_(distances):
            # encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            sampled_dist, encoding_indices = torch.min(distances, dim=1)
            encoding_indices = encoding_indices.view(input_shape[0], -1)
            sampled_dist = sampled_dist.view(input_shape[0], -1)

            # import pdb;pdb.set_trace()
            encoding_indices = encoding_indices.view(-1)
            # encoding_indices = encoding_indices[torch.argsort(sampled_dist, dim=1, descending=False).view(-1)]
            encoding_indices = encoding_indices.unsqueeze(1)
            return encoding_indices

        # Flatten features
        features = features.view(-1, self.codebook_dim)


        # Update prior with previosuly wrt principle components
        # This will ensure that codebook indicies that are not in idxs wont be sampled
        # hacky way need to find a better way of implementing this
        key_codebook = self._embedding.weight.clone()
        if not (idxs is None):
            idxs = torch.unique(idxs).reshape(-1, 1)
            for i in range(self._num_embeddings):
                if not (i in idxs): 
                    key_codebook[i, :] = 2*torch.max(features) 


        # Quantize and unflatten
        distances = self._get_distance(features, key_codebook)
        encoding_indices = _min_encoding_(distances)

        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=features.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight)
    
        # =========
        slots = None
        # slot sampling
        if self.qk:
            slots = self.qkclass.sample_slots(quantized, 
                                                encodings, 
                                                input_shape,
                                                MCsamples=MCsamples)


        quantized = self.project_out(quantized)
        quantized = self.norm_out(quantized)
        quantized = quantized.view(input_shape[0], input_shape[1], -1)
        return quantized, encoding_indices, encodings, slots, None


    def gumble_sample(self, features, 
                        hard=False, 
                        idxs = None, 
                        MCsamples = 1, 
                        final=False):
        
        input_shape = features.shape

        # force hard = True when we are in eval mode, as we must quantize
        logits = self.gumble_proj(features)

        # Update prior with previosuly wrt principle components
        if not (idxs is None):
            mask_idxs = torch.zeros_like(logits)

            for b in range(input_shape[0]):
                mask_idxs[b, :, idxs[b]] = 1
            logits = mask_idxs*logits


        soft_one_hot = F.gumbel_softmax(logits, 
                                        tau=self.temperature, 
                                        dim=1, hard=hard)
        
        quantized = torch.einsum('b t k, k d -> b t d', 
                                        soft_one_hot, 
                                        self._embedding.weight)            



        encoding_indices = soft_one_hot.argmax(dim=-1).view(-1, 1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=features.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        
        # ==========================
        slots = None
        # slot sampling
        if self.qk:
            slots = self.qkclass.sample_slots(quantized, 
                                                    encodings, 
                                                    input_shape,
                                                    MCsamples=MCsamples)

        quantized = self.project_out(quantized)
        quantized = self.norm_out(quantized)
        return quantized, encoding_indices, encodings, slots, logits



    def sample(self, features, 
                        hard=False, 
                        idxs = None,
                        MCsamples = 1, 
                        from_train = False):
        if not from_train:
            features = self.project_in(features)
            # layer norm on features
            features = self.norm_in(features)


        if self.gumble:
            return self.gumble_sample(features, 
                                        MCsamples=MCsamples,
                                        hard=hard, 
                                        idxs=idxs)
        else:
            return self.vq_sample(features, 
                                    MCsamples=MCsamples,
                                    hard=hard, 
                                    idxs=idxs)


    def compute_baseloss(self, quantized, inputs, logits=None, loss_type=0):
        if self.gumble:
            # + kl divergence to the prior loss
            if (logits is None):
                loss = 0
            else:
                print (logits.min(), logits.max(), logits.mean(), '===========')
                qy = F.softmax(logits, dim=-1)
                loss = self.kld_scale * torch.sum(qy * torch.log(qy * self._num_embeddings + 1e-10), dim=-1).mean()

        else:
             # Loss
            e_latent_loss = self.loss_fn(quantized.detach(), inputs)
            q_latent_loss = self.loss_fn(quantized, inputs.detach())


            if loss_type == 0:
                loss = q_latent_loss + self._commitment_cost * e_latent_loss
            elif loss_type == 1:
                loss = q_latent_loss
            else:
                loss = self._commitment_cost * e_latent_loss

        return loss 


    def forward(self, *args):
        return NotImplementedError()



class VectorQuantizer(BaseVectorQuantizer):
    def __init__(self, num_embeddings, 
                        embedding_dim, 
                        codebook_dim=8,
                        nhidden=128,
                        decay=0.8, 
                        epsilon=1e-5,
                        commitment_cost=0.0, 
                        usage_threshold=1.0e-9,
                        qk=False,
                        cosine=False,
                        gumble=False,
                        temperature=1.0,
                        kld_scale=1.0):

        super(VectorQuantizer, self).__init__(num_embeddings = num_embeddings, 
                                                    embedding_dim = embedding_dim,
                                                    codebook_dim = codebook_dim,
                                                    nhidden = nhidden, 
                                                    commitment_cost = commitment_cost,
                                                    usage_threshold = usage_threshold,
                                                    qk = qk,
                                                    cosine = cosine,
                                                    gumble = gumble,
                                                    temperature = temperature,
                                                    kld_scale = kld_scale)
        
        self._embedding_dim = embedding_dim
        self.codebook_dim = codebook_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._usage_threshold = usage_threshold
        self._cosine = cosine
        self.qk = qk

       

        # ======================
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.codebook_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon


    def forward(self, inputs, 
                        update=False,
                        loss_type=0,
                        idxs=None,
                        MCsamples = 1,
                        reset_usage=False):
        input_shape = inputs.shape

        features = self.project_in(inputs)
        # layer norm on features
        features = self.norm_in(features)

        # Flatten input
        flat_input = features.view(-1, self.codebook_dim)
        
        
        # update data stats
        if self.training:
            self.data_mean = 0.9*self.data_mean + 0.1*features.clone().detach().mean(1).mean(0)
            self.data_std = 0.9*self.data_std + 0.1*features.clone().detach().std(1).mean(0)



        quantized, encoding_indices, encodings, slots, logits = self.sample(features, 
                                                                            idxs=idxs, 
                                                                            hard = True,
                                                                            MCsamples = MCsamples,
                                                                            from_train=True)



        # ===================================
        # Tricks to prevent codebook collapse---
        # Restart vectors
        if update:
            if np.random.uniform() > 0.99: self.random_restart()
            if reset_usage: self.reset_usage()


        # Reset unused cb vectors...
        if update: self.update_usage(encoding_indices)


        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon)
                                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        

        # ============================
        loss = self.compute_baseloss(quantized, inputs, logits, loss_type)
        unique_code_ids = torch.unique(encoding_indices)




        # Regularization terms ==========================
        # loss += get_cb_variance(self._embedding.weight[unique_code_ids])

        if not (loss_type == 2):
            if self.qk:
                qkloss = self.qkclass.compute_qkloss(encoding_indices, 
                                                            inputs)
            
                loss += qkloss


        # print (f'feature: {inputs.max()}, qfeatures: {quantized.max(), quantized.min(), quantized.mean()}, quant loss: {loss}, QKloss: {qkloss}')
        
        # Straight Through Estimator
        if not self.gumble: quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        encoding_indices = encoding_indices.view(input_shape[0], -1)
        
        return quantized, encoding_indices, loss, perplexity, slots