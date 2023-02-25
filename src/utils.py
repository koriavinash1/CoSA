import torch
import os
import random
from torch import nn
import numpy as np
from contextlib import contextmanager, ExitStack
import torch.nn.functional as F

import cv2
import scipy
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from pymatting.util.util import row_sum
from pymatting.util.kdtree import knn

from einops import rearrange
import matplotlib.pyplot as plt
from torch import nn, einsum
import torch.distributed as distributed

from einops import rearrange, repeat
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere


try:
    from torch.cuda import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool
    return model



def create_histogram(k, img_size, cbidx):
    image = np.zeros((2*k + 2, 2*k + 2, 3), dtype='uint8')
    
    cbidx = cbidx.detach().cpu().numpy()
    for uidx in np.unique(cbidx):
        count = np.sum(cbidx == uidx)
        image[:int(2*count), int(2*uidx):int(2*uidx+2), :] = (125, 255, 25)
    
    image = cv2.resize(image, img_size, cv2.INTER_NEAREST)
    image = image*1.0/255
    image = np.flipud(image)
    return 1 - image.transpose(2, 0, 1)


def visualize(image, recon_orig, attns, cbidxs, max_slots, N=8):
    _, _, H, W = image.shape
    attns = attns.permute(0, 1, 4, 2, 3)
    image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    recon_orig = recon_orig[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    attns = attns[:N].expand(-1, -1, 3, H, W)

    histograms = np.array([create_histogram(max_slots, (W, H), idxs) for idxs in cbidxs[:N]])
    histograms = torch.from_numpy(histograms).to(image.device).type(image.dtype).unsqueeze(1)
    
    return torch.cat((image, recon_orig, attns, histograms), dim=1).view(-1, 3, H, W)


def linear_warmup(step, start_value, final_value, start_step, final_step):
    
    assert start_value <= final_value
    assert start_step <= final_step
    
    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b
    
    return value


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def noop(*args, **kwargs):
    pass

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def hsphere_init(codebook_dim, emb_dim):
    sphere = Hypersphere(dim=emb_dim - 1)
    points_in_manifold = torch.Tensor(sphere.random_uniform(n_samples=codebook_dim))
    return points_in_manifold


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    if temperature == 0:
        return t.argmax(dim = dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def laplace_smoothing(x, n_categories, eps = 1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

def pad_shape(shape, size, dim = 0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype = torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)

def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)

def all_gather_variably_sized(x, sizes, dim = 0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src = i, async_op = True)
        all_x.append(t)

    distributed.barrier()
    return all_x


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src = 0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    out = torch.cat(all_samples, dim = 0)

    return rearrange(out, '... -> 1 ...')

def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -torch.cdist(samples, means, p = 2)

        buckets = torch.argmax(dists, dim = -1)
        bins = batched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins

def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d = dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch)
    return embeds.gather(2, indices)

# regularization losses

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)




def unique_sampling_fn(distances, nunique=-1):
    # distance: Bxntokensxncbtokens

    B, S, N = distances.shape
    if not (isinstance(nunique, list) or isinstance(nunique, np.ndarray)):
        if (nunique == -1): 
            nunique = min(S, N)
        nunique = [nunique]*B
    
    nunique = np.minimum(nunique, N)
    batch_sampled_vectors = []
    for b in range(B):
        distance_vector = distances[b, ...]
        sorted_idx = torch.argsort(distance_vector, dim=-1, descending=False)
        # Create a bin over codebook direction of distance vectors
        # with nunique bins and sample based on that...
        # 
        #  
        sampled_vectors = []
        sampled_distances = []
        for i in range(S):

            if i < nunique[b]:
                for k in range(N):
                    current_idx = sorted_idx[i, k]
                    if not (current_idx in sampled_vectors):
                        sampled_vectors.append(current_idx.unsqueeze(0))
                        sampled_distances.append(distance_vector[i, current_idx].unsqueeze(0))
                        break
            else:
                current_idx = sorted_idx[i, 0]
                sampled_vectors.append(current_idx.unsqueeze(0))
                sampled_distances.append(distance_vector[i, current_idx].unsqueeze(0))
 
        
        sampled_vectors = torch.cat(sampled_vectors, 0)
        sampled_distances = torch.cat(sampled_distances, 0)
        sampled_vectors = sampled_vectors[torch.argsort(sampled_distances, descending=False)]
        batch_sampled_vectors.append(sampled_vectors.unsqueeze(0))

    batch_sampled_vectors = torch.cat(batch_sampled_vectors, 0)
    return batch_sampled_vectors.view(-1)

    
def get_euclidian_distance(u, v):
    # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
    d = torch.sum(u**2, dim=1, keepdim=True) + \
        torch.sum(v**2, dim=1) - 2 * torch.matmul(u, v.t())
    return d



def sorting_idxs(quantized, cidxs):
    B, N, _ = quantized.shape

    batch_sampled_vectors = []
    for b in range(B):
        sampled_idxs = cidxs[b, ...]

        st_pointer = -1
        end_pointer = 0
        unique = torch.zeros_like(sampled_idxs)
        for unique_idx in torch.sort(torch.unique(sampled_idxs)):
            idxs = torch.argwhere(sampled_idxs == unique_idx)
            
            st_pointer += 1
            end_pointer -= len(idxs[1:])

            unique[st_pointer] = idxs[0]
            unique_idx[end_pointer : end_pointer + len(idxs[1:])] = idxs[1:]

            pass 


    return quantized, cidxs


def get_cosine_distance(u, v):
    # distance on sphere
    d = torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))
    ed1 = torch.sqrt(torch.sum(u**2, dim=1, keepdim=True))
    ed2 = torch.sqrt(torch.sum(v**2, dim=1, keepdim=True))
    ed3 = torch.einsum('bd,dn->bn', ed1, rearrange(ed2, 'n d  -> d n'))
    geod = torch.clamp(d/(ed3), min=-0.99999, max=0.99999)
    return torch.acos(torch.abs(geod))/(2.0*np.pi)


def get_cb_variance(cb):
    # cb = cb / (1e-5 + torch.norm(cb, dim=1, keepdim=True))
    cd = get_cosine_distance(cb, cb)
    return 1 - torch.mean(torch.var(cd, 1)) 



class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    @property
    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n




# Implementations adopted from https://github.com/lukemelas/deep-spectral-segmentation
def get_diagonal(W: scipy.sparse.csr_matrix, threshold: float = 1e-12):
    # See normalize_rows in pymatting.util.util
    D = row_sum(W)
    D[D < threshold] = 1.0  # Prevent division by zero.
    D = scipy.sparse.diags(D)
    return D


# Implementations adopted from https://github.com/lukemelas/deep-spectral-segmentation
def knn_affinity(image, n_neighbors=[2, 1], distance_weights=[2.0, 0.1]):
    """Computes a KNN-based affinity matrix. Note that this function requires pymatting"""

    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h

    x = np.tile(np.linspace(0, 1, w), h)
    y = np.repeat(np.linspace(0, 1, h), w)

    i, j = [], []

    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((n, 5), dtype=np.float32),
        )

        distances, neighbors = knn(f, f, k=k)

        i.append(np.repeat(np.arange(n), k))
        j.append(neighbors.flatten())

    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.ones(2 * sum(n_neighbors) * n)

    # This is our affinity matrix
    W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n, n))
    return W


# Implementations adopted from https://github.com/lukemelas/deep-spectral-segmentation
@torch.no_grad()
def compute_eigen(
    feats,
    image = None,
    K: int = 4, 
    which_matrix: str = 'affinity_torch',
    normalize: bool = True,
    binarize: bool = True,
    lapnorm: bool = True,
    threshold_at_zero: bool = False,
    image_color_lambda: float = 10,
):

    if normalize:
        feats = F.normalize(feats, p=2, dim=-1)
    

    W = feats @ feats.T
    if threshold_at_zero:
        W = (W * (W > 0))


    # if binarize:
    #     # apply softmax on token dimension 
    #     W = torch.softmax(W, dim = 1)
        
        # W = torch.sigmoid(W)
        # W[W >= 0.5] = 1
        # W[W < 0.5] = 0


    # Eigenvectors of affinity matrix
    if which_matrix == 'affinity_torch':
        eigenvalues, eigenvectors = torch.eig(W, eigenvectors=True)
        eigenvalues = eigenvalues[-K:] 
        eigenvectors = eigenvectors[:, -K:].T
    
    # Eigenvectors of affinity matrix with scipy
    elif which_matrix == 'affinity_svd':        
        USV = torch.linalg.svd(W, full_matrices=False)
        eigenvectors = USV[0][:, -K:].T #.to('cpu', non_blocking=True)
        eigenvalues = USV[1][-K:] #.to('cpu', non_blocking=True)

    # Eigenvectors of affinity matrix with scipy
    elif which_matrix == 'affinity':
        W = W.cpu().numpy()
        eigenvalues, eigenvectors = eigsh(W, which='LM', k=K)

    # Eigenvectors of matting laplacian matrix
    elif which_matrix in ['matting_laplacian', 'laplacian']:

        ### Feature affinities 
        W_feat = W.cpu().numpy()
        
        # Combine
        if image_color_lambda > 0:
            if image is None:
                raise ValueError('Image argument is required for laplacian based eigen decomposition')
            # Load image
            H = int(feats.shape[0]**0.5)
            image_lr = F.interpolate(image.unsqueeze(0), 
                size=(H, H), mode='bilinear', align_corners=False
            ).squeeze(0).cpu().numpy().transpose(1,2,0)
            

            # Color affinities (of type scipy.sparse.csr_matrix)
            W_lr = knn_affinity(image_lr)

            # Convert to dense numpy array
            W_color = np.array(W_lr.todense().astype(np.float32))
            W_color *=0.25
        else:

            # No color affinity
            W_color = 0
            
        
        W_comb = W_feat + W_color * image_color_lambda  # combination
        D_comb = np.array(get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check

        # Extract eigenvectors
        if lapnorm:
            try:
                eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM', M=D_comb)
            except:
                eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM', M=D_comb)
        else:
            try:
                eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM')
            except:
                eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, which='SM')
        eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()

    # Sign ambiguity
    for k in range(eigenvectors.shape[0]):
        if 0.5 < torch.mean((eigenvectors[k] > 0).float()).item() < 1.0:  # reverse segment
            eigenvectors[k] = 0 - eigenvectors[k]


    # normalization of eigen vectors
    # eigenvectors = eigenvectors/torch.norm(eigenvectors, dim=-1, keepdim=True)

    # Save dict
    # _, indices = torch.sort(eigenvalues, 0)
    # if len(indices.shape) > 1:
    #     indices = indices[:, 0]

    # eigenvalues = eigenvalues[indices]
    # eigenvectors = eigenvectors[indices, :]
    return eigenvectors, eigenvalues



@torch.no_grad()
def plot_vis(img, eigenvectors, K=10):
    n = min(eigenvectors.shape[0], K) + 1
    img = img.cpu().numpy().transpose(1,2,0)
    
    plt.figure(figsize=(3*n, 3))
    plt.subplot(1, n, 1)
    plt.imshow(img)
    
    for i in range(1, n):
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.imshow(eigenvectors[i-1].squeeze().cpu().numpy(), cmap='coolwarm', vmax=1.0, vmin=0.0, alpha=0.7)
    
    plt.show()
    

def process_eigen(eigenvectors, img=None, visual= False, binarize=0.0):
    eigen_reshape = int(eigenvectors.shape[-1]**0.5)
    
    print (eigen_reshape, eigenvectors.shape)
    # transpose as col correspond to eigen vectors
    eigenvectors = eigenvectors.T.view(-1, 1, eigen_reshape, eigen_reshape)
    # eigenvectors = (eigenvectors - torch.min(eigenvectors))/ (torch.max(eigenvectors) - torch.min(eigenvectors))

    if visual:
        if img is None: 
            raise ValueError()
        _, H, W = img.shape
        eigenvectors = F.interpolate(
                    eigenvectors, 
                    size=(H, W), mode='bilinear', align_corners=True
                )
    
    return eigenvectors



def visual_concepts(eigenvectors, data, binarize=0.0):
    B, c, H,W = data.shape
    b, k, c, h, w = eigenvectors.shape
    visual = []
    for i in range(int(k)):
        conecpts = eigenvectors[:, i, ...]
        resized =  F.interpolate(
                        conecpts, 
                        size=(H, W), 
                        mode='bilinear', 
                        align_corners=True
                    )
        visual.append(resized.unsqueeze(1))
    visual = torch.cat(visual, 1)
    return visual