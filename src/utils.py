import torch
import random
from random import random
from torch import nn
import numpy as np
from contextlib import contextmanager, ExitStack
import torch.nn.functional as F

import scipy
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from pymatting.util.util import row_sum
from pymatting.util.kdtree import knn


import matplotlib.pyplot as plt


try:
    from torch.cuda import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool
    return model



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


    if binarize:
        # apply softmax on token dimension 
        W = torch.softmax(W, dim = 1)
        
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

        else:

            # No color affinity
            W_color = 0
            
        
        import pdb;pdb.set_trace()
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