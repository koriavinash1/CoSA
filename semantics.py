import os
import argparse
from dataset import *
from model import *
from tqdm import tqdm
import time, math
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument
parser.add_argument('--model_dir', default='Logs', type=str, help='where to save models' )
parser.add_argument('--data_root', default='/vol/biomedic2/agk21/PhDLogs/datasets/CLEVR/CLEVR-Hans3', type=str, help='where to save models' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=32, type=int)

parser.add_argument('--num_slots', default=10, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--max_slots', default=64, type=int, help='Maximum number of plausible slots in dataset.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)

parser.add_argument('--img_size', default=128, type=int)
parser.add_argument('--encoder_res', default=8, type=int)
parser.add_argument('--decoder_res', default=8, type=int)

parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')
parser.add_argument('--hyperspherical', default=False, type=bool, help='use geodesic distance')
parser.add_argument('--unique_sampling', default=False, type=bool, help='use unique sampling')
parser.add_argument('--use_gumblesampling', default=False, type=bool, help='use traditional VQ or gumble based quantization')


parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000)


# python train.py --exp_name 4img64 --img_size 64 --encoder_res 4 --decoder_res 4 --max_slots 12 --num_slots 10 --hyperspherical True --use_gumblesampling True


opt = parser.parse_args()
resolution = (opt.img_size, opt.img_size)
os.makedirs(os.path.join(os.path.dirname(opt.model_dir), 'semantics'), exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SlotAttentionAutoEncoder(resolution, 
                                    opt.num_slots, 
                                    opt.num_iterations, 
                                    opt.hid_dim,
                                    opt.max_slots,
                                    opt.hyperspherical,
                                    opt.unique_sampling,
                                    opt.use_gumblesampling,
                                    opt.encoder_res,
                                    opt.decoder_res).to(device)
model.load_state_dict(torch.load(opt.model_dir)['model_state_dict'])
model.eval()

val_set = DataGenerator(root=opt.data_root, mode='val',  resolution=resolution)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)


val_epoch_size = len(val_dataloader)

total_loss = 0
start = time.time()

Listconcepts = []; Listimages = []; Listindices=[]

for sample in tqdm(val_dataloader):

    image = sample['image'].to(device)
    recon_combined, recons, masks, slots, cbidxs, qloss = model(image, epoch=0)
    recon_loss = ((image - recon_combined) ** 2).sum() / image.shape[0]

    loss = recon_loss + qloss
    total_loss += loss.item()*1./val_epoch_size

    Listconcepts.append(masks.detach().cpu().numpy())
    Listimages.append(image.detach().cpu().numpy())
    Listindices.append(cbidxs.detach().cpu().numpy())



Listconcepts = np.concatenate(Listconcepts, 0)
Listimages = np.concatenate(Listimages, 0)
Listindices = np.concatenate(Listindices, 0)


print ("Loss: {}, CB_VAR: {}, Time: {}".format(total_loss, 
                            torch.var(1.0*torch.unique(cbidxs)).item(),
                            timedelta(seconds=time.time() - start)))


def plot_concepts(images, concepts, indices):

    unique_indicies = np.unique(indices)
    plot_nmaps = 10

    for concept_idx in unique_indicies:
        image_idx, map_idx = np.where(indices == concept_idx)
        nmap_counter = 1
        
        binary_concepts = []
        plt.figure(figsize=(3*plot_nmaps, 3*plot_nmaps))
        for imap, (iidx, midx) in enumerate(zip(image_idx, map_idx)):
            if imap >= plot_nmaps**2: break
            if midx >= opt.num_slots: break
            
            img = images[iidx]
            cmap = concepts[iidx, midx]
            cmap = (cmap - np.min(cmap))/ (np.max(cmap) - np.min(cmap) + 1e-3)
            binary_concepts.append(img*(1.0*(cmap >= 0.5))[..., None])
            plt.subplot(plot_nmaps, plot_nmaps, nmap_counter)
            plt.imshow(img)
            plt.imshow(cmap, cmap='coolwarm', vmax=1.0, vmin=0.0, alpha=0.5)
            plt.title(f"Image: {iidx}, Concept: {midx}")
            nmap_counter += 1

        binary_concepts = np.array(binary_concepts)
        plt.suptitle(f"Visual semantics for codebook embedding: {concept_idx}, Total concept variance: {np.mean(np.var(binary_concepts, 0))}")
        plt.savefig(os.path.join(os.path.dirname(opt.model_dir), f'semantics/CBEmbedding-{concept_idx}.png'))
    pass 


Listimages = Listimages.transpose(0, 2, 3, 1)
print (np.unique(Listindices))
plot_concepts(Listimages, Listconcepts, Listindices)