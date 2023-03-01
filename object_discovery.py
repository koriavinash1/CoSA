import os
import argparse
from src.dataset import DataGenerator
from src.model import SlotAttentionAutoEncoder
from src.metrics import calculate_fid, calculate_sfid, dice_loss
from src.utils import seed_everything, get_cb_variance, create_histogram, linear_warmup, visualize

from tqdm import tqdm
import time, math
from datetime import datetime, timedelta
from torch.nn.utils import clip_grad_norm_

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import json
import cv2
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true', '1')

# exp setup information
parser.add_argument('--model_dir', default='Logs', type=str, help='where to save models' )
parser.add_argument('--exp_name', default='test', type=str, help='where to save models' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--nunique_objects', type=int, default=8, help='total number of unique objects in the dataset')


# dataset information
parser.add_argument('--dataset_name', default='clevr', type=str, help='where to save models' )
parser.add_argument('--variant', default='hans3', type=str, help='where to save models' )
parser.add_argument('--img_size', default=128, type=int, help='image size')


# model information
parser.add_argument('--kernel_size', default=5, type=int, help='convolutional kernel size')
parser.add_argument('--encoder_res', default=8, type=int, help='encoder latent size')
parser.add_argument('--decoder_res', default=8, type=int, help='decoder init size')

# SA parameters
parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=5, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')

# BSA parameters
parser.add_argument('--max_slots', default=64, type=int, help='Maximum number of plausible slots in dataset.')
parser.add_argument('--quantize', default=False, type=str2bool, help='perform slot quantization')
parser.add_argument('--cosine', default=False, type=str2bool, help='use geodesic distance')
parser.add_argument('--cb_decay', type=float, default=0.999, help='EMA decay for codebook')
parser.add_argument('--cb_qk', type=str2bool, default=False, help='use slot dictionary structure')
parser.add_argument('--eigen_quantizer', type=str2bool, default=False, help='quantize principle components')
parser.add_argument('--restart_cbstats', type=str2bool, default=False, help='random restart codebook stats to prevent codebook collapse')
parser.add_argument('--gumble', type=str2bool, default=False, help='use gumple softmax trick for continious sampling')
parser.add_argument('--temperature', type=float, default=2.0, help='sampling temperature')
parser.add_argument('--kld_scale', type=float, default=1.0, help='kl distance weightage')


# training parameters
parser.add_argument('--batch_size', default=16, type=int, help='training mini-batch size')
parser.add_argument('--learning_rate', default=0.0004, type=float, help='training learning rate')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--implicit', type=str2bool, default=False, help="use implicit neumann's approximation for computing fixed point")
parser.add_argument('--overlap_weightage', type=float, default=0.0, help='use mask overlap as a regularization constraints')


# unused parameters
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')




opt = parser.parse_args()
opt.model_dir = os.path.join(opt.model_dir, 'ObjectDiscovery', opt.exp_name)
seed_everything(opt.seed)

# dataset path setting =====================================

# set information based on dataset and it variant
if opt.dataset_name == 'bitmoji':
    opt.variant ='default'
    opt.encoder_res = 8
    opt.decoder_res = 8
    opt.img_size = 64
    opt.max_slots = 64
    opt.nunique_objects = 9
    opt.kernel_size = 5
    opt.num_slots = 7
    opt.data_root = '/vol/biomedic3/agk21/datasets/bitmoji'

elif opt.dataset_name == 'clevr':
    opt.encoder_res = 8
    opt.decoder_res = 8
    opt.img_size = 64
    opt.max_slots = 64
    opt.kernel_size = 5
    opt.num_slots = 7
    opt.nunique_objects = 16

    if opt.variant == 'hans3':
        opt.data_root = '/vol/biomedic2/agk21/PhDLogs/datasets/CLEVR/CLEVR-Hans3'
    elif opt.variant == 'hans7':
        opt.data_root = '/vol/biomedic2/agk21/PhDLogs/datasets/CLEVR/CLEVR-Hans7'
    else:
        opt.data_root = '/vol/biomedic3/agk21/datasets/multi-objects/RawData-subset/clevr_with_masks'


elif opt.dataset_name == 'floatingMNIST':
    opt.encoder_res = 8
    opt.decoder_res = 8
    opt.img_size = 64
    opt.max_slots = 11
    opt.kernel_size = 5
    opt.nunique_objects = 11


    if opt.variant == 'n3':
        opt.num_slots = 4
        opt.data_root = '/vol/biomedic3/agk21/datasets/FloatingMNIST3'
    else:
        opt.num_slots = 3
        opt.data_root = '/vol/biomedic3/agk21/datasets/FloatingMNIST2'


elif opt.dataset_name == 'multi_dsprites':
    opt.encoder_res = 4
    opt.decoder_res = 4
    opt.img_size = 32
    opt.max_slots = 5    
    opt.num_slots = 5
    opt.nunique_objects = 5
    opt.kernel_size = 3

    if opt.variant == 'colored_on_colored':
        opt.data_root = '/vol/biomedic3/agk21/datasets/multi-objects/RawData-subset/multi_dsprites/colored_on_colored'
    else:
        opt.data_root = '/vol/biomedic3/agk21/datasets/multi-objects/RawData-subset/multi_dsprites/colored_on_grayscale'


elif opt.dataset_name == 'objects_room':
    opt.variant ='default'
    opt.encoder_res = 8
    opt.decoder_res = 8
    opt.img_size = 32
    opt.max_slots = 32
    opt.num_slots = 6
    opt.nunique_objects = 8
    opt.kernel_size = 3
    opt.data_root = '/vol/biomedic3/agk21/datasets/multi-objects/RawData-subset/objects_room/default'


elif opt.dataset_name == 'tetrominoes':
    opt.variant ='default'
    opt.encoder_res = 8
    opt.decoder_res = 8
    opt.img_size = 32
    opt.max_slots = 32
    opt.nunique_objects = 16
    opt.num_slots = 5
    opt.kernel_size = 3
    opt.data_root = '/vol/biomedic3/agk21/datasets/multi-objects/RawData-subset/tetrominoes'


elif opt.dataset_name == 'ffhq':
    opt.variant ='default'
    opt.encoder_res = 8
    opt.decoder_res = 8
    opt.img_size = 64
    opt.max_slots = 64
    opt.num_slots = 7
    opt.nunique_objects = 15
    opt.kernel_size = 5
    opt.data_root = '/vol/biomedic2/agk21/PhDLogs/datasets/FFHQ/data'

else:
    raise ValueError('Invalid dataset found')


# =========================================================
# Dump exp config into json and tensorboard logs....

resolution = (opt.img_size, opt.img_size)
arg_str_list = ['{}={}'.format(k, v) for k, v in vars(opt).items()]
arg_str = '__'.join(arg_str_list)


os.makedirs(opt.model_dir, exist_ok=True)
 # save all parameters in the exp directory
json.dump(vars(opt), 
        open(os.path.join(opt.model_dir, 'exp-parameters.json'), 'w'), 
        indent=4)
print ('Parameters saved in: ', os.path.join(opt.model_dir, 'exp-parameters.json'))

log_dir = os.path.join(opt.model_dir, datetime.today().isoformat())


os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)

# ======================================================
# model init


model = SlotAttentionAutoEncoder(resolution = resolution, 
                                    num_slots =  opt.num_slots, 
                                    num_iterations =  opt.num_iterations, 
                                    hid_dim =  opt.hid_dim,
                                    max_slots =  opt.max_slots,
                                    nunique_slots =  opt.nunique_objects,
                                    quantize = opt.quantize,
                                    cosine = opt.cosine,
                                    cb_decay = opt.cb_decay,
                                    encoder_res =  opt.encoder_res,
                                    decoder_res = opt.decoder_res,
                                    kernel_size = opt.kernel_size,
                                    cb_qk = opt.cb_qk,
                                    eigen_quantizer =  opt.eigen_quantizer,
                                    restart_cbstats =  opt.restart_cbstats,
                                    implicit = opt.implicit,
                                    gumble = opt.gumble,
                                    temperature = opt.temperature,
                                    kld_scale = opt.kld_scale).to(device)
model.device = device


params = [{'params': model.parameters()}]


# ======================================================
# dataloader init

train_set = DataGenerator(root=opt.data_root, mode='train', resolution=resolution)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers, drop_last=True)

val_set = DataGenerator(root=opt.data_root, mode='val',  resolution=resolution)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size,
                        shuffle=False, num_workers=opt.num_workers, drop_last=True)

test_set = DataGenerator(root=opt.data_root, mode='test',  resolution=resolution)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                        shuffle=False, num_workers=opt.num_workers, drop_last=True)


train_epoch_size = min(50000, len(train_dataloader))
val_epoch_size = min(10000, len(val_dataloader))

# ======================================================
# optimizer init

optimizer = optim.Adam(params, lr=opt.learning_rate)
opt.log_interval = train_epoch_size // 5





def training_step(model, optimizer, epoch, opt):
    idxs = []
    total_loss = 0; recon_loss = 0; quant_loss = 0; overlap_loss = 0
    for ibatch, sample in tqdm(enumerate(train_dataloader)):
        if ibatch > train_epoch_size: break
        global_step = epoch * train_epoch_size + ibatch
        image = sample['image'].to(device)
        recon_combined, recons, masks, slots, cbidxs, qloss, perplexity = model(image, 
                                                                                MCsamples = 1,
                                                                                epoch=epoch, 
                                                                                batch=ibatch)
        recon_loss_ = ((image - recon_combined)**2).mean()
        
        overlap_loss_ = dice_loss(masks)
        loss = recon_loss_ + qloss + opt.overlap_weightage*overlap_loss_

        total_loss += loss.item()
        recon_loss += recon_loss_.item()
        quant_loss += qloss.item()
        overlap_loss += overlap_loss_.item()


        optimizer.zero_grad()
        loss.backward()

        # clip_grad_norm_(model.parameters(), 10.0, 'inf')
        optimizer.step()

        idxs.append(cbidxs)

        with torch.no_grad():
            if ibatch % opt.log_interval == 0:            
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/mse', recon_loss_.item(), global_step)
                writer.add_scalar('TRAIN/opi', overlap_loss_.item(), global_step)
                if opt.quantize: writer.add_scalar('TRAIN/quant', qloss.item(), global_step)
                if opt.quantize: writer.add_scalar('TRAIN/cbp', perplexity.item(), global_step)
                if opt.quantize: writer.add_scalar('TRAIN/cbd', get_cb_variance(model.slot_attention.slot_quantizer._embedding.weight).item(), global_step)
                if opt.quantize: writer.add_scalar('TRAIN/Samplingvar', len(torch.unique(cbidxs)), global_step)
                writer.add_scalar('TRAIN/lr', optimizer.param_groups[0]['lr'], global_step)
        

    with torch.no_grad():
        attns = recons * masks + (1 - masks)
        vis_recon = visualize(image, recon_combined, attns, cbidxs, opt.max_slots, N=32)
        grid = vutils.make_grid(vis_recon, nrow=opt.num_slots + 3, pad_value=0.2)[:, 2:-2, 2:-2]
        writer.add_image('TRAIN/recon_epoch={:03}'.format(epoch+1), grid)

        del recons, masks, slots

    idxs = torch.cat(idxs, 0)
    return_stats = {'total_loss': total_loss*1.0/train_epoch_size,
                        'recon_loss': recon_loss*1.0/train_epoch_size,
                        'quant_loss': quant_loss*1.0/train_epoch_size,
                        'overlap_loss': overlap_loss*1.0/train_epoch_size,
                        'unique_idxs': torch.unique(idxs).cpu().numpy()}

    return return_stats


@torch.no_grad()
def validation_step(model, optimizer, epoch, opt):
    idxs = []
    total_loss = 0; recon_loss = 0; quant_loss = 0; overlap_loss = 0
    for ibatch, sample in tqdm(enumerate(val_dataloader)):
        if ibatch > val_epoch_size: break
        global_step = epoch * val_epoch_size + ibatch
        image = sample['image'].to(device)
        recon_combined, recons, masks, slots, cbidxs, qloss, perplexity = model(image, 
                                                                                MCsamples = 10,
                                                                                epoch=epoch, 
                                                                                batch=ibatch)

        recon_loss_ = ((image - recon_combined)**2).mean()
        
        overlap_loss_ = dice_loss(masks)
        loss = recon_loss_ + qloss + opt.overlap_weightage*overlap_loss_

        total_loss += loss.item()
        recon_loss += recon_loss_.item()
        quant_loss += qloss.item()
        overlap_loss += overlap_loss_.item()


        idxs.append(cbidxs)

        if ibatch % opt.log_interval == 0:            
            writer.add_scalar('VALID/loss', loss.item(), global_step)
            writer.add_scalar('VALID/mse', recon_loss_.item(), global_step)
            writer.add_scalar('VALID/opi', overlap_loss_.item(), global_step)
            if opt.quantize: writer.add_scalar('VALID/quant', qloss.item(), global_step)
            if opt.quantize: writer.add_scalar('VALID/cbp', perplexity.item(), global_step)
            if opt.quantize: writer.add_scalar('VALID/cbd', get_cb_variance(model.slot_attention.slot_quantizer._embedding.weight).item(), global_step)
            if opt.quantize: writer.add_scalar('VALID/Samplingvar', len(torch.unique(cbidxs)), global_step)
            writer.add_scalar('VALID/lr', optimizer.param_groups[0]['lr'], global_step)


    attns = recons * masks + (1 - masks)
    vis_recon = visualize(image, recon_combined, attns, cbidxs, opt.max_slots, N=32)
    grid = vutils.make_grid(vis_recon, nrow=opt.num_slots + 3, pad_value=0.2)[:, 2:-2, 2:-2]
    writer.add_image('VALID/recon_epoch={:03}'.format(epoch+1), grid)

    del recons, masks, slots

    idxs = torch.cat(idxs, 0)
    return_stats = {'total_loss': total_loss*1.0/val_epoch_size,
                        'recon_loss': recon_loss*1.0/val_epoch_size,
                        'quant_loss': quant_loss*1.0/val_epoch_size,
                        'overlap_loss': overlap_loss*1.0/val_epoch_size,
                        'unique_idxs': torch.unique(idxs).cpu().numpy()}
    return return_stats



start = time.time()
min_recon_loss = 1000000.0
counter = 0
patience = 5
lrscheduler = ReduceLROnPlateau(optimizer, 'min', 
                                patience=patience,
                                factor=0.5,
                                verbose = True)


for epoch in range(opt.num_epochs):
    model.train()
   
    training_stats = training_step(model, optimizer, epoch, opt)
    print ("TrainingLogs Time Taken:{} --- EXP: {}, Epoch: {}, Stats: {}".format(timedelta(seconds=time.time() - start),
                                                        opt.exp_name,
                                                        epoch, 
                                                        training_stats
                                                        ))


    model.eval()
    validation_stats = validation_step(model, optimizer, epoch, opt)
    print ("ValidationLogs Time Taken:{} --- EXP: {}, Epoch: {}, Stats: {}".format(timedelta(seconds=time.time() - start),
                                                        opt.exp_name,
                                                        epoch, 
                                                        validation_stats
                                                        ))
    print ('='*150)


    # warm up learning rate setup
    if epoch*train_epoch_size < opt.warmup_steps: 
        learning_rate = opt.learning_rate *(epoch * train_epoch_size / opt.warmup_steps)
    else:
        learning_rate = opt.learning_rate
   
    learning_rate = learning_rate * (opt.decay_rate ** (epoch * train_epoch_size / opt.decay_steps))
    optimizer.param_groups[0]['lr'] = learning_rate

    

    if epoch*train_epoch_size > opt.warmup_steps:
        lrscheduler.step(validation_stats['recon_loss'])

        
        if min_recon_loss > validation_stats['recon_loss']:
            counter = 0
            min_recon_loss = validation_stats['recon_loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optm_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'vstats': validation_stats,
                'tstats': training_stats, 
                }, os.path.join(opt.model_dir, f'discovery_best.pth'))
        else:
            counter +=1 
            if counter > patience:
                ckpt = torch.load(os.path.join(opt.model_dir, f'discovery_best.pth'))
                model.load_state_dict(ckpt['model_state_dict'])
            
            if counter > 5*patience:
                print('Early Stopping: --------------')
                break


        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'vstats': validation_stats,
            'tstats': training_stats, 
            }, os.path.join(opt.model_dir, f'discovery_last.pth'))



    if epoch > 5:
        opt.overlap_weightage *= (1 + 10*epoch/opt.num_epochs)



    # Evaluation metrics....
    with torch.no_grad():
        every_nepoch = 10
        if not epoch  % every_nepoch:
            model.eval()
            # For evaluating the AP score, we get a batch from the validation dataset.
            fid = calculate_fid(test_dataloader, model, opt.batch_size, 25, opt.model_dir)
            print(f'FID Score: {fid}')

            writer.add_scalar('VALID/FID', fid, epoch//every_nepoch)

            sfid = calculate_sfid(test_dataloader, model, opt.batch_size, 25, opt.model_dir)
            print(f'SFID Score: {sfid}')

            writer.add_scalar('VALID/SFID', sfid, epoch//every_nepoch)