import os
import argparse
from src.dataset import *
from src.model import SlotAttentionReasoning
from src.metrics import average_precision_clevr, accuracy, dice_loss, calculate_fid
from src.utils import seed_everything, get_cb_variance, create_histogram, linear_warmup, visualize

from tqdm import tqdm
import time, math
from datetime import datetime, timedelta
from torch.nn.utils import clip_grad_norm_

import torch.optim as optim
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import linear_sum_assignment
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true', '1')


# exp setup information
parser.add_argument('--model_dir', default='Logs', type=str, help='where to save models' )
parser.add_argument('--exp_name', default='test', type=str, help='where to save models' )
parser.add_argument('--seed', default=0, type=int, help='random seed')

# dataset information
parser.add_argument('--dataset_name', default='clevr', type=str, help='where to save models' )
parser.add_argument('--variant', default='hans3', type=str, help='where to save models' )
parser.add_argument('--img_size', default=128, type=int, help='image size')
parser.add_argument('--nproperties', default=19, type=int, help='total number properties in clevr dataset')
parser.add_argument('--nclasses', default=3, type=int, help='total number classes clevr dataset')

# model information
parser.add_argument('--kernel_size', default=5, type=int, help='convolutional kernel size')
parser.add_argument('--encoder_res', default=8, type=int, help='encoder latent size')
parser.add_argument('--decoder_res', default=8, type=int, help='decoder init size')

# SA parameters
parser.add_argument('--num_slots', default=10, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
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
parser.add_argument('--learning_rate', default=0.001, type=float, help='training learning rate')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--implicit', type=str2bool, default=False, help="use implicit neumann's approximation for computing fixed point")

# unused parameters
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')


opt = parser.parse_args()
opt.model_dir = os.path.join(opt.model_dir, 'Reasoning', opt.exp_name)


seed_everything(opt.seed)

# dataset path setting =====================================

# set information based on dataset and it variant
if opt.dataset_name == 'clevr':
    opt.encoder_res = 4
    opt.decoder_res = 4
    opt.img_size = 128
    opt.max_slots = 64
    opt.kernel_size = 5
    opt.num_slots = 7
    opt.nunique_objects = 16
    opt.nproperties = 19

    if opt.variant == 'hans3':
        opt.nclasses = 3
        opt.data_root = '/vol/biomedic2/agk21/PhDLogs/datasets/CLEVR/CLEVR-Hans3'
    elif opt.variant == 'hans7':
        opt.nclasses = 7
        opt.data_root = '/vol/biomedic2/agk21/PhDLogs/datasets/CLEVR/CLEVR-Hans7'

elif opt.dataset_name == 'ffhq':
    opt.variant ='default'
    opt.encoder_res = 8
    opt.decoder_res = 8
    opt.img_size = 64
    opt.max_slots = 64
    opt.num_slots = 7
    opt.nunique_objects = 15
    opt.kernel_size = 5
    opt.nproperties = 21
    opt.nclasses = 2
    opt.data_root = '/vol/biomedic2/agk21/PhDLogs/datasets/FFHQ/data'


elif opt.dataset_name == 'floatingMNIST':
    opt.encoder_res = 8
    opt.decoder_res = 8
    opt.img_size = 64
    opt.kernel_size = 3
    opt.max_slots = 11
    opt.nproperties = 11

    if opt.variant == 'n2':
        opt.num_slots = 3
        opt.nunique_objects = 3
        opt.data_root = '/vol/biomedic3/agk21/datasets/FloatingMNIST2'
        if opt.reasoning_type == 'diff':
            opt.nclasses = 10
        elif opt.reasoning_type == 'mixed':
            opt.nclasses = 25
        else:
            opt.nclasses = 19

    elif opt.variant == 'n3':
        opt.num_slots = 4
        opt.nunique_objects = 4
        opt.data_root = '/vol/biomedic3/agk21/datasets/FloatingMNIST3'
        if opt.reasoning_type == 'diff':
            opt.nclasses = 28
        elif opt.reasoning_type == 'mixed':
            opt.nclasses = 38
        else:
            opt.nclasses = 28
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

model = SlotAttentionReasoning(resolution = resolution, 
                                    num_slots =  opt.num_slots, 
                                    num_iterations =  opt.num_iterations, 
                                    hid_dim =  opt.hid_dim,
                                    nproperties = opt.nproperties,
                                    nclasses = opt.nclasses,
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
                                    kld_scale = opt.kld_scale,
                                    deeper=True).to(device)
model.device = device


# ===========================================
# Optimizer setup
optimizer = optim.Adam(list(model.parameters()), lr=opt.learning_rate)
lrscheduler = ReduceLROnPlateau(optimizer, 'min')

# ============================================
# dataloader

train_set = DataGenerator(root=opt.data_root, 
                            mode='train', 
                            max_objects = opt.num_slots,
                            properties=False,
                            class_info=True,
                            resolution=resolution)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers, drop_last=True)

val_set = DataGenerator(root=opt.data_root, 
                            mode='val',  
                            max_objects = opt.num_slots,
                            properties=False,
                            class_info=True,
                            resolution=resolution)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size,
                        shuffle=False, num_workers=opt.num_workers, drop_last=True)

test_set = DataGenerator(root=opt.data_root, 
                            mode='test',  
                            max_objects = opt.num_slots,
                            properties=False,
                            class_info=True,
                            resolution=resolution)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                        shuffle=False, num_workers=opt.num_workers, drop_last=True)


train_epoch_size = min(50000, len(train_dataloader))
val_epoch_size = min(10000, len(val_dataloader))
test_epoch_size = min(10000, len(test_dataloader))

# ==============================================

opt.log_interval = train_epoch_size // 5




def training_step(model, optimizer, epoch, opt):
    idxs = []
    total_loss = 0; reasoning_loss = 0; recon_loss = 0; property_loss = 0; quant_loss = 0; overlap_loss = 0
    for ibatch, sample in tqdm(enumerate(train_dataloader)):
        global_step = epoch * train_epoch_size + ibatch
        image = sample['image'].to(device)
        labels = sample['target'].to(device)

        MCsamples = 1
        # ================
        predictions, cbidxs, qloss, perplexity = model(image, 
                                                        MCsamples = MCsamples,
                                                        epoch=epoch, 
                                                        batch=ibatch)
        
        rloss = F.cross_entropy(predictions, labels)
        loss = rloss + qloss

        reasoning_loss += rloss.item()
        quant_loss += qloss.item()
        total_loss += loss.item()

        # =================

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ==================
        idxs.append(cbidxs)

        with torch.no_grad():
            if ibatch % opt.log_interval == 0:            
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/rloss', rloss.item(), global_step)
                if opt.quantize: writer.add_scalar('TRAIN/quant', qloss.item(), global_step)
                if opt.quantize: writer.add_scalar('TRAIN/cbp', perplexity.item(), global_step)
                if opt.quantize: writer.add_scalar('TRAIN/cbd', get_cb_variance(model.slot_attention.slot_quantizer._embedding.weight).item(), global_step)
                if opt.quantize: writer.add_scalar('TRAIN/Samplingvar', len(torch.unique(cbidxs)), global_step)
                writer.add_scalar('TRAIN/lr', optimizer.param_groups[0]['lr'], global_step)
        

    idxs = torch.cat(idxs, 0)
    return_stats = {'total_loss': total_loss*1.0/train_epoch_size,
                        'reasoning_loss': reasoning_loss*1.0/train_epoch_size,
                        'quant_loss': quant_loss*1.0/train_epoch_size,
                        'unique_idxs': torch.unique(idxs).cpu().numpy()}

    return return_stats


@torch.no_grad()
def validation_step(model, optimizer, epoch, opt):
    idxs = []
    total_loss = 0; reasoning_loss = 0; recon_loss = 0; property_loss = 0; quant_loss = 0; overlap_loss = 0
    for ibatch, sample in tqdm(enumerate(train_dataloader)):
        global_step = epoch * train_epoch_size + ibatch
        image = sample['image'].to(device)
        labels = sample['target'].to(device)
        MCsamples = 1

        # ================
        predictions, cbidxs, qloss, perplexity = model(image, 
                                                        MCsamples = MCsamples,
                                                        epoch=epoch, 
                                                        batch=ibatch)
        
        rloss = F.cross_entropy(predictions, labels)
        loss = rloss + qloss


        reasoning_loss += rloss.item()
        quant_loss += qloss.item()
        total_loss += loss.item()
        # =================

        idxs.append(cbidxs)

        if ibatch % opt.log_interval == 0:            
            writer.add_scalar('VALID/loss', loss.item(), global_step)
            writer.add_scalar('VALID/rloss', rloss.item(), global_step)
            if opt.quantize: writer.add_scalar('VALID/quant', qloss.item(), global_step)
            if opt.quantize: writer.add_scalar('VALID/cbp', perplexity.item(), global_step)
            if opt.quantize: writer.add_scalar('VALID/cbd', get_cb_variance(model.slot_attention.slot_quantizer._embedding.weight).item(), global_step)
            if opt.quantize: writer.add_scalar('VALID/Samplingvar', len(torch.unique(cbidxs)), global_step)
            writer.add_scalar('VALID/lr', optimizer.param_groups[0]['lr'], global_step)


    idxs = torch.cat(idxs, 0)
    return_stats = {'total_loss': total_loss*1.0/train_epoch_size,
                        'reasoning_loss': reasoning_loss*1.0/train_epoch_size,
                        'quant_loss': quant_loss*1.0/train_epoch_size,
                        'unique_idxs': torch.unique(idxs).cpu().numpy()}
    return return_stats


# =========================

start = time.time()
min_rloss = 1000000.0
min_recon_loss = 1000000.0
counter = 0
patience = 15


for epoch in range(opt.num_epochs):
    model.train()
    training_stats = training_step(model, optimizer, epoch, opt)
    print ("TrainingLogs Setprediction Time Taken:{} --- EXP: {}, Epoch: {}, Stats: {}".format(timedelta(seconds=time.time() - start),
                                                        opt.exp_name,
                                                        epoch, 
                                                        training_stats
                                                        ))


    model.eval()
    validation_stats = validation_step(model, optimizer, epoch, opt)
    print ("ValidationLogs Setprediction Time Taken:{} --- EXP: {}, Epoch: {}, Stats: {}".format(timedelta(seconds=time.time() - start),
                                                        opt.exp_name,
                                                        epoch, 
                                                        validation_stats
                                                        ))
    print ('='*50)


     # warm up learning rate setup
    if epoch*train_epoch_size < opt.warmup_steps: 
        learning_rate = opt.learning_rate *(epoch * train_epoch_size / opt.warmup_steps)
    else:
        learning_rate = opt.learning_rate
   
    learning_rate = learning_rate * (opt.decay_rate ** (epoch * train_epoch_size / opt.decay_steps))
    optimizer.param_groups[0]['lr'] = learning_rate



    if epoch*train_epoch_size > opt.warmup_steps:
        lrscheduler.step(validation_stats['reasoning_loss'])

        if min_rloss > validation_stats['reasoning_loss']:
            min_rloss = validation_stats['reasoning_loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'vstats': validation_stats,
                'tstats': training_stats, 
                }, os.path.join(opt.model_dir, f'reasoning_best.pth'))
        else:
            counter +=1 
            if counter > patience:
                ckpt = torch.load(os.path.join(opt.model_dir, f'reasoning_best.pth'))
                model.load_state_dict(ckpt['model_state_dict'])
            
            if counter > 5*patience:
                print('Early Stopping: --------------')
                break

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'vstats': validation_stats,
        'tstats': training_stats, 
        }, os.path.join(opt.model_dir, f'reasoning_last.pth'))



    # Evaluation metrics....
    with torch.no_grad():
        every_nepoch = 10
        if not epoch  % every_nepoch:
            model.eval()

            pred = []; attributes = []
            for batch_num, samples in tqdm(enumerate(test_dataloader), desc='calculating Acc-F1 '):
                if batch_num > 50: break

                image = samples['image'].to(model.device)
                targets = samples['target'].to(model.device)

                logits, cbidxs, qloss, perplexity = model(image, MCsamples=1)

                attributes.append(targets)
                pred.append(torch.argmax(logits, -1))

            attributes = torch.cat(attributes, 0).cpu().numpy()
            pred = torch.cat(pred, 0).cpu().numpy()

            print (attributes.shape, pred.shape)
            # For evaluating the AP score, we get a batch from the validation dataset.
           
            acc = accuracy_score(attributes, pred)
            f1  = f1_score(attributes, pred, average='macro')

            print("Acc.: {}, F1: {}".format(acc, f1))

            writer.add_scalar('VALID/Acc', acc, epoch//every_nepoch)
            writer.add_scalar('VALID/F1', f1, epoch//every_nepoch)