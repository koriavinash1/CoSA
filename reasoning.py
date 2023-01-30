import os
import argparse
from src.dataset import *
from src.model import *
from tqdm import tqdm
import time, math
from datetime import datetime, timedelta
from torch.nn.utils import clip_grad_norm_

import torch.optim as optim
import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true', '1')


parser.add_argument('--model_dir', default='Logs', type=str, help='where to save models' )
parser.add_argument('--exp_name', default='test', type=str, help='where to save models' )
parser.add_argument('--data_root', default='/vol/biomedic2/agk21/PhDLogs/datasets/CLEVR/CLEVR-Hans3', type=str, help='where to save models' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=32, type=int)

parser.add_argument('--num_slots', default=10, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--max_slots', default=64, type=int, help='Maximum number of plausible slots in dataset.')
parser.add_argument('--num_iterations', default=5, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=128, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)

parser.add_argument('--img_size', default=128, type=int)
parser.add_argument('--encoder_res', default=8, type=int)
parser.add_argument('--decoder_res', default=8, type=int)

parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=10, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')

parser.add_argument('--quantize', default=False, type=str2bool, help='perform slot quantization')
parser.add_argument('--cosine', default=False, type=str2bool, help='use geodesic distance')
parser.add_argument('--unique_sampling', default=False, type=str2bool, help='use unique sampling')


parser.add_argument('--nunique_objects', type=int, default=8)
parser.add_argument('--variational', type=str2bool, default=False)
parser.add_argument('--binarize', type=str2bool, default=False)
parser.add_argument('--eigen_noposition', type=str2bool, default=True)

parser.add_argument('--overlap_weightage', type=float, default=0.0)
parser.add_argument('--cb_decay', type=float, default=0.0)

parser.add_argument('--cb_qk', type=str2bool, default=False)
parser.add_argument('--eigen_quantizer', type=str2bool, default=False)
parser.add_argument('--restart_cbstats', type=str2bool, default=False)

parser.add_argument('--implicit', type=str2bool, default=True)
parser.add_argument('--gumble', type=str2bool, default=False)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--kld_scale', type=float, default=1.0)




opt = parser.parse_args()
resolution = (opt.img_size, opt.img_size)

opt.model_dir = os.path.join(opt.model_dir, opt.exp_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(opt).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(opt.model_dir, datetime.today().isoformat())

os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)

model = SlotAttentionAutoEncoder(resolution, 
                                    opt.num_slots, 
                                    opt.num_iterations, 
                                    opt.hid_dim,
                                    opt.max_slots,
                                    opt.nunique_objects,
                                    opt.quantize,
                                    opt.cosine,
                                    opt.cb_decay,
                                    opt.encoder_res,
                                    opt.decoder_res,
                                    opt.variational,
                                    opt.binarize,
                                    opt.cb_qk,
                                    opt.eigen_quantizer,
                                    opt.restart_cbstats,
                                    opt.implicit,
                                    opt.gumble,
                                    opt.temperature,
                                    opt.kld_scale).to(device)
# model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])

criterion = nn.MSELoss()

params = [{'params': model.parameters()}]

train_set = DataGenerator(root=opt.data_root, mode='train', resolution=resolution)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)

val_set = DataGenerator(root=opt.data_root, mode='val',  resolution=resolution)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)


train_epoch_size = len(train_dataloader)
val_epoch_size = len(val_dataloader)

opt.log_interval = train_epoch_size // 5
optimizer = optim.Adam(params, lr=opt.learning_rate)
lrscheduler = ReduceLROnPlateau(optimizer, 'min')

# criterion = nn.MSELoss()

def visualize(image, recon_orig, attns, N=8):
    _, _, H, W = image.shape
    attns = attns.permute(0, 1, 4, 2, 3)
    image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    recon_orig = recon_orig[:N].expand(-1, 3, H, W).unsqueeze(dim=1)
    attns = attns[:N].expand(-1, -1, 3, H, W)
    return torch.cat((image, recon_orig, attns), dim=1).view(-1, 3, H, W)

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



class SoftDiceLossV1(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=0.001):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = (2 * numer + self.smooth) / (denor + self.smooth)
        return loss
    

_dice_loss_ = SoftDiceLossV1()

def dice_loss(masks):
    shape = masks.shape
    idxs = np.arange(shape[1])

    masks = (masks - masks.min(dim=1, keepdim=True)[0] + 1e-4)/(1e-4 + masks.max(dim=1, keepdim=True)[0] - masks.min(dim=1, keepdim=True)[0])
    gt_masks = masks.clone().detach()
    gt_masks[gt_masks >= 0.5] = 1.0
    gt_masks[gt_masks < 0.5] = 0.0
    
    loss = 0
    for i in idxs:
        _idxs_ = list(np.arange(shape[1]))
        del _idxs_[i]
        loss += _dice_loss_(masks[:, _idxs_, ...].reshape(-1, shape[2], shape[3], shape[4]),
            gt_masks[:, i, ...].unsqueeze(1).repeat(1, len(idxs) -1, 1, 1, 1).reshape(-1, shape[2], shape[3], shape[4]))

    return loss*1.0/len(idxs)


def training_step(model, optimizer, epoch, opt):
    idxs = []
    total_loss = 0; recon_loss = 0; quant_loss = 0; overlap_loss = 0
    for ibatch, sample in tqdm(enumerate(train_dataloader)):
        global_step = epoch * train_epoch_size + ibatch
        image = sample['image'].to(device)
        recon_combined, recons, masks, slots, cbidxs, qloss, perplexity = model(image, 
                                                                                epoch=epoch, 
                                                                                batch=ibatch)
        recon_loss_ = ((image - recon_combined)**2).mean()
        
        overlap_loss_ = dice_loss(masks)
        loss = recon_loss_ + 0.5*qloss + opt.overlap_weightage*overlap_loss_

        total_loss += loss.item()
        recon_loss += recon_loss_.item()
        quant_loss += qloss.item()
        overlap_loss += overlap_loss_.item()


        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 10.0, 'inf')
        optimizer.step()

        idxs.append(cbidxs)

        with torch.no_grad():
            if ibatch % opt.log_interval == 0:            
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/recon', recon_loss_.item(), global_step)
                writer.add_scalar('TRAIN/overlap', overlap_loss_.item(), global_step)
                writer.add_scalar('TRAIN/perplexity', perplexity.item(), global_step)
                writer.add_scalar('TRAIN/quant', qloss.item(), global_step)
                writer.add_scalar('TRAIN/Samplingvar', len(torch.unique(cbidxs)), global_step)
                writer.add_scalar('TRAIN/lr', optimizer.param_groups[0]['lr'], global_step)

    with torch.no_grad():
        attns = recons * masks + (1 - masks)
        vis_recon = visualize(image, recon_combined, attns, N=32)
        grid = vutils.make_grid(vis_recon, nrow=opt.num_slots + 2, pad_value=0.2)[:, 2:-2, 2:-2]
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
        global_step = epoch * val_epoch_size + ibatch
        image = sample['image'].to(device)
        recon_combined, recons, masks, slots, cbidxs, qloss, perplexity = model(image, 
                                                                                epoch=epoch, 
                                                                                batch=ibatch)
        recon_loss_ = ((image - recon_combined)**2).mean()
        
        overlap_loss_ = dice_loss(masks)
        loss = recon_loss_ + 0.5*qloss + opt.overlap_weightage*overlap_loss_

        total_loss += loss.item()
        recon_loss += recon_loss_.item()
        quant_loss += qloss.item()
        overlap_loss += overlap_loss_.item()


        idxs.append(cbidxs)

        if ibatch % opt.log_interval == 0:            
            writer.add_scalar('VALID/loss', loss.item(), global_step)
            writer.add_scalar('VALID/recon', recon_loss_.item(), global_step)
            writer.add_scalar('VALID/overlap', overlap_loss_.item(), global_step)
            writer.add_scalar('VALID/perplexity', perplexity.item(), global_step)
            writer.add_scalar('VALID/quant', qloss.item(), global_step)
            writer.add_scalar('VALID/Samplingvar', len(torch.unique(cbidxs)), global_step)
            writer.add_scalar('VALID/lr', optimizer.param_groups[0]['lr'], global_step)


    attns = recons * masks + (1 - masks)
    vis_recon = visualize(image, recon_combined, attns, N=32)
    grid = vutils.make_grid(vis_recon, nrow=opt.num_slots + 2, pad_value=0.2)[:, 2:-2, 2:-2]
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
    print ('='*50)

    lrscheduler.step(validation_stats['recon_loss'])

    if min_recon_loss > validation_stats['recon_loss']:
        min_recon_loss = validation_stats['recon_loss']
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'vstats': validation_stats,
            'tstats': training_stats, 
            }, os.path.join(opt.model_dir, f'best.pth'))

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'vstats': validation_stats,
        'tstats': training_stats, 
        }, os.path.join(opt.model_dir, f'last.pth'))

    if epoch > 5:
        opt.overlap_weightage *= (1 + 10*epoch/opt.num_epochs)
