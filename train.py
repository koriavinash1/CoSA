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
                                    opt.unique_sampling,
                                    opt.cb_decay,
                                    opt.encoder_res,
                                    opt.decoder_res,
                                    opt.variational,
                                    opt.binarize,
                                    opt.eigen_noposition,
                                    opt.cb_qk,
                                    opt.eigen_quantizer,
                                    opt.restart_cbstats).to(device)
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

log_interval = train_epoch_size // 5

optimizer = optim.Adam(params, lr=opt.learning_rate)

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

start = time.time()
i = 0
lr_decay_factor = 1
recon_history = [0]*5

for epoch in range(opt.num_epochs):
    model.train()

    total_loss = 0
    idxs = []

    for ibatch, sample in tqdm(enumerate(train_dataloader)):
        i += 1
        global_step = epoch * train_epoch_size + i

        # lr_warmup_factor = linear_warmup(
        #     global_step,
        #     0.,
        #     1.0,
        #     0,
        #     opt.warmup_steps)

        # if i < opt.warmup_steps:
        #     learning_rate = opt.learning_rate * (i / opt.warmup_steps)
        # else:
        #     learning_rate = opt.learning_rate

        # learning_rate = learning_rate * (opt.decay_rate ** (
        #                                     i / opt.decay_steps))

        lr_warmup_factor = 1
        learning_rate = lr_decay_factor * lr_warmup_factor * opt.learning_rate
        optimizer.param_groups[0]['lr'] = learning_rate
        
        image = sample['image'].to(device)
        recon_combined, recons, masks, slots, cbidxs, qloss, perplexity = model(image, epoch=epoch, batch=ibatch)
        recon_loss = ((image - recon_combined)**2).mean()
        
        overlap_loss = dice_loss(masks)
        loss = recon_loss + 0.5*qloss + opt.overlap_weightage*overlap_loss

        total_loss += loss.item()


        optimizer.zero_grad()
        loss.backward()
<<<<<<< HEAD
        # clip_grad_norm_(model.parameters(), 1.0, 'inf')
=======
        # clip_grad_norm_(model.parameters(), 10.0, 2)
>>>>>>> 3f97515a306b0dd8c02775aeecc68ba07eae2128
        optimizer.step()

        idxs.append(cbidxs)

        with torch.no_grad():
            if i % log_interval == 0:            
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/recon', recon_loss.item(), global_step)
                writer.add_scalar('TRAIN/overlap', overlap_loss.item(), global_step)
                writer.add_scalar('TRAIN/perplexity', perplexity.item(), global_step)
                writer.add_scalar('TRAIN/quant', qloss.item(), global_step)
                writer.add_scalar('TRAIN/Samplingvar', len(torch.unique(cbidxs)), global_step)
                writer.add_scalar('TRAIN/lr', optimizer.param_groups[0]['lr'], global_step)

            if (i % train_epoch_size) == train_epoch_size - 1:
                attns = recons * masks + (1 - masks)
                vis_recon = visualize(image, recon_combined, attns, N=32)
                grid = vutils.make_grid(vis_recon, nrow=opt.num_slots + 2, pad_value=0.2)[:, 2:-2, 2:-2]
                writer.add_image('TRAIN_recon/epoch={:03}'.format(epoch+1), grid)

        del recons, masks, slots

    total_loss /= len(train_dataloader)
    
    recon_history.append(total_loss)
    recon_history = recon_history[1:]


    idxs = torch.cat(idxs, 0)
    print ("EXP: {}, Epoch: {}, RECON:{}, Loss: {}, CB_VAR: {}, Time: {}".format(opt.exp_name,
                                                        epoch, recon_loss.item(), total_loss, 
                                                        torch.unique(idxs),
                                                        timedelta(seconds=time.time() - start)))


    if epoch > 5:
        opt.overlap_weightage *= (1 + 10*epoch/opt.num_epochs)
    
    if not epoch % 10: # (np.mean(recon_history) > total_loss):   
        print (recon_history, np.mean(recon_history),  total_loss)
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch
            }, os.path.join(opt.model_dir, f'ckpt_{epoch}.pth'))


<<<<<<< HEAD
    if (epoch > 10) and (np.mean(recon_history) < total_loss):
        lr_decay_factor *= 0.5


    # if epoch % 50 == 49:
    #     lr_decay_factor *= 0.5
=======
    # if (epoch > 10) and (np.mean(recon_history) < total_loss):
    #     lr_decay_factor *= 0.5


    if epoch % 50 == 49:
        lr_decay_factor *= 0.5
>>>>>>> 3f97515a306b0dd8c02775aeecc68ba07eae2128
