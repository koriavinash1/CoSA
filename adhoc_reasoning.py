import os
import argparse
from src.dataset import *
from src.model import ReasoningClassifier, SlotAttentionAutoEncoder
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


parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--nproperties', default=19, type=int)
parser.add_argument('--nclasses', default=3, type=int)
parser.add_argument('--config', type=str)
parser.add_argument('--reasoning_type', default='default', type=str)


parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')

opt = parser.parse_args()


# load parameters....
exp_arguments = json.load(open(opt.config, 'r'))
print(exp_arguments)
print ('='*25)

opt.overlap_weightage = exp_arguments['overlap_weightage']
opt.quantize = exp_arguments['quantize']
opt.batch_size = exp_arguments['batch_size']
opt.max_slots = exp_arguments['max_slots']
opt.num_slots = exp_arguments['num_slots']
seed_everything(exp_arguments['seed'])

# dataset path setting =====================================

# set information based on dataset and it variant
if exp_arguments['dataset_name'] == 'clevr':
    opt.nproperties = 19

    if exp_arguments['variant'] == 'hans3':
        opt.nclasses = 3
    elif exp_arguments['variant'] == 'hans7':
        opt.nclasses = 7

elif exp_arguments['dataset_name'] == 'ffhq':
    opt.nproperties = 21
    opt.nclasses = 2

elif exp_arguments['dataset_name'] == 'floatingMNIST':
    opt.nproperties = 11
    if exp_arguments['variant'] == 'n2':
        if opt.reasoning_type == 'diff':
            opt.nclasses = 10
        elif opt.reasoning_type == 'mixed':
            opt.nclasses = 25
        else:
            opt.nclasses = 19

    elif exp_arguments['variant'] == 'n3':
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

exp_arguments['img_size'] = 128
exp_arguments['encoder_res'] = 4
exp_arguments['decoder_res'] = 16

resolution = (exp_arguments['img_size'], exp_arguments['img_size'])
opt.model_dir = exp_arguments['model_dir'].replace('ObjectDiscovery', 'Reasoning')


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

# ===========================================
# model init
AEmodel = SlotAttentionAutoEncoder(resolution, 
                                    exp_arguments['num_slots'], 
                                    exp_arguments['num_iterations'], 
                                    exp_arguments['hid_dim'],
                                    exp_arguments['max_slots'],
                                    exp_arguments['nunique_objects'],
                                    exp_arguments['quantize'],
                                    exp_arguments['cosine'],
                                    exp_arguments['cb_decay'],
                                    exp_arguments['encoder_res'],
                                    exp_arguments['decoder_res'],
                                    exp_arguments['kernel_size'],
                                    exp_arguments['cb_qk'],
                                    exp_arguments['eigen_quantizer'],
                                    exp_arguments['restart_cbstats'],
                                    exp_arguments['implicit'],
                                    exp_arguments['gumble'],
                                    exp_arguments['temperature'],
                                    exp_arguments['kld_scale'],
                                    deeper=True).to(device)


# ckpt=torch.load(os.path.join(exp_arguments['model_dir'], 'discovery_best.pth' ))
# AEmodel.load_state_dict(ckpt['model_state_dict'])
AEmodel.device = device


Rmodel = ReasoningClassifier(slot_dim=exp_arguments['hid_dim'],
                                    max_slots=exp_arguments['max_slots'],
                                    nproperties=opt.nproperties,
                                    nclasses=opt.nclasses).to(device)


# ===========================================
# Optimizer setup
finetune_factor = 0.01
# AEoptimizer = optim.Adam(AEmodel.parameters(), lr=finetune_factor*opt.learning_rate)
Roptimizer = optim.Adam(list(AEmodel.parameters()) + list(Rmodel.parameters()), lr=opt.learning_rate)

Rlrscheduler = ReduceLROnPlateau(Roptimizer, 'min')
# AElrscheduler = ReduceLROnPlateau(AEoptimizer, 'min')

# ============================================


train_set = DataGenerator(root=exp_arguments['data_root'], 
                            mode='train', 
                            class_info=True,
                            resolution=resolution)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers, drop_last=True)

val_set = DataGenerator(root=exp_arguments['data_root'], 
                            mode='val', 
                            class_info=True,
                            resolution=resolution)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers, drop_last=True)

test_set = DataGenerator(root=exp_arguments['data_root'], 
                            mode='test', 
                            class_info=True,
                            resolution=resolution)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers, drop_last=True)

train_epoch_size = min(50000, len(train_dataloader))
val_epoch_size = min(10000, len(val_dataloader))
test_epoch_size = min(10000, len(test_dataloader))

# ==============================================

opt.log_interval = train_epoch_size // 5




def training_step(AEmodel, Rmodel, Roptimizer, epoch, opt):
    idxs = []
    total_loss = 0; reasoning_loss = 0; recon_loss = 0; property_loss = 0; quant_loss = 0; overlap_loss = 0
    for ibatch, sample in tqdm(enumerate(train_dataloader)):
        global_step = epoch * train_epoch_size + ibatch
        image = sample['image'].to(device)
        labels = sample['target'].to(device)
        MCsamples = 1
        # ================
        recon_combined, recons, masks, slots, cbidxs, qloss, perplexity = AEmodel(image, 
                                                                                MCsamples = MCsamples,
                                                                                epoch=epoch, 
                                                                                batch=ibatch)
        recon_loss_ = ((image - recon_combined)**2).mean()
        
        overlap_loss_ = dice_loss(masks)
        loss = recon_loss_ + qloss + opt.overlap_weightage*overlap_loss_

        recon_loss += recon_loss_.item()
        quant_loss += qloss.item()
        overlap_loss += overlap_loss_.item()

        # =================
        logits = Rmodel(slots, cbidxs, 
                            epoch=epoch, batch=ibatch)
        rloss = F.cross_entropy(logits, labels)
        reasoning_loss += rloss.item()
        loss = 0.01*loss + rloss
        # =================
        total_loss += loss.item()

        # AEoptimizer.zero_grad()
        # loss.backward()
        # AEoptimizer.step()

        Roptimizer.zero_grad()
        loss.backward()
        Roptimizer.step()

        # ==================
        idxs.append(cbidxs)

        with torch.no_grad():
            if ibatch % opt.log_interval == 0:            
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/mse', recon_loss_.item(), global_step)
                writer.add_scalar('TRAIN/opi', overlap_loss_.item(), global_step)
                writer.add_scalar('TRAIN/rloss', rloss.item(), global_step)
                if opt.quantize: writer.add_scalar('TRAIN/quant', qloss.item(), global_step)
                if opt.quantize: writer.add_scalar('TRAIN/cbp', perplexity.item(), global_step)
                if opt.quantize: writer.add_scalar('TRAIN/cbd', get_cb_variance(AEmodel.slot_attention.slot_quantizer._embedding.weight).item(), global_step)
                if opt.quantize: writer.add_scalar('TRAIN/Samplingvar', len(torch.unique(cbidxs)), global_step)
                writer.add_scalar('TRAIN/Rlr', Roptimizer.param_groups[0]['lr'], global_step)
                # writer.add_scalar('TRAIN/AElr', AEoptimizer.param_groups[0]['lr'], global_step)
        

    with torch.no_grad():
        attns = recons * masks + (1 - masks)
        vis_recon = visualize(image, recon_combined, attns, cbidxs, opt.max_slots, N=32)
        grid = vutils.make_grid(vis_recon, nrow=opt.num_slots + 3, pad_value=0.2)[:, 2:-2, 2:-2]
        writer.add_image('TRAIN/recon_epoch={:03}'.format(epoch+1), grid)

        del recons, masks, slots

    idxs = torch.cat(idxs, 0)
    return_stats = {'total_loss': total_loss*1.0/train_epoch_size,
                        'reasoning_loss': reasoning_loss*1.0/train_epoch_size,
                        'recon_loss': recon_loss*1.0/train_epoch_size,
                        'quant_loss': quant_loss*1.0/train_epoch_size,
                        'overlap_loss': overlap_loss*1.0/train_epoch_size,
                        'unique_idxs': torch.unique(idxs).cpu().numpy()}

    return return_stats


@torch.no_grad()
def validation_step(AEmodel, Rmodel, Roptimizer, epoch, opt):
    idxs = []
    total_loss = 0; reasoning_loss = 0; recon_loss = 0; property_loss = 0; quant_loss = 0; overlap_loss = 0
    for ibatch, sample in tqdm(enumerate(train_dataloader)):
        global_step = epoch * train_epoch_size + ibatch
        image = sample['image'].to(device)
        labels = sample['target'].to(device)
        MCsamples = 1

         # ================
        recon_combined, recons, masks, slots, cbidxs, qloss, perplexity = AEmodel(image, 
                                                                                MCsamples = MCsamples,
                                                                                epoch=epoch, 
                                                                                batch=ibatch)
        recon_loss_ = ((image - recon_combined)**2).mean()
        
        overlap_loss_ = dice_loss(masks)
        loss = recon_loss_ + qloss + opt.overlap_weightage*overlap_loss_

        
        recon_loss += recon_loss_.item()
        quant_loss += qloss.item()
        overlap_loss += overlap_loss_.item()

        # =================
        logits = Rmodel(slots, cbidxs, 
                            epoch=epoch, batch=ibatch)
        rloss = F.cross_entropy(logits, labels)
        reasoning_loss += rloss.item()
        loss += rloss
        # =================

        total_loss += loss.item()
        idxs.append(cbidxs)

        if ibatch % opt.log_interval == 0:            
            writer.add_scalar('VALID/loss', loss.item(), global_step)
            writer.add_scalar('VALID/mse', recon_loss_.item(), global_step)
            writer.add_scalar('VALID/opi', overlap_loss_.item(), global_step)
            writer.add_scalar('VALID/rloss', rloss.item(), global_step)
            if opt.quantize: writer.add_scalar('VALID/quant', qloss.item(), global_step)
            if opt.quantize: writer.add_scalar('VALID/cbp', perplexity.item(), global_step)
            if opt.quantize: writer.add_scalar('VALID/cbd', get_cb_variance(AEmodel.slot_attention.slot_quantizer._embedding.weight).item(), global_step)
            if opt.quantize: writer.add_scalar('VALID/Samplingvar', len(torch.unique(cbidxs)), global_step)
            writer.add_scalar('VALID/Rlr', Roptimizer.param_groups[0]['lr'], global_step)
            # writer.add_scalar('VALID/AElr', AEoptimizer.param_groups[0]['lr'], global_step)


    attns = recons * masks + (1 - masks)
    vis_recon = visualize(image, recon_combined, attns, cbidxs, opt.max_slots, N=32)
    grid = vutils.make_grid(vis_recon, nrow=opt.num_slots + 3, pad_value=0.2)[:, 2:-2, 2:-2]
    writer.add_image('VALID/recon_epoch={:03}'.format(epoch+1), grid)

    del recons, masks, slots

    idxs = torch.cat(idxs, 0)
    return_stats = {'total_loss': total_loss*1.0/train_epoch_size,
                        'reasoning_loss': reasoning_loss*1.0/train_epoch_size,
                        'recon_loss': recon_loss*1.0/train_epoch_size,
                        'quant_loss': quant_loss*1.0/train_epoch_size,
                        'overlap_loss': overlap_loss*1.0/train_epoch_size,
                        'unique_idxs': torch.unique(idxs).cpu().numpy()}
    return return_stats


# =========================

start = time.time()
min_rloss = 1000000.0
min_recon_loss = 1000000.0
counter = 0
patience = 15


for epoch in range(opt.num_epochs):
    AEmodel.train(); Rmodel.train()
    training_stats = training_step(AEmodel, Rmodel, Roptimizer, epoch, opt)
    print ("TrainingLogs Setprediction Time Taken:{} --- EXP: {}, Epoch: {}, Stats: {}".format(timedelta(seconds=time.time() - start),
                                                        exp_arguments['exp_name'],
                                                        epoch, 
                                                        training_stats
                                                        ))


    AEmodel.eval(); Rmodel.eval()
    validation_stats = validation_step(AEmodel, Rmodel, Roptimizer, epoch, opt)
    print ("ValidationLogs Setprediction Time Taken:{} --- EXP: {}, Epoch: {}, Stats: {}".format(timedelta(seconds=time.time() - start),
                                                        exp_arguments['exp_name'],
                                                        epoch, 
                                                        validation_stats
                                                        ))
    print ('='*50)


    # warm up learning rate setup
    if epoch*train_epoch_size < exp_arguments['warmup_steps']: 
        learning_rate = exp_arguments['learning_rate'] *(epoch * train_epoch_size / exp_arguments['warmup_steps'])
    else:
        learning_rate = exp_arguments['learning_rate']
   
    learning_rate = learning_rate * (exp_arguments['decay_rate'] ** (epoch * train_epoch_size / exp_arguments['decay_steps']))
    # AEoptimizer.param_groups[0]['lr'] = finetune_factor*learning_rate
    Roptimizer.param_groups[0]['lr'] = learning_rate


    if epoch*train_epoch_size > exp_arguments['warmup_steps']:
        # AElrscheduler.step(validation_stats['recon_loss'])
        Rlrscheduler.step(validation_stats['reasoning_loss'])

        if min_rloss > validation_stats['reasoning_loss']:
            min_rloss = validation_stats['reasoning_loss']
            torch.save({
                'AEmodel_state_dict': AEmodel.state_dict(),
                'Rmodel_state_dict': Rmodel.state_dict(),
                'epoch': epoch,
                'vstats': validation_stats,
                'tstats': training_stats, 
                }, os.path.join(opt.model_dir, f'reasoning_best.pth'))
        else:
            counter +=1 
            if counter > patience:
                ckpt = torch.load(os.path.join(opt.model_dir, f'reasoning_best.pth'))
                AEmodel.load_state_dict(ckpt['AEmodel_state_dict'])
                Rmodel.load_state_dict(ckpt['Rmodel_state_dict'])
            
            if counter > 5*patience:
                print('Early Stopping: --------------')
                break

    torch.save({
        'AEmodel_state_dict': AEmodel.state_dict(),
        'Rmodel_state_dict': Rmodel.state_dict(),
        'epoch': epoch,
        'vstats': validation_stats,
        'tstats': training_stats, 
        }, os.path.join(opt.model_dir, f'reasoning_last.pth'))

    if epoch > 5:
        opt.overlap_weightage *= (1 + 10*epoch/opt.num_epochs)



    # Evaluation metrics....
    with torch.no_grad():
        every_nepoch = 10
        if not epoch  % every_nepoch:
            AEmodel.eval(); Rmodel.eval()

            fid = calculate_fid(test_dataloader, AEmodel, opt.batch_size, 25, opt.model_dir)
            print(f'FID Score: {fid}')

            writer.add_scalar('VALID/FID', fid, epoch//every_nepoch)


            pred = []; attributes = []
            for batch_num, samples in tqdm(enumerate(test_dataloader), desc='calculating Acc-F1 '):
                if batch_num > 50: break

                image = samples['image'].to(AEmodel.device)
                targets = samples['target'].to(AEmodel.device)

                recon_combined, recons, masks, slots, cbidxs, qloss, perplexity = AEmodel(image)
                logits = Rmodel(slots, cbidxs)

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