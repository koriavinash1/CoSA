import os
import argparse
from src.dataset import *
from src.model import DefaultCNN
from src.metrics import average_precision_clevr
from src.utils import get_cb_variance, seed_everything
from tqdm import tqdm
import time, math
from datetime import datetime, timedelta
from torch.nn.utils import clip_grad_norm_

import torch.optim as optim
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
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

# model information
parser.add_argument('--kernel_size', default=5, type=int, help='convolutional kernel size')
parser.add_argument('--encoder_res', default=8, type=int, help='encoder latent size')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')

# training parameters
parser.add_argument('--batch_size', default=16, type=int, help='training mini-batch size')
parser.add_argument('--learning_rate', default=0.001, type=float, help='training learning rate')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')

# unused parameters
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')


opt = parser.parse_args()
opt.model_dir = os.path.join(opt.model_dir, 'BaselineClassifier', opt.exp_name)


seed_everything(opt.seed)

# dataset path setting =====================================

# set information based on dataset and it variant
if opt.dataset_name == 'clevr':
    opt.encoder_res = 8
    opt.decoder_res = 8
    opt.img_size = 64
    opt.max_slots = 19
    opt.kernel_size = 5
    opt.num_slots = 7
    opt.nunique_objects = 16

    if opt.variant == 'hans3':
        opt.data_root = '/vol/biomedic2/agk21/PhDLogs/datasets/CLEVR/CLEVR-Hans3'
    elif opt.variant == 'hans7':
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
    opt.data_root = '/vol/biomedic2/agk21/PhDLogs/datasets/FFHQ/data'


elif opt.dataset_name == 'floatingMNIST':
    opt.encoder_res = 8
    opt.decoder_res = 8
    opt.img_size = 64
    opt.kernel_size = 3
    opt.max_slots = 11

    if opt.variant == 'n2':
        opt.num_slots = 3
        opt.nunique_objects = 3
        opt.data_root = '/vol/biomedic3/agk21/datasets/FloatingMNIST2'
    elif opt.variant == 'n3':
        opt.num_slots = 4
        opt.nunique_objects = 4
        opt.data_root = '/vol/biomedic3/agk21/datasets/FloatingMNIST3'

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

model = DefaultCNN(resolution = resolution, 
                        hid_dim =  opt.hid_dim,
                        encoder_res =  opt.encoder_res,
                        kernel_size = opt.kernel_size).to(device)


model.device = device
params = [{'params': model.parameters()}]

# ======================================================
# dataloader init

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
                        shuffle=True, num_workers=opt.num_workers, drop_last=True)

test_set = DataGenerator(root=opt.data_root, 
                            mode='test',  
                            max_objects = opt.num_slots,
                            properties=False,
                            class_info=True,
                            resolution=resolution)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers, drop_last=True)


train_epoch_size = min(50000, len(train_dataloader))
val_epoch_size = min(10000, len(val_dataloader))

# ======================================================

opt.log_interval = train_epoch_size // 5
optimizer = optim.Adam(params, lr=opt.learning_rate)


def training_step(model, optimizer, epoch, opt):
    idxs = []
    total_loss = 0; property_loss = 0; qloss = 0
    for ibatch, sample in tqdm(enumerate(train_dataloader)):
        global_step = epoch * train_epoch_size + ibatch
        image = sample['image'].to(device)
        labels = sample['target'].to(device)

        predictions = model(image, 
                                epoch=epoch, 
                                batch=ibatch)


        loss = F.cross_entropy(predictions, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()

        # clip_grad_norm_(model.parameters(), 10.0, 'inf')
        optimizer.step()


        with torch.no_grad():
            if ibatch % opt.log_interval == 0:            
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/lr', optimizer.param_groups[0]['lr'], global_step)


    return_stats = {'total_loss': total_loss*1.0/train_epoch_size}

    return return_stats


@torch.no_grad()
def validation_step(model, optimizer, epoch, opt):
    idxs = []
    total_loss = 0; qloss = 0; property_loss = 0;
    for ibatch, sample in tqdm(enumerate(val_dataloader)):
        global_step = epoch * val_epoch_size + ibatch
        image = sample['image'].to(device)
        labels = sample['target'].to(device)

        predictions = model(image, 
                                epoch=epoch, 
                                batch=ibatch)

        loss = F.cross_entropy(predictions, labels)
        total_loss += loss.item()


        if ibatch % opt.log_interval == 0:            
            writer.add_scalar('VALID/loss', loss.item(), global_step)
            writer.add_scalar('VALID/lr', optimizer.param_groups[0]['lr'], global_step)


    return_stats = {'total_loss': total_loss*1.0/val_epoch_size}
    return return_stats



start = time.time()
min_recon_loss = 1000000.0
counter = 0
patience = 15
lrscheduler = ReduceLROnPlateau(optimizer, 'min', 
                                patience=patience,
                                factor=0.5,
                                verbose = True)


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
    print ('='*150)
    

     # warm up learning rate setup
    if epoch*train_epoch_size < opt.warmup_steps: 
        learning_rate = opt.learning_rate *(epoch * train_epoch_size / opt.warmup_steps)
    else:
        learning_rate = opt.learning_rate
   
    learning_rate = learning_rate * (opt.decay_rate ** (epoch * train_epoch_size / opt.decay_steps))
    optimizer.param_groups[0]['lr'] = learning_rate

    
    if epoch*train_epoch_size > opt.warmup_steps:
        lrscheduler.step(validation_stats['total_loss'])

        if min_recon_loss > validation_stats['total_loss']:
            min_recon_loss = validation_stats['total_loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'vstats': validation_stats,
                'tstats': training_stats, 
                }, os.path.join(opt.model_dir, f'baseline_classifier_best.pth'))
        else:
            counter +=1 
            if counter > patience:
                ckpt = torch.load(os.path.join(opt.model_dir, f'baseline_classifier_best.pth'))
                model.load_state_dict(ckpt['model_state_dict'])
            
            if counter > 5*patience:
                print('Early Stopping: --------------')
                break

        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'vstats': validation_stats,
            'tstats': training_stats, 
            }, os.path.join(opt.model_dir, f'baseline_classifier_last.pth'))



    # Evaluation metrics....
    with torch.no_grad():
        every_nepoch = 10
        if not epoch  % every_nepoch:
            model.eval()

            pred = []; attributes = []
            for batch_num, samples in tqdm(enumerate(test_dataloader), desc='calculating Acc-F1 '):
                image = samples['image'].to(model.device)
                targets = samples['target'].to(model.device)

                predictions = model(image, 
                                        epoch=0, 
                                        batch=batch_num)

                attributes.append(targets)
                pred.append(torch.argmax(predictions, -1))

            attributes = torch.cat(attributes, 0).cpu().numpy()
            pred = torch.cat(pred, 0).cpu().numpy()

            print (attributes.shape, pred.shape)
            # For evaluating the AP score, we get a batch from the validation dataset.
           
            acc = accuracy_score(attributes, pred)
            f1  = f1_score(attributes, pred, average='macro')

            print("Acc.: {}, F1: {}".format(acc, f1))

            writer.add_scalar('VALID/Acc', acc, epoch//every_nepoch)
            writer.add_scalar('VALID/F1', f1, epoch//every_nepoch)