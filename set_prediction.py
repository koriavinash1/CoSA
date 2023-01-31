import os
import argparse
from src.dataset import *
from src.model import *
from src.metrics import average_precision_clevr
from tqdm import tqdm
import time, math
from datetime import datetime, timedelta
from torch.nn.utils import clip_grad_norm_

import torch.optim as optim
import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import linear_sum_assignment
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true', '1')


parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--config', type=str)

parser.add_argument('--num_workers', default=10, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')

opt = parser.parse_args()


# load parameters....
exp_arguments = json.load(open(opt.config, 'r'))
print(exp_arguments)
print ('='*25)



resolution = (exp_arguments['img_size'], exp_arguments['img_size'])
opt.model_dir = exp_arguments['model_dir']
opt.overlap_weightage = exp_arguments['overlap_weightage']

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(opt).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(opt.model_dir, datetime.today().isoformat())

os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)


model = SlotAttentionClassifier(resolution, 
                                    exp_arguments['num_slots'], 
                                    exp_arguments['num_iterations'], 
                                    exp_arguments['hid_dim'],
                                    19, #nproperties...
                                    exp_arguments['max_slots'],
                                    exp_arguments['nunique_objects'],
                                    exp_arguments['quantize'],
                                    exp_arguments['cosine'],
                                    exp_arguments['cb_decay'],
                                    exp_arguments['encoder_res'],
                                    exp_arguments['decoder_res'],
                                    exp_arguments['variational'],
                                    exp_arguments['binarize'],
                                    exp_arguments['cb_qk'],
                                    exp_arguments['eigen_quantizer'],
                                    exp_arguments['restart_cbstats'],
                                    exp_arguments['implicit'],
                                    exp_arguments['gumble'],
                                    exp_arguments['temperature'],
                                    exp_arguments['kld_scale']).to(device)

ckpt=torch.load(os.path.join(opt.model_dir, 'discovery_best.pth' ))
keys = ckpt['model_state_dict'].keys()

model.encoder_cnn.load_state_dict({k[12:]: ckpt['model_state_dict'][k] for k in keys if k.__contains__('encoder_cnn')})
model.slot_attention.load_state_dict({k[15:]: ckpt['model_state_dict'][k] for k in keys if k.__contains__('slot_attention')})
model.device = device
params = [{'params': model.parameters()}]

train_set = DataGenerator(root=exp_arguments['data_root'], 
                                mode='train',
                                max_objects = 10,
                                properties=True,
                                class_info=False, 
                                resolution=resolution)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)

val_set = DataGenerator(root=exp_arguments['data_root'], 
                            mode='val',  
                            max_objects = 10,
                            properties=True,
                            class_info=False,
                            resolution=resolution)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)

test_set = DataGenerator(root=exp_arguments['data_root'], 
                                mode='test',  
                                max_objects = 10,
                                properties=True,
                                class_info=False,
                                resolution=resolution)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)


train_epoch_size = len(train_dataloader)
val_epoch_size = len(val_dataloader)

opt.exp_name = exp_arguments['exp_name']
opt.log_interval = train_epoch_size // 5
optimizer = optim.Adam(params, lr=opt.learning_rate)
lrscheduler = ReduceLROnPlateau(optimizer, 'min')


def hungarian_huber_loss(x, y):
    """
    code conversion from: https://github.com/google-research/google-research/blob/c3bef5045a2d4777a80a1fb333d31e03474222fb/slot_attention/utils.py#L26
    
    Huber loss for sets, matching elements with the Hungarian algorithm.
    This loss is used as reconstruction loss in the paper 'Deep Set Prediction
    Networks' https://arxiv.org/abs/1906.06565, see Eq. 2. For each element in the
    batches we wish to compute min_{pi} ||y_i - x_{pi(i)}||^2 where pi is a
    permutation of the set elements. We first compute the pairwise distances
    between each point in both sets and then match the elements using the scipy
    implementation of the Hungarian algorithm. This is applied for every set in
    the two batches. Note that if the number of points does not match, some of the
    elements will not be matched. As distance function we use the Huber loss.
    Args:
    x: Batch of sets of size [batch_size, n_points, dim_points]. Each set in the
        batch contains n_points many points, each represented as a vector of
        dimension dim_points.
    y: Batch of sets of size [batch_size, n_points, dim_points].
    Returns:
    Average distance between all sets in the two batches.
    """

    # adjust shape for x and y
    x = x.unsqueeze(-3)
    y = y.unsqueeze(-2)


    pairwise_cost = F.huber_loss(x, y, reduction='none')
    pairwise_cost = pairwise_cost.mean(-1)

    indices = np.array(list(map(linear_sum_assignment, pairwise_cost.clone().detach().cpu().numpy())))
    transposed_indices = np.transpose(indices, axes=(0, 2, 1))

    transposed_indices = torch.tensor(transposed_indices).to(x.device)

    actual_costs = torch.gather(pairwise_cost, dim = 1, index = transposed_indices)
    
    return actual_costs.sum(1).mean()



def training_step(model, optimizer, epoch, opt):
    idxs = []
    total_loss = 0; property_loss = 0; qloss = 0
    for ibatch, sample in tqdm(enumerate(train_dataloader)):
        global_step = epoch * train_epoch_size + ibatch
        image = sample['image'].to(device)
        labels = sample['properties'].to(device)

        predictions, cbidxs, qerror, perplexity = model(image, 
                                                        epoch=epoch, 
                                                        batch=ibatch)
        property_error = hungarian_huber_loss(predictions, labels)

        loss = property_error + qerror

        total_loss += loss.item()
        property_loss += property_error.item()
        qloss += qerror.item()




        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 10.0, 'inf')
        optimizer.step()

        idxs.append(cbidxs)

        with torch.no_grad():
            if ibatch % opt.log_interval == 0:            
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/property_loss', property_error.item(), global_step)
                writer.add_scalar('TRAIN/quant_loss', qerror.item(), global_step)
                writer.add_scalar('TRAIN/perplexity', perplexity.item(), global_step)
                writer.add_scalar('TRAIN/Samplingvar', len(torch.unique(cbidxs)), global_step)
                writer.add_scalar('TRAIN/lr', optimizer.param_groups[0]['lr'], global_step)


    idxs = torch.cat(idxs, 0)
    return_stats = {'total_loss': total_loss*1.0/train_epoch_size,
                        'property_loss': property_loss*1.0/train_epoch_size,
                        'qloss': qloss*1.0/train_epoch_size,
                        'unique_idxs': torch.unique(idxs).cpu().numpy()}

    return return_stats


@torch.no_grad()
def validation_step(model, optimizer, epoch, opt):
    idxs = []
    total_loss = 0; qloss = 0; property_loss = 0;
    for ibatch, sample in tqdm(enumerate(val_dataloader)):
        global_step = epoch * val_epoch_size + ibatch
        image = sample['image'].to(device)
        labels = sample['properties'].to(device)

        predictions, cbidxs, qerror, perplexity = model(image, 
                                                        epoch=epoch, 
                                                        batch=ibatch)
        property_error = hungarian_huber_loss(predictions, labels)

        loss = property_error + qerror

        total_loss += loss.item()
        property_loss += property_error.item()
        qloss += qerror.item()

        idxs.append(cbidxs)

        if ibatch % opt.log_interval == 0:            
            writer.add_scalar('VALID/loss', loss.item(), global_step)
            writer.add_scalar('VALID/quant_loss', qerror.item(), global_step)
            writer.add_scalar('VALID/property_loss', property_error.item(), global_step)
            writer.add_scalar('VALID/perplexity', perplexity.item(), global_step)
            writer.add_scalar('VALID/Samplingvar', len(torch.unique(cbidxs)), global_step)
            writer.add_scalar('VALID/lr', optimizer.param_groups[0]['lr'], global_step)


    idxs = torch.cat(idxs, 0)
    return_stats = {'total_loss': total_loss*1.0/val_epoch_size,
                        'qloss': qloss*1.0/val_epoch_size,
                        'property_loss': property_loss*1.0/val_epoch_size,
                        'unique_idxs': torch.unique(idxs).cpu().numpy()}
    return return_stats



start = time.time()
min_recon_loss = 1000000.0

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

    lrscheduler.step(validation_stats['property_loss'])

    if min_recon_loss > validation_stats['property_loss']:
        min_recon_loss = validation_stats['property_loss']
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'vstats': validation_stats,
            'tstats': training_stats, 
            }, os.path.join(opt.model_dir, f'setprediction_best.pth'))

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'vstats': validation_stats,
        'tstats': training_stats, 
        }, os.path.join(opt.model_dir, f'setprediction_last.pth'))

    if epoch > 5:
        opt.overlap_weightage *= (1 + 10*epoch/opt.num_epochs)




    # Evaluation metrics....
    with torch.no_grad():
        every_nepoch = 10
        if not epoch  % every_nepoch:
            model.eval()

            pred = []; attributes = []
            for batch_num in tqdm(range(25), desc='calculating APR '):
                samples = next(iter(test_dataloader))

                image = samples['image'].to(model.device)
                properties = samples['properties'].to(model.device)

                predictions, *_ = model(image, 
                                        epoch=0, 
                                        batch=batch_num)

                attributes.append(properties)
                pred.append(predictions)

            attributes = torch.cat(attributes, 0).cpu().numpy()
            pred = torch.cat(pred, 0).cpu().numpy()

            # For evaluating the AP score, we get a batch from the validation dataset.
            ap = [
                average_precision_clevr(pred, attributes, d)
                for d in [-1., 1., 0.5, 0.25, 0.125]
            ]
            print(
                "AP@inf: %.2f, AP@1: %.2f, AP@0.5: %.2f, AP@0.25: %.2f, AP@0.125: %.2f",
                ap[0], ap[1], ap[2], ap[3], ap[4])

            writer.add_scalar('VALID/AP@inf', ap[0], epoch//every_nepoch)
            writer.add_scalar('VALID/AP@1', ap[1], epoch//every_nepoch)
            writer.add_scalar('VALID/AP@0.5', ap[2], epoch//every_nepoch)
            writer.add_scalar('VALID/AP@0.25', ap[3], epoch//every_nepoch)
            writer.add_scalar('VALID/AP@0.125', ap[4], epoch//every_nepoch)