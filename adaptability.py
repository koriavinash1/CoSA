import os
import argparse
from src.dataset import *
from src.model import SlotAttentionReasoning, DefaultCNN
from src.metrics import compositional_fid, ReliableReasoningIndex
from sklearn.metrics import accuracy_score, f1_score

from src.utils import seed_everything

from tqdm import tqdm
import time, math
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
import json
import pandas as pd 
import torch.optim as optim
from itertools import cycle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true', '1')


parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_batches', default=10, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')

opt = parser.parse_args()


def get_dataroot(dataset, variant, reasoning_type):
    info = {'dataroot': '',
                'nclasses': 0}

    # set information based on dataset and it variant
    if dataset == 'clevr':
        if variant == 'hans3':
            nclasses = 3
            data_root = '/vol/biomedic2/agk21/PhDLogs/datasets/CLEVR/CLEVR-Hans3'
        elif variant == 'hans7':
            nclasses = 7
            data_root = '/vol/biomedic2/agk21/PhDLogs/datasets/CLEVR/CLEVR-Hans7'

    elif dataset == 'floatingMNIST':
        if variant == 'n2':
            data_root = '/vol/biomedic3/agk21/datasets/FloatingMNIST2'
            if reasoning_type == 'diff': nclasses = 10
            elif reasoning_type == 'mixed': nclasses = 29
            else: nclasses = 19

        elif variant == 'n3':
            data_root = '/vol/biomedic3/agk21/datasets/FloatingMNIST3'
            if reasoning_type == 'diff': nclasses = 19
            elif reasoning_type == 'mixed': nclasses = 44
            else: nclasses = 28
    else:
        raise ValueError('Invalid dataset found')


    info['dataroot'] = data_root
    info['nclasses'] = nclasses
    info['variants'] = variant
    info['reasoning_type'] = reasoning_type
    return info



def training_step(train_dataloader, model, optimizer, epoch=0):
    total_loss = 0
    for ibatch, sample in tqdm(enumerate(cycle(train_dataloader))):
        if ibatch > opt.num_batches: break
        image = sample['image'].to(device)
        labels = sample['target'].to(device)

        MCsamples = 1
        # ================
        predictions, properties, cbidxs, qloss, perplexity = model(image, 
                                                        MCsamples = MCsamples,
                                                        epoch=epoch, 
                                                        batch=ibatch)
        
        loss = F.cross_entropy(predictions, labels)

        total_loss += loss.item()

        # =================

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return_stats = {'total_loss': total_loss*1.0/opt.num_batches}

    return return_stats




def get_task_adaptability_results(info):
    # load parameters....
    print (info)
    exp_arguments = json.load(open(info['config'], 'r'))
    print(exp_arguments)
    print ('='*25)

    seed_everything(exp_arguments['seed'])

    resolution = (exp_arguments['img_size'], exp_arguments['img_size'])
    # ===========================================
    # model init
    if info['config'].__contains__('DefaultCNN'):
        model = DefaultCNN(resolution = resolution,
                            hid_dim = exp_arguments['nproperties'],
                            kernel_size = exp_arguments['kernel_size'],
                            encoder_res = exp_arguments['encoder_res'],
                            nclasses = info['nclasses']).to(device)
        ckpt=torch.load(os.path.join(exp_arguments['model_dir'], 'baseline_classifier_best.pth'), map_location=device)
    else:
        model = SlotAttentionReasoning(resolution = resolution, 
                                        num_slots = exp_arguments['num_slots'], 
                                        num_iterations = exp_arguments['num_iterations'], 
                                        hid_dim = exp_arguments['hid_dim'],
                                        nproperties = exp_arguments['nproperties'],
                                        nclasses = info['nclasses'],
                                        max_slots = exp_arguments['max_slots'],
                                        nunique_slots = exp_arguments['nunique_objects'],
                                        quantize = exp_arguments['quantize'],
                                        cosine = exp_arguments['cosine'],
                                        cb_decay = exp_arguments['cb_decay'],
                                        encoder_res = exp_arguments['encoder_res'],
                                        decoder_res = exp_arguments['decoder_res'],
                                        kernel_size = exp_arguments['kernel_size'],
                                        cb_qk = exp_arguments['cb_qk'],
                                        eigen_quantizer = exp_arguments['eigen_quantizer'],
                                        restart_cbstats = exp_arguments['restart_cbstats'],
                                        implicit = exp_arguments['implicit'],
                                        gumble = exp_arguments['gumble'],
                                        temperature = exp_arguments['temperature'],
                                        kld_scale = exp_arguments['kld_scale'],
                                        deeper=True).to(device)
        ckpt=torch.load(os.path.join(exp_arguments['model_dir'], 'reasoning_best.pth'), map_location=device)
    
    
    model_state_dict = {k: ckpt['model_state_dict'][k] for k in ckpt['model_state_dict'].keys() if not k.__contains__('classifier')}
    model.load_state_dict(model_state_dict, strict=False)

    for n, p in model.named_parameters():
        if not n.__contains__('classifier'): 
            p.requires_grad = False
        else:
            p.requires_grad = True


    model.device = device
    optimizer = optim.Adam(list(model.classifier.parameters()), lr=opt.learning_rate)

    # =====================================================================================
    train_set = DataGenerator(root=info['dataroot'], 
                            mode='train', 
                            max_objects = exp_arguments['num_slots'],
                            reasoning_type = info['reasoning_type'],
                            properties=True,
                            class_info=True,
                            resolution=resolution)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers, drop_last=True)

    val_set = DataGenerator(root=info['dataroot'], 
                                mode='val',  
                                max_objects = exp_arguments['num_slots'],
                                reasoning_type = info['reasoning_type'],
                                properties=True,
                                class_info=True,
                                resolution=resolution)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers, drop_last=True)

    test_set = DataGenerator(root=info['dataroot'], 
                                mode='test',  
                                max_objects = exp_arguments['num_slots'],
                                reasoning_type = info['reasoning_type'],
                                properties=True,
                                class_info=True,
                                resolution=resolution)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers, drop_last=True)

    # =====================================================================================
    
    model.train()
    train_stats = training_step(train_dataloader, model, optimizer)

    # =====================================================================================

    model.eval()
    # Evaluation metrics....
    with torch.no_grad():
        model.eval()
        hmc = 0.0
        pred = []; attributes = []; properties = []; labels = []
        for batch_num, samples in tqdm(enumerate(test_dataloader), desc='calculating Acc-F1 '):
            if batch_num*opt.batch_size > 2500: break

            image = samples['image'].to(model.device)
            ppty  = samples['properties'].to(model.device)
            targets = samples['target'].to(model.device)

            logits, pred_properties, cbidxs, qloss, perplexity = model(image, MCsamples=1)

            labels.append(targets)
            attributes.append(ppty)
            properties.append(pred_properties)
            pred.append(torch.argmax(logits, -1))

            if info['config'].__contains__('DefaultCNN'):
                hmc += 0.0
            else:
                hmc  += ReliableReasoningIndex(pred_properties, ppty).item()

        pred = torch.cat(pred, 0).cpu().numpy()
        labels = torch.cat(labels, 0).cpu().numpy()

        hmc = hmc*1.0/(batch_num+1)

       

        acc = accuracy_score(labels, pred)
        f1  = f1_score(labels, pred, average='macro')

        logs = {'Accuracy': acc, 'F1': f1, 'HMC': hmc}
        
        return logs



if __name__ == '__main__':
    infos = [
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/clevrhans3default/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('clevr', 'hans7', 'default')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/clevrhans7default/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('clevr', 'hans3', 'default')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/clevrhans3default/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('clevr', 'hans7', 'default')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/clevrhans7default/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('clevr', 'hans3', 'default')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/clevrhans3default/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('clevr', 'hans7', 'default')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/clevrhans7default/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('clevr', 'hans3', 'default')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/clevrhans3default/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('clevr', 'hans7', 'default')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/clevrhans7default/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('clevr', 'hans3', 'default')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/clevrhans3default/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('clevr', 'hans7', 'default')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/clevrhans7default/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('clevr', 'hans3', 'default')
        },

        #===================================================== n2

        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2add/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2add/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2diff/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'add')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2diff/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2mixed/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2mixed/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'add')
        },


        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2add/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2add/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2diff/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'add')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2diff/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2mixed/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2mixed/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'add')
        },

        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2add/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2add/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2diff/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'add')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2diff/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2mixed/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2mixed/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'add')
        },

        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2add/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2add/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2diff/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'add')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2diff/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2mixed/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2mixed/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'add')
        },

        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2add/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2add/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2diff/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'add')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2diff/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2mixed/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn2mixed/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n2', 'add')
        },


        #=====================================================n3
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3add/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3add/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3diff/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'add')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3diff/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3mixed/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3mixed/Reasoning/Baseline/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'add')
        },


        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3add/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3add/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3diff/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'add')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3diff/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3mixed/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3mixed/Reasoning/DefaultCNN/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'add')
        },

        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3add/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3add/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3diff/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'add')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3diff/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3mixed/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3mixed/Reasoning/Gumble/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'add')
        },

        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3add/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3add/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3diff/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'add')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3diff/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3mixed/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3mixed/Reasoning/Cosine/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'add')
        },

        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3add/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3add/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3diff/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'add')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3diff/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'mixed')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3mixed/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'diff')
        },
        {
            'config': '/vol/biomedic3/agk21/testEigenSlots2/LOGS-RESASONING-RRI/floatingMNISTn3mixed/Reasoning/Euclidian/exp-parameters.json',
            **get_dataroot('floatingMNIST', 'n3', 'add')
        },

    ]
    

    configs  = []
    variants = []
    reasoning_types = []
    accs = []
    hmcs = []
    f1s  = []

    for i, info in enumerate(infos):
        logs = get_task_adaptability_results(info)

        configs.append(info['config'])
        variants.append(info['variants'])
        reasoning_types.append(info['reasoning_type'])
        accs.append(logs['Accuracy'])
        hmcs.append(logs['HMC'])
        f1s.append(logs['F1'])
        
        df = pd.DataFrame({'config': configs, 'reasoning_type': reasoning_types, 'variants': variants, 'acc': accs, 'hmcs': hmcs, 'f1s': f1s})
        df.to_csv(f'/vol/biomedic3/agk21/testEigenSlots2/CSVS/Adaptability_test_k={opt.num_batches}.csv')