import os
import argparse
from src.dataset import *
from src.model import SlotAttentionAutoEncoder
from src.metrics import compositional_fid

from tqdm import tqdm
import time, math
from datetime import datetime, timedelta

import torch
import json
import pandas as pd 



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true', '1')


parser.add_argument('--batch_size', default=40, type=int)
parser.add_argument('--num_batches', default=1000, type=int)
parser.add_argument('--num_slots', default=4, type=int)
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')

opt = parser.parse_args()



def get_computational_fid(config):
    # load parameters....
    exp_arguments = json.load(open(config, 'r'))
    print(exp_arguments)
    print ('='*25)

    resolution = (exp_arguments['img_size'], exp_arguments['img_size'])
    # ===========================================
    # model init
    model = SlotAttentionAutoEncoder(resolution, 
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
                                        exp_arguments['kld_scale']).to(device)


    ckpt=torch.load(os.path.join(exp_arguments['model_dir'], 'discovery_best.pth' ))
    model.load_state_dict(ckpt['model_state_dict'])
    model.device = device


    test_set = DataGenerator(root=exp_arguments['data_root'], 
                                mode='test',
                                resolution=resolution)
    test_dataloader = torch.utils.data.DataLoader(test_set, 
                                        batch_size=opt.batch_size,
                                        shuffle=True, 
                                        num_workers=opt.num_workers, 
                                        drop_last=True)

    test_epoch_size = min(10000, len(test_dataloader))

    CFID, CSFID = compositional_fid(test_dataloader,
                                sfid        = True,
                                model       = model,
                                ns          = opt.num_slots,
                                device      = device,
                                batch_size  = opt.batch_size,
                                num_batches = opt.num_batches,
                                fid_dir     = os.path.join(f'LOGS-IMAGECOMPOSITION-ns-{opt.num_slots}', exp_arguments['exp_name']))

    print(CFID, CSFID, '====================')
    return CFID, CSFID


if __name__ == '__main__':
    configs = [
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/bitmojidefaultCosine/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/bitmojidefaultEuclidian/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/bitmojidefaultGumble/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/clevrdefaultEuclidian/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/clevrdefaultCosine/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/clevrdefaultGumble/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/objects_roomdefaultCosine/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/objects_roomdefaultEuclidian/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/objects_roomdefaultGumble/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/tetrominoesdefaultCosine/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/tetrominoesdefaultEuclidian/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/tetrominoesdefaultGumble/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/ffhqdefaultCosine/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/ffhqdefaultEuclidian/exp-parameters.json',
                '/vol/biomedic3/agk21/testEigenSlots2/LOGS-IMPLICIT2/ObjectDiscovery/ffhqdefaultGumble/exp-parameters.json',
                ]
    
    cfids = []
    csfids = []

    for i, config in enumerate(configs):
        fid, sfid = get_computational_fid(config)

        cfids.append(fid)
        csfids.append(sfid)

        df = pd.DataFrame({'config': configs[:i+1], 'CFID': cfids, 'CSFID': csfids})
        df.to_csv(f'/vol/biomedic3/agk21/testEigenSlots2/CSVS/compositional_fids_{opt.num_slots}.csv')