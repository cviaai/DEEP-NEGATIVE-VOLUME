import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
import time
import pandas as pd
import argparse
import torch
import json
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from early_stopping import EarlyStopping
from losses import DiceCrossEntropyLoss, hausdorff_distance
from dataloader import get_train_loaders 
from losses import hausdorff_distance
from vnet import VNet
from model import UNet3D, UNet3D_attention




def iter_to_patient(i, test_patient_ids):
    patient_id = test_patient_ids[i%len(test_patient_ids)] 
    if i//len(test_patient_ids) == 0:
        side = 'L'
    else:
        side = 'R'
    return patient_id, side


def get_model(config):
    model_name = config['training']['model_name']
    device = config['training']['device']
    if model_name == 'UNet':
        model = UNet3D(in_channels=1, out_channels=2,layer_order='crg',f_maps=32,
                             num_groups=8,final_sigmoid=False,device=device)
    elif model_name == "UNetAtt":
        model = UNet3D_attention(in_channels=1, out_channels=2,layer_order='crg',f_maps=32,
                             num_groups=8,final_sigmoid=False,device=device)
    else: # model_name == VNet
        model = VNet()
    return model

def weights_init(m):
    classname = m.__class__.__name__
    

def predict(model, val_loader, config, criterion):
    
    test_results = pd.DataFrame(columns = ['patient_id','side','loss', 'ce', 'dice','IoU','HD'])
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    model_path = config['predict']['model_path'] 
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    

    hd_losses, l_losses, ce_losses, dice_losses, iou_losses = [], [], [], [], []
    test_patient_ids = config['loaders']['val_patient_ids']
    model.train(False)
    for p, input in enumerate(val_loader):
        images, labels = input
        patient_id, side = iter_to_patient(p, test_patient_ids)
        outputs = model(images[None,:].to(device,torch.float32))

        hd = hausdorff_distance(outputs, labels)
        hd_losses.append(hd)
        l = criterion(outputs,labels.cuda(outputs.device))
        l_losses.append(l.item())
        ce_losses.append(criterion.ce)
        dice_losses.append(criterion.dice)
        iou_losses.append(criterion.IoU)

        test_results.loc[len(test_results)] = [patient_id, side, l.item(), criterion.ce, \
                                               criterion.dice,criterion.IoU, hd]

    print('Test patient {}'.format(test_patient_ids))   
    print('Loss: {:.3f} +\- {:.3f}'.format(np.mean(l_losses), np.std(l_losses)))
    print('CrossEntropy: {:.3f} +\- {:.3f}'.format(np.mean(ce_losses), np.std(ce_losses)))
    print('Dice: {:.3f} +\- {:.3f}'.format(np.mean(dice_losses), np.std(dice_losses)))
    print('IoU: {:.3f}  +\-{:.3f}'.format(np.mean(iou_losses), np.std(iou_losses)))
    print('Hausdorff Distance: {:.3f} +\- {:.3f}'.format(np.mean(hd_losses), np.std(hd_losses)))
                
    test_results.to_csv('test_results.csv', encoding='utf-8', index=None) 

def load_config():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as infile:
        config = json.load(infile)
        
    return config
  
    
def main():
    # Load experiment configuration
    config = load_config()
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = get_model(config)


    loss_type=config['loss']['loss_type']
    w0, w1 = config['loss']['w0'],config['loss']['w1']
    ce_weights = [w0, w1]
    dce_w=config['loss']['dce_w']
    nll = config['loss']['nll']
    criterion = DiceCrossEntropyLoss(loss=loss_type, logging_name=None, ce_weights = ce_weights, \
                                         dce_weight=dce_w, nll=nll)
    loaders = get_train_loaders(config)
    val_loader = loaders['val']

    predict(model, val_loader, config, criterion)
    
    
if __name__ == '__main__':
    main()