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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias != None:
            m.bias.data.zero_()
        
def choose_test_sample(test_patient_ids, config, train_size=8):
    all_patient_ids = list(np.arange(1,11))
    train_patient_ids = np.delete(all_patient_ids, np.array(test_patient_ids) - 1)
    exclude = np.random.choice(train_patient_ids,10-len(test_patient_ids)-train_size,replace=False)
    for i in exclude:
        train_patient_ids = list(train_patient_ids)
        train_patient_ids.remove(i)

    config['loaders']['val_patient_ids'] = test_patient_ids
    config['loaders']['train_patient_ids'] = train_patient_ids
    print ('Train patient ids:',train_patient_ids)
    print ('Test patient ids:',test_patient_ids)
    return config

def k_folds(k = 5):
    folds = []
    for i in range(k):
        test_patient_ids = [2*i+1,2*i+2]
        folds.append(test_patient_ids)
    return folds

def iter_to_patient(i, test_patient_ids):
    patient_id = test_patient_ids[i%len(test_patient_ids)] 
    if i//len(test_patient_ids) == 0:
        side = 'L'
    else:
        side = 'R'
    return patient_id, side

def class_weights(loaders):
    w1_vals = []
    for images, labels in loaders['train']:
        w1 = labels.mean()
        w1_vals.append(w1)
    w1_mean = np.mean(w1_vals)
    w0 = np.round(w1_mean,2)
    w1 = 1-w1_mean
    w1 = np.round(w0,2)
    return w0,w1


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

def train(model, config, optimizer, scheduler, criterion, early_stopping=None):
    loss_convergence = pd.DataFrame(columns = ['k', 'epoch', 'mode','iter', 'loss', 'ce', 'dice','IoU'])
    test_results = pd.DataFrame(columns = ['k', 'final_epoch', 'patient_id','side','loss', 'ce', 'dice','IoU','HD'])
    batch_size = config['training']['batch_size']
    k = config['training']['k_follds_number']
    folds = k_folds(k)
    num_epochs = config['training']['num_epochs']
    model_name = config['training']['model_name']
    for f, test_patient_ids in enumerate(folds):
        print ('{} of {}-fold tests starts...'.format(f+1, k))
        trainsize = config['training']['trainsize']
        config = choose_test_sample(test_patient_ids, config, trainsize)
        loaders = get_train_loaders(config)
        
        model.apply(weights_init)
        device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_dice, val_dice = [], []
        train_IoU, val_IoU = [], []
        train_ce, val_ce = [], []
        train_loss, val_loss = [], []
        
        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            print('Training: Epoch {}, LR {:.8f}'.format(epoch+1, scheduler.get_lr()[0]))

            b = 0
            for images, labels in loaders['train']:
                images = images[None,:].to(device, torch.float32)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss =  criterion(outputs, labels) 

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

                train_dice.append(criterion.dice)
                train_IoU.append(criterion.IoU)
                train_ce.append(criterion.ce)

                loss_convergence.loc[len(loss_convergence)] = [f+1,epoch+1,'train',b,loss.item(),criterion.ce,criterion.dice,criterion.IoU]
                b+=1


            print('Validation starts... ')
            model.train(False)
            b = 0
            for images, labels in loaders['val']:
                images = images[None,:].to(device, torch.float32)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss.append(loss.data.cpu().numpy())
                val_dice.append(criterion.dice)
                val_IoU.append(criterion.IoU)
                val_ce.append(criterion.ce)

                loss_convergence.loc[len(loss_convergence)] = [f+1,epoch+1,'test',b,loss.item(),criterion.ce,criterion.dice,criterion.IoU]
                b+=1

            val_l = np.mean(val_loss[-len(loaders['val']) // batch_size :])

            scheduler.step()
  
            print("Epoch {}/{} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
            print("train loss : \t{:.3f} +\- {:.3f}\t".format(np.mean(train_loss[-len(loaders['train']) // batch_size :]),np.std(train_loss[-len(loaders['train']) // batch_size :])))
            print("val loss: \t{:.3f} +\- {:.3f}".format(np.mean(val_loss[-len(loaders['val']) // batch_size :]),np.std(val_loss[-len(loaders['val']) // batch_size :])))
            print("train Dice : \t{:.3f} +\- {:.3f}\t".format(np.mean(train_dice[-len(loaders['train']) // batch_size :]),np.std(train_dice[-len(loaders['train']) // batch_size :])))
            print("val Dice: \t{:.3f} +\- {:.3f}".format(np.mean(val_dice[-len(loaders['val']) // batch_size :]),np.std(val_dice[-len(loaders['val']) // batch_size :])))
            print("train IoU : \t{:.3f} +\- {:.3f}\t".format(np.mean(train_IoU[-len(loaders['train']) // batch_size :]),np.std(train_IoU[-len(loaders['train']) // batch_size :])))
            print("val IoU: \t{:.3f} +\- {:.3f}".format(np.mean(val_IoU[-len(loaders['val']) // batch_size :]),np.std(val_IoU[-len(loaders['val']) // batch_size :])))
            print("train CE : \t{:.3f} +\- {:.3f}\t".format(np.mean(train_ce[-len(loaders['train']) // batch_size :]),np.std(train_ce[-len(loaders['train']) // batch_size :])))
            print("val CE: \t{:.3f} +\- {:.3f}".format(np.mean(val_ce[-len(loaders['val']) // batch_size :]),np.std(val_ce[-len(loaders['val']) // batch_size :])))
            print('-' * 50)


            if early_stopping is not None:
                early_stopping(val_l, model, epoch+1)

                if early_stopping.early_stop:
                    print("Early stopping!")
                    break

        
        path = F"./{str(model_name)+'_'+str(epoch+1)+'e_'+str(f+1)+'.pth'}"
        torch.save(model.state_dict(), path)

        p = 0
        hd_losses, l_losses, ce_losses, dice_losses, iou_losses = [], [], [], [], []
        model.train(False)
        for images, labels in loaders['val']:
            patient_id, side = iter_to_patient(p, test_patient_ids)
            outputs = model(images[None,:].to(device,torch.float32))

            hd = hausdorff_distance(outputs, labels)
            hd_losses.append(hd)
            l = criterion(outputs,labels.cuda(outputs.device))
            l_losses.append(l.item())
            ce_losses.append(criterion.ce)
            dice_losses.append(criterion.dice)
            iou_losses.append(criterion.IoU)

            test_results.loc[len(test_results)] = [f+1, epoch+1, patient_id, side, l.item(), criterion.ce, \
                                                   criterion.dice,criterion.IoU, hd]
            p+=1

        print('Test patient {}'.format(test_patient_ids))   
        print('Loss: {:.3f} +\- {:.3f}'.format(np.mean(l_losses), np.std(l_losses)))
        print('CrossEntropy: {:.3f} +\- {:.3f}'.format(np.mean(ce_losses), np.std(ce_losses)))
        print('Dice: {:.3f} +\- {:.3f}'.format(np.mean(dice_losses), np.std(dice_losses)))
        print('IoU: {:.3f} +\- {:.3f}'.format(np.mean(iou_losses), np.std(iou_losses)))
        print('Hausdorff Distance: {:.3f} +\- {:.3f}'.format(np.mean(hd_losses), np.std(hd_losses)))
        print('-' * 50)
                
    loss_convergence.to_csv('loss_convergence.csv', encoding='utf-8', index=None)  
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

    # Create the model
    device = config['training']['device']
    model = get_model(config)
    
    learning_rate = config['training']['learning_rate']
    # momentum = config['training']['momentum']
    wd=config['training']['wd']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd) 
    # betas=(0.9, 0.999), eps=1e-08, amsgrad=False
    step_size = config['training']['step_size']
    gamma = config['training']['gamma']
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=-1)

    patience = config['training']['patience']
    delta = config['training']['delta']
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta, checkpoint_path=None)
    # Create loss criterion
    loss_type=config['loss']['loss_type']
    w0, w1 = config['loss']['w0'],config['loss']['w1']
    ce_weights = [w0, w1]
    dce_w=config['loss']['dce_w']
    nll = config['loss']['nll']
    criterion = DiceCrossEntropyLoss(loss=loss_type, logging_name=None, ce_weights = ce_weights, \
                                         dce_weight=dce_w, nll=nll)

    # Start training
    train(model, config, optimizer, scheduler, criterion, early_stopping=None)
    
    
if __name__ == '__main__':
    main()