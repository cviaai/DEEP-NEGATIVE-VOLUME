import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.autograd import Function
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk

def mean_dist(box_output, box_gt):
    mean_distance = 0
    for p in range(len(box_output)):
        q = np.sqrt(((np.array(box_output[p]) - np.array(box_gt[p]))**2).sum())
        mean_distance +=q
    mean_distance = mean_distance/len(box_output)
    return mean_distance

def hausdorff_distance(input, target):
    _, result = input.max(1)
    result = torch.squeeze(result)
    target = torch.squeeze(target)
    result_np = result.data.cpu().numpy()
    label_np = target.data.cpu().numpy()

    output_indexes = np.where(result_np == 1.0)
    sitk_output = sitk.GetImageFromArray(result_np)

    label_indexes = np.where(label_np == 1.0)
    sitk_label = sitk.GetImageFromArray(label_np)
    
    if (result_np.sum()==0) or (label_np.sum()==0):
         h_dist = 0
    else:      
        # Note the reversed order of access between SimpleITK and numpy (z,y,x)
        if len(output_indexes) == 3:
            physical_points_output = [sitk_output.TransformIndexToPhysicalPoint([int(x), int(y), int(z)]) \
                       for z,y,x in zip(output_indexes[0], output_indexes[1], output_indexes[2])]

            physical_points_label = [sitk_label.TransformIndexToPhysicalPoint([int(x), int(y), int(z)]) \
                       for z,y,x in zip(label_indexes[0], label_indexes[1], label_indexes[2])]
        if len(output_indexes) == 2:
            physical_points_output = [sitk_output.TransformIndexToPhysicalPoint([int(x), int(y)]) \
                       for y,x in zip(output_indexes[0], output_indexes[1])]

            physical_points_label = [sitk_label.TransformIndexToPhysicalPoint([int(x), int(y)]) \
                       for y,x in zip(label_indexes[0], label_indexes[1])]

        h_dist_lo, u_ind, v_ind  = directed_hausdorff(u = physical_points_label, v = physical_points_output)
        h_dist_ol, u_ind, v_ind  = directed_hausdorff(u = physical_points_output, v = physical_points_label)
        h_dist = max(h_dist_lo, h_dist_ol)
    
    return h_dist
        

class DiceLoss(Function):
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def forward(ctx, input, target):    
        eps = 1e-6
        _, result = input.max(1)
        result = torch.squeeze(result)
        target = torch.squeeze(target)
        if (target.is_cuda) or (input.is_cuda):
            result = torch.cuda.FloatTensor(result.to(torch.float32)) 
            target = torch.cuda.FloatTensor(target.to(result.device,torch.float32)) 
        else:
            result = torch.FloatTensor(result.to(torch.float32))
            target = torch.FloatTensor(target.to(torch.float32))
        ctx.target = target
        intersect = (result*target).sum()
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        uni = (result+target) > 0
        union = torch.sum(uni.to(torch.float32))
        sum_of_pixels = result_sum + target_sum + (2*eps)
        ctx.save_for_backward(input, target, intersect, sum_of_pixels)
        ctx.IoU = intersect / (union + eps)
        dice = 2*intersect/sum_of_pixels
        ctx.dice = dice
        out = torch.FloatTensor(1).fill_(ctx.dice).to(input.device)
        ctx.intersect, ctx.union, ctx.sum_of_pixels = intersect, union, sum_of_pixels
        tn_ind = (result+target) == 0
        tn = torch.sum(tn_ind.to(torch.float32))
        subtraction_of_pixels = result_sum - target_sum 
        ctx.VS = 1-torch.abs(subtraction_of_pixels/sum_of_pixels)
        ctx.sensitivity = intersect/(target_sum + eps)
        ctx.specificity = tn/(tn+result_sum-intersect)
        return out  
    
    @staticmethod
    def backward(ctx, grad_output):
        input, target, intersect, sum_of_pixels = ctx.saved_tensors
        gt = torch.div(target, sum_of_pixels)
        IoU2 = intersect/(sum_of_pixels*sum_of_pixels)
        pred = IoU2*input[0, 1]
        dDice = 2*gt-4*pred
        grad_input = torch.cat((torch.mul(dDice, -grad_output[0])[None,...], 
                                torch.mul(dDice, grad_output[0])[None,...]),0)[None,...]

        return grad_input, None
      
    @classmethod
    def metrics(cls, input, target):
        eps = 1e-6
        _, result = input.max(1)
        result = torch.squeeze(result)
        target = torch.squeeze(target)
        if (target.is_cuda) or (input.is_cuda):
            result = torch.cuda.FloatTensor(result.to(torch.float32)) 
            target = torch.cuda.FloatTensor(target.to(result.device,torch.float32)) 
        else:
            result = torch.FloatTensor(result.to(torch.float32))
            target = torch.FloatTensor(target.to(torch.float32))
        cls.target = target
        intersect = (result*target).sum()
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        uni = (result+target) > 0
        union = torch.sum(uni.to(torch.float32))
        sum_of_pixels = result_sum + target_sum + (2*eps)
        iou = intersect / (union + eps)
        cls.IoU = iou
        dice = 2*intersect/sum_of_pixels
        cls.dice = dice  
        return dice, iou


class DiceCrossEntropyLoss(nn.Module):
    """This criterion represents linear compination of dice lossand cross-entropy.
    Args:
        loss ('DCE','CE','D'): type of loss, 'CE' for cross-entropy, 'D' for dice loss and 'DCE' - their combination
        logging_name (str): path to the logging file.
        ce_weights (list): a manual rescaling weight given to each class. Default is [1, 1].
        dce_weight: a weight given to dice part of the loss. Default is 1.
        nll (True, False): if it is True, then nn.NLL function is used as cross-entropy, otherwise nn.CrossEntropy. 
        The value should be True for current modification of  Unet3d
    """
    def __init__(self, loss='CE', logging_name=None, ce_weights=[1., 1.], dce_weight = 1., nll = False,**kwargs):
        super(DiceCrossEntropyLoss, self).__init__()
        self.loss = loss
        self.logging_name = logging_name
        self.ce_weights = ce_weights
        self.dce_weight = dce_weight
        self.nll = nll
        pass
        
    def forward(self, input, target):
        d_loss = DiceLoss()
        dl = 1- d_loss.apply(input, target)
        m = d_loss.metrics(input, target)
        dice, IoU = m[0], m[1] # d_loss.dice.item()
        # self.IoU = d_loss.IoU.item()
        self.dice, self.IoU =  dice.item(), IoU.item() 
        # self.specificity = d_loss.specificity.item()
        # self.sensitivity = d_loss.sensitivity.item()
        # self.VS = d_loss.VS.item()
        
        if self.nll:
            ce_loss = nn.NLLLoss(weight=torch.Tensor(self.ce_weights).to(input.device)) #[0.25, 0.75]
            ce = ce_loss(torch.log(input), target.long())
        else: 
            ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor(self.ce_weights).to(input.device))
            ce = ce_loss(input, target.long())
        self.ce = ce.item()
 
        if self.loss == 'CE':
            out = ce      
        elif self.loss == 'D':
            out = dl  
        else: # self.loss == 'DCE'
            out = torch.add(dl, self.dce_weight*ce)
            
        if self.logging_name is not None:
            file = open(self.logging_name, 'a')
            file.write('CrossEntropy: {:.3f}\t Dice: {:.3f}\n'.format(ce.item(),self.dice))
            file.close()
             
        return out
    
  