import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
from torch.utils.data import Dataset, DataLoader, ConcatDataset, RandomSampler
import torchvision
import imageio
import importlib
import random
import glob
import os

import transforms_3d 
    
class patch_DS(Dataset):
    """Implementation of torch.utils.data.Dataset for set of .tiff files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """      
    def __init__(self, root_dcm, root_mask, phase, transformer_config, patient_ids, patch_shape, stride_shape, patch_builder_cls,
                 voi_shape, precrop, seed_fn=None):
        """
        Args:
            root_dcm: path to directory containing raw data.
            root_mask: path to directory containing label data.
            phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase.
            transformer_config: dictionary of transformations and parameters for data augmentation.
            patient_ids: set of patients' ids for dataset during the phase.
            patch_shape: the shape of the patch DxHxW.
            stride_shape: the shape of the stride DxHxW.
            slice_builder_cls: defines how to sample patches from the image.
            voi_shape: shape of each image DxHxW.
            precrop: necessity of precroppping.
        """   
        self.root_dcm = root_dcm
        self.root_mask = root_mask
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.transformer_config = transformer_config
        self.patient_ids = patient_ids
        self.patch_shape = patch_shape
        self.stride_shape = stride_shape
        self.patch_builder_cls = patch_builder_cls
        self.voi_shape = voi_shape
        self.precrop = precrop
        self.seed_fn = seed_fn

        self.to_tensor_transform = torchvision.transforms.ToTensor()
        self.filenames_dcm = []
        self.filenames_mask = []
        self.raws = []
        self.labels = []
        for i in patient_ids:
            filenames_img = glob.glob(os.path.join(root_dcm+str(i), '*.tiff'))
            filenames_img.sort()
            filenames_m = [x.replace('dicom','mask') for x in filenames_img]
            self.filenames_dcm.append(filenames_img)
            self.filenames_mask.append(filenames_m)
            depth = len(filenames_img)
            if self.precrop:
                z1 = (depth-self.voi_shape[2])//2
                z2 = z1+self.voi_shape[2]
            else:
                z1 = 0
                z2 = depth
            # read raw scan
            raw_img = np.zeros(self.voi_shape, dtype='uint8') # create zero image  
            for fn in filenames_img[z1:z2]:
                img = imageio.imread(fn) 
                if self.precrop:
                    img = self._center_crop(img,self.voi_shape[:2])
                raw_img[:, :, filenames_img[z1:z2].index(fn)] = img
            self.raws.append(raw_img)
            # read mask for scan
            label_img = np.zeros(self.voi_shape, dtype='uint8') # create zero mask 
            for fn in filenames_m[z1:z2]:
                m = imageio.imread(fn) 
                if self.precrop:
                    m = self._center_crop(m,self.voi_shape[:2])
                label_img[:, :, filenames_m[z1:z2].index(fn)] = m
            self.labels.append(label_img)
        self.raws = np.array(self.raws)
        self.labels = np.array(self.labels)
        
        min_value, max_value, mean, std = self._calculate_stats(self.raws)
        print (f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')
        self.transformer = transforms_3d.get_transformer(self.transformer_config, min_value=min_value, max_value=max_value,
                                                          mean=mean, std=std, phase=self.phase)
        self.raw_transform = self.transformer.raw_transform()
        self.label_transform = self.transformer.label_transform()
        patch_builder = patch_builder_cls(self.raws, self.labels, patch_shape, stride_shape)
        self.raw_patches = patch_builder.raw_patches
        self.label_patches= patch_builder.label_patches
        self.len = len(self.raw_patches)

    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if self.seed_fn:
            self.seed_fn(seed)
            
    @staticmethod
    def _calculate_stats(inputs):
        return np.min(inputs), np.max(inputs), np.mean(inputs), np.std(inputs)
            
    def __getitem__(self, index):
        raw_idx = self.raw_patches[index]
        label_idx = self.label_patches[index]
        
        image = self.raws[raw_idx]
        image = image.reshape(self.patch_shape)
        mask = self.labels[label_idx]
        mask = mask.reshape(self.patch_shape)

        seed = random.randint(0, 2**32)
        self._set_seed(seed)
        image = self.raw_transform(image)
        image = self.to_tensor_transform(image)
            
        self._set_seed(seed)
        mask = self.label_transform(mask)
        mask = self.to_tensor_transform(mask)
        
        return image, mask

    def _center_crop(self, img, roi_shape):
        y_size, x_size = roi_shape
        y1 = (img.shape[0]-y_size)//2
        x1 = (img.shape[1]-x_size)//2  
        return img[y1:y1+y_size, x1:x1+x_size]

    def __len__(self):
        return self.len 


class PatchBuilder:
    """Sample patches from the image."""
    def __init__(self, raw_dataset, label_dataset, patch_shape, stride_shape):
        """
        Args:
            raw_dataset: array of raw data.
            label_dataset: array of label data.
            patch_shape: the shape of the patch DxHxW.
            stride_shape: the shape of the stride DxHxW.
        """   
        self._raw_patches = self._build_patches(raw_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            self._label_patches = None
        else:
            self._label_patches = self._build_patches(label_dataset, patch_shape, stride_shape)

    @property
    def raw_patches(self):
        return self._raw_patches

    @property
    def label_patches(self):
        return self._label_patches

    @staticmethod
    def _build_patches(dataset, patch_shape, stride_shape):
        """Iterate over a given dataset patch-by-patch with a given stride and builds an array of slice positions.
        
        Args:
            dataset: array of label data.
            patch_shape: the shape of the patch DxHxW.
            stride_shape: the shape of the stride DxHxW.
            
        Returns:
            list of slices [(slice, slice, slice, slice), ...] 
        """
        slices = []
        assert len(dataset.shape) == 4, 'Supports only 4D (NxDxHxW)'
        num_patients, i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        for p in range(num_patients):
            z_steps = PatchBuilder._gen_indices(i_z, k_z, s_z)
            for z in z_steps:
                y_steps = PatchBuilder._gen_indices(i_y, k_y, s_y)
                for y in y_steps:
                    x_steps = PatchBuilder._gen_indices(i_x, k_x, s_x)
                    for x in x_steps:
                        slice_idx = (
                            slice(z, z + k_z),
                            slice(y, y + k_y),
                            slice(x, x + k_x)
                        )
                        slice_idx = (slice(p, p+1),) + slice_idx # patient id
                        slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        """
        Args:
            i (int): image size.
            k (int): patch size.
            s (int): stride size.
        Returns:
            generator of slides start positions
        """
        assert i >= k, 'Sample size should be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if (j + k < i)&(i!=s):
            yield i - k      
            
            
def _get_patch_builder_cls(class_name):
    m = importlib.import_module('dataloader')
    clazz = getattr(m, class_name)
    return clazz


def get_train_loaders(config):
    """Return dictionary containing the training and validation loaders (torch.utils.data.DataLoader).
    
    Args:
        config: a top level configuration object containing the 'loaders' key.
    
    Returns: dict {'train': <train_loader>, 'val': <val_loader>}: dictionary containing the training and validation loaders.
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']
    print ('Creating training and validation set loaders...')
    # get train and validation files
    objects = loaders_config['objects']
    assert isinstance(objects, list)
    voi_shape = loaders_config['voi_shape']
    dicom_path = loaders_config['dicom_path']
    mask_path = loaders_config['mask_path']
    train_ids = tuple(loaders_config['train_patient_ids'])
    train_patch = tuple(loaders_config['train_patch'])
    train_stride = tuple(loaders_config['train_stride'])
    val_ids = tuple(loaders_config['val_patient_ids'])
    val_patch = tuple(loaders_config['val_patch'])
    val_stride = tuple(loaders_config['val_stride'])
    transformer_config = loaders_config['transformer']
    precrop = loaders_config['precrop']
    # get train slice_builder_cls
    train_patch_builder_str = loaders_config.get('train_patch_builder', 'PatchBuilder')
    print (f'Train s builder class: {train_patch_builder_str}')
    train_patch_builder_cls = _get_patch_builder_cls(train_patch_builder_str)
    
    train_datasets = []
    for obj in objects:
        root_dcm = dicom_path+'_'+obj+ '/'
        root_mask = mask_path+'_'+obj+ '/'
        try:
            print (f'Loading training set from: {root_dcm}...')
            train_dataset = patch_DS(root_dcm, root_mask, 'train', transformer_config, 
                                     train_ids, train_patch, train_stride,
                                     train_patch_builder_cls, voi_shape,precrop, seed_fn=None)
            train_datasets.append(train_dataset)
        except Exception:
            print (f'Skipping training set: {root_dcm}')

    # get val slice_builder_cls
    val_patch_builder_str = loaders_config.get('val_patch_builder', 'PatchBuilder')
    print (f'Val patch builder class: {val_patch_builder_str}')
    val_patch_builder_cls = _get_patch_builder_cls(val_patch_builder_str)

    val_datasets = []
    for obj in objects:
        root_dcm = dicom_path+'_'+obj+ '/'
        root_mask = mask_path+'_'+obj+ '/'
        try:
            print (f'Loading val set from: {root_dcm}...')
            val_dataset = patch_DS(root_dcm, root_mask, 'val', transformer_config, 
                                   val_ids, val_patch, val_stride,
                                   val_patch_builder_cls, voi_shape, precrop, seed_fn=None)
            val_datasets.append(val_dataset)
        except Exception:
            print(f'Skipping val set: {root_dcm}')

    num_workers = loaders_config.get('num_workers', 1)
    print (f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    print (f'Batch size for train/val loader: {batch_size}')
    train_dataset_size = loaders_config.get('train_dataset_size', 1)
    train_rand_sampler = RandomSampler(ConcatDataset(train_datasets), replacement=True, num_samples=train_dataset_size)
    return {'train': DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=False, sampler=train_rand_sampler,
                            num_workers=num_workers),
            'val': DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False, num_workers=num_workers)
           }
           
        