# Based on: https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/augment/transforms.py
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torchvision
from torchvision.transforms import Compose
from scipy.ndimage import rotate, zoom, map_coordinates, gaussian_filter
from scipy.ndimage.filters import convolve
from skimage.filters import gaussian
import importlib

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
GLOBAL_RANDOM_STATE = np.random.RandomState(47)

class CenterCrop:
    """Crop the given image at the center.

    Args:
        size (sequence or int): desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size, **kwargs):
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (array): image to be cropped.

        Returns:
            img (array): cropped image.
        """
        assert img.ndim in [3], 'Supports only 3D (DxHxW)'
        t0 = torchvision.transforms.ToPILImage()
        t = torchvision.transforms.CenterCrop(self.size)
        depth = img.shape[2]
        s1 = (depth-self.size)//2
        s2 = s1+self.size
        channels = [t(t0(img[:,:,c])) for c in range(s1, s2)]
        img = np.stack(channels, axis=2)
        return img
    
class Resize:
    """Resize the input image to the given size with interpolation PIL.Image.BILINEAR

    Args:
        resize_f: scale factor along the axes for resizing.
        size (sequence or int): desired output size. If size is a sequence like
        (h, w), output size will be matched to this. If size is an int, 
        smaller edge of the image will be matched to this number.
        i.e, if height > width, then image will be rescaled to
        (size * height / width, size). Default size is None.
        
    """
    def __init__(self, resize_f, size=None, **kwargs):
        self.resize_f = resize_f
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (array): image to be scaled.

        Returns:
            img (array): rescaled image.
        """
        assert img.ndim in [3], 'Supports only 3D (DxHxW)'
        if self.size is None:
            depth = img.shape[2]
            size = depth//self.resize_f
            img = img[:,:,range(0,depth,self.resize_f)]
            t0 = torchvision.transforms.ToPILImage()
            t = torchvision.transforms.Resize(size)
            channels = [t(t0(img[:,:,c])) for c in range(img.shape[2])]
            img = np.stack(channels, axis=2)
        else:
            depth = img.shape[2]
            resize_f = depth//self.size
            img = img[:,:,range(0,depth,resize_f)]
            t0 = torchvision.transforms.ToPILImage()
            t = torchvision.transforms.Resize(self.size)
            channels = [t(t0(img[:,:,c])) for c in range(img.shape[2])]
            img = np.stack(channels, axis=2)     
        return img
    
class Normalize:
    """ Apply simple min-max scaling to a given input image, shrink the range of the data in a fixed range of [-1, 1].
    
    Args:
        min_value: minimal value in the image
        max_value: maximal value in the image
    """

    def __init__(self, min_value, max_value, **kwargs):
        assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value 

    def __call__(self, img):
        """
        Args:
            img (array): image of size to be normalized.

        Returns:
            img (array): normalized image.
        """
        norm_0_1 = (img - self.min_value) / (self.max_value - self.min_value)
        return 2 * norm_0_1 - 1
    
class Standardize:
    """Normalize a image with mean and standard deviation, i.e. re-scaling the values to be 0-mean and 1-std.
    Mean and std parameter have to be provided explicitly.
    
    Args:
        mean (float): mean value of the image.
        std (float): standard deviation of the image.
    """

    def __init__(self, mean, std, **kwargs):
        self.mean = mean
        self.std = std
        self.eps = 1e-6

    def __call__(self, img):
        """
        Args:
            img (array): image to be standardized.

        Returns:
            img (array): standardized image.
        """
        self.mean = img.mean()
        self.std = img.std()
        res = (img - self.mean) / np.clip(self.std, a_min=self.eps, a_max=None)
        return res
    
class RandomFlip:
    """Flip the image across the axes randomly with a given probability. Image should be 3D (DxHxW).
    
    Args:
        axes (list): the list of axes across the image being flipped. Default value is [1] - horizontal flip.
        execution_probability (float): probability of the image being flipped. Default value is 0.5.
    """
    def __init__(self, random_state, axes = [1], execution_probability=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = axes
        self.execution_probability = execution_probability

    def __call__(self, img):
        """
        Args:
            img (array): image to be flipped.

        Returns:
            img (array): randomly flipped image.
        """
        assert img.ndim in [3], 'Supports only 3D (DxHxW) images'
        for axis in self.axes:
            if self.random_state.uniform() < self.execution_probability:
                img = np.flip(img, axis)
        return img

class RandomRotate:
    """
    Rotate the image by angle from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation plane is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=10, axes=None, mode='constant', order=0, **kwargs):
        """
        Args:
            angle_spectrum (float or int): Range of degrees to select from, the range of degrees
            will be (-degrees, +degrees). Default value is 0.
            axes (list): list of pairs of axes that define the planes of rotation. 
            mode ('reflect', 'constant', 'nearest', 'mirror', 'wrap'): the parameter determines how the input array is extended beyond its
            boundaries. Default is 'constant'. 
            order (int): the order of the spline interpolation, default is 0. The order has to be in the range 0-5.
        """
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, img):
        """
        Args:
            img (array): image to be rotated.

        Returns:
            img (array): rotated image.
        """
        assert img.ndim in [3], 'Supports only 3D (DxHxW)'
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)
        img = rotate(img, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        return img

class RandomContrast:
    """Adjust contrast by scaling each voxel to alpha.
    
    Args:
        alpha (tuple of float (min, max)): how much to change the contrast.
        execution_probability (float): probability of the image being changed. Default value is 0.1.
    
    """
    def __init__(self, random_state, alpha=(0.5, 1.), execution_probability=0.1, **kwargs):
        self.random_state = random_state
        self.alpha = alpha
        self.execution_probability = execution_probability

    def __call__(self, img):
        """
        Args:
            img (array): image for changing the contrast.

        Returns:
            img (array): image with randomly changed contrast.
        """
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            self.mean = img.mean() 
            result = self.mean + alpha * (img - self.mean)
            return np.clip(result, -1, 1)
        else:
            return img

class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    It's relatively slow, so use multiple workers in the DataLoader.
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """
    def __init__(self, random_state, spline_order, alpha=15, sigma=3, execution_probability=0.1, **kwargs):
        """
        Args:
            spline_order: the order of spline interpolation (use 0 for labeled images).
            alpha: scaling factor for deformations.
            sigma: smoothing factor for Gaussian filter.
            execution_probability (float): probability of the image being deformed. Default value is 0.1.
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, img):
        if self.random_state.uniform()<self.execution_probability:
            assert img.ndim in [3, 4], 'Supports only 3D (DxHxW) and 4D (CxDxHxW) images'
            if img.ndim == 3:
                volume_shape = img.shape
            else:
                volume_shape = img[0].shape

            dz = gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="constant",
                                 cval=0) * self.alpha
            dy = gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="constant",
                                 cval=0) * self.alpha
            dx = gaussian_filter(self.random_state.randn(*volume_shape), self.sigma, mode="constant",
                                 cval=0) * self.alpha

            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx

            if img.ndim == 3:
                return map_coordinates(img, indices, order=self.spline_order, mode='reflect')
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in img]
                return np.stack(channels, axis=0)
        return img
        
def get_transformer(config, min_value, max_value, mean, std, phase):
    if phase == 'val':
        phase = 'test'
    assert phase in config, f'Cannot find transformer config for phase: {phase}'
    phase_config = config[phase]
    base_config = {'min_value': min_value, 'max_value': max_value, 'mean': mean, 'std': std}
    return Transformer(phase_config, base_config)


class Transformer:
    def __init__(self, phase_config, base_config):
        self.phase_config = phase_config
        self.config_base = base_config
        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)

    def raw_transform(self):
        return self._create_transform('raw')

    def label_transform(self):
        return self._create_transform('label')

    @staticmethod
    def _transformer_class(class_name):
        module = importlib.import_module('transforms_3d')
        clazz = getattr(module, class_name)
        return clazz

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config[name]
        ])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        config['random_state'] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config['name'])
        return aug_class(**config)

