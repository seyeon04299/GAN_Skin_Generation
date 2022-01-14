from torchvision import datasets, transforms
from base import BaseDataLoader
from base import BaseDataLoader_basic
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from .datasets import *
import numpy as np



######################################################################
####### Data Loader For ProGAN
######################################################################



class KNUskinDataLoader_ProGAN(BaseDataLoader):
    """
    KNU data loading for GAN
    """
    def __init__(self, data_dir, batch_size, input_size, subject_map, cell_type = 'BCC', shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.subject_map = pd.read_csv(subject_map)

        if training==False:
            mode = 'test'
        if training==True:
            mode = 'train'

        train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        #ToTensor() changes (nrow, ncol, nchannel) to (nchannel, nrow, ncol)

        self.data_dir = data_dir
        
        # self.dataset = KNUskinDataset(self.subject_map, transform=train_transform)
        self.dataset = KNUskinDataset_ProGAN(self.subject_map, transform=train_transform, mode=mode, cell_type=cell_type)
        self.subject_map = self.dataset.subject_map
        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)




######################################################################
####### Data Loader For GAN - DCGAN, WGAN, WGANGP, ...
######################################################################



class KNUskinDataLoader_GAN(BaseDataLoader):
    """
    KNU data loading for GAN
    """
    def __init__(self, data_dir, batch_size, subject_map, cell_type = 'BCC', shuffle=True, validation_split=0.0, num_workers=1, training=True):
        input_size = 224
        self.subject_map = pd.read_csv(subject_map)

        if training==False:
            mode = 'test'
        if training==True:
            mode = 'train'

        train_transform = transforms.Compose([ZFImage(),
                                            transforms.Resize((input_size,input_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        #ToTensor() changes (nrow, ncol, nchannel) to (nchannel, nrow, ncol)

        self.data_dir = data_dir
        
        # self.dataset = KNUskinDataset(self.subject_map, transform=train_transform)
        self.dataset = KNUskinDataset_GAN(self.subject_map, transform=train_transform, mode=mode,cell_type=cell_type)
        self.subject_map = self.dataset.subject_map
        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)



def get_padding(image):
    w, h = image.size
    max_wh = np.max([w,h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


    
class ZFImage(object):
    '''
    Z-fill image ; Only the shorter side is filled.
    '''
    def __init__(self, fill=0, padding_mode='constant'):
            self.fill = fill
            self.padding_mode = padding_mode
            
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return transforms.functional.pad(img, get_padding(img), self.fill, self.padding_mode)




######################################################################
####### Data Loader For Classification
######################################################################



class KNUskinDataLoader(BaseDataLoader):
    """
    KNU data loading demo using all patient image
    """
    def __init__(self, data_dir, batch_size, subject_map, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        input_size = 224
        self.subject_map = pd.read_csv(subject_map)

        if training==False:
            mode = 'test'
        if training==True:
            mode = 'train'

        train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(20),
                                            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        #ToTensor() changes (nrow, ncol, nchannel) to (nchannel, nrow, ncol)

        self.data_dir = data_dir
        # test code
        # self.dataset = KNUskinDataset(self.subject_map, transform=train_transform)
        self.dataset = KNUskinDataset(self.subject_map, transform=train_transform, mode=mode)
        self.subject_map = self.dataset.subject_map
        
        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)




class KNUskinDataLoader_numclass2(BaseDataLoader):
    """
    KNU data loading demo using all patient image
    """
    def __init__(self, data_dir, batch_size, subject_map, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        input_size = 224
        self.subject_map = pd.read_csv(subject_map)

        if training==False:
            mode = 'test'
        if training==True:
            mode = 'train'

        train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(20),
                                            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        #ToTensor() changes (nrow, ncol, nchannel) to (nchannel, nrow, ncol)

        self.data_dir = data_dir
        # test code
        # self.dataset = KNUskinDataset(self.subject_map, transform=train_transform)
        self.dataset = KNUskinDataset_numclass2(self.subject_map, transform=train_transform, mode=mode)
        self.subject_map = self.dataset.subject_map
        
        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)



class KNUskinDataLoader_siteLabel(BaseDataLoader):
    """
    KNU data loading demo using all patient image
    """
    def __init__(self, data_dir, batch_size, subject_map, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        input_size = 224
        self.subject_map = pd.read_csv(subject_map)

        if training==False:
            mode = 'test'
        if training==True:
            mode = 'train'

        train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(20),
                                            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        #ToTensor() changes (nrow, ncol, nchannel) to (nchannel, nrow, ncol)

        self.data_dir = data_dir
        # test code
        # self.dataset = KNUskinDataset(self.subject_map, transform=train_transform)
        self.dataset = KNUskinDataset_siteLabel(self.subject_map, transform=train_transform, mode=mode)
        self.subject_map = self.dataset.subject_map
        
        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)



class KNUskinDataLoader_del_duplic(BaseDataLoader_basic):
    """
    KNU data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, subject_map, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        input_size = 224
        self.subject_map = pd.read_csv(subject_map)

        if training==False:
            mode = 'test'
        if training==True:
            mode = 'train'

        train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(20),
                                            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        #ToTensor() changes (nrow, ncol, nchannel) to (nchannel, nrow, ncol)

        self.data_dir = data_dir
        
        # self.dataset = KNUskinDataset(self.subject_map, transform=train_transform)
        self.dataset = KNUskinDataset_1imgperpat(self.subject_map, transform=train_transform, mode=mode)
        self.subject_map = self.dataset.subject_map

        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)




class KNUskinDataLoader_ZF(BaseDataLoader):
    """
    KNU data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, subject_map, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        input_size = 224
        self.subject_map = pd.read_csv(subject_map)

        if training==False:
            mode = 'test'
        if training==True:
            mode = 'train'

        train_transform = transforms.Compose([ZFImage(),
                                            transforms.Resize((input_size,input_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(20),
                                            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        #ToTensor() changes (nrow, ncol, nchannel) to (nchannel, nrow, ncol)

        self.data_dir = data_dir
        
        # self.dataset = KNUskinDataset(self.subject_map, transform=train_transform)
        self.dataset = KNUskinDataset_1imgperpat(self.subject_map, transform=train_transform,mode=mode)
        self.subject_map = self.dataset.subject_map
        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)

def get_padding(image):
    w, h = image.size
    max_wh = np.max([w,h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding
    
class ZFImage(object):
    '''
    Z-fill image ; Only the shorter side is filled.
    '''
    def __init__(self, fill=0, padding_mode='constant'):
            self.fill = fill
            self.padding_mode = padding_mode
            
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return transforms.functional.pad(img, get_padding(img), self.fill, self.padding_mode)




class HAM10000DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, subject_map, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        input_size = 224
        self.subject_map = pd.read_csv(subject_map)

        train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(20),
                                            transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        #ToTensor() changes (nrow, ncol, nchannel) to (nchannel, nrow, ncol)

        self.data_dir = data_dir
        
        self.dataset = HAM10000Dataset(self.subject_map, transform=train_transform)
        super().__init__(self.dataset, batch_size, self.subject_map, shuffle, validation_split, num_workers)


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)