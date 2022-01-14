
from torch import float32
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class KNUskinDataset_ProGAN(Dataset): 

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map,transform, mode, cell_type):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        # self.subject_map

        self.mode = mode
        self.cell_type = cell_type
        ## USE ALL DATA FOR GAN
        # self.subject_map = self.subject_map.loc[self.subject_map['mode']==self.mode]
        self.subject_map = self.subject_map.loc[self.subject_map['cell_type']==self.cell_type]
        
        #randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index()
        
        self.len = len(self.subject_map)
        self.transform = transform

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수
    def __getitem__(self, index):
        row = self.subject_map.iloc[index]
        y = row['label']
        img_path = row['path']

        # Load image
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return (image, y)

class KNUskinDataset_GAN(Dataset): 

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map,transform, mode, cell_type):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        # self.subject_map

        self.mode = mode
        self.cell_type = cell_type
        ## USE ALL DATA FOR GAN
        # self.subject_map = self.subject_map.loc[self.subject_map['mode']==self.mode]
        self.subject_map = self.subject_map.loc[self.subject_map['cell_type']==self.cell_type]
        print("Dataset")
        print(self.subject_map.shape)
        
        #randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index()
        
        self.len = len(self.subject_map)
        self.transform = transform

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수
    def __getitem__(self, index):
        row = self.subject_map.iloc[index]
        y = row['label']
        img_path = row['path']

        # Load image
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return (image, y)

class KNUskinDataset(Dataset): 

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map,transform, mode, cell_type):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        # self.subject_map

        self.mode = mode
        self.cell_type = cell_type
        self.subject_map = self.subject_map.loc[self.subject_map['mode']==self.mode]
        self.subject_map = self.subject_map.loc[self.subject_map['cell_type']==self.cell_type]
        print("Dataset")
        print(self.subject_map.shape)
        
        #randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index()
        
        self.len = len(self.subject_map)
        self.transform = transform

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수
    def __getitem__(self, index):
        row = self.subject_map.iloc[index]
        y = row['label']
        img_path = row['path']

        # Load image
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return (image, y)




class KNUskinDataset_numclass2(Dataset): 

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map,transform, mode):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        # self.subject_map

        self.mode = mode
        self.subject_map = self.subject_map.loc[self.subject_map['mode']==self.mode]
        print("Dataset")
        print(self.subject_map.shape)
        #randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index()
        self.len = len(self.subject_map)
        self.transform = transform

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수
    def __getitem__(self, index):
        row = self.subject_map.iloc[index]
        y = int(row['label']>0)

        img_path = row['path']

        # Load image
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return (image, y)



class KNUskinDataset_1imgperpat(Dataset): 

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map,transform, mode):
        # self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        self.subject_map = subject_map.drop_duplicates(subset=['subject_no','label']).reset_index()

        self.mode = mode
        self.subject_map = self.subject_map.loc[self.subject_map['mode']==self.mode]
        
        self.len = len(self.subject_map)
        self.transform = transform

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수
    def __getitem__(self, index):
        row = self.subject_map.iloc[index]
        y = row['label']
        img_path = row['path']

        # Load image
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return (image, y)




class KNUskinDataset_siteLabel(Dataset): 

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map,transform, mode):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        # self.subject_map

        self.mode = mode
        self.subject_map = self.subject_map.loc[self.subject_map['mode']==self.mode]
        print("Dataset")
        print(self.subject_map.shape)
        
        #randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index()
        self.site_map = pd.get_dummies(self.subject_map['site'],prefix='site',dtype=float)
        
        self.len = len(self.subject_map)
        self.transform = transform

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수
    def __getitem__(self, index):
        row = self.subject_map.iloc[index]
        site = self.site_map.iloc[index].values
        y = row['label']
        img_path = row['path']

        # Load image
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return (image, y, site)



class HAM10000Dataset(Dataset): 

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map,transform, mode = 'train',):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        self.mode = mode
        self.len = len(self.subject_map)
        self.transform = transform

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수
    def __getitem__(self, index):
        row = self.subject_map.iloc[index]
        y = row['label']
        img_path = row['path']

        # Load image
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return (image, y)
