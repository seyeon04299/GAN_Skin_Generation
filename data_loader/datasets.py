
from torch import float32
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class Dataset_UNCGAN_withNoise(Dataset): 

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map,transform, transform_noise, mode, cell_type):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        # self.subject_map

        self.mode = mode
        self.cell_type = cell_type
        
        #randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index()
        
        ## USE ALL DATA FOR GAN
        self.subject_map = self.subject_map.loc[self.subject_map['cell_type']==self.cell_type]
        
        
        print("Dataset")

        
        self.len = len(self.subject_map)
        self.transform = transform
        self.transform_noise = transform_noise

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수 (Image1 from label1, Image2 from label2)
    def __getitem__(self, index):
        row1 = self.subject_map.iloc[index]

        
        # y1 = row1['label']
        img_path1 = row1['path']

        # Load image
        image1 = Image.open(img_path1)
        image1 = image1.convert("RGB")

        if self.transform is not None:
            normal_img = self.transform(image1)

        noise_img = self.transform_noise(image1)
        # image2 = image2.convert("RGB")

        # if self.transform is not None:
        #     image2 = self.transform(image2)
        return (noise_img, normal_img)




class Dataset_UNCGAN_withSegmentation(Dataset): 

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map,transform, transform_segmented, mode, cell_type):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        # self.subject_map

        self.mode = mode
        self.cell_type = cell_type
        # self.cell_type2 = cell_type2
        
        #randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index()
        
        ## USE ALL DATA FOR GAN
        # self.subject_map = self.subject_map.loc[self.subject_map['mode']==self.mode]
        self.subject_map = self.subject_map.loc[self.subject_map['cell_type']==self.cell_type]
        # self.subject_map2 = self.subject_map.loc[self.subject_map['cell_type']==self.cell_type2]
        # if self.subject_map1.shape[0]>self.subject_map2.shape[0]:
        #     self.subject_map1 = self.subject_map1[:self.subject_map2.shape[0]]
        # if self.subject_map2.shape[0]>self.subject_map1.shape[0]:
        #     self.subject_map2 = self.subject_map2[:self.subject_map1.shape[0]]

        
        print("Dataset")
        # print(self.subject_map1.shape)
        # print(self.subject_map2.shape)
        
        
        
        self.len = len(self.subject_map)
        self.transform = transform
        self.transform_segmented = transform_segmented

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수 (Image1 from label1, Image2 from label2)
    def __getitem__(self, index):
        row1 = self.subject_map.iloc[index]

        
        # y1 = row1['label']
        img_path1 = row1['path']

        # Load image
        image1 = Image.open(img_path1)
        image1 = image1.convert("RGB")

        if self.transform is not None:
            image1 = self.transform(image1)

        segmented_img = row1['path_segmentation']
        # row2 = self.subject_map2.iloc[index]
        # y2 = row2['label']
        # img_path2 = row2['path']

        # Load image
        segmented_img = Image.open(segmented_img)
        segmented_img = self.transform_segmented(segmented_img)
        # image2 = image2.convert("RGB")

        # if self.transform is not None:
        #     image2 = self.transform(image2)
        return (segmented_img, image1)

class Dataset_UNCGAN(Dataset): 

    #데이터셋의 전처리를 해주는 부분
    def __init__ (self, subject_map,transform, mode, cell_type1, cell_type2):
        self.subject_map = subject_map   # 데이터 정보를 포함한 dataframe
        # self.subject_map

        self.mode = mode
        self.cell_type1 = cell_type1
        self.cell_type2 = cell_type2
        
        #randomize row
        self.subject_map = self.subject_map.sample(frac =1).reset_index()
        
        ## USE ALL DATA FOR GAN
        # self.subject_map = self.subject_map.loc[self.subject_map['mode']==self.mode]
        self.subject_map1 = self.subject_map.loc[self.subject_map['cell_type']==self.cell_type1]
        self.subject_map2 = self.subject_map.loc[self.subject_map['cell_type']==self.cell_type2]
        if self.subject_map1.shape[0]>self.subject_map2.shape[0]:
            self.subject_map1 = self.subject_map1[:self.subject_map2.shape[0]]
        if self.subject_map2.shape[0]>self.subject_map1.shape[0]:
            self.subject_map2 = self.subject_map2[:self.subject_map1.shape[0]]

        
        print("Dataset")
        print(self.subject_map1.shape)
        print(self.subject_map2.shape)
        
        
        
        self.len = len(self.subject_map1)
        self.transform = transform

    #데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return self.len

    #데이터셋에서 특정 1개의 샘플을 가져오는 함수 (Image1 from label1, Image2 from label2)
    def __getitem__(self, index):
        row1 = self.subject_map1.iloc[index]
        y1 = row1['label']
        img_path1 = row1['path']

        # Load image
        image1 = Image.open(img_path1)
        image1 = image1.convert("RGB")

        if self.transform is not None:
            image1 = self.transform(image1)

        row2 = self.subject_map2.iloc[index]
        y2 = row2['label']
        img_path2 = row2['path']

        # Load image
        image2 = Image.open(img_path2)
        image2 = image2.convert("RGB")

        if self.transform is not None:
            image2 = self.transform(image2)
        return (image1, image2)



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
