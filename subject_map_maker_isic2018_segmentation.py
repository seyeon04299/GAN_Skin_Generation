import os
import pandas as pd
from glob import glob

def get_data(base_dir, imageid_path_train_dic, imageid_path_val_dic):

    # df_original = pd.DataFrame(columns = ['orig_name','subject_no','image_no','lesion_no','path','site','site_label','cell_type','label','mode'])

    # df_original['orig_name'] = imageid_path_dict.keys()
    # df_original['path'] = df_original['orig_name'].map(imageid_path_dict.get)

    # df_original['subject_no'] = df_original['orig_name'].apply(lambda x: int(x.split('_')[1]))
    # df_original['image_no'] = df_original['orig_name'].apply(lambda x: int(x.split('_')[2]))
    # df_original['lesion_no'] = df_original['orig_name'].apply(lambda x: int(x.split('_')[3]))
    # df_original['site'] = df_original['orig_name'].apply(lambda x: x.split('_')[4])
    # df_original['site_label'] = df_original['site'].map(site_type_dict.get)
    # df_original['cell_type'] = df_original['orig_name'].apply(lambda x: x.split('_')[0])
    # df_original['label'] = df_original['cell_type'].map(label_type_dict.get)

    # # Train-Test Split
    # df_permutated = df_original.sample(frac=1,random_state=0)              # 섞기
    # split = 0.8
    

    # train_ind_list=[]
    # test_ind_list =[]

    df_train = pd.DataFrame()
    df_val = pd.DataFrame()

    df_train['orig_name'] =  imageid_path_train_dict.keys()
    df_val['orig_name'] =  imageid_path_val_dict.keys()

    df_train['path'] = df_train['orig_name'].map(imageid_path_train_dict.get)
    df_val['path'] = df_val['orig_name'].map(imageid_path_val_dict.get)

    df_train['label'] = base_dir + "/ISIC2018_Task1_Training_GroundTruth/" + df_train['orig_name'] + "_segmentation.png"
    df_val['label'] = base_dir + "/ISIC2018_Task1_Validation_GroundTruth/" + df_val['orig_name'] + "_segmentation.png"
    
    # df_train['points'] = '/home/administrator/DATA/KNU_GAN/knu_skin_gan/data/ISIC2018/Training_Points/' + df_train['orig_name'] + "_segmentation.npy"
    # df_val['points'] = 'home/administrator/DATA/KNU_GAN/knu_skin_gan/data/ISIC2018/Validation_Points' + df_val['orig_name'] + "_segmentation.npy"
    
    # df_train['batunet_prd'] = '/home/administrator/Data/skin_segmentation/data/ISIC2018/train_seg_predi_batransunet/' + df_train['orig_name'] + ".npy"
    # df_val['batunet_prd'] = '/home/administrator/Data/skin_segmentation/data/ISIC2018/valid_seg_predi_batransunet/' + df_train['orig_name'] + ".npy"
    
    df_train['mode'] = "train"
    df_val["mode"] ="test"

    df_undup = pd.concat([df_train,df_val])
    df_undup = df_undup.sample(frac =1, random_state = 1)
    
    return df_undup


if __name__ == '__main__':
    base_dir =os.path.join('//home/administrator/DATA/KNU_GAN/knu_skin_gan/data/ISIC2018/')
    sub_dir = os.walk(base_dir)

    image_path_train = glob(os.path.join(base_dir + "ISIC2018_Task1-2_Training_Input/",  '*.jpg'))
    image_path_val = glob(os.path.join(base_dir + "ISIC2018_Task1-2_Validation_Input/", '*.jpg'))


    imageid_path_train_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in image_path_train}
    imageid_path_val_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in image_path_val}


    print(imageid_path_train_dict)
    subject_map = get_data(base_dir, imageid_path_train_dict , imageid_path_val_dict )

    subject_map.to_csv(base_dir+'/ISIC2018_subjectmap.csv',index=False)
