import os
import pandas as pd
from glob import glob


'''
Subjectmap column = orig_name,subject_no,image_no,lesion_no,path,site,site_label,cell_type,label,mode
'''

def get_data(base_dir, imageid_path_dict):
    
    lesion_type_dict = {
        'MEL': 'melanoma',
        'SEK': 'Seborrheic Keratosis',
        'NEV': 'Nevus'
    }


    labels_dict = {
        'SEK': 0,
        'MEL': 1,
        'NEV': 0
    }
    

    
    
    
    df = pd.DataFrame(columns = ['orig_name','subject_no','image_no','lesion_no','path','site','site_label','cell_type','label','mode'])

    df_train = pd.read_csv(os.path.join(base_dir, 'ISIC-2017_Training_Part3_GroundTruth.csv'))
    df_test = pd.read_csv(os.path.join(base_dir, 'ISIC-2017_Test_v2_Part3_GroundTruth.csv'))
    df_valid = pd.read_csv(os.path.join(base_dir, 'ISIC-2017_Validation_Part3_GroundTruth.csv'))
    
    df_train["mode"] = "train"
    df_valid["mode"] = "train"
    df_test["mode"] = "test"
    
    df_original = pd.concat([df_train, df_valid, df_test], axis= 0)
    
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    
    df_original['cell_type'] = "NEV"
    df_original.loc[df_original["melanoma"] ==1, "cell_type"] = "MEL"
    df_original.loc[df_original["seborrheic_keratosis"] ==1, "cell_type"]= "SEK"
    
    df_original['label'] = df_original['cell_type'].map(labels_dict.get)\
    
    df["label"] = df_original["label"]
    df["orig_name"] = df_original["image_id"]
    df["subject_no"] = df_original["image_id"]
    df["image_no"] = df_original["image_id"]
    df["lesion_no"] = df_original["image_id"]
    df["path"] = df_original["path"]
    df["site"] = "Not available"
    df["site_label"] = 0
    df["cell_type"] = df_original["cell_type"]
    df["label"] = df_original["label"]

    df = df.sample(frac =1)
    

    return(df)



 



if __name__ == '__main__':
    
    # base_dir = os.path.join('/media/disk/data/DrAnswer2_KNU/knu_skin/data/ISIC_2017/')
    
    base_dir =os.path.join('C:/Data/닥터앤서2.0피부질환/Program/knu_skin/data/ISIC_2017/')
    
    all_image_path = glob(os.path.join(base_dir, '*', '*.jpg'))
    
    imageid_path_dict = {os.path.basename(x)[:-4]: x for x in all_image_path}
    print(imageid_path_dict)
    subject_map = get_data(base_dir, imageid_path_dict)
    subject_map.to_csv(base_dir+'ISIC_2017_subjectmap.csv',index=False)


