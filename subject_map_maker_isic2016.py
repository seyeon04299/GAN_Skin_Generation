import os
import pandas as pd
from glob import glob


'''
Subjectmap column = orig_name,subject_no,image_no,lesion_no,path,site,site_label,cell_type,label,mode
'''

def get_data(base_dir, imageid_path_dict):
    

    labels_dict = {
        'Malignant': 1,
        'Benign': 0
    }
    

    
    
    
    df = pd.DataFrame(columns = ['orig_name','subject_no','image_no','lesion_no','path','site','site_label','cell_type','label','mode'])

    df_train = pd.read_csv(os.path.join(base_dir, 'ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'), names = ["image_id", "label"])
    df_test = pd.read_csv(os.path.join(base_dir, 'ISBI2016_ISIC_Part3B_Test_GroundTruth.csv'), names = ["image_id", "label"])
    
    df_train["mode"] = "train"
    df_train["label"] = [1 if x =="malignant" else 0 for x in df_train["label"]] 
    
    df_test["mode"] = "test"

    
    df_original = pd.concat([df_train,  df_test], axis= 0)
    
    print(df_original)
    
    
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    
    df_original['cell_type'] = ["Malignant" if x ==1 else "Benign" for x in df_original["label"]] 
    
    
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
    
    # base_dir = os.path.join('/media/disk/data/DrAnswer2_KNU/knu_skin/data/ISIC_2016/' )
    
    base_dir =os.path.join('C:/Data/닥터앤서2.0피부질환/Program/knu_skin/data/ISIC_2016/')
    
    all_image_path = glob(os.path.join(base_dir, '*', '*.jpg'))
    
    imageid_path_dict = {os.path.basename(x)[:-4]: x for x in all_image_path}
    subject_map = get_data(base_dir, imageid_path_dict)
    subject_map.to_csv(base_dir+'ISIC_2016_subjectmap.csv',index=False)



