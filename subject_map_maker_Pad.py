import os
import pandas as pd
from glob import glob


'''
Subjectmap column = orig_name,subject_no,image_no,lesion_no,path,site,site_label,cell_type,label,mode
'''

def get_data(base_dir, imageid_path_dict):
    
    lesion_type_dict = {
        'SCC': 'Squamous Cell Carcinoma',
        'ACK': 'Actinic Keratosis',
        'SEK': 'Seborrheic Keratosis',
        'BCC': 'Basal cell carcinoma',
        'BOD': 'Bowens disease',
        'MEL': 'Melanoma',
        'NEV': 'Nevus'
    }


    labels_dict = {
        'SCC': 2,
        'BCC': 1,
        'MEL': 3,
        'NEV': 0,
        'SEK': 0,
        'BOD': 0,
        'ACK': 0 
    }
    
    site_type_dict = {
    'head&neck': 0,
    'extremity': 1,
    'trunk': 2,
    'acral': 3
    }
    
    
    
    
    df = pd.DataFrame(columns = ['orig_name','subject_no','image_no','lesion_no','path','site','site_label','cell_type','label','mode'])


    df_original = pd.read_csv(os.path.join(base_dir, 'PADUFES20_metadata.csv'))
    df_original['path'] = df_original['img_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['diagnostic'].map(lesion_type_dict.get)
    df_original['label'] = df_original['diagnostic'].map(labels_dict.get)

    ## Drop nas in label
    df_original = df_original.dropna(subset=['label'])
    df_original['label'] = df_original["label"].astype(int)
    
    
    df["label"] = df_original["label"]
    df["orig_name"] = df_original["patient_id"]
    df["subject_no"] = df_original["patient_id"]
    df["image_no"] = df_original["patient_id"]
    df["lesion_no"] = df_original["lesion_id"]
    df["path"] = df_original["path"]
    df["site"] = df_original['region']
    df["site_label"] = df_original['region']
    df["cell_type"] = df_original["cell_type"]
    df["label"] = df_original["label"]
    
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    
    for label in df["label"].unique():
        
        df_label = df.loc[df['label']==label]
        patients = list(df_label['subject_no'].unique())
        len_train = int(len(patients)*0.8)
        patients_train = patients[:len_train]
        patients_test = patients[len_train:]
        train = df_label.loc[df_label['subject_no'].isin(patients_train)]
        test = df_label.loc[df_label['subject_no'].isin(patients_test)]
        
        df_train = df_train.append(train)
        df_test = df_test.append(test)
    
    df_train['mode']='train'
    df_test['mode']='test'
    
    df_undup = pd.concat([df_train, df_test])
    
    return(df_undup)



 



if __name__ == '__main__':
    
    
    base_dir = os.path.join('C:/Data/닥터앤서2.0피부질환/Program/knu_skin/data/PADUFES20/')
    # base_dir = os.path.join('/media/disk/data/DrAnswer2_KNU/knu_skin/data/PADUFES20/')
    '''
    base_dir = os.path.join('C:/Data/Project/SkinCancer/smartphone/data/PADUFES20/')'''
    all_image_path = glob(os.path.join(base_dir, '*', '*.png'))
    imageid_path_dict = {os.path.basename(x)[:-4]: x for x in all_image_path}
    print(imageid_path_dict)
    subject_map = get_data(base_dir, imageid_path_dict)
    subject_map.to_csv(base_dir+'PADUFES20_subjectmap.csv',index=False)


