import os
import pandas as pd
from glob import glob

def get_data(base_dir, imageid_path_dict):
    # image_id_path_dict = {'SCC_9_3_1_head&neck':'C:/Users/User/Desktop/김세연/SKINCANCER_DRANSWER/HAM_TEST_Project/data/KNU\\data\\SCC_9_3_1_head&neck.jpg', ...}
    label_type_dict = {
    'BCC': 1,
    'SCC': 2,
    'Melanoma': 3,
    'Benign': 0
    }
    site_type_dict = {
    'head&neck': 0,
    'extremity': 1,
    'trunk': 2,
    'acral': 3
    }
    df_original = pd.DataFrame(columns = ['orig_name','subject_no','image_no','lesion_no','path','site','site_label','cell_type','label','mode'])

    df_original['orig_name'] = imageid_path_dict.keys()
    df_original['path'] = df_original['orig_name'].map(imageid_path_dict.get)

    df_original['subject_no'] = df_original['orig_name'].apply(lambda x: int(x.split('_')[1]))
    df_original['image_no'] = df_original['orig_name'].apply(lambda x: int(x.split('_')[2]))
    df_original['lesion_no'] = df_original['orig_name'].apply(lambda x: int(x.split('_')[3]))
    df_original['site'] = df_original['orig_name'].apply(lambda x: x.split('_')[4])
    df_original['site_label'] = df_original['site'].map(site_type_dict.get)
    df_original['cell_type'] = df_original['orig_name'].apply(lambda x: x.split('_')[0])
    df_original['label'] = df_original['cell_type'].map(label_type_dict.get)

    # Train-Test Split
    df_permutated = df_original.sample(frac=1,random_state=0)              # 섞기
    split = 0.8
    

    train_ind_list=[]
    test_ind_list =[]

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for label in label_type_dict.keys():
        df_label = df_original.loc[df_original['cell_type']==label]
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

    df_undup = pd.concat([df_train,df_test])
    return df_undup


    # df_original[['label', 'cell_type']].sort_values('label').drop_duplicates()

    # # this will tell us how many images are associated with each lesion_id
    # df_undup = df_original.groupby('lesion_id').count()
    # # now we filter out lesion_id's that have only one image associated with it
    # df_undup = df_undup[df_undup['image_id'] == 1]
    # df_undup.reset_index(inplace=True)

    # # here we identify lesion_id's that have duplicate images and those that have only one image.
    # def get_duplicates(x):
    #     unique_list = list(df_undup['lesion_id'])
    #     if x in unique_list:
    #         return 'unduplicated'
    #     else:
    #         return 'duplicated'

    # # create a new colum that is a copy of the lesion_id column
    # df_original['duplicates'] = df_original['lesion_id']

    # # apply the function to this new column
    # df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

    # # now we filter out images that don't have duplicates
    # df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    # df_undup = df_undup.sample(frac = 1)
    # return df_undup



if __name__ == '__main__':
    base_dir = os.path.join('//home/administrator/DATA/KNU_GAN/knu_skin_gan/data/KNU_DSLR/')
    all_image_path = glob(os.path.join(base_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    # print(imageid_path_dict)
    subject_map = get_data(base_dir, imageid_path_dict)
    subject_map.to_csv(base_dir+'KNU_DSLR_subjectmap.csv',index=False)

