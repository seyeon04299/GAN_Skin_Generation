import os
import pandas as pd
from glob import glob

def get_data(base_dir, imageid_path_dict):
    lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
    }

    df_original = pd.read_csv(os.path.join(base_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['label'] = pd.Categorical(df_original['cell_type']).codes
    # df_original.head()

    df_original[['label', 'cell_type']].sort_values('label').drop_duplicates()

    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']

    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

    # now we filter out images that don't have duplicates
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    df_undup = df_undup.sample(frac = 1)
    return df_undup



if __name__ == '__main__':
    base_dir = os.path.join('C:/Users/User/Desktop/김세연/SKINCANCER_DRANSWER/HAM_TEST_Project/data/HAM10000/' )
    all_image_path = glob(os.path.join(base_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    subject_map = get_data(base_dir, imageid_path_dict)
    subject_map.to_csv(base_dir+'HAM10000_subjectmap.csv',index=False)

