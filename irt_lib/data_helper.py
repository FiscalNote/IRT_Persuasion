import os

import pandas as pd

BASE_PATH = '/home/ec2-user/final_paper_data_v2/debates/'

def create_full_data(label_type='bin', feature_types=['style'], base_path=BASE_PATH):
    
    vote_data = pd.read_csv(os.path.join(base_path, 'vote_data.csv'))

    file = feature_types[0]
    all_style_data = pd.read_csv(os.path.join(base_path, f'{file}.csv'), index_col=0)
    
    for file in feature_types[1:]:
        data = pd.read_csv(os.path.join(base_path, f'{file}.csv'), index_col=0)
        all_style_data = pd.merge(all_style_data, data, left_index=True, right_index=True)

     # Now actually assemble the data
    assembled_data = []

    for _, row in vote_data.iterrows():
        feats = all_style_data.loc[row.doc_id]
        user_id = row.user_id

        try:
            label = row[f'y_{label_type}']
        except KeyError:
            raise ValueError("Unknown label - pick from", row.to_dict())
            
        assembled_data.append({'feats': list(feats), 'user_id': user_id, 'label': label, 'doc_id': row.doc_id})
        
    return pd.DataFrame(assembled_data)


    