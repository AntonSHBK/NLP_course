import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class CategoricalLabelEncoder():
    def __init__(self, dataframe: pd.DataFrame, categorical_columns: list[str]) -> None:
        self.encoders = {}
        
        for column in categorical_columns:
            encoder = LabelEncoder()
            dataframe[column] = encoder.fit_transform(dataframe[column])
            self.encoders[column] = encoder
        
    def decode(self, label: str, code: list[int]):
        return self.encoders[label].inverse_transform([code])
    
    def get_classes(self, label):
        categories_list = self.encoders[label].classes_
        print("Список категорий в 'categoria':", categories_list)
        return categories_list
    

class CustomDataset(Dataset):
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 tokenizer: GPT2Tokenizer,
                 max_length=512):
        
        self.dataframe = dataframe
        
        self.type_ids = torch.tensor(
            dataframe['type_problem'].astype(int).tolist()
        )

        self.source_encodings = tokenizer(
            dataframe['source'].tolist(), 
            padding='max_length', 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        self.target_encodings = tokenizer(
            dataframe['target'].tolist(), 
            padding='max_length', 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )      

    def __len__(self):
        return len(self.type_ids)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.source_encodings.items()}
        item['target_ids'] = self.target_encodings['input_ids'][idx]
        item['target_attention_mask'] = self.target_encodings['attention_mask'][idx]
        item['type_ids'] = self.type_ids[idx]
        return item
