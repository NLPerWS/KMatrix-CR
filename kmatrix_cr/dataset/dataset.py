import json
import pickle
from typing import Optional


class Dataset:
    
    def __init__(
        self,
        dataset_path: Optional[str] = "",
        dataset_name: Optional[str] = "",
        load_data: bool = True,
        data_list = [],
    ):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.data_list = data_list
        if load_data:
            self.load_data()
            # self.data_list = self.data_list[0:5]

    def load_data(self):
        
        if self.dataset_path == "":
            return []
        if len(self.data_list) != 0:
            return self.data_list
        
        if self.dataset_path.endswith(".json"):
            try:
                with open(self.dataset_path,'r',encoding='utf-8') as f:
                    temp_data = json.load(f)
            except Exception as e:
                with open(self.dataset_path,'r',encoding='utf-8') as f:
                    temp_data = []
                    for line in f:
                        temp_data.append(json.loads(line.strip()))
            
        elif self.dataset_path.endswith(".jsonl"):
            with open(self.dataset_path,'r',encoding='utf-8') as f:
                temp_data = []
                for line in f:
                    temp_data.append(json.loads(line.strip()))

        elif self.dataset_path.endswith(".pkl"):
            try:
                with open(self.dataset_path, 'rb') as f:
                    temp_data = pickle.load(f)
                temp_data = temp_data.to_dict(orient='records')
                
            except Exception as e:
                with open(self.dataset_path,'r',encoding='utf-8') as f:
                    temp_data = []
                    for line in f:
                        temp_data.append(json.loads(line.strip()))
        else:
            print("Invalid file format. Please provide an (.json .jsonl .pkl) file.")
            temp_data = []
        
        self.data_list = temp_data
        return temp_data
    