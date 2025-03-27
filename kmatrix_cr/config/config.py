

from typing import Any, List
from kmatrix_cr.dataset.dataset import Dataset
from kmatrix_cr.generator.root_generator import RootGenerator

class Config:
    
    def __init__(self, 
        dataset:Dataset,
        model:RootGenerator = None,
        llm_model:RootGenerator = None,
        openai_model:RootGenerator = None,
        metrics:List[str] = []
    ):
        self.dataset = dataset
        self.model = model
        self.llm_model = llm_model
        self.openai_model = openai_model
        self.metrics = metrics
     
     
