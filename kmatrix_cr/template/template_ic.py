import argparse
import json
import os
from typing import Literal
from kmatrix_cr.config.config import Config

class ICTemplate:
    ALLOWED_CONFLICT_METHODS = ["ExternalKnowledgeConflicts","DAA"]
    def __init__(self,
                config : Config,
                conflict_method: Literal["ExternalKnowledgeConflicts","DAA"],
                args_kwargs: dict = {}
    ):
        if conflict_method not in self.ALLOWED_CONFLICT_METHODS:
            raise ValueError(f"Invalid conflict_method: {conflict_method}. Allowed values are {self.ALLOWED_CONFLICT_METHODS}")
        
        self.config = config
        self.conflict_method = conflict_method
        self.dataset = self.config.dataset
        self.llm_model = self.config.llm_model
        self.openai_model = self.config.openai_model
        self.metrics = self.config.metrics
        self.args_kwargs = args_kwargs
        self.data_list = self.dataset.data_list

    def run(self,do_eval=True,output_path=""):
                
        if self.conflict_method == "ExternalKnowledgeConflicts":
            from kmatrix_cr.toolkit.ExternalKnowledgeConflicts.ExternalKnowledgeConflicts import ExternalKnowledgeConflicts 
            
            for data in self.data_list:
                if "ctxs_content_list" not in data:
                    data['ctxs_content_list'] = data.get("text","")
            ex_con = ExternalKnowledgeConflicts(llm = self.openai_model)
            filter_data_list = ex_con.run(query_obj_list=self.data_list)
            prompt_list = []
            for data in filter_data_list:
                
                facts = '\n'.join(data['filter_ctxs_list'])
                prompt = f"Given the following information:\n{facts}\nAnswer the following question based on the given information with one or few words: {data['question']}\nAnswer:"
                
                # prompt = """
                # Please refer to the knowledge I have provided below:
                
                # """ + '\n'.join(data['filter_ctxs_list']) + """
                
                # Answer my questions:
                # """+ data['question'] +"""
                
                # Just reply with the answer, do not provide any additional information.
                # """
                prompt_list.append(prompt)
            
            print("-----------prompt_list generated-----------------")
            result_list = self.llm_model.run(prompt_list = prompt_list)
            result = {}
            if do_eval:
                if "acc" in self.metrics:
                    rex_count_ok = 0
                    in_count_ok = 0
                
                    for dataobj,resobj in zip(self.data_list,result_list):
                        dataobj['gen_result'] = resobj['content']
                        if resobj['content'] in dataobj['ground_truth']:
                            rex_count_ok += 1
                        for ground_truth in dataobj['ground_truth']:
                            if ground_truth in resobj['content']:
                                in_count_ok += 1
                            
                    result['rex_acc'] = f"{rex_count_ok/len(self.data_list):.2%}"
                    result['in_acc'] = f"{in_count_ok/len(self.data_list):.2%}"
                    
            result['result'] = self.data_list
            
        
        elif self.conflict_method == "DAA" :
            prompt_list = []
            for data in self.data_list:
                
                facts = []
                for index,text in enumerate(data['text']):
                    facts.append(f"Passage {index+1}: {text}")
                facts = "\n\n".join(facts)
                
                prompt = """
""" + facts + """

Some of the above Passage may be incorrect or irrelevant information.
If there are any incorrect or irrelevant Passage, identify and ignore them when generating the correct answer.
Finally, please answer my question with one or a few words.

Question: 
""" + data['question']  + """

Answer:
                """
                
                print("---------------------------------------------------")
                print(len(prompt))                
                
                prompt_list.append(prompt)
            
            results = self.llm_model.run(prompt_list = prompt_list)
            for data,res in zip(self.data_list,results):
                data['gen_result'] = res['content']
            
            result = {
                "result":self.data_list
            }
            
                    
        else:
            result = {}
        
        if output_path != "":
            if result == {}:
                print("result is empty  ")
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f,ensure_ascii=False)
            
        return result
        