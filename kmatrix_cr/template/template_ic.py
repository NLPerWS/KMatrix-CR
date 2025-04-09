import argparse
import json
import os
from typing import Literal
from kmatrix_cr.config.config import Config
from kmatrix_cr.utils.common_utils import eval

class ICTemplate:
    ALLOWED_CONFLICT_METHODS = ["ICL-seprate","ICL-whole","factool"]
    def __init__(self,
                config : Config,
                conflict_method: Literal["ICL-seprate","ICL-whole"],
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
                
        if self.conflict_method == "ICL-seprate":
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
            
            result_list = self.llm_model.run(prompt_list = prompt_list)
            for dataobj,resobj in zip(self.data_list,result_list):
                dataobj['gen_answer'] = resobj['content']
            result = {
                "result": self.data_list
            }
            
        
        elif self.conflict_method == "ICL-whole" :
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
                data['gen_answer'] = res['content']
            
            result = {
                "result":self.data_list
            }
        
        
        elif self.conflict_method == "factool":
            
            from kmatrix_cr.toolkit.factool.factool import Factool
            
            for data in self.data_list:
                data['category'] = "kbqa"
                data['prompt'] = data['question']
                data['response'] = data['ground_truth'][0]
                
            
            factool = Factool(foundation_model="gpt-3.5-turbo",data_list=self.data_list)
            res = factool.run(inputs=self.data_list)
            
            result = {
                "result":res
            }    
                    
        else:
            result = {}
        
        if do_eval:
            eval_obj = eval(metrics=self.metrics,data=result,data_path="")
            print(eval_obj)
            
            print(eval_obj['log_str'])
            for key in eval_obj:
                if key != "log_str":
                    result[key] = eval_obj[key]
                    
        if output_path != "":
            if result == {}:
                print("result is empty  ")
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f,ensure_ascii=False)
            
        return result
        