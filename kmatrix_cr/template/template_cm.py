import argparse
import json
import pathlib
import os
from typing import Literal
from kmatrix_cr.config.config import Config

class CMTemplate:
    ALLOWED_CONFLICT_METHODS = ["coiecd","retrieveorgenerated","llms_believe_the_earth_is_flat"]
    
    def __init__(self,
                config : Config,
                conflict_method: Literal["coiecd",'retrieveorgenerated','llms_believe_the_earth_is_flat'],
                args_kwargs: dict = {}
    ):
        if conflict_method not in self.ALLOWED_CONFLICT_METHODS:
            raise ValueError(f"Invalid conflict_method: {conflict_method}. Allowed values are {self.ALLOWED_CONFLICT_METHODS}")
         
        self.config = config
        self.conflict_method = conflict_method
        self.dataset = self.config.dataset
        self.model = self.config.model
        self.metrics = self.config.metrics
        self.args_kwargs = args_kwargs
        self.data_list = self.dataset.data_list
        
    def run(self,do_eval=True,output_path=""):
        
        if self.conflict_method == "coiecd":
            from kmatrix_cr.toolkit.coiecd_greedy.llama_generate_nq_coiecd_greedy import main as coiecd_greedy_main
            if self.model.model is None or self.model.tokenizer is None:
                self.model.load_model()
                if self.model.model is None or self.model.tokenizer is None:
                    raise ValueError("This generator does not have model or tokenizer and does not support the method. Please switch generator or conflict_method and try again.")
            result = coiecd_greedy_main(model=self.model.model,tokenizer=self.model.tokenizer,data_list=self.data_list,metrics=self.metrics,do_eval=do_eval)

        elif self.conflict_method == "llms_believe_the_earth_is_flat":
            from kmatrix_cr.toolkit.llms_believe_the_earth_is_flat.run_exp import main
            
            if self.model.model is None or self.model.tokenizer is None:
                            self.model.load_model()
                            if self.model.model is None or self.model.tokenizer is None:
                                raise ValueError("This generator does not have model or tokenizer and does not support the method. Please switch generator or conflict_method and try again.")
                
            parser = argparse.ArgumentParser(description='all-in-one experiment on boolq, nq, and truthfulqa')
            parser.add_argument('-m', '--model', type=str, default='')
            parser.add_argument('-n', '--num_turns', type=int, default=4)
            parser.add_argument('-f', '--failure', default=3) # max num of tries if the output format is illegal
            parser.add_argument('--tprob', default=0.2) # default temperature for probing
            parser.add_argument('--tnorm', default=0.8) # default temperature for (response) generation
            args = parser.parse_args()
            args.data_list = self.data_list
            args.dataset_name = self.dataset.dataset_name
            args.model_name = self.model.model_name
            args.model = self.model.model
            args.tokenizer = self.model.tokenizer
            args.metrics = self.metrics
            args.do_eval = do_eval
            
            result = main(args)

        elif self.conflict_method == "retrieveorgenerated":
            from kmatrix_cr.toolkit.retrieveorgenerated.src.combine import context_conflicting_dataset
            # step1
            
            
            
            # step2
            # 7b-chat 13b-chat gpt-3.5-turbo-0613 gpt-4-0613
            # nq tqa
            root_path = self.dataset.dataset_path
            reader = self.model.model_name
            generator = self.model.model_name
            dataset = self.dataset.dataset_name
            def full_name(name):
                if name in ['7b-chat', '13b-chat']:
                    return f"llama_model/{name}"
                else:
                    return name
            
            path = {
                'com_path':root_path + '/Answer-with-{}/{}/prompt_similar_length/Retrieved-contriever-1-Generated-{}-p1-trunclen-0.jsonl'.format(full_name(reader), dataset, generator),
                'gen_path':root_path + '/Answer-with-{}/{}/prompt_similar_length/Retrieved-none-0-Generated-{}-p1-trunclen-0.jsonl'.format(full_name(reader), dataset, generator),
                'ir_path':root_path + '/Answer-with-{}/{}/prompt_similar_length/Retrieved-contriever-1-Generated-none-p1-trunclen-0.jsonl'.format(full_name(reader), dataset),
                'llm_path': root_path + f'/Answer-with-{full_name(reader)}/{dataset}/prompt_similar_length/Retrieved-none-1-Generated-none-p3-trunclen-0.jsonl'
            }
            data = context_conflicting_dataset(**path)
            # print("\n\nDiffGR : \n", data.get_diffgr())
            result = {
            }
            if do_eval:
                if "em" in self.metrics:
                    result['em'] = data.em
                if "preference" in self.metrics:
                    result['preference'] = data.preference
                    

        else:
            result = {}
        
        if output_path != "":
            if result == {}:
                print("result is empty  ")
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f,ensure_ascii=False)
            
        return result
        
        