import time
import requests
import traceback
from typing import Any, Dict, List, Optional
from kmatrix_cr.generator.root_generator import RootGenerator

class DeepSeekGenerator(RootGenerator):
    def __init__(self,api_key,model_version:str='deepseek-chat',generation_kwargs: Dict[str, Any] = {
        "max_tokens":1000,
        "temperature":0
    }):
        self.api_key = api_key
        self.model_version = model_version
        self.generation_kwargs = generation_kwargs
        self.model = None
        self.tokenizer = None
        self.model_name = model_version
        
    def load_model(self):
        pass
        
    def chatWithDeepseekByRequest(self,prompt,proxy=None,retry=3,timeout=20):
        api_url = 'https://api.deepseek.com/chat/completions'
        for i in range(retry):
            try:
                param = {
                    'model': self.model_version,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
                param.update(self.generation_kwargs)
                
                response = requests.post(api_url, json=param,
                            headers={'Authorization': f'Bearer {self.api_key}',"Content-Type": "application/json"}, proxies=proxy)
                return response.json()['choices'][0]['message']['content']
            except:
                traceback.print_exc()
                print("--------------------------------------response-----------------------------\n",response)
                time.sleep(timeout)
                continue
        
        raise Exception("DeepSeek request error ...")
        # return ""

    def run(
        self,
        prompt: str = "",
        prompt_list: List[str] = [],
        sampling_params: Any = None
    ):
        if prompt != "" and len(prompt_list) == 0:
            prompt_list = [prompt]
        if sampling_params is not None:
            self.generation_kwargs.update(sampling_params)
        
        final_result = []
        for one_prompt in prompt_list:
            content = self.chatWithDeepseekByRequest(one_prompt)
            final_result.append({
                "prompt":one_prompt,
                "content":content,
                "meta":{},    
            })    
        return final_result

