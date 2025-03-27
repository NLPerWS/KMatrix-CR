import time
import requests
import traceback
from typing import Any, Dict, List, Optional
from kmatrix_cr.generator.root_generator import RootGenerator

class OpenAiGenerator(RootGenerator):
    def __init__(self,api_key,model_version:str='gpt-3.5-turbo',generation_kwargs: Dict[str, Any] = {
        "max_tokens":100,
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
        
    def chatWithOpenAiByRequest(self,prompt,proxy=None,retry=1,timeout=20):
        api_url = 'https://api.openai.com/v1/chat/completions'
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
                time.sleep(timeout)
                continue
        
        raise Exception("OpenAi request error ...")
        # return ""

    def embeddingWithOpenAiByRequest(self,prompt,proxy=None,retry=3,timeout=20):
        api_url = 'https://api.openai.com/v1/embeddings'
        for i in range(retry):
            try:
                response = requests.post(api_url, json={
                    'model': self.model_version,
                    "encoding_format": "float",
                    'input': prompt
                }, headers={'Authorization': f'Bearer {self.api_key}',"Content-Type": "application/json"}, proxies=proxy)
                return response.json()['data'][0]['embedding']
            except Exception as e:
                traceback.print_exc()
                time.sleep(timeout)
                continue
        return []
    
    def run(
        self,
        prompt: str = "",
        prompt_list: List[str] = [],
    ):
        if prompt != "" and len(prompt_list) == 0:
            prompt_list = [prompt]
        
        
        final_result = []
        for index,one_prompt in enumerate(prompt_list):
            while True:
                try:
                    content = self.chatWithOpenAiByRequest(one_prompt)
                    break
                except Exception as e:
                    time.sleep(10)
                    continue
                
            final_result.append({
                "prompt":one_prompt,
                "content":content,
                "meta":{},    
            })    
            print(f"{index+1}/{len(prompt_list)} OK ... ")
        
        return final_result

