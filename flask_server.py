import os
from root_config import RootConfig
os.environ["CUDA_VISIBLE_DEVICES"] = RootConfig.CUDA_VISIBLE_DEVICES
os.environ["SERPER_API_KEY"] = RootConfig.SERPER_API_KEY
os.environ['OPENAI_API_KEY'] =  RootConfig.OPENAI_API_KEY
os.environ['DEEPSEEK_API_KEY'] =  RootConfig.DEEPSEEK_API_KEY
# os.environ.pop('HF_HOME', None)

import sys
import json
import traceback
from flask import Flask, jsonify, request
from flask_cors import *
import torch
from kmatrix_cr.utils.common_utils import do_initCatch

# import ray
# ray.init(
#     num_gpus=torch.cuda.device_count(),
#     ignore_reinit_error=True,
#     include_dashboard=False  # 禁用不需要的仪表盘服务
# )

from kmatrix_cr.config.config import Config
from kmatrix_cr.dataset.dataset import Dataset
from kmatrix_cr.generator.llm_generator import LLmGenerator
from kmatrix_cr.generator.openai_generator import OpenAiGenerator
from kmatrix_cr.generator.deepseek_generator import DeepSeekGenerator
from kmatrix_cr.template.template_cm import CMTemplate
from kmatrix_cr.template.template_ic import ICTemplate
from kmatrix_cr.template.template_im import IMTemplate
from kmatrix_cr.utils.common_utils import eval


class Kninjllm_Flask:
    
    def __init__(self,flask_name,
                 default_datasets_path:str,
                 default_models_path:str,
                 upload_datasets_json_path:str,
                 upload_datasets_dir_path:str,
                ):
        
        # init flask_app
        self.app = Flask(flask_name)
        self.app = Flask(__name__, static_url_path='')
        self.app.config['JSON_AS_ASCII'] = False
        
        CORS(self.app, resources={r"/*": {"origins": "*"}}, send_wildcard=True)

        # init dir
        self.default_datasets_path = default_datasets_path
        self.default_models_path = default_models_path
        self.upload_datasets_json_path = upload_datasets_json_path
        self.upload_datasets_dir_path = upload_datasets_dir_path
        self.current_method = ""
        
        self.model_config = {
                "Llama-2-7b-chat-hf":RootConfig.LLMAM2_7B_CHAT_HF_MODEL_PATH,
                "Llama-2-13b-chat-hf":RootConfig.LLMAM2_13B_CHAT_HF_MODEL_PATH,
                "Baichuan2-7B-Chat":RootConfig.BAICHUAN2_7B_CHAT_MODEL_PATH,
                "Baichuan2-13B-Chat":RootConfig.BAICHUAN2_13B_CHAT_MODEL_PATH,
                "gpt-3.5-turbo":"gpt-3.5-turbo",
                "gpt-4o":"gpt-4o",
                "DeepSeek R1":"deepseek-chat"
            }
        
        
        if not os.path.exists(self.default_datasets_path):
            os.mkdir(self.default_datasets_path)
            
        if not os.path.exists(self.default_models_path):
            os.mkdir(self.default_models_path)
                
        if not os.path.exists(self.upload_datasets_dir_path):
            os.mkdir(self.upload_datasets_dir_path)
            
        if not os.path.exists(self.upload_datasets_json_path):
            with open(self.upload_datasets_json_path, 'w',encoding='utf-8') as f:
                json.dump([], f,ensure_ascii=False)
        
        # 上传 - 任务的数据集
        @self.app.route('/uploadDataset', methods=['POST'])
        def uploadDataset():
            
            option = request.form.get("option")
            uploadTime = request.form.get("uploadTime")
            print(option)
            
            if 'file' not in request.files:
                return jsonify({'error': 'No file part','code':500}) 

            files = request.files.getlist('file')
            if not files:
                return jsonify({'error': 'No file uploaded','code':500})

            with open(self.upload_datasets_json_path,'r',encoding='utf-8') as f:
                origin_datasetDataList = json.load(f)

            for file in files:
                filename = file.filename 
                
                has_flag = False
                # 如果存在就更新上传时间 不存在就添加
                for index,d in enumerate(origin_datasetDataList):
                    if d['fileName'] == filename:
                        has_flag = True
                        origin_datasetDataList[index]['uploadTime'] = uploadTime
                        origin_datasetDataList[index]['option'] = option
                if has_flag == False:
                    print("添加",filename)
                    origin_datasetDataList.append({
                        'fileName':filename,
                        'uploadTime':uploadTime,
                        'option':option,
                    })

                # 保存文件到服务器
                print(filename)
                savePath = self.upload_datasets_dir_path + filename
                file.save(savePath)  
            with open(self.upload_datasets_json_path,'w',encoding='utf-8') as f:
                json.dump(origin_datasetDataList, f, ensure_ascii=False)

            return jsonify({'message': 'Files successfully uploaded','code':200})
        
        
        # 获取模型参数知识
        @self.app.post('/get_parameters_by_question')
        def get_parameters_by_question():
            jsondata = request.get_json()
            question = jsondata["question"]
            option_params = jsondata["option_params"]
            try:
                option_params = json.loads(option_params)
            except:
                pass
            model_name = option_params['model_name']
            
            if model_name in self.model_config:
                llama_model_path = self.model_config[model_name]
            else:
                return jsonify({"data": f"Unsupported model: {llama_model_path}", "code": 500})
            
            if "gpt" in llama_model_path:
                llm_model = OpenAiGenerator(api_key=os.getenv('OPENAI_API_KEY'),model_version=llama_model_path)  # any
            elif "deepseek" in llama_model_path:
                llm_model = DeepSeekGenerator(api_key=os.getenv('DEEPSEEK_API_KEY'),model_version=llama_model_path)  # any
            else:
                llm_model = LLmGenerator(model_path=llama_model_path,load_model=True,load_model_mode="llm")  # any
            
            res1 = llm_model.run(prompt_list=[question],sampling_params={"max_tokens":500,"temperature":0.0})[0]['content']
            res2 = llm_model.run(prompt_list=[question],sampling_params={"max_tokens":500,"temperature":0.5})[0]['content']
            res3 = llm_model.run(prompt_list=[question],sampling_params={"max_tokens":500,"temperature":1.0})[0]['content']
                
            result = res1 + "\n\n" + res2 + "\n\n" + res3
            
            return jsonify({"data": result, "code": 200})
        
        # 开始运行任务
        @self.app.post('/chat')
        def chat():
            jsondata = request.get_json()
            
            input_obj = jsondata["input"]
            try:
                input_obj = json.loads(input_obj)
            except:
                pass
            
            # 获取 model_name
            model_name = jsondata["option"]["model_name"]

            if model_name in self.model_config:
                llama_model_path = self.model_config[model_name]
            else:
                return jsonify({"data": f"Unsupported model: {llama_model_path}", "code": 500})
            openai_model_version = "gpt-3.5-turbo"
            
            # 获取 method 
            root_conflict_avoidance_strategy = jsondata["option"]["root_conflict_avoidance_strategy"]
            if "IC" in root_conflict_avoidance_strategy[0]:
                input_obj['text'] = input_obj['c_text']
            if "CM" in root_conflict_avoidance_strategy[0]:
                input_obj['text'] = input_obj['m_text']
                
            if isinstance(root_conflict_avoidance_strategy,list) and len(root_conflict_avoidance_strategy) == 3:
                root_conflict_avoidance_strategy_method = root_conflict_avoidance_strategy[2]
            if isinstance(root_conflict_avoidance_strategy,list) and len(root_conflict_avoidance_strategy) == 1:
                root_conflict_avoidance_strategy_method = root_conflict_avoidance_strategy[0]
            if isinstance(root_conflict_avoidance_strategy,str):
                root_conflict_avoidance_strategy_method = root_conflict_avoidance_strategy
            print("---------------------root_conflict_avoidance_strategy_method----------------\n",root_conflict_avoidance_strategy_method)
                
            # method and model 匹配校验(有些方法必须使用LLM)  Disent QA    llms-believe-the_earth_is_flat     Concord
            must_llm_list = ["Coiecd","Aware-Decoding","Contrastive-Decoding","Dola","Disent QA","llms-believe-the_earth_is_flat"]
            if root_conflict_avoidance_strategy_method in must_llm_list and ("gpt" in llama_model_path or "deepseek" in llama_model_path):
                return jsonify({"data": f"{must_llm_list} it is imperative to use LLM", "code": 500})
                
            max_tokens = jsondata["option"]["max_tokens"]
            temperature = jsondata["option"]["temperature"]
            
            # # 显存处理
            # if self.current_method != root_conflict_avoidance_strategy_method and self.current_method != "":
            #     try:
            #         do_initCatch(clean_knowledge=True,clean_model=True)
            #         # ray.init(
            #         #     num_gpus=torch.cuda.device_count(),
            #         #     ignore_reinit_error=True,
            #         #     include_dashboard=False  # 禁用不需要的仪表盘服务
            #         # )
            #     except Exception as e:
            #         print(e)
            
            
            
            self.current_method = root_conflict_avoidance_strategy_method
            
            # 无方法选择
            if root_conflict_avoidance_strategy_method == "None":
                if "gpt" in llama_model_path:
                    llm_model = OpenAiGenerator(api_key=os.getenv('OPENAI_API_KEY'),model_version=llama_model_path)  # any
                elif "deepseek" in llama_model_path:
                    llm_model = DeepSeekGenerator(api_key=os.getenv('DEEPSEEK_API_KEY'),model_version=llama_model_path)  # any
                else:
                    llm_model = LLmGenerator(model_path=llama_model_path,load_model=True,load_model_mode="llm")  # any
                
                res = llm_model.run(prompt_list=[input_obj['question']],sampling_params={"max_tokens":max_tokens,"temperature":temperature})[0]
                
                result = {
                    "result":[
                        {
                            "quesion":input_obj['question'],
                            "gen_answer":res['content']
                        }
                    ]
                }

            else:
                dataset = Dataset(data_list=[input_obj])
                # ------------------------------CM----------------------------------------
                if root_conflict_avoidance_strategy_method == "Coiecd":
                    print(f'------------------------CM-coiecd测试-----------------------------')
                    llm_model = LLmGenerator(model_path=llama_model_path)  # any
                    config = Config(dataset=dataset,
                                    llm_model=llm_model,
                                    metrics = ["acc"])
                    template = CMTemplate(config=config,conflict_method="coiecd")
                    result = template.run(do_eval=False)
                    
                if root_conflict_avoidance_strategy_method == "Context-Faithful":
                        print(f'------------------------CM-Context-Faithful 测试-----------------------------')
                        # llm_model = LLmGenerator(model_path=llama_model_path,load_model=True,load_model_mode="llm")  # any
                        if "gpt" in llama_model_path:
                            llm_model = OpenAiGenerator(api_key=os.getenv('OPENAI_API_KEY'),model_version=llama_model_path,generation_kwargs={"max_tokens":max_tokens,"temperature":temperature})  # any
                        elif "deepseek" in llama_model_path:
                            llm_model = DeepSeekGenerator(api_key=os.getenv('DEEPSEEK_API_KEY'),model_version=llama_model_path,generation_kwargs={"max_tokens":max_tokens,"temperature":temperature})  # any
                        else:
                            llm_model = LLmGenerator(model_path=llama_model_path,generation_kwargs={"max_tokens":max_tokens,"temperature":temperature},load_model=True,load_model_mode="llm")  # any
                        
                        
                        config = Config(dataset=dataset,
                                        llm_model=llm_model,
                                        metrics = ["acc"])
                        template = CMTemplate(config=config,conflict_method="context-faithful")
                        result = template.run(do_eval=False)
                        
                if root_conflict_avoidance_strategy_method == "Aware-Decoding":
                        print(f'------------------------CM-Aware-Decoding 测试-----------------------------')
                        llm_model = LLmGenerator(model_path=llama_model_path,load_model=False)  # any
                        config = Config(dataset=dataset,
                                        llm_model=llm_model,
                                        metrics = ["acc"])
                        template = CMTemplate(config=config,conflict_method="aware-decoding")
                        result = template.run(do_eval=False)
                        
    
                if root_conflict_avoidance_strategy_method == "Contrastive-Decoding":
                        print(f'------------------------CM-Contrastive-Decoding 测试-----------------------------')
                        llm_model = LLmGenerator(model_path="gpt2-xl",load_model=False)  # any
                        config = Config(dataset=dataset,
                                        llm_model=llm_model,
                                        metrics = ["acc"])
                        template = CMTemplate(config=config,conflict_method="ContrastiveDecoding")
                        result = template.run(do_eval=False)
                
                if root_conflict_avoidance_strategy_method == "Disent QA":
                    
                    # return jsonify({"data": "Disent QA 输入输出没有统一, 会跑自带的数据集,比较慢,跳过 (功能已经接入了)...", "code": 200})
                    
                    print(f'------------------------CM-Disent QA 测试-----------------------------')
                    # llm_model = LLmGenerator(model_path=self.default_models_path + "disent_qa/t5-small_disent_qa_train_contextual_baseline_85540_b128_lr0.0001_e20_smax256.ckpt",load_model=False)  # any
                    llm_model = LLmGenerator(model_path=llama_model_path,load_model=True,load_model_mode="llm",generation_kwargs={"max_tokens":max_tokens,"temperature":temperature})  # any
                    config = Config(dataset=dataset,
                                    llm_model=llm_model,
                                    metrics = ["acc"])
                    template = CMTemplate(config=config,conflict_method="Disent_QA")
                    result = template.run(do_eval=False)
                    
                if root_conflict_avoidance_strategy_method == "llms-believe-the_earth_is_flat":
                    
                    return jsonify({"data": "llms-believe-the_earth_is_flat 输入输出没有统一, 会跑自带的数据集,比较慢,跳过(功能已经接入了)...", "code": 200})
                    dataset = Dataset(dataset_path=self.default_datasets_path + "CM/llms_believe_the_earth_is_flat_NQ1.jsonl")
                    llm_model = LLmGenerator(model_path=llama_model_path,load_model_mode="llm") # ["meta-llama/Llama-2-7b-chat-hf","meta-llama/Llama-2-13b-chat-hf","lmsys/vicuna-7b-v1.5","lmsys/vicuna-13b-v1.5"]
                    config = Config(dataset=dataset,
                                    llm_model=llm_model,
                                    metrics = ["em","preference","acc"],
                                    )
                    template = CMTemplate(config=config,conflict_method="llms_believe_the_earth_is_flat")
                    result = template.run(do_eval=False)
                    
                # -----------------------------IC-----------------------------------------------
                if root_conflict_avoidance_strategy_method == "ICL-seprate":
                    print(f'------------------ic ICL-seprate (ICL-seprate) 测试------------------------')
                    openai_model=OpenAiGenerator(api_key=os.getenv('OPENAI_API_KEY'),model_version=openai_model_version)  # openai
                    # llm_model = LLmGenerator(model_path=llama_model_path,load_model=True,load_model_mode="llm")  # any
                    if "gpt" in llama_model_path:
                        llm_model = OpenAiGenerator(api_key=os.getenv('OPENAI_API_KEY'),model_version=llama_model_path,generation_kwargs={"max_tokens":max_tokens,"temperature":temperature})  # any
                    elif "deepseek" in llama_model_path:
                        llm_model = DeepSeekGenerator(api_key=os.getenv('DEEPSEEK_API_KEY'),model_version=llama_model_path,generation_kwargs={"max_tokens":max_tokens,"temperature":temperature})  # any
                    else:
                        llm_model = LLmGenerator(model_path=llama_model_path,generation_kwargs={"max_tokens":max_tokens,"temperature":temperature},load_model=True,load_model_mode="llm")  # any
                    
                    config = Config(dataset=dataset,
                                    openai_model=openai_model,
                                    llm_model=llm_model,
                                    metrics = ["acc"],
                                    )
                    template = ICTemplate(config=config,conflict_method="ICL-seprate")
                    result = template.run(do_eval=False)
                    
                if root_conflict_avoidance_strategy_method == "ICL-whole":
                    print(f'----------------ic ICL-whole (ICL-whole) 测试------------------------')
                    # llm_model = LLmGenerator(model_path=llama_model_path,load_model=True,load_model_mode="llm")  # any
                    if "gpt" in llama_model_path:
                        llm_model = OpenAiGenerator(api_key=os.getenv('OPENAI_API_KEY'),model_version=llama_model_path,generation_kwargs={"max_tokens":max_tokens,"temperature":temperature})  # any
                    elif "deepseek" in llama_model_path:
                        llm_model = DeepSeekGenerator(api_key=os.getenv('DEEPSEEK_API_KEY'),model_version=llama_model_path,generation_kwargs={"max_tokens":max_tokens,"temperature":temperature})  # any
                    else:
                        llm_model = LLmGenerator(model_path=llama_model_path,generation_kwargs={"max_tokens":max_tokens,"temperature":temperature},load_model=True,load_model_mode="llm")  # any
                    
                    
                    
                    config = Config(dataset=dataset,
                                    llm_model=llm_model,
                                    metrics = ["acc"],
                                    )
                    template = ICTemplate(config=config,conflict_method="ICL-whole")
                    result = template.run(do_eval=False)
                    
                # -----------------------------IM-----------------------------------------------
                if root_conflict_avoidance_strategy_method == "Dola":
                    print(f'----------------IM  Dola 测试------------------------')
                    llm_model = LLmGenerator(model_path=llama_model_path,load_model=False)    # any "huggyllama/llama-7b"
                    config = Config(dataset=dataset,
                                    llm_model=llm_model,
                                    metrics = ["acc"],
                                    )
                    template = IMTemplate(config=config,conflict_method="dola")
                    result = template.run(do_eval=False)
                    

            # print('----------------result-------------------\n',result)
            
            # 精简结果
            del_filed_list = [
                "c_text",
                "m_text",
                "input_index",
                "assigned_model",
                "article",
                "filter_p",
                "gold_answers",
                "model_name",
                "context-faithful-prompt",
                "text",
                "ctxs_content_list",    # ICL-seprate
                # "filter_ctxs_list",     # ICL-seprate
                "detailed_information", # factool
                "is_correct",
                "full_input_text",
                "model_completion",
                "should_flip",
                "bad_details",
                "good_details",
            ]
            
            
            if isinstance(result['result'],list):
                for res in result['result']:
                    # 遍历字典的键并删除在数组中的键
                    for key in list(res.keys()):  # 使用 list() 来复制键列表，以避免在迭代时修改字典
                        if key in del_filed_list:
                            del res[key]
            else:
                for key in list(result['result'].keys()):  # 使用 list() 来复制键列表，以避免在迭代时修改字典
                    if key in del_filed_list:
                        del result['result'][key]
                        
            return jsonify({"data": result, "code": 200})
        
    # 去重字典数组
    def remove_duplicates(self,dict_list):
        seen = set()
        unique_dicts = []
        for d in dict_list:
            items = tuple(d.items())
            if items not in seen:
                seen.add(items)
                unique_dicts.append(d)
        return unique_dicts

    # run 
    def run(self, host, port):
        self.app.run(host=host, port=port,threaded=True)

if __name__ == "__main__":
    my_flask_app = Kninjllm_Flask(flask_name='my_flask_app',
                                  default_datasets_path=RootConfig.root_path +"kmatrix_cr_datasets/",
                                  default_models_path=RootConfig.root_path + "kmatrix_cr_models/",
                                  upload_datasets_json_path=RootConfig.root_path + "kmatrix_cr_upload_datasets.json",
                                  upload_datasets_dir_path=RootConfig.root_path + "kmatrix_cr_upload_datasets/",
                                  )
    my_flask_app.run(host='0.0.0.0', port=10026)
    
    
