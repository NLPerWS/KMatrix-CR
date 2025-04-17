import hashlib
import json
import subprocess
from root_config import RootConfig
def get_random_id_from_string(input_string):
    hash_object = hashlib.md5(input_string.encode())
    return str(hash_object.hexdigest())

def do_initCatch(clean_knowledge=True,clean_model=True):

    def get_gpu_memory_usage(gpu_index):
        """Helper function to query GPU memory usage for a specific GPU."""
        try:
            # Execute 'nvidia-smi' command and parse the output
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                stdout=subprocess.PIPE,
                check=True
            )
            # Decode the byte string, strip extra whitespace, and split by line
            usage_lines = result.stdout.decode('utf-8').strip().split('\n')
            # Parse the desired GPU's usage line to get the memory usage as integer
            if gpu_index < len(usage_lines):
                return int(usage_lines[gpu_index].strip())
            else:
                raise ValueError(f"GPU index {gpu_index} out of range, found {len(usage_lines)} GPUs")
        except Exception as e:
            print(f"Error querying GPU memory usage: {e}")
            return None

    if clean_model:
        del RootConfig.tempModelCatch
        RootConfig.tempModelCatch = []
        
        import gc
        # import ray
        import torch
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

        # 终止模型并行进程
        destroy_model_parallel()
        # 判断并终止分布式进程组
        try:
            torch.distributed.destroy_process_group()
        except Exception as e:
            pass
        # 清除CUDA缓存和同步设备
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # 关闭Ray进程
        # ray.shutdown()
        # 进行垃圾收集
        gc.collect()
        
        # # Wait for GPU memory to be released
        # initial_memory_usage = get_gpu_memory_usage(gpu_index=int(RootConfig.CUDA_VISIBLE_DEVICES))
        # print("-----------initial_memory_usage--------\n",initial_memory_usage)
        
        # timeout = 60  # set a timeout in seconds
        # elapsed_time = 0
        
        # while elapsed_time < timeout:
        #     time.sleep(5)  # check every second
        #     current_memory_usage = get_gpu_memory_usage(gpu_index=int(RootConfig.CUDA_VISIBLE_DEVICES))
        #     print("-----------current_memory_usage--------\n",current_memory_usage)
        #     if current_memory_usage is not None and current_memory_usage <= initial_memory_usage - 30000: 
        #         break
        #     elapsed_time += 1

    return True

def eval(metrics=["acc"],data={},data_path=""):
    
    if data_path != "":
        try:
            with open(data_path,'r',encoding='utf-8') as f:
                data = json.load(f)
        except:
            print(f"{data_path} is not json file")
            return
    else:
        data = data
    
    if data == {}:
        print("data is empty")
        return ""
        
    try:
        data = data['result']
    except:
        data = data
        
    log_str = ""
    if data_path != "":
        log_str += f"------------------------------{data_path}----------------------------\n"
    
    eval_obj = {
        "log_str":"",
    }
    
    if "acc" in metrics:
        rex_count_ok = 0
        in_count_ok = 0
        
        for index,d in enumerate(data):

            try:
                data_gen_answer = d['gen_answer']
            except:
                data_gen_answer = d['gen_result']
            
            data_ground_truth_list = d['ground_truth']
            
            data_gen_answer = str(data_gen_answer).strip().lower()
            # if data_gen_answer.endswith('.'):
                # data_gen_answer = data_gen_answer[:-1]
            
            data_ground_truth_list = [str(ground_truth).strip().lower() for ground_truth in data_ground_truth_list]
            
            if data_gen_answer in data_ground_truth_list:
                rex_count_ok += 1
                
            for ground_truth in data_ground_truth_list:
                if ground_truth in data_gen_answer:
                    in_count_ok += 1
                    break
        
        rex_acc = rex_count_ok / len(data)  * 100
        in_acc = in_count_ok / len(data) * 100

        log_str += f'sum count: {len(data)}\n'
        log_str += f'rex_acc:  {rex_acc:.2f}%\n'
        log_str += f'in_acc:  {in_acc:.2f}%\n'
        
        eval_obj['rex_acc'] = f"{rex_acc:.2f}%"
        eval_obj['in_acc'] = f"{in_acc:.2f}%"

    eval_obj['log_str'] = log_str
    return eval_obj
        
        