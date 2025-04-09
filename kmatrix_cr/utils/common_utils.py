import hashlib
import json
def get_random_id_from_string(input_string):
    hash_object = hashlib.md5(input_string.encode())
    return str(hash_object.hexdigest())


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
        