from transformers import AutoTokenizer, LlamaForCausalLM
import json
from transformers import  TopKLogitsWarper, TopPLogitsWarper
from tqdm import tqdm
import torch
import numpy as np

def load_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
        # data = json.load(f)
    return data

def save_file(data, path):
    with open(path, 'w', encoding='utf-8') as w:
        for unit in data:
            output = json.dumps(unit)
            w.write(output + "\n")
        w.close()

def entropy_from_scores(logits):
    # logits = torch.nn.functional.log_softmax(scores, dim=-1)
    logits = logits - logits.logsumexp(dim=-1, keepdims=True)
    entropy = (-1 * logits.exp() * logits).sum(-1)
    return entropy

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 0.9,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits

def typical_sample_mask(scores, logits, mass=0.9, threshold_ratio=4):
    # calculate entropy
    normalized = torch.nn.functional.log_softmax(logits, dim=-1)
    p = torch.exp(normalized)
    ent = -(normalized * p).nansum(-1, keepdim=True)

    # shift and sort
    normalized = torch.nn.functional.log_softmax(scores, dim=-1)
    shifted_scores = (-normalized) - ent

    # constraint
    scores_normalized = shifted_scores.log_softmax(dim=-1) 
    probs_min = torch.min(scores_normalized, dim=-1).values
    probs_thresh = probs_min + np.log(threshold_ratio)
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_filter = probs_max - np.log(threshold_ratio)
    probs_filter = probs_filter.unsqueeze(-1)
    mask_filter = [scores_normalized > probs_filter]
    
    # print(min_thresh, probs_thresh)
    # probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    mask = [scores_normalized > probs_thresh]
    count_mask = [scores_normalized < probs_thresh]
    if count_mask[0].sum() == 1:
        mask = torch.ones(scores.shape[-1], dtype=torch.bool).unsqueeze(0)
    
    return mask, mask_filter

def coiecd_constraint(logits_cond, logits_uncond, alpha=1.0):
    
    logits_diff = logits_cond - logits_uncond
    
    typical_mask, mask_filter = typical_sample_mask(logits_cond, logits_uncond)
    constraint_list = torch.ones_like(logits_diff)

    alpha_list = torch.ones_like(logits_diff) * alpha

    constraint_list[typical_mask] = float(0)
    constraint_list[mask_filter] = float(1)
    _constraint_list = 1- constraint_list
    
    logits_merged = constraint_list * logits_cond + _constraint_list * logits_uncond + logits_diff * alpha_list
    
    return logits_merged


"""
def model_generate(model, input_ids, attention_mask, tgt_len, past_key_values=None):
    ans = torch.tensor([], dtype=torch.int64, device=device)
    n = input_ids.shape[0]
    for i in range(tgt_len):
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                            use_cache=True,
                            past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
        
        logits = outputs.logits[:, -1, :] 
        logits = logits - logits.logsumexp(dim=-1, keepdims=True) 
        probs = torch.nn.functional.softmax(logits, dim=-1)

        next_tokens = torch.argmax(probs, dim=-1)
        ans = torch.cat([ans, next_tokens], dim=-1)
        if next_tokens[0] == tokenizer.eos_token_id:
            break
        # prepare for next iteration
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1)
    answer = tokenizer.decode(ans, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return answer
"""

def model_answer(model, tokenizer, question, facts, tgt_len):
    device = 'cuda'
    # Generate
    context = f'Given the following information:{facts}\nAnswer the following question based on the given information with one or few words: {question}\nAnswer:'

    prompt = f'Answer the following question based on your internal knowledge with one or few words: {question}\nAnswer:'


    batch = [context, prompt]
    inputs = tokenizer(batch, padding=True, return_tensors='pt', truncation=True, max_length=2048).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask 
    

    #------------------coie output--------------------------#
    past_key_values = None    
    ans = torch.tensor([], dtype=torch.int64, device=device)
    alpha = 1.0
    n = input_ids.shape[0]
    for i in range(tgt_len):
    # for i in range(tgt_len):
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                            use_cache=True,
                            past_key_values=past_key_values
                        )
            past_key_values = outputs.past_key_values
        
        
        logits = outputs.logits[:, -1, :] 
        logits = logits - logits.logsumexp(dim=-1, keepdims=True) # scale logits for numerical stability of exp(logits) operation and keep the value of softmax(logits) unchanged

        logits_cond = logits[0].unsqueeze(0)
        logits_uncond = logits[1].unsqueeze(0)
        
        logits_merged = coiecd_constraint(logits_cond, logits_uncond, alpha)
        
        # logits_merged = top_k_top_p_filtering(logits_merged, top_k=0, top_p=0.9)
        probs = torch.nn.functional.softmax(logits_merged, dim=-1)
        # next_tokens = torch.multinomial(probs, num_samples=1)[0]
        next_tokens = torch.argmax(probs, dim=-1)
        ans = torch.cat([ans, next_tokens], dim=-1)

        if next_tokens[0] == tokenizer.eos_token_id:
            break
        # prepare for next iteration
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1)   
    gen_answer = tokenizer.decode(ans, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return gen_answer #, cond_answer, uncond_answer

def main(model,tokenizer,data_list):
    
    tokenizer.pad_token_id = 0
    filter_value = -float("Inf")
    #------------json file----------------------------#
    data = data_list
    # data = data[:10]
    final_data_list = []
    
    for index,line in enumerate(tqdm(data)):
        try:
            line = json.loads(line)
        except:
            pass
        question = line['question']
        
        # merge 
        # try:
        #     ground_truth = line['ground_truth'][0]
        #     tgt_len = len(tokenizer.encode(ground_truth, add_special_tokens=False))
        # except:
        #     ground_truth = ""
        #     tgt_len = 200
        
        tgt_len = 200
        context = "\n\n".join(line['c_text']) 
        
        
        gen_answer = model_answer(model, tokenizer, question, context, tgt_len)
        output_data = {'gen_answer': gen_answer}
        output_data.update(line)
        final_data_list.append(output_data)

    return final_data_list


if __name__ == '__main__':
    PATH_TO_CONVERTED_WEIGHTS = '/mnt/publiccache/huggingface/Qwen2.5-14B-Instruct'
    PATH_TO_CONVERTED_TOKENIZER = '/mnt/publiccache/huggingface/Qwen2.5-14B-Instruct'
    DATA_PATH = ''
    model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
    data_list = load_json_data(DATA_PATH)
    main(model=model,tokenizer=tokenizer,data_list=data_list)