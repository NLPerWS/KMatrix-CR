import json
import time
import re
from kmatrix_cr.toolkit.disent_qa.run_nq_fine_tuning import NaturalQuestionsModel
import os
from pathlib import Path
import torch
from transformers import T5Tokenizer
import pandas as pd
import argparse

print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
torch.cuda.empty_cache()

answer_types = {"f": "factual", "cf": "counterfactual", "cb": "closed_book", "rc": "random_context", "all": None}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def df_chunks_generator(lst, n, input_col):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        small_df = lst.iloc[i:i + n]
        input_items = small_df[input_col].to_list()
        # print('-----------------------------input_items----------------------\n',input_items)
        yield input_items


def generate_answer(question,question_and_context,input_max_length, output_max_length, repetition_penalty, length_penalty,
                    num_beams):
    """
    pass a batch of question_and_context together (in the format question: <question>\ncontext: <context>) to the model
    to generate an answer batch
    """
    
    try:
        # with content
        source_encoding_with_content = tokenizer(question_and_context, max_length=input_max_length,
                                    padding="max_length", truncation=True, return_attention_mask=True,
                                    add_special_tokens=True, return_tensors="pt")
        source_encoding_with_content = {key: tensor.to(device) for key, tensor in source_encoding_with_content.items()}
        generated_ids_with_content = model.model.generate(input_ids=source_encoding_with_content["input_ids"],
                                            attention_mask=source_encoding_with_content["attention_mask"], num_beams=num_beams,
                                            max_length=output_max_length, repetition_penalty=repetition_penalty,
                                            length_penalty=length_penalty,
                                            early_stopping=True, use_cache=True)
        preds_with_content = [tokenizer.decode(generated_id_with_content, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                generated_id_with_content in generated_ids_with_content]
        
        # only paramerter
        source_encoding = tokenizer(question, max_length=input_max_length,
                                    padding="max_length", truncation=True, return_attention_mask=True,
                                    add_special_tokens=True, return_tensors="pt")
        source_encoding = {key: tensor.to(device) for key, tensor in source_encoding.items()}
        generated_ids = model.model.generate(input_ids=source_encoding["input_ids"],
                                            attention_mask=source_encoding["attention_mask"], num_beams=num_beams,
                                            max_length=output_max_length, repetition_penalty=repetition_penalty,
                                            length_penalty=length_penalty,
                                            early_stopping=True, use_cache=True)
        preds_only_model = [tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                generated_id in generated_ids]
        
    except:
        # with content
        preds_with_content = llm_model.run(prompt_list=[question_and_context])[0]['content']
        # only paramerter
        preds_only_model =  llm_model.run(prompt_list=[question])[0]['content']
    
    return str(preds_with_content).strip(),str(preds_only_model).strip()


def query_model(data_list, config_dict, answer_type=""):
    """
    gets a path to an evaluation set and an answer type and queries the global loaded model,
    then saves the results to a csv file.
    """
        
    for data in data_list:
        
        question_only_model = "Please use your own parameter knowledge to answer my question\n\nQuestion: " + data['question'] + "\n\nAnswer:"
        question_with_content = "You must strictly adhere to the knowledge I provide when answering questions.\n\nknowledge:\n "+"\n".join(data['context']) +"\n\nQuestion:\n"+data['question'] + "\n\nAnswer:"
        
        model_output_with_content,model_output = generate_answer(question_only_model,question_with_content,  # pass in batches
                                       input_max_length=config_dict['input_max_length'],
                                       output_max_length=config_dict['output_max_length'],
                                       repetition_penalty=config_dict['repetition_penalty'],
                                       length_penalty=config_dict['length_penalty'],
                                       num_beams=config_dict['num_beams'])
        
        
        data['gen_answer'] = f"contextual: {model_output_with_content}\n\nparametric: {model_output}"
        
    return data_list

def main(args):
    global tokenizer, model, llm_model

    query_answer_type = answer_types[args.answer_type] if args.answer_type else ""
    f = open(os.path.dirname(__file__) + "/config.json")
    config = json.load(f)["query_model"]
    
    if args.llm_model != None:
        llm_model = args.llm_model
        model = args.llm_model.model
        tokenizer = args.llm_model.tokenizer
    else:
        print("loading the {} model".format(args.checkpoint_name))
        # tokenizer = T5Tokenizer.from_pretrained(config["model_name"], cache_dir="models/")
        tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
        # load checkpoint from a path in the form of "checkpoints/" + checkpoint_names[args.model_name] + ".ckpt")
        trained_model = NaturalQuestionsModel.load_from_checkpoint(
            args.checkpoint_name,
            map_location=device,
            model_name=config['model_name'],
            learning_rate=config['learning_rate']
            
            )
            
        trained_model.freeze()
        model = trained_model
        
    data_list = query_model(data_list=args.data_list,answer_type=query_answer_type,config_dict=config)
    
    return data_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False, help="Path to a csv file")
    parser.add_argument("--checkpoint_name", type=str, required=False, help="checkpoint_name - path with .ckpt")
    parser.add_argument("--answer_type", type=str, required=False, help="f, cf, rc, or cb.")
    args = parser.parse_args()
    main(args)