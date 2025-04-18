def main(args):
    
    data_list = args.data_list
    llm_model = args.llm_model
    
    prompt_list = []
    for data in data_list:
        prompt = "Please use your knowledge to answer my Question.\nQuestion:\n" + data['question'] + "\n\nAnswer:\n"
        prompt_list.append(prompt)
        
    res_list = llm_model.run(prompt_list=prompt_list)
    
    for data,res in zip(data_list,res_list):
        data['gen_answer'] = res['content']
        
    return data_list