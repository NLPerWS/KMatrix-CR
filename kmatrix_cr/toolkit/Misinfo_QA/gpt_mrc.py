print_example = False

no_context_prompts = [
    # "{question} \n\n The answer is", # cited from GenRead
    "{question} \n\n The answer in just a few words: "
]

context_prompts = [
    "Refer to the passages below and answer the following question with just a few words. \n\n Passages: {passages} \n\n Question: {question} \n\n Answer:",
]

multiple_answers_prompts = [
    "Refer to the passages below and answer the following question with just a few words. Sometimes the passages might support different answers, in which case you could give multiple answers separated by semicolons.  \n\n Passages: {passages} \n\n Question: {question} \n\n Answer:",
]

disinformation_aware_heads = [
    "Refer to the passages below and answer the following question with just a few words. Note that a small portion of the passages is crafted to mislead you.",
    "Answer the question below using just a few words after reading the passages provided. Beware that a small portion of the passages is designed to deceive you.",
    "After reading the passages given, use only a few words to answer the following question. Be cautious since some parts of the passages may mislead you.",
    "In the provided passages, some text is intended to mislead you, but read them nonetheless and answer the question with a few words. What is the answer to the question?",
    "Your task is to answer a question using only a few words after reading the given passages. Beware that some parts of the passages are meant to deceive you.",
    "Using only a few words, answer the question that follows after reading the passages provided. However, keep in mind that some of the passages are crafted to mislead you."
]

disinformation_aware_tails = [
    "\n\n Passages: {passages} \n\n Question: {question} \n\n Answer:", 
]

disinformation_aware_prompts = [
    head + " " + tail for head in disinformation_aware_heads for tail in disinformation_aware_tails
]

extraction_prompts = [
    "Summarize the 10 snippets of text below relevant to the question in 100 words: \n\n Question: {question} \n\n Snippets: {snippets} \n\n",
]

context_with_holdback_prompts = [
    "Refer to the passages below and answer the following question with just a few words. If the passages support zero or multiple answers, output NEI. \n\n Passages: {passages} \n\n Question: {question} \n\n Answer:",
]

vote_prompts = [
    "Refer to the following responses produced by different individuals to a question. Only considering the responses not refraining from answering, output the majority opinion within five words. \n\n Responses: {responses} \n\n Question: {question} \n\n Answer:",
]

def gpt_gen(filled_prompt, engine):
    res = engine.run(prompt_list=[filled_prompt])[0]['content']
    return res
    

def gpt_vote(responses, question,engine):
    return gpt_gen(vote_prompts[0].format(responses="\n\n".join(responses), question=question),engine)

def get_gpt_answer(question, context, ctx_per_call, multi_answer, disinformation_aware, holdback, prompt_idx, engine): 
    if holdback:
        filled_prompt = context_with_holdback_prompts[prompt_idx].format(passages="\n\n".join(context[:ctx_per_call]), question=question)
    elif disinformation_aware:
        filled_prompt = disinformation_aware_prompts[prompt_idx].format(passages="\n\n".join(context[:ctx_per_call]), question=question)
    elif multi_answer:
        # print("multi_answer")p
        filled_prompt = multiple_answers_prompts[prompt_idx].format(passages="\n\n".join(context[:ctx_per_call]), question=question)
    elif ctx_per_call != 0:
        filled_prompt = context_prompts[prompt_idx].format(passages="\n\n".join(context[:ctx_per_call]), question=question)
    else:
        filled_prompt = no_context_prompts[prompt_idx].format(question=question)
    if print_example:
        print(filled_prompt)
    
    response = gpt_gen(filled_prompt, engine=engine)
    return response.strip()

def extraction(prompt, snippets, top_k, engine):
    # if not random:
    output = []
    for i in range(0, len(snippets), top_k):
        filled_prompt = extraction_prompts[0].format(question=prompt, snippets="\n\n".join(snippets[i:i+top_k]))
        response = gpt_gen(filled_prompt, engine=engine)
        output.append(response.strip())
    return output

def get_question_and_context(qac, title=False):
    if title:
        return qac['question'], [c['title'] + '[SEP]' + c['text'] for c in qac['ctxs']]
    else:
        return qac['question'], [c['text'] for c in qac['ctxs']]
    # return qac['question'], [c['title'] + '[SEP]' + c['text'] for c in qac['ctxs']]
    # return qac['question'], [c['text'] for c in qac['ctxs']]

def extract_then_read(data_list, top_k, multi_answer, disinformation_aware, size_limit,engine):
    output = []
    import json
    dpr_output = data_list
    from tqdm import tqdm
    for qac in tqdm(dpr_output[:size_limit]):
        question, snippets = get_question_and_context(qac)
        context = extraction(question, snippets, top_k, engine)
        answer = get_gpt_answer(question=question, context=context,ctx_per_call=top_k, multi_answer=multi_answer, disinformation_aware=disinformation_aware,engine=engine)
        output.append(answer.replace('\n', ' ').strip())
    return output

def multi_reader_vote(data_list, top_k, multi_answer, disinformation_aware, size_limit, holdback, sample_question,engine, ctx_per_call=4):
    output = []
    import json
    dpr_output = data_list
    if sample_question:
            dpr_output = dpr_output[-300:]
    print('Using {} per call, totalled {} contexts, for {} questions'.format(ctx_per_call, top_k, len(dpr_output)))
    from tqdm import tqdm
    for qac in tqdm(dpr_output[:size_limit]):
        question, context = get_question_and_context(qac)
        answers = []
        for i in range(0, top_k, ctx_per_call):
            answer = get_gpt_answer(question, (context[i:i+ctx_per_call]), ctx_per_call, multi_answer, disinformation_aware, holdback, prompt_idx=0, engine=engine)
            answers.append(answer.replace('\n', ' ').strip())
        output.append(answers)
    return output

def read_dpr_output(data_list, top_k, multi_answer, disinformation_aware, extract_then_read_switch, size_limit, holdback, sample_question, prompt_idx,engine):
    output = []
    import json
    if not extract_then_read_switch:
        dpr_output = data_list
        if sample_question:
            dpr_output = dpr_output[-300:]
        from tqdm import tqdm
        limited_input = [i for i in dpr_output[:size_limit]]
        for qac in tqdm(limited_input):
            question, context = get_question_and_context(qac)
            answer = get_gpt_answer(question, context, top_k, multi_answer, disinformation_aware, holdback, prompt_idx, engine=engine)
            output.append(answer.replace('\n', ' ').strip())
    else:
        output = extract_then_read(data_list, top_k, multi_answer, disinformation_aware, size_limit,engine)
    return output