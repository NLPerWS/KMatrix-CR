import argparse
import json
from tqdm import tqdm

from kmatrix_cr.toolkit.Misinfo_QA.gpt_mrc import gpt_vote,read_dpr_output,multi_reader_vote

def voted(indiv_answers_list, questions, naive,engine):
    # this function receives a list of lists of answers produced by ChatGPT doing ODQA
    # and returns a list of answers with voting, conducted implicitly by GPT-3.5 if not naive
    voted_answers = []
    
    for i in range(len(indiv_answers_list)):
        if not naive:
            # produce majority-voted answers through GPT-3.5
            response = gpt_vote(indiv_answers_list[i], questions[i],engine)
            voted_answer = response
        else:
            # produce majority-voted answers through naive voting
            voted_answer = max((indiv_answers_list[i]), key=lambda x: (indiv_answers_list.count(x), -indiv_answers_list.index(x)))
        voted_answers.append(voted_answer)
    return voted_answers

def read_llm(args, prompt_idx=0):
    
    if not args.vote:
        output = read_dpr_output(args.data_list, int(args.top_k), args.multi_answer, args.disinfo, args.extract_and_read, args.size_limit, args.holdback, args.sample, prompt_idx,args.llm_model)
    else:
        output = multi_reader_vote(data_list=args.data_list, top_k=int(args.top_k), multi_answer=args.multi_answer, disinformation_aware=args.disinfo, size_limit=args.size_limit, holdback=args.holdback, sample_question=args.sample,engine=args.llm_model)
        with open('before-vote.txt', 'w') as f:
            for i in output:
                f.write(str(i) + '\n')
        questions = [data['question'] for data in args.data_list]
        output = voted(output, questions, args.naive_vote,args.llm_model)
    # print("----------------------output-----------------------------\n",output)
    return output


def execute(args):
    if len(args.data_list) == 0:
        return args.data_list
    
    res_list = read_llm(args)
    for data,res in zip(args.data_list,res_list):
        data['gen_answer'] = res

    return args.data_list
    

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, help='number of \
        passages to retrieve/number of retrieved passages to read', default=100)
    parser.add_argument('--multi_answer', action='store_true', help='whether \
                        to reveal to readers that the answer to produce list of answers')
    parser.add_argument('--disinfo', action='store_true', help='whether \
                        to reveal to readers that the passages contain disinformation')
    parser.add_argument('--extract_and_read', action='store_true', help='whether \
                        to extract passages from the retrieved passages and read them')
    parser.add_argument('--size_limit', type=int, help='size limit of the \
                        input questions to the reader')
    parser.add_argument('--vote', action='store_true', help='whether to use \
                        voting to aggregate the answers')
    parser.add_argument('--holdback', action='store_true', help='whether to \
                        give instructions to holdback answering in the prompts')
    parser.add_argument('--sample', action='store_true', help='whether to \
                        sample the questions to do QA, instead of reading all passages. Here \
                        we use the last 300 questions as the sample.')
    parser.add_argument('--naive_vote', action='store_true', help='whether to  \
                        use naive voting to aggregate the answers')
    
    args = parser.parse_args()
    args.data_list = []
    args.llm_model = None

    res_list = execute(args)




