import torch
from transformers import *
import pdb
import operator
from collections import OrderedDict
import sys
import traceback
import argparse
import string



DEFAULT_MODEL_PATH="./model"
TERM1 = "Drug"
TERM2 = "Target"
PAD = 0


def init_tokenizer(model_path,to_lower):
    tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=to_lower)
    return tokenizer


def map_label(field):
    return "1" if field == "Target" else "0"

def find_all_pos(sent,term,pos_arr):
    index = 0
    while (True):
        pos = sent[index:].find(term)
        if (pos == -1):
            break
        pos += index
        pos_arr.append(pos)
        pos_arr.append(pos + len(term))
        index = pos + len(term)
   
        

def truncated_sentence(term1,term2,sent):
    pos_arr = []
    find_all_pos(sent,term1,pos_arr) 
    find_all_pos(sent,term2,pos_arr) 
    min_pos = min(pos_arr)
    max_pos = max(pos_arr)
    return sent[min_pos:max_pos + PAD]



def tokenize_input(params):
    input_file = params.input
    output_file = params.output
    model_path = params.model
    #tokenizer = init_tokenizer(model_path,False)
    out_fp = open(output_file,"w")
    
    #out_fp.write('text1\ttext2\tlabel\n')
    out_fp.write('text\tlabel\n')
    with open(input_file) as fp:
        for line in fp:
            line = line.rstrip('\n')
            fields_arr = line.split('\t')
            if (len(fields_arr) == 6):
                #print(fields_arr[0],fields_arr[1],fields_arr[3],fields_arr[5])
                #text = '[CLS] ' + fields_arr[0] + ' [SEP] '  +  fields_arr[1] + ';' + fields_arr[3] + ' [SEP]' 
                #tokenized_text = tokenizer.tokenize(text)
                #assert(len(tokenized_text) < 512)
                #print(' '.join(tokenized_text) + '|' + map_label(fields_arr[5]))
                #out_fp.write(' '.join(tokenized_text) + '|' + map_label(fields_arr[5]) + '\n')
                #text = fields_arr[0] + '\t' + fields_arr[1] + ';' +  fields_arr[3] + '\t' + map_label(fields_arr[5])
                text = fields_arr[0].replace(fields_arr[1],TERM1).replace(fields_arr[3],TERM2)
                #print(text)
                #text = truncated_sentence(TERM1,TERM2,text) +  '\t' + map_label(fields_arr[5])
                text = text + '\t' + map_label(fields_arr[5])
                #tokenized_text = tokenizer.tokenize(text)
                #assert(len(tokenized_text) < 128)
                #if (len(tokenized_text) >= 128):
                #    print(tokenized_text)
                print(text)
                out_fp.write(text + '\n')
            
    out_fp.close()
                
              
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare training data by tokenizing the input ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-input', action="store", dest="input",required=True, help='Input file')
    parser.add_argument('-output', action="store", dest="output",required=True, help='Output file')

    results = parser.parse_args()
    try:
        tokenize_input(results)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
