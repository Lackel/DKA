import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import pandas as pd
from ast import literal_eval

import json
import random

from source import evaluation
from source.ok_vqa_in_context_learning import val_in_context_learning_ok_vqa_with_mcan
from source.a_ok_vqa_in_context_learning import val_in_context_learning_a_ok_vqa_with_mcan
from source.a_ok_vqa_in_context_learning import test_in_context_learning_a_ok_vqa_with_mcan

from config import get_config
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoProcessor, BlipForImageTextRetrieval
from transformers import set_seed


#get confing variables 
cnf = get_config(sys.argv)

#set up device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#unfolding common params
dataset_to_use = cnf.dataset
train_images_dir = cnf.train_images_dir
val_images_dir = cnf.val_images_dir
test_images_dir = cnf.test_images_dir

n_shots = cnf.n_shots
k_ensemble = cnf.k_ensemble
no_of_captions = cnf.no_of_captions
path_to_save_preds = cnf.path_to_save_preds 
decomposed_caption_path = cnf.decomposed_caption_path
decomposed_knowledge_path = cnf.decomposed_knowledge_path
#load Llama model

llama_model = LlamaForCausalLM.from_pretrained(cnf.llama_path, use_safetensors=False)
llama_tokenizer = LlamaTokenizer.from_pretrained(cnf.llama_path, use_safetensors=False)
llama_model = llama_model.to(device, dtype=torch.float16)

#load the blip model 
blip_model = BlipForImageTextRetrieval.from_pretrained(cnf.blip_path)
blip_processor = AutoProcessor.from_pretrained(cnf.blip_path)
blip_model = blip_model.to(device)

#load annotations
train_annotations_df = pd.read_csv(cnf.train_annotations_path)
if cnf.evaluation_set == "val":
    val_annotations_df = pd.read_csv(cnf.val_annotations_path)
else:
    test_annotations_df = pd.read_csv(cnf.test_annotations_path)


#load the mcan context examples 
with open(cnf.mcan_examples_path, "rb") as input:
    examples = json.load(input)
mcan_examples_df = pd.DataFrame({'question_id' : examples.keys(), 'similar_examples' : examples.values()})


#load captions 
train_captions = pd.read_csv(cnf.train_captions_path)
train_captions.captions = train_captions.captions.apply(literal_eval)

if cnf.evaluation_set == 'val':
    val_captions = pd.read_csv(cnf.val_captions_path)
    val_captions.captions = val_captions.captions.apply(literal_eval)
else:
    test_captions = pd.read_csv(cnf.test_captions_path)
    test_captions.captions = test_captions.captions.apply(literal_eval)


if __name__ == "__main__":
    if dataset_to_use == "ok_vqa":
        #apply literal eval to the answers
        train_annotations_df.answers = train_annotations_df.answers.apply(literal_eval)
        val_annotations_df.answers = val_annotations_df.answers.apply(literal_eval)

        
        mcan_examples_df['question_id'] = mcan_examples_df['question_id'].astype('int')
        results_df = val_in_context_learning_ok_vqa_with_mcan(llama_model=llama_model, 
                                                              llama_tokenizer=llama_tokenizer,
                                                              blip_model=blip_model,
                                                              blip_processor=blip_processor,
                                                              train_annotations_df=train_annotations_df,
                                                              val_annotations_df=val_annotations_df,
                                                              train_captions=train_captions, 
                                                              val_captions=val_captions,
                                                              context_examples_df=mcan_examples_df, 
                                                              train_images_dir=train_images_dir, 
                                                              val_images_dir=val_images_dir,
                                                              decomposed_caption_path = decomposed_caption_path,
                                                              decomposed_knowledge_path = decomposed_caption_path,
                                                              n_shots=n_shots, 
                                                              k_ensemble=k_ensemble,
                                                              MAX_CAPTION_LEN=30,
                                                              NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                              path_to_save_preds=path_to_save_preds,
                                                              device=device)
                                            
    
    elif cnf.dataset == "a_ok_vqa":
        #apply literal eval to the answers
        train_annotations_df.direct_answers = train_annotations_df.direct_answers.apply(literal_eval)

        if cnf.evaluation_set == "val":
            #apply literal eval to the answers
            val_annotations_df.direct_answers = val_annotations_df.direct_answers.apply(literal_eval)
        
            results_df = val_in_context_learning_a_ok_vqa_with_mcan(llama_model=llama_model,
                                                                    llama_tokenizer=llama_tokenizer,
                                                                    blip_model=blip_model,
                                                                    blip_processor=blip_processor,
                                                                    train_annotations_df=train_annotations_df,
                                                                    val_annotations_df=val_annotations_df, 
                                                                    train_captions=train_captions, 
                                                                    val_captions=val_captions,
                                                                    context_examples_df=mcan_examples_df, 
                                                                    train_images_dir=train_images_dir, 
                                                                    val_images_dir=val_images_dir,
                                                                    decomposed_caption_path = decomposed_caption_path,
                                                                    decomposed_knowledge_path = decomposed_caption_path,
                                                                    n_shots=n_shots, 
                                                                    k_ensemble=k_ensemble, 
                                                                    MAX_CAPTION_LEN=30, 
                                                                    NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                                    path_to_save_preds=path_to_save_preds,
                                                                    device=device)
            
        elif cnf.evaluation_set == "test":
            results_df = test_in_context_learning_a_ok_vqa_with_mcan(llama_model=llama_model,
                                                                    llama_tokenizer=llama_tokenizer,
                                                                    blip_model=blip_model,
                                                                    blip_processor=blip_processor,
                                                                    train_annotations_df=train_annotations_df,
                                                                    test_annotations_df=test_annotations_df, 
                                                                    train_captions=train_captions, 
                                                                    test_captions=test_captions,
                                                                    context_examples_df=mcan_examples_df, 
                                                                    train_images_dir=train_images_dir, 
                                                                    test_images_dir=test_images_dir,
                                                                    decomposed_caption_path = decomposed_caption_path,
                                                                    decomposed_knowledge_path = decomposed_caption_path,
                                                                    n_shots=n_shots, 
                                                                    k_ensemble=k_ensemble, 
                                                                    MAX_CAPTION_LEN=30, 
                                                                    NO_OF_CAPTIONS_AS_CONTEXT=no_of_captions,
                                                                    path_to_save_preds=path_to_save_preds,
                                                                    device=device)
    
    
    #evaluate the predictions (only for val sets)
    if cnf.evaluation_set == "val": 
        results_df = pd.merge(val_annotations_df, results_df, on = 'question_id')
        if cnf.dataset == "ok_vqa":
            results_df['acc'] = results_df.apply(lambda row: evaluation.okvqa_ems(row['llama_answer'], row['answers'], row['question_id']),axis = 1)
        else:
            results_df['acc'] = results_df.apply(lambda row: evaluation.okvqa_ems(row['llama_answer'], row['direct_answers'], row['question_id']), axis = 1)
        print("\n========")
        print("VQA acc: ", np.round(results_df.acc.mean(),3))
        print("==========")
    


    





                   


  