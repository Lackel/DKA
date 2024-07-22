import pandas as pd
from ast import literal_eval
from source import evaluation
import numpy as np

val_annotations_df = pd.read_csv('annotations/ok_vqa/val_annots_fixed.csv')
val_annotations_df.answers = val_annotations_df.answers.apply(literal_eval)
results_df = pd.read_csv('results/ok_vqa_val_with_mcan_llama2.csv')
results_df = pd.merge(val_annotations_df, results_df, on = 'question_id')
results_df['acc'] = results_df.apply(lambda row: evaluation.okvqa_ems(row['llama_answer'], row['answers'], row['question_id']),axis = 1)
print("VQA acc: ", np.round(results_df.acc.mean(),3))
