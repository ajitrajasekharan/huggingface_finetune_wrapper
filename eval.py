import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import pdb
import argparse
import traceback
import sys
DEFAULT_SEQUENCE_LENGTH=512

DEFAULT_MODEL_PATH="./model"
DEFAULT_OUTPUT_FILE="./fine_tuned_results.txt"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])




# ----- 3. Predict -----#

def test_fine_tune(params):
    input_file = params.input
    model_path = params.model
    output_file = params.output
    paired = params.paired
    seq_length = params.seq_length

    # Load trained model
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.eval()

    # Load test data
    test_data = pd.read_csv(input_file,sep='\t')
    if (paired):
        X1 = list(test_data["text1"])
        X2 = list(test_data["text2"])
        X_test_tokenized = tokenizer(text = X1,text_pair = X2, padding=True, truncation=True, max_length=seq_length)
    else:
        X = list(test_data["text"])
        X_test_tokenized = tokenizer(X, padding=True, truncation=True, max_length=seq_length)
    y = list(test_data["label"])

    # Create torch dataset
    test_dataset = Dataset(X_test_tokenized)

    # Define test trainer
    test_trainer = Trainer(model)

    # Make prediction
    raw_pred, _, _ = test_trainer.predict(test_dataset)

    pdb.set_trace()
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)
    confusion_matrix = {}
    confusion_matrix[0] = {0:0,1:0}
    confusion_matrix[1] = {0:0,1:0}
    assert(len(y) == len(y_pred))
    fp = open("failed.txt","w")
    for i in range(len(y)):
        confusion_matrix[y[i]][y_pred[i]] += 1
        if (y[i] != y_pred[i]):
            fp.write(str(i+2) + '] ' + str(X[i])+ '\t' + str(y[i]) + '\n')
            #fp.write(str(X[i])+ '\t' + str(y[i]) + '\n')
    fp.close()

    print(confusion_matrix)
    f1_dict = {}
    f1_dict["precision"] = {}
    f1_dict["recall"] = {}
    f1_dict["precision"]["no_relation"] = round(float(confusion_matrix[0][0])/(float(confusion_matrix[0][0]) + float(confusion_matrix[1][0])),2)
    f1_dict["precision"]["target"] = round(float(confusion_matrix[1][1])/(float(confusion_matrix[1][1]) + float(confusion_matrix[0][1])),2)
    f1_dict["recall"]["no_relation"] = round(float(confusion_matrix[0][0])/(float(confusion_matrix[0][0]) + float(confusion_matrix[0][1])),2)
    f1_dict["recall"]["target"] = round(float(confusion_matrix[1][1])/(float(confusion_matrix[1][0]) + float(confusion_matrix[1][1])),2)
    print(f1_dict)
    final_f1_score = {}
    final_f1_score["no_relation"] = round((2*f1_dict["precision"]["no_relation"]*f1_dict["recall"]["no_relation"])/(f1_dict["recall"]["no_relation"] + f1_dict["precision"]["no_relation"]),2)
    final_f1_score["target"] = round((2*f1_dict["precision"]["target"]*f1_dict["recall"]["target"])/(f1_dict["recall"]["target"] + f1_dict["precision"]["target"]),2)
    print(final_f1_score)
    with open(output_file,"w") as fp:
        fp.write(str(confusion_matrix) + '\n')
        fp.write(str(f1_dict) + '\n')
        fp.write(str(final_f1_score) + '\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Fine tuned model ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-input', action="store", dest="input",required=True, help='Input test data file')
    parser.add_argument('-output', action="store", dest="output",default=DEFAULT_OUTPUT_FILE, help='Output results')
    parser.add_argument('-seq_length', action="store", dest="seq_length",type=int,default=DEFAULT_SEQUENCE_LENGTH, help='Default max sequence length of input')
    parser.add_argument('-paired', dest="paired", action='store_true',help='Input is expected to be **pairs** of sentences')
    parser.add_argument('-no-paired', dest="paired", action='store_false',help='Input is expected to be **single** sentence - not pairs of sentences')
    parser.set_defaults(paired=False)

    results = parser.parse_args()
    try:
        test_fine_tune(results)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
