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


DEFAULT_MODEL_PATH="./model"
DEFAULT_OUTPUT_DIR="./output"
DEFAULT_SEQUENCE_LENGTH=512


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


# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def fine_tune(params):
    input_file = params.input
    model_name_or_path = params.model
    output_dir = params.output
    paired = params.paired
    seq_length = params.seq_length

    # Read data
    #data = pd.read_csv("data/tokenized_train.csv",sep='\t')
    data = pd.read_csv(input_file,sep='\t')

    # Define pretrained tokenizer and model
    #model_name = "bert-large-cased"
    model_name = model_name_or_path
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    # ----- 1. Preprocess data -----#
    # Preprocess data
    if (paired):
        X1 = list(data["text1"])
        X2 = list(data["text2"])
        assert(len(X1) == len(X2))
        X = []
        for i in range(len(X1)):
            X.append(X1[i] + '\t' + X2[i])
    else:
        X = list(data["text"])
    y = list(data["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01)
    if (paired):
        X1 = []
        X2  = []
        for i in range(len(X_train)):
            arr = X_train[i].split('\t')
            assert(len(arr) == 2)
            X1.append(arr[0])
            X2.append(arr[1])
    #pdb.set_trace()
        X_train_tokenized = tokenizer(text=X1, text_pair = X2, padding=True, truncation=True, max_length=seq_length)
    else:
        X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=seq_length)

    if (paired):
        X1 = []
        X2 = []
        for i in range(len(X_val)):
            arr = X_val[i].split('\t')
            assert(len(arr) == 2)
            X1.append(arr[0])
            X2.append(arr[1])
        X_val_tokenized = tokenizer(text = X1, text_pair = X2, padding=True, truncation=True, max_length=seq_length)
    else:
        X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=seq_length)

    # Create torch dataset

    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    # ----- 2. Fine-tune pretrained model -----#

    # Define Trainer
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        warmup_steps=500,                
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        #learning_rate = 1e-5,
        num_train_epochs=5,
        #weight_decay=0.01,
        seed=0,
        load_best_model_at_end=True,
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        metric_for_best_model="accuracy"
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train pre-trained model
    trainer.train()

    trainer.save_model(output_dir)

    print("Model saved. Training complete")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine tune model ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-input', action="store", dest="input",required=True, help='Input train data file')
    parser.add_argument('-output', action="store", dest="output",default=DEFAULT_OUTPUT_DIR, help='Output directory where model is saved')
    parser.add_argument('-seq_length', action="store", dest="seq_length",type=int,default=DEFAULT_SEQUENCE_LENGTH, help='Default max sequence length of input')
    parser.add_argument('-paired', dest="paired", action='store_true',help='Input is expected to be **pairs** of sentences')
    parser.add_argument('-no-paired', dest="paired", action='store_false',help='Input is expected to be **single** sentence - not pairs of sentences')
    parser.set_defaults(paired=False)

    results = parser.parse_args()
    try:
        torch.cuda.empty_cache()
        fine_tune(results)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
