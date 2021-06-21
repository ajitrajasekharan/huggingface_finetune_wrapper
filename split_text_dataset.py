import pandas as pd
from sklearn.model_selection import train_test_split
import pdb
import argparse
import traceback
import sys



DEFAULT_RATIO=.11
DEFAULT_PREFIX=""
DEFAULT_SEPARATOR='\t'

def dump_file(file_name,field1,field2,x_label,y_label,separator):
    with open(file_name,"w") as fp:
        fp.write(x_label + separator + y_label + '\n')
        for i in range(len(field1)):
            fp.write(str(field1[i]) + str(separator) + str(field2[i]) + '\n')

def split_data(params):
    input_file = params.input
    output_prefix = params.output_prefix
    separator = params.separator
    ratio = params.ratio

    with open(input_file) as fp:
        arr = fp.readline().rstrip('\n').split(separator)
        assert(len(arr) == 2)
        x_label = arr[0]
        y_label = arr[1]
        

    # Read data
    data = pd.read_csv(input_file,sep=separator)

    X = list(data[x_label])
    y = list(data[y_label])
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=ratio)
    assert(len(X_train) == len(y_train))
    assert(len(X_dev) == len(y_dev))
    train_percent = round(float(len(X_train))/(len(X_train) + len(X_dev)),2)*100
    print("Train percent:",train_percent,"Test percent:",100 - train_percent,"Split ratio:",ratio)
    train_file_name = (output_prefix + "_train.csv") if (len(output_prefix) > 1) else "train.csv"
    dev_file_name = (output_prefix + "_dev.csv") if (len(output_prefix) > 1) else "dev.csv"
    dump_file(train_file_name,X_train,y_train,x_label,y_label,separator)
    dump_file(dev_file_name,X_dev,y_dev,x_label,y_label,separator)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split single line text data set with two columns into train and dev. Default split  train,dev,test (80,10,10) assumes test split has already been done. ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input",required=True, help='Input file')
    parser.add_argument('-output_prefix', action="store", dest="output_prefix",default=DEFAULT_PREFIX, help='Output file prefix p (optional). If specified, Output will be p_dev.txt and p_test.txt')
    parser.add_argument('-ratio', action="store", dest="ratio",type=float,default=DEFAULT_RATIO, help='Split ratio')
    parser.add_argument('-separator', action="store", dest="separator",default=DEFAULT_SEPARATOR, help='Field separator')

    results = parser.parse_args()
    try:
        split_data(results)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
