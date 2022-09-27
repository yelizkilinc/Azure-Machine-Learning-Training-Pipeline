from azureml.core import Run

import pandas as pd 
import numpy as np 
import argparse

def prepare_data(df):
    #any preparation code
    return df

parser=argparse.ArgumentParser()
parser.add_argument('--output_path', dest='output_path',required=True)
args=parser.parse_args()

train_ds=Run.get_context().input_datasets['AutoMLE2EPipeline_Classification_train']
df=train_ds.to_pandas_dataframe()

df=prepare_data(df)
df.to_csv(os.path.join(args.output_path,"prepped_data_classification.csv"))

print(f"Wrote prepped data to {args.output_path}/prepped_data.csv")