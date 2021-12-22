import argparse
import os
import sys 
import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import subprocess
import joblib

# Workaround for dependencies
def install_pip(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
# Workaround for dependencies
def install_conda(package):
    subprocess.check_call(["conda","config","--add","channels","conda-forge"])
    subprocess.check_call(["conda","install",package])

#install_pip('joblib')
#import joblib

if __name__ == "__main__":
    
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--scaler", type=str, default='robust')
    args, _ = parser.parse_known_args()
    print(f"Received arguments {args}")

    # read in data
    input_data_path = os.path.join("/opt/ml/processing/rawdata", "covtype.data")
    header = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
       'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
       'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
       'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
       'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
       'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
       'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
       'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
       'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
       'Soil_Type39', 'Soil_Type40', 'Cover_Type']
    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path,names=header)

    # move target column to the first column
    df = pd.concat([df['Cover_Type'], df.drop(['Cover_Type'], axis=1)], axis=1)
    
    # reduce the target column by one so it is in the range [0,num_classes)
    df['Cover_Type'] = df['Cover_Type'] - 1
    
    # split data using stratified folds
    train_df, temp_df = train_test_split(
        df, 
        test_size=(args.val_split+args.test_split), 
        random_state=4321, 
        shuffle=True, 
        stratify=df['Cover_Type']
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(args.test_split/(args.val_split+args.test_split)), 
        random_state=4321, 
        shuffle=True, 
        stratify=temp_df['Cover_Type']
    )
 
    # scale the data based on the training dataset
    scale_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points']
    
    if args.scaler.lower() == 'robust':
        scaler = RobustScaler()
    elif args.scaler.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif args.scaler.lower() == 'standard':
        scaler = StandardScaler()
    else:
        print("Unknown scaling passed as arguments, selecting standard scaling")
        scaler = StandardScaler()

    # fit scaler
    scaler.fit(train_df[scale_cols].to_numpy())
    
    # make copies of dataframes
    train_df_ = train_df.copy()
    val_df_ = val_df.copy()
    test_df_ = test_df.copy()
        
    # apply scaler
    train_df_.loc[:,scale_cols] = scaler.transform(train_df[scale_cols].values)
    val_df_.loc[:,scale_cols] = scaler.transform(val_df[scale_cols].values)
    test_df_.loc[:,scale_cols] = scaler.transform(test_df[scale_cols].values)
    
    # save scaler
    scaler_filename = "/opt/ml/processing/scaler/scaler.save"
    joblib.dump(scaler, scaler_filename) 
        
    print(f"Train data shape after preprocessing: {train_df.shape}")
    print(f"Validation data shape after preprocessing: {val_df.shape}")
    print(f"Test data shape after preprocessing: {test_df.shape}")

    train_output_path = os.path.join("/opt/ml/processing/train", "train.csv")
    test_output_path = os.path.join("/opt/ml/processing/test", "test.csv")
    val_output_path = os.path.join("/opt/ml/processing/val", "val.csv")

    print("Saving training data to {}".format(train_output_path))
    train_df_.to_csv(train_output_path, header=False, index=False)
    
    print("Saving val data to {}".format(val_output_path))
    val_df_.to_csv(val_output_path, header=False, index=False)

    print("Saving test data to {}".format(test_output_path))
    test_df_.to_csv(test_output_path, header=False, index=False)