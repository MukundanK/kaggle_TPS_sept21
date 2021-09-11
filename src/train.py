import os
import config
import joblib
import numpy as np
import pandas as pd
import model
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
from sklearn.metrics import accuracy_score

def run(df, fold, target_col, model_name, file):
    
    print ('starting Fold:', fold)

    # train data where kfold is not equal to fold
    df_train = df[df.kfold != fold].reset_index(drop = True)

    # validation data where kfold is equal to fold
    df_valid = df[df.kfold == fold].reset_index(drop = True)

    # split train into feature and target
    X_train = df_train.drop(target_col, axis =1).values
    y_train = df_train[target_col].values

    # split validation data into feature and target
    X_valid = df_valid.drop(target_col, axis=1).values
    y_valid = df_valid[target_col].values
    
    print ('created train and validation data')
    print ('impute missing values')

    # pipeline
    pipe = Pipeline(steps=[('impute', SimpleImputer(strategy='mean')), ('scaler',MinMaxScaler()), ('clf', model.model_dict[model_name])])

    # fit model on training data
    pipe.fit(X_train, y_train)

    print ('predicting on validation data')

    # predict on validation data
    y_pred = pipe.predict(X_valid)

    print ('calculating score')

    #  calculate accuracy score and write to file
    accuracy = accuracy_score(y_valid, y_pred)
    
    file.write(f'Fold: {fold} ,')
    file.write(f'Accuracy: {accuracy}\n')

    # save model
    print ('saving model')
    joblib.dump(pipe['clf'], os.path.join(config.MODEL_OUTPUT, f'{model_name}_{fold}.bin'))

    print (f'Fold: {fold} completed')
    return None


if __name__ == '__main__':
    
    # obtain model and filename from user
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--filename', type=str)

    args = parser.parse_args()

    # read train data with kfolds
    train_kfold_path = os.path.join(config.PROCESSED_DATA,'train_kfold.csv')
    train_kfold = pd.read_csv(train_kfold_path)

    # drop id column
    train_kfold.drop(columns=['id'], inplace=True)

    target_col = config.TARGET
    model_name = args.model

    # open a file to write scores
    file = open(os.path.join(config.OUTPUT, args.filename), 'w')

    # number of kfolds in train data
    num_folds = train_kfold['kfold'].nunique()

    for fold in np.arange(0,num_folds, 1):
        run(train_kfold, fold, target_col, model_name, file) 

    file.close()









