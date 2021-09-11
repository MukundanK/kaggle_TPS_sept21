import os
import config
import joblib
import numpy as np
import pandas as pd
import model
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def run(df_train, df_test, target_col, model_name):

    print ('create X_train, y_train')

    # split train into feature and target
    X_train = df_train.drop(target_col, axis =1).values
    y_train = df_train[target_col].values
    
    # initiate model
    # pipeline
    pipe = Pipeline(steps=[('impute', SimpleImputer(strategy='mean')), ('scaler',StandardScaler()), ('clf', model.model_dict[model_name])])

    print ('fitting classifier')

    # fit model on training data
    pipe.fit(X_train, y_train.ravel())

    print ('predicting on test data')

    # predict on test data
    y_pred = pipe.predict_proba(df_test)

    # save model
    print ('saving model')
    joblib.dump(pipe['clf'], os.path.join(config.MODEL_OUTPUT, f'{model_name}.bin'))

    print (f'completed')
    return y_pred

if __name__ == '__main__':

     # obtain model and filename from user
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--filename', type=str)

    args = parser.parse_args()

    # read data
    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)

    # drop id
    test_id = test['id'].values
    test.drop('id', axis=1, inplace=True)
    train.drop(['id'], axis=1, inplace=True)

    model_name = args.model
    filename = args.filename
    target_col = config.TARGET

    y_pred = run(train, test, target_col, model_name) 

    # select probability to be in class 1
    data = {'id':test_id, 'claim':y_pred[:,1]}
    test_pred = pd.DataFrame(data, columns = ['id','claim'])

    output = os.path.join(config.OUTPUT, filename)
    test_pred.to_csv(output, index=None)









