
# import
import pandas as pd
import config
import os
from sklearn import model_selection

def create_folds(data, target_col, output_dir):

    # create kfold column and fill with 0
    data['kfold'] = 0

    # shuffle data and reset index
    data = data.sample(frac = 1).reset_index(drop = True)

    # kfold
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X = data, y = data[target_col].values)):
        data.loc[v_, 'kfold'] = f

    output_file = os.path.join(output_dir, 'train_kfold.csv')
    data.to_csv(output_file, index = None)

    return None

if __name__ == '__main__':

    data = config.TRAIN_CSV
    target_col = config.TARGET
    output_dir = config.PROCESSED_DATA

    train = pd.read_csv(data)
    create_folds(train, target_col, output_dir)
