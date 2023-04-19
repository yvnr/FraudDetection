import pandas as pd
from util import *

if __name__ == '__main__':
    df = pd.read_csv('./../data/creditcard.csv')
    df = drop_unwanted_columns(df)
    create_the_train_and_test_sets(df)
