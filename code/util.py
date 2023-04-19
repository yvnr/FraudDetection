import random

import pandas as pd
from collections import defaultdict
import os


def drop_unwanted_columns(df):
    df = df.drop('Time', axis=1)
    return df


def create_the_train_and_test_sets(df, k=5, class_column="Class"):
    folds = defaultdict(lambda: pd.DataFrame())
    instances = defaultdict(lambda: pd.DataFrame())
    columns = list(df.columns.values)
    classes = set(df[class_column])
    counts = defaultdict(lambda: 0)
    for class_instance in classes:
        instances[class_instance] = df[df[class_column] == class_instance]
        counts[class_instance] = int(round(len(instances[class_instance]) / k, 0))
    for i in range(k - 1):
        t_df = pd.DataFrame(columns=columns)
        for class_instance in classes:
            t_df1 = instances[class_instance].sample(n=counts[class_instance], random_state=0)
            instances[class_instance] = instances[class_instance].drop(t_df1.index)
            t_df = pd.concat([t_df, t_df1], ignore_index=True)
        folds[i] = t_df
    t_df = pd.DataFrame(columns=columns)
    for instance in instances:
        t_df1 = instances[instance]
        instances[instance] = instances[instance].drop(t_df1.index)
        t_df = pd.concat([t_df, t_df1], ignore_index=True)
    folds[k - 1] = t_df

    for test_index in range(k):
        test_df = pd.DataFrame()
        train_df = pd.DataFrame()
        for stratified_fold in folds:
            if stratified_fold == test_index:
                test_df = pd.concat([test_df, folds[stratified_fold]])
            else:
                train_df = pd.concat([train_df, folds[stratified_fold]])

        train_df.reset_index(inplace=True, drop=True)
        test_df.reset_index(inplace=True, drop=True)

        current_directory = os.getcwd()
        relative_path = '../data/fold_{}'.format(test_index)
        target_directory = os.path.join(current_directory, relative_path)
        os.mkdir(target_directory)

        train_df.to_csv(target_directory + '/train.csv'.format(test_index), index=False)
        test_df.to_csv(target_directory + '/test.csv'.format(test_index), index=False)
