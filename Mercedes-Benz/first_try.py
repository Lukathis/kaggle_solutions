import pandas as pd
import numpy as np

path = '/Users/Chi/PycharmProjects/kaggle_solutions/Mercedes-Benz/data/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
print(train.head())
print(test.head())

print(train.info())
print(test.info())

