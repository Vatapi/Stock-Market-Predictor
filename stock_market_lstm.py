import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import LSTM

dataset = pd.read_csv("EOD-MSFT.csv")
dataset['Date'] = pd.to_datetime(dataset.Date , format='%Y-%m-%d')
dataset.index = dataset['Date']
dataset = dataset.reindex(index = dataset.index[::-1])
df = dataset['Close']
