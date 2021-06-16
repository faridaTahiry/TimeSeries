import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense,Dropout, Flatten,LSTM,RepeatVector,TimeDistributed,Conv1D,MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from livelossplot.keras import PlotLossesCallback

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

print("done importing")

# dataPath = "/Users/ftahiry/desktop/TimeSeries/data"
#
# for item in dataPath:
#     print("item")

df = pd.read_csv("merged_data65.csv")
print(df.head())

fig, ax = plt.subplots()
df.plot(legend=False, ax=ax)
plt.show()

training_mean = df.mean()
training_std = df.std()
df_training_value = (df - training_mean) / training_std
print("Number of training samples:", len(df_training_value))

# Let's make it function for further usage
def parse_and_standardize(df: pd.DataFrame, scaler: StandardScaler = None):
    df['Force Sensor 3'] = pd.to_datetime(df['Force Sensor 3'])
    print("woohoo")
    df['Audio'] = df['Audio']
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(df['Audio'].values.reshape(-1, 1))
    df['Audio'] = scaler.transform(df['Audio'].values.reshape(-1, 1))
    print(df["Audio"])
    return scaler

data_scaler = parse_and_standardize(df)
