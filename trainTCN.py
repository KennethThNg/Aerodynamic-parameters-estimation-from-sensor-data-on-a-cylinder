## import libraries-------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from helpers import *
from nets import *
from pathlib import Path

# new column names
column_names = {'node_position [°]': 'node_position',
                'angle_attack [°]': 'angle_of_attack',
                'moving_drag_force_x [N]': 'moving_drag_force_x',
                'moving_drag_force_y [N]': 'moving_drag_force_y',
                'drag_force [N]': 'drag_force',
                'moving_lift_force_x [N]': 'moving_lift_force_x',
                'moving_lift_force_y [N]': 'moving_lift_force_y',
                'time [s]': 'time',
                'inlet_velocity [m/s]': 'inlet_velocity',
                'rotational_speed [rad/s]': 'rotational_speed',
                'coef_drag [-]': 'coef_drag',
                'coef_lift [-]': 'coef_lift',
                'drag_force [N]': 'drag_force',
                'lift_force [N]': 'lift_force',
                'acoustic-source-power-db': 'acoustic_power'}

## load training data -----------------------------------------------------------------
print('Fetching data...  0%', end='\r')
data_path = '../DATA-FOR-TRAINING/'

df_chunk = pd.read_csv(data_path + r'training_data_study1_32dp.csv', chunksize=1e6)
chunk_list = []  # append each chunk df here

# Each chunk is in df format
max_chunks = 1  # maximum number of chunks to import
for i, chunk in enumerate(df_chunk):
    # rename and drop columns
    chunk = chunk.drop(columns=['sensor2_pos [°]',
                                'sensor1_pos [°]',
                                'sensor0_pos [°]',
                                'x-velocity',
                                'y-velocity',
                                'x-coordinate',
                                'y-coordinate',
                                'velocity-angle',
                                'nodenumber']).rename(columns=column_names)

    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk)

    if not i < max_chunks:
        break;

    print(f'Fetching data... {int((i + 1) / max_chunks * 100)}%', end='\r')

data = pd.concat(chunk_list)
print('Fetching data... Done!')
## ------------------------------------------------------------------------------------
## generate time series of all experiments---------------------------------------------
sensors = ['pressure', 'velocity-magnitude', 'acoustic_power']
output_labels = ['drag_force', 'lift_force', 'angle_of_attack']
sensor_angles = [0, 45, 90]

nb_experiments = len(data['inlet_velocity'].unique())
nb_measurements = len(sensor_angles) * len(sensors)
exp_length = int(len(data['time'].unique()) / 6)

timeseries = np.zeros((nb_experiments, exp_length, len(sensor_angles) * 3))
labels = np.zeros((nb_experiments, exp_length, 3))

print('Generating timeseries from data...  0%', end='\r')

for i, vel in enumerate(data['inlet_velocity'].unique()):
    timeseries[i], labels[i] = getTimeSeries(data, sensor_angles=sensor_angles,
                                             sensors=sensors, output_labels=output_labels,
                                             inlet_velocity=vel, length=exp_length)
    print(f'Generating timeseries from data... {int((i + 1) / nb_experiments * 100)}%', end='\r')

print('Generating timeseries from data... Done!')

print(f'timeseries: {timeseries.shape}\n'
      f'labels: {labels.shape}')
## ------------------------------------------------------------------------------------
## generate train and test sets--------------------------------------------------------

print('Generating train and test set... ', end='\r')

train_input, train_target, test_input, test_target = train_test_set(timeseries, labels)

print('Generating train and test set... Done!')
print(f'train_input: {train_input.shape}\n'
      f'train_target: {train_target.shape}\n'
      f'test_input: {test_input.shape}\n'
      f'test_target: {test_target.shape}')
## ------------------------------------------------------------------------------------
## standardize data--------------------------------------------------------------------
print('Standardizing data... ', end='\r')

features_train, mean_train, std_train = standardize(train_input)
features_train[np.isnan(features_train)] = 0

features_test = fit_standardize(test_input, mean_train, std_train)
features_test[np.isnan(features_test)] = 0

print('Standardizing data... Done!')
## ------------------------------------------------------------------------------------
## creating tensors---------------------------------------------------------------------
print('Creating tensors... ', end='\r')
conv_size = 36  # size of temporal convolution

train_TCN_input = torch.zeros(features_train.shape[0], 1, features_train.shape[1], features_train.shape[2])
test_TCN_input = torch.zeros(features_test.shape[0], 1, features_test.shape[1], features_test.shape[2])
train_target_tensor = torch.zeros(train_target.shape[0], train_target.shape[1], train_target.shape[2])
test_target_tensor = torch.zeros(test_target.shape[0], test_target.shape[1], test_target.shape[2])

for i in range(features_train.shape[0]):
    train_TCN_input[i, 0] = torch.from_numpy(features_train[i]).float()
    train_target_tensor[i] = torch.from_numpy(train_target[i]).float()

for i in range(features_test.shape[0]):
    test_TCN_input[i, 0] = torch.from_numpy(features_test[i]).float()
    test_target_tensor[i] = torch.from_numpy(test_target[i]).float()

print('Creating tensors... Done!')
print(f'train_TCN_input: {train_TCN_input.shape}\n'
      f'test_TCN_input: {test_TCN_input.shape}\n'
      f'train_target_tensor: {train_target_tensor.shape}\n'
      f'test_target_tensor: {test_target_tensor.shape}')
## ------------------------------------------------------------------------------------
## train model-------------------------------------------------------------------------
hidden = 100
out = train_target_tensor.shape[2]
in_dim = train_TCN_input.shape[3]

model = TemporalConvNet(in_dim=in_dim, hid_dim=hidden, out_dim=out, time_len=exp_length)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.01)
n_epoch = 100
batch_size = 1

train_losses = []
test_losses = []

for epoch in range(n_epoch):
    train_loss = train(model, train_TCN_input, train_target_tensor, loss_fn, optimizer, batch_size)
    train_losses.append(train_loss)
    test_loss = test(model, test_TCN_input, test_target_tensor, loss_fn)
    test_losses.append(test_loss)
    print('epoch:', epoch + 1, 'train_loss:', ', ', float(train_loss), 'test_loss:', float(test_loss))

plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.show()
## ------------------------------------------------------------------------------------

