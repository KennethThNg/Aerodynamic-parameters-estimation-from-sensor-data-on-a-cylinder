## Script usage:
# as argument, give the data path followed by the number of chunks to load or
# '-s' instead of the data path, to load the time series directly from the save file
# if no argument is given the entire four data sets are loaded and converted to time series to train the model
# ------------------------------------------------------------------------------------

# Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from helpers import *
from nets import *
import logging

fh = logging.FileHandler('nn_training.log')
fh.setLevel(logging.DEBUG)

fh.debug("Starded python code logging")

column_names = {'node_position [°]': 'node_position',
                'angle_attack [°]': 'angle_of_attack',
                'moving_drag_force_x [N]': 'moving_drag_force_x',
                'moving_drag_force_y [N]': 'moving_drag_force_y',
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

## Handle command line arguments-------------------------------------------------------
NB_DATA_FILES = 4
LOAD_FROM_FILE = False
LOAD_ALL_CHUNKS = True
NO_ARGS_GIVEN = True
BASE_PATH = '../DATA-FOR-TRAINING/'  # data folder path
DATA_PATH = BASE_PATH + 'training_data1.csv'  # default data path
max_chunks = 2  # maximum number of chunks to import
nb_of_input_args = len(sys.argv)
if nb_of_input_args > 1:
    NO_ARGS_GIVEN = False
    if str(sys.argv[1]) == "-s":
        LOAD_FROM_FILE = True
    else:
        DATA_PATH = str(sys.argv[1])
        if nb_of_input_args > 2:
            max_chunks = int(sys.argv[2])
            LOAD_ALL_CHUNKS = False

## ------------------------------------------------------------------------------------
## load training data -----------------------------------------------------------------
if LOAD_FROM_FILE is not True:
    fh.debug('Fetching data...  0%', end='\r')
    if NO_ARGS_GIVEN is False:
        df_chunk = pd.read_csv(DATA_PATH, chunksize=1e6)
        chunk_list = []  # append each chunk df here

        # Each chunk is in df format
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

            if not i < max_chunks and LOAD_ALL_CHUNKS is False:
                break
            if LOAD_ALL_CHUNKS is False:
                fh.debug(f'Fetching data... {int((i + 1) / max_chunks * 100)}%', end='\r')
        data = pd.concat(chunk_list)
        grouped_data = data.groupby('time', as_index=False)
    else:
        DATA_PATH1 = BASE_PATH + 'training_data1.csv'
        DATA_PATH2 = BASE_PATH + 'training_data2.csv'
        DATA_PATH3 = BASE_PATH + 'training_data3.csv'
        DATA_PATH4 = BASE_PATH + 'training_data4.csv'
        df_chunk1 = pd.read_csv(DATA_PATH1, chunksize=1e6)
        df_chunk2 = pd.read_csv(DATA_PATH2, chunksize=1e6)
        df_chunk3 = pd.read_csv(DATA_PATH3, chunksize=1e6)
        df_chunk4 = pd.read_csv(DATA_PATH4, chunksize=1e6)
        chunk_list1 = []  # append each chunk df here
        chunk_list2 = []  # append each chunk df here
        chunk_list3 = []  # append each chunk df here
        chunk_list4 = []  # append each chunk df here

        # Each chunk is in df format
        for i, (chunk1, chunk2, chunk3, chunk4) in enumerate(zip(df_chunk1, df_chunk2, df_chunk3, df_chunk4)):
            # rename and drop columns
            chunk1 = chunk1.drop(columns=['sensor2_pos [°]',
                                          'sensor1_pos [°]',
                                          'sensor0_pos [°]',
                                          'x-velocity',
                                          'y-velocity',
                                          'x-coordinate',
                                          'y-coordinate',
                                          'velocity-angle',
                                          'nodenumber']).rename(columns=column_names)
            chunk2 = chunk2.drop(columns=['sensor2_pos [°]',
                                          'sensor1_pos [°]',
                                          'sensor0_pos [°]',
                                          'x-velocity',
                                          'y-velocity',
                                          'x-coordinate',
                                          'y-coordinate',
                                          'velocity-angle',
                                          'nodenumber']).rename(columns=column_names)
            chunk3 = chunk3.drop(columns=['sensor2_pos [°]',
                                          'sensor1_pos [°]',
                                          'sensor0_pos [°]',
                                          'x-velocity',
                                          'y-velocity',
                                          'x-coordinate',
                                          'y-coordinate',
                                          'velocity-angle',
                                          'nodenumber']).rename(columns=column_names)
            chunk4 = chunk4.drop(columns=['sensor2_pos [°]',
                                          'sensor1_pos [°]',
                                          'sensor0_pos [°]',
                                          'x-velocity',
                                          'y-velocity',
                                          'x-coordinate',
                                          'y-coordinate',
                                          'velocity-angle',
                                          'nodenumber']).rename(columns=column_names)

            # Once the data filtering is done, append the chunk to list
            chunk_list1.append(chunk1)
            chunk_list2.append(chunk2)
            chunk_list3.append(chunk3)
            chunk_list4.append(chunk4)

        data1 = pd.concat(chunk_list1)
        grouped_data1 = data1.groupby('time', as_index=False)
        data2 = pd.concat(chunk_list2)
        grouped_data2 = data2.groupby('time', as_index=False)
        data3 = pd.concat(chunk_list3)
        grouped_data3 = data3.groupby('time', as_index=False)
        data4 = pd.concat(chunk_list4)
        grouped_data4 = data4.groupby('time', as_index=False)

    fh.debug('Fetching data... Done!')
## ------------------------------------------------------------------------------------
## generate time series of all experiments---------------------------------------------
if LOAD_FROM_FILE is not True:
    sensor_angles = [45, 135, 225, 315]
    sensors = ['pressure', 'velocity-magnitude']
    output_labels = ['angle_of_attack', 'drag_force', 'lift_force', 'coef_drag', 'coef_lift', 'inlet_velocity']
    if NO_ARGS_GIVEN is False:
        nb_experiments = int(len(grouped_data.groups[data['time'].unique()[0]])/343) - 1

        fh.debug('Generating timeseries...  0%', end='\r')
        timeseries = np.zeros((nb_experiments, len(data['time'].unique()), len(sensors) * len(sensor_angles)))
        labels = np.zeros((nb_experiments, len(data['time'].unique()), len(output_labels)))
        for i in range(nb_experiments):
            if i == nb_experiments:
                break
            timeseries[i], labels[i] = getTimeSeries(data,
                                                     sensor_angles=sensor_angles,
                                                     sensors=sensors,
                                                     experiment_number=i,
                                                     output_labels=output_labels)
            fh.debug(f'generating timeseries... {int((i + 1) / nb_experiments * 100)}% ', end='\r')
        fh.debug('Generating timeseries... Done!')
    else:
        nb_experiments1 = len(grouped_data1.groups[data1['time'].unique()[0]])/343 - 1
        fh.debug('Generating timeseries for file 1...  0%', end='\r')
        timeseries1 = np.zeros((nb_experiments1, len(data1['time'].unique()), len(sensors) * len(sensor_angles)))
        labels1 = np.zeros((nb_experiments1, len(data1['time'].unique()), len(output_labels)))
        for i in range(nb_experiments1):
            if i == nb_experiments1:
                break
            timeseries1[i], labels1[i] = getTimeSeries(data1,
                                                       sensor_angles=sensor_angles,
                                                       sensors=sensors,
                                                       experiment_number=i,
                                                       output_labels=output_labels)
            fh.debug(f'generating timeseries for file 1... {int((i + 1) / nb_experiments1 * 100)}% ', end='\r')
        fh.debug('Generating timeseries... Done!')

        nb_experiments2 = len(grouped_data2.groups[data2['time'].unique()[0]])/343 - 1
        fh.debug('Generating timeseries for file 2...  0%', end='\r')
        timeseries2 = np.zeros((nb_experiments2, len(data2['time'].unique()), len(sensors) * len(sensor_angles)))
        labels2 = np.zeros((nb_experiments2, len(data2['time'].unique()), len(output_labels)))
        for i in range(nb_experiments2):
            if i == nb_experiments2:
                break
            timeseries2[i], labels2[i] = getTimeSeries(data2,
                                                       sensor_angles=sensor_angles,
                                                       sensors=sensors,
                                                       experiment_number=i,
                                                       output_labels=output_labels)
            fh.debug(f'generating timeseries for file 2... {int((i + 1) / nb_experiments2 * 100)}% ', end='\r')
        fh.debug('Generating timeseries... Done!')

        nb_experiments3 = len(grouped_data3.groups[data3['time'].unique()[0]])/343 - 1
        fh.debug('Generating timeseries for file 3...  0%', end='\r')
        timeseries3 = np.zeros((nb_experiments3, len(data3['time'].unique()), len(sensors) * len(sensor_angles)))
        labels3 = np.zeros((nb_experiments3, len(data3['time'].unique()), len(output_labels)))
        for i in range(nb_experiments3):
            if i == nb_experiments3:
                break
            timeseries3[i], labels3[i] = getTimeSeries(data3,
                                                       sensor_angles=sensor_angles,
                                                       sensors=sensors,
                                                       experiment_number=i,
                                                       output_labels=output_labels)
            fh.debug(f'generating timeseries for file 3... {int((i + 1) / nb_experiments3 * 100)}% ', end='\r')
        fh.debug('Generating timeseries... Done!')

        nb_experiments4 = len(grouped_data4.groups[data4['time'].unique()[0]])/343 - 1
        fh.debug('Generating timeseries for file 4...  0%', end='\r')
        timeseries4 = np.zeros((nb_experiments4, len(data4['time'].unique()), len(sensors) * len(sensor_angles)))
        labels4 = np.zeros((nb_experiments4, len(data4['time'].unique()), len(output_labels)))
        for i in range(nb_experiments4):
            if i == nb_experiments4:
                break
            timeseries4[i], labels4[i] = getTimeSeries(data4,
                                                       sensor_angles=sensor_angles,
                                                       sensors=sensors,
                                                       experiment_number=i,
                                                       output_labels=output_labels)
            fh.debug(f'generating timeseries for file 4... {int((i + 1) / nb_experiments4 * 100)}% ', end='\r')
        fh.debug('Generating timeseries... Done!')

        fh.debug('Concatenating timeseries...  0%', end='\r')
        tmpt = np.concatenate((timeseries1, timeseries2), axis=0)
        tmpl = np.concatenate((labels1, labels2), axis=0)
        fh.debug('Concatenating timeseries... 33%', end='\r')
        tmpt = np.concatenate((tmpt, timeseries3), axis=0)
        tmpl = np.concatenate((tmpl, labels3), axis=0)
        fh.debug('Concatenating timeseries... 66%', end='\r')
        timeseries = np.concatenate((tmpt, timeseries4), axis=0)
        labels = np.concatenate((tmpl, labels4), axis=0)
        fh.debug('Concatenating timeseries... Done!')

    fh.debug(f'timeseries: {timeseries.shape}\n'
          f'labels: {labels.shape}')
## ------------------------------------------------------------------------------------
## generate train and test sets--------------------------------------------------------
if LOAD_FROM_FILE is not True:
    fh.debug('Generating train and test set... ', end='\r')

    train_input_set, train_target_set, test_input_set, test_target_set = train_test_set(timeseries, labels)

    fh.debug('Generating train and test set... Done!')
    fh.debug(f'train_input: {train_input_set.shape}\n'
          f'train_target: {train_target_set.shape}\n'
          f'test_input: {test_input_set.shape}\n'
          f'test_target: {test_target_set.shape}')
## ------------------------------------------------------------------------------------
## standardize data--------------------------------------------------------------------
if LOAD_FROM_FILE is not True:
    fh.debug('Standardizing data... ', end='\r')

    features_train, mean_train, std_train = standardize(train_input_set)
    features_train[np.isnan(features_train)] = 0
    features_train = features_train.reshape(-1, features_train.shape[2])

    features_test = fit_standardize(test_input_set, mean_train, std_train)
    features_test[np.isnan(features_test)] = 0
    features_test = features_test.reshape(-1, features_test.shape[2])

    labels_train = train_target_set.reshape(-1, train_target_set.shape[2])
    labels_test = test_target_set.reshape(-1, test_target_set.shape[2])

    fh.debug('Standardizing data... Done!')
## ------------------------------------------------------------------------------------
## Save data to file-------------------------------------------------------------------
if LOAD_FROM_FILE is not True:
    fh.debug("Saving data to file... ", end='\r')
    np.savetxt('features_train.txt', features_train)
    np.savetxt('labels_train.txt', labels_train)
    np.savetxt('features_test.txt', features_test)
    np.savetxt('labels_test.txt', labels_test)
    fh.debug("Saving data to file... Done!")
## ------------------------------------------------------------------------------------
## Load data from file-----------------------------------------------------------------
if LOAD_FROM_FILE is True:
    fh.debug("Loading data from file... ", end='\r')
    features_train = np.loadtxt('features_train.txt')
    labels_train = np.loadtxt('labels_train.txt')
    features_test = np.loadtxt('features_test.txt')
    labels_test = np.loadtxt('labels_test.txt')
    fh.debug("Loading data from file... Done!")
    fh.debug(f'train_input: {features_train.shape}\n'
          f'train_target: {labels_train.shape}\n'
          f'test_input: {features_test.shape}\n'
          f'test_target: {labels_test.shape}')
## ------------------------------------------------------------------------------------
## train model-------------------------------------------------------------------------
conv_size = 64
batch_size = 100
n_epoch = 50
for conv_size in [8, 16, 32, 64, 128]:
    model = Wavenet(conv_size=conv_size, nb_of_measurements=features_train.shape[1], nb_of_outputs=labels_train.shape[1])
    model.reset_parameters()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.01)

    train_inputs = getWavenetInputsFromTimeseries(features_train, conv_size=conv_size)
    train_targets = torch.from_numpy(labels_train).float()
    test_inputs = getWavenetInputsFromTimeseries(features_test, conv_size=conv_size)
    test_targets = torch.from_numpy(labels_test).float()

    train_losses = []
    test_losses = []
    fh.debug('Training Start')
    for epoch in range(n_epoch):
        running_loss = 0
        for batch_labels, batch_inputs in batch_iter(train_targets, train_inputs, batch_size=batch_size,
                                                     num_batches=int(train_inputs.shape[0] / batch_size)):
            # zero the parameter gradients
            optimizer.zero_grad()
            # calculate outputs
            outputs = model(batch_inputs)
            # calculate loss
            train_loss = loss_fn(outputs, batch_labels)
            # perform backpropagation
            train_loss.backward()
            # optimize
            optimizer.step()

            running_loss += train_loss.item()

        epoch_loss = running_loss / int(train_inputs.shape[0] / batch_size)

        with torch.no_grad():
            pred = model(test_inputs)
            test_loss = loss_fn(pred, test_targets)

        train_losses.append(epoch_loss)
        test_losses.append(test_loss.item())

        # fh.debug statistics
        fh.debug(f'epoch: {epoch}, train loss: {epoch_loss}, test loss: {test_loss.item()}')
    fh.debug('Training finished!')
    ## ------------------------------------------------------------------------------------
    ## Saving the trained model to file----------------------------------------------------
    PATH = f'Wavenet_trained_cz{conv_size}.pt'
    fh.debug('Saving trained model to ' + PATH + '... ', end='\r')
    torch.save(model, PATH)
    fh.debug('Saving trained model to ' + PATH + '... Done!')


    ## save losses
    fh.debug('Saving losses... ', end='\r')
    np.savetxt(f'train_loss{conv_size}.txt', train_losses)
    np.savetxt(f'test_loss.txt{conv_size}', test_losses)
    fh.debug('Saving losses... Done!')
    ## ------------------------------------------------------------------------------------
