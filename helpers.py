import numpy as np
import pandas as pd
import torch


def standardize(x):
    std_data = np.zeros(x.shape)
    if len(x.shape) > 2:
        mean = np.mean(x.reshape((x.shape[0]*x.shape[1], x.shape[2])), axis=0)
        std = np.std(x.reshape((x.shape[0]*x.shape[1], x.shape[2])), axis=0)
        for i in range(x.shape[0]):
            centered_data = x[i] - mean
            std_data[i] = centered_data / std
    else:
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        centered_data = x - mean
        std_data = centered_data / std
    return std_data, mean, std


def fit_standardize(x, mean_stand, std_stand):
    std_data = np.zeros(x.shape)
    if len(x.shape) > 2:
        for i in range(x.shape[0]):
            std_data[i] = (x[i] - mean_stand) / std_stand
    else:
        std_data = (x - mean_stand) / std_stand
    return std_data


# finds the index of the nearest value in an array and its immediate neighbors
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return [idx - 1, idx, idx + 1]


# returns the measurement at a given angle of the cylinder
def getMeasureFromAngle(timestep, angle):
    timestep = pd.DataFrame(timestep)
    # find 3 nearest measurement nodes
    nearest_nodes = find_nearest_idx(timestep['node_position'].unique(), angle)
    # calc mean of 3 nearest measurment nodes
    nodes = timestep[timestep['node_position'] == timestep['node_position'].unique()[nearest_nodes[0]]].append(
        timestep[timestep['node_position'] == timestep['node_position'].unique()[nearest_nodes[1]]])
    nodes = nodes.append(timestep[timestep['node_position'] == timestep['node_position'].unique()[nearest_nodes[1]]])
    return nodes.mean(axis=0)


# constructs timeseries of one experiment from complete data structure
def getTimeSeries(df, sensor_angles=np.linspace(0, 359, 15),
                  sensors=['pressure', 'velocity-magnitude', 'acoustic_power'],
                  output_labels=['drag_force', 'lift_force', 'angle_of_attack'],
                  experiment_number=0,
                  length=None):

    experiment = df.groupby('time', as_index=False).nth(list(range(experiment_number * 343, (experiment_number + 1) * 343)))
    timevector = experiment['time']
    length = len(timevector.unique())
    
    if length is None:
        length = experiment['time'].unique().size

    features = np.zeros((length, len(sensors) * len(sensor_angles)))
    labels = np.zeros((length, len(output_labels)))
    for i, time in enumerate(timevector.unique()):
        if i >= length:
            break
        timestep = experiment[timevector == time]
        angle_of_attack = timestep.iloc[0]['angle_of_attack']
        f_row = []
        for k, sensor_angle in enumerate(sensor_angles):
            angle_step = sensor_angle + angle_of_attack
            if angle_step < 0:
                angle_step += 360
            elif angle_step >= 360:
                angle_step -= 360
            f_row = np.append(f_row, getMeasureFromAngle(timestep, angle_step)[sensors])
        features[i] = f_row
        labels[i] = timestep.iloc[0][output_labels]
    return features, labels


def train_test_set(feature, label, ratio=0.8):
    """Data splitting"""
    train_sep = int(feature.shape[0] * ratio)
    train_input = feature[:train_sep]
    train_target = label[:train_sep]

    test_input = feature[train_sep:]
    test_target = label[train_sep:]

    return train_input, train_target, test_input, test_target


def getTcnInputsFromTimeseries(timeseries, timestep: int):
    out = torch.zeros(1, timeseries.shape[1])

    aug_timeseries = np.append(np.zeros((conv_size - 1, timeseries.shape[1])), timeseries, axis=0)
    out[0, :, :] = torch.from_numpy(aug_timeseries[timestep:timestep + conv_size])
    return out


def getWavenetInputsFromTimeseries(timeseries, conv_size: int):
    out = torch.zeros(len(timeseries), timeseries.shape[1], conv_size)
    
    aug_timeseries = np.append(np.zeros((conv_size-1, timeseries.shape[1])), timeseries, axis=0)
    for i in range(len(timeseries)):
        for k in range(timeseries.shape[1]):
            out[i, k, :] = torch.from_numpy(aug_timeseries[i:i+conv_size,k])
    return out


def create_TCN_feature(feature, conv_size):
    T = int(feature.shape[0])
    out = torch.zeros(T, conv_size, feature.shape[1])
    for i in range(T):
        out[i, :, :] = getTcnInputsFromTimeseries(feature, conv_size, i)
    return out


def train(model, train_input, train_target, loss_fn, optimizer, batch_size=1):
    model.train()
    train_loss = 0

    for b in range(0, train_input.size(0), batch_size):
        model.zero_grad()
        #print(train_input.narrow(0, b, batch_size).shape)
        pred = model(train_input.narrow(0, b, batch_size))
        loss = loss_fn(pred, train_target.narrow(0, b, batch_size))  # +ridge ?
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss / train_input.shape[0]


def test(model, test_input, test_target, loss_fn, batch_size=1):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for b in range(0, test_input.size(0), batch_size):
            pred = model(test_input.narrow(0, b, batch_size))
            loss = loss_fn(pred, test_target.narrow(0, b, batch_size))
            test_loss += loss.item()

    return test_loss / test_input.shape[0]

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = torch.randperm(data_size)
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]