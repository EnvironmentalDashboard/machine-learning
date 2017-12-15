"""
Will build models for all meters using data that defaults to hour resolution but can be specified via the first command line arguemnt
If a second command line option is given, the program will chart the data and prediction using matplotlib
"""
import datetime
import sys
import os
import random
import csv
import config
import pymysql.cursors
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pandas import Series
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dense, Dropout, LSTM
from math import sqrt

def create_model(layer1, layer2, layer3, layer4):  # do neural net stuff
    model = Sequential()
    model.add(LSTM(
        input_shape=(layer2, layer1),
        units=layer2,
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(layer3, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=layer4))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="adam")
    return model

def make_batches(batch_size, data_set):
    x = [data_set[i:batch_size + i] for i in range(0, len(data_set) - batch_size)]
    y = [data_set[batch_size + i] for i in range(0, len(data_set) - batch_size)]
    return x, y

def predict_sequences_multiple(model, data, window_size):
    # Predict sequence of window_size steps before shifting prediction run forward by window_size steps
    prediction_seqs = []
    # print('length of data', len(data), len(data[0]), '\n=========data=======\n', data)
    for i in range(int(len(data) / window_size)):
        curr_frame = data[i * window_size]
        predicted = []
        for j in range(window_size):
            predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_results_multiple(predicted_data, true_data, prediction_len, name, MSE, NRMSD, res):
    fig = plt.figure(facecolor='white')
    plt.title('MeterID=%s Epochs=%s WS=%s NN=%s %s'%name)
    plt.annotate("MSE = %s\nNRMSD = %s"%(round(MSE, 3), round(NRMSD, 3)), xy = (0.75, 0.05), xycoords = 'axes fraction')
    plt.xlabel("Unit = %s"%res)
    plt.ylabel("Standardized Usage")
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Actual Data')
    plt.legend()
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
    plt.show()
    return fig


def query_db(cur, res, specific_meter):
    # load data from database
    # REMEMBER TO REMOVE LIMITs IN FINAL CODE!!!
    instances = {}
    if specific_meter == None:
        cur.execute("SELECT id FROM meters") # we're going to build a seperate network for each meter
        for meter in cur.fetchall():
            instances[meter[0]] = []
            cur.execute("SELECT value FROM meter_data WHERE meter_id = %s AND resolution = %s ORDER BY recorded DESC", (int(meter[0]), res))
            last_point = 0
            for data_point in cur.fetchall():
                val = data_point[0]
                if val == None:  # very few data points are null so just fill in the ones that are
                    val = last_point
                instances[meter[0]].append(val)
                last_point = val
    else:
        instances[specific_meter] = []
        cur.execute("SELECT value FROM meter_data WHERE meter_id = %s AND resolution = %s ORDER BY recorded DESC", (int(specific_meter), res))
        last_point = 0
        for data_point in cur.fetchall():
            val = data_point[0]
            if val == None:  # very few data points are null so just fill in the ones that are
                val = last_point
            instances[specific_meter].append(val)
            last_point = val
    return instances

def normalize_data(data):
    series = Series(data)
    series_values = series.values
    series_values = series_values.reshape((len(series_values), 1))
    # train the normalization
    scaler = StandardScaler()
    scaler = scaler.fit(series_values)
    standardized = scaler.transform(series_values)
    return standardized

def convertRange(val, old_min, old_max, new_min, new_max):
    if old_max == old_min:
        return 0
    return (((new_max - new_min) * (val - old_min)) / (old_max - old_min)) + new_min

def build_train_and_test_data(data, window_size, training_pct, normal_in_window):
    test_set = []
    training_set = []
    actual_labels = []
    meter_id = data[0]
    meter_array = data[1]
    train_array = meter_array[:int(len(meter_array)*training_pct)]
    test_array = meter_array[int(len(meter_array)*training_pct):]
    old_max_train = max(train_array)
    old_min_train = min(train_array)
    old_max_test = max(test_array)
    old_min_test = min(test_array)
    """
    for i in range(len(meter_array) - window_size):
        if random.randint(0, 100) < training_pct:
            for tmp in meter_array[i:(i + window_size - 1)]:
                training_set.append(tmp)
        else:
            for tmp in meter_array[i:(i + window_size - 1)]:
                test_set.append(tmp)
        actual_labels.append(meter_array[i + window_size])
    """
    # if normal_in_window:
    #     x_train, _ = make_batches(window_size, train_array)
    #     x_test, _ = make_batches(window_size, test_array)
    #     for i in range(len(x_train)):
    #         x_train[i] = normalize_data(x_train[i])
    #     y_train = [[x_train[i][-1]] for i in range(1, len(x_train))] + [x_train[-1][-1]]
    #     for i in range(len(x_test)):
    #         x_test[i] = normalize_data(x_test[i])
    #     y_test = [[x_test[i][-1]] for i in range(1, len(x_test))] + [x_test[-1][-1]]
    # else:

    training_set = normalize_data(train_array)
    print("===========normalized training=========", training_set)
    test_set = normalize_data(test_array)
    x_train, y_train = make_batches(window_size, training_set)
    x_test, y_test = make_batches(window_size, test_set)
    normalization = 'StandardScaler'

    # for i in range(len(train_array)):
    #     train_array[i] = [convertRange(train_array[i], old_min_train, old_max_train, 0, 1)]
    # for i in range(len(test_array)):
    #     test_array[i] = [convertRange(test_array[i], old_min_test, old_max_test, 0, 1)]
    # x_train, y_train = make_batches(window_size, train_array)
    # x_test, y_test = make_batches(window_size, test_array)
    # print('x_train =====', x_train)
    # print('y_train =====', y_train)
    # normalization = 'CovertRange'

    diff = max(y_test)[0]-min(y_test)[0]


    x_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    x_test = np.array(x_test, dtype=float)
    y_test = np.array(y_test, dtype=float)

    print('x_train =====', x_train)
    print('y_train =====', y_train)
    print('diff: ',diff)
    return x_train, y_train, x_test, y_test, diff, normalization

def windowSize(resolution):
    if resolution == 'day':
        return 7
    if resolution == 'hour':
        return 24
    else:
        return 10

def main():
    # Res, Chart, MeterID, Epochs, Training_Percent, NN
    args = len(sys.argv)
    epochs = 5
    path = os.getcwd()
    NN = 50
    training_pct = 0.9
    val_pct = 0.1
    specific_meter = None
    chart = False
    if args == 1: # no args
        res = 'hour'
    elif args == 2: # only resolution provided
        res = sys.argv[1]
    elif args == 3:
        res = sys.argv[1]
        if sys.argv[2] == 'chart':
            chart = True
    elif args == 4:
        res = sys.argv[1]
        if sys.argv[2] == 'chart':
            chart = True
        specific_meter = int(sys.argv[3])
    elif args == 5:
        res = sys.argv[1]
        if sys.argv[2] == 'chart':
            chart = True
        specific_meter = int(sys.argv[3])
        epochs = int(sys.argv[4])
    elif args == 6:
        res = sys.argv[1]
        if sys.argv[2] == 'chart':
            chart = True
        specific_meter = int(sys.argv[3])
        epochs = int(sys.argv[4])
        training_pct = float(sys.argv[5])
    elif args == 7:
        res = sys.argv[1]
        if sys.argv[2] == 'chart':
            chart = True
        specific_meter = int(sys.argv[3])
        epochs = int(sys.argv[4])
        training_pct = float(sys.argv[5])
        NN = int(sys.argv[6])
    window_size = windowSize(res)
    db = pymysql.connect(host="67.205.179.187", port=3306, user=config.username, password=config.password, db="csci374", autocommit=True)
    cur = db.cursor()
    instances = query_db(cur, res, specific_meter)
    # print(len(instances), len(instances[1]))
    for meter in instances.items():
        print("Processing meter", meter[0])
        if len(meter[1]) == 0:
            print(meter[0], "has no data")
            continue
        x_train, y_train, x_test, y_test, diff, normalization= build_train_and_test_data(meter, window_size, training_pct, True)

        model = create_model(1, window_size, NN, 1)

        model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_split= val_pct, shuffle=False)
        MSE = model.evaluate(x_test, y_test)
        print('Accuracy/Mean Squared Error: ', MSE)
        NRMSD = sqrt(MSE)/float(diff)
        print("NRMSD: ", NRMSD)
        if chart:
            predictions = predict_sequences_multiple(model, x_test, window_size)
            # print(len(x_test), len(y_test), len(predictions))
            name = 266, epochs, window_size, NN, normalization
            fig = plot_results_multiple(predictions, y_test, window_size, name, MSE, NRMSD, res)
            fig.savefig('id%s_%s_epochs%s_ws%s_nn%s_%s'%(266, res, epochs, window_size, NN, normalization))

        with open('recorder.csv', mode='a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',')
            filewriter.writerow([MSE, NRMSD, epochs, window_size, NN, normalization, training_pct, val_pct])

        # See https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        model_json = model.to_json()
        model.save_weights(path + "/model.h5") # serialize weights to HDF5 to read from later
        cur.execute("INSERT INTO models (meter_id, res, model, weights, MSE, NRMSD) VALUES (%s, %s, %s, %s, %s, %s)", (meter[0], res, model_json, open(path + "/model.h5", "rb").read(), np.asscalar(MSE), NRMSD))
    try:
        os.remove(path + "/model.h5")
    except OSError:
        pass
    db.close()

if __name__ == '__main__':
    main()
