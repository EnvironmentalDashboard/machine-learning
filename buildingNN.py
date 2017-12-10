import datetime
import sys
import random
import config
import pymysql.cursors  # install with `pip install PyMySQL`
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt


def create_time_list():
    timelist = []
    for month in range(1, 13):
        for day in range(1, 32):
            if month < 10:
                if day < 10:
                    time = '0' + str(month) + '-' + '0' + str(day)
                else:
                    time = '0' + str(month) + '-' + str(day)
            else:
                if day < 10:
                    time = str(month) + '-' + '0' + str(day)
                else:
                    time = str(month) + '-' + str(day)
            timelist.append(time)
    return timelist


def onehot(value, list):
    onehot_value = [0] * len(list)
    i = list.index(value)
    onehot_value[i] = 1
    return onehot_value


def create_model(layer1, layer2, layer3, layer4):  # do neural net stuff
    model = Sequential()
    model.add(LSTM(
        input_shape=(layer2, layer1),
        output_dim=layer2,
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(layer3, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=layer4))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="adam")
    return model


def normalize_windows(window_data):
    normalised_data = []
    for window in window_data:
        try:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        except ZeroDivisionError:
            normalised_window = 0  # lol
        normalised_data.append(normalised_window)
    return normalised_data


def make_batches(batch_size, data_set):
    x = [data_set[i:batch_size + i] for i in range(0, len(data_set) - batch_size)]
    y = [data_set[batch_size + i] for i in range(0, len(data_set) - batch_size)]
    return x, y


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    # print('length of data', len(data), len(data[0]), '\n=========data=======\n', data)
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        # plt.legend()
    plt.show()


def download_data():
    # load data from BuildingOS
    instances = {}
    db = pymysql.connect(host="67.205.179.187", port=3306, user=config.username, password=config.password, db="csci374")
    cur = db.cursor()
    cur.execute("SELECT id FROM meters ORDER BY RAND() LIMIT 3") # we're going to build a seperate network for each meter
    for meter in cur.fetchall():
        instances[meter[0]] = []
        cur.execute("SELECT value FROM meter_data WHERE meter_id = %s ORDER BY recorded DESC", int(meter[0]))
        last_point = 0
        for data_point in cur.fetchall():
            val = data_point[0]
            if val == None:  # very few data points are null so just fill in the ones that are
                val = last_point
            instances[meter[0]].append(val)
            last_point = val
    db.close()
    return instances


def build_train_and_test_data(data, window_size):
    test_set = []
    training_set = []
    actual_labels = []
    meter_id = data[0]
    meter_array = data[1]
    for i in range(len(meter_array) - window_size):
        if random.randint(0, 100) < 90:
            for tmp in meter_array[i:(i + window_size - 1)]:
                training_set.append([tmp])
        else:
            for tmp in meter_array[i:(i + window_size - 1)]:
                test_set.append([tmp])
        actual_labels.append(meter_array[i + window_size])
        # training_set = normalize_windows(training_set)
        # test_set = normalize_windows(test_set)
    x_train, y_train = make_batches(window_size, training_set)
    x_test, y_test = make_batches(window_size, test_set)

    x_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    x_test = np.array(x_test, dtype=float)
    y_test = np.array(y_test, dtype=float)

    return x_train, y_train, x_test, y_test


def main():
    epochs = 1
    window_size = 7
    instances = download_data()
    for meter in instances.items():
        print("Processing meter", meter[0])
        if len(meter[1]) == 0:
            print(meter[0], "has no data")
            continue
        x_train, y_train, x_test, y_test = build_train_and_test_data(meter, window_size)

        model = create_model(1, window_size, 100, 1)

        model.fit(x_train, y_train, batch_size=512, nb_epoch=epochs, validation_split=0.05, shuffle=False)
        predictions = predict_sequences_multiple(model, x_test, window_size, 7)
        # print(len(x_test), len(y_test), len(predictions))
        plot_results_multiple(predictions, y_test, 7)


main()
