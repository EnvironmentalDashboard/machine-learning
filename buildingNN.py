import datetime
import sys
import os
import random
import config
import pymysql.cursors
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pandas import Series
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dense, Dropout, LSTM

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

def make_batches(batch_size, data_set):
    x = [data_set[i:batch_size + i] for i in range(0, len(data_set) - batch_size)]
    y = [data_set[batch_size + i] for i in range(0, len(data_set) - batch_size)]
    return x, y

def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of window_size steps before shifting prediction run forward by window_size steps
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


def query_db(cur):
    # load data from database
    # REMEMBER TO REMOVE LIMITs IN FINAL CODE!!!
    instances = {}
    cur.execute("SELECT id FROM meters LIMIT 3") # we're going to build a seperate network for each meter
    for meter in cur.fetchall():
        instances[meter[0]] = []
        cur.execute("SELECT value FROM meter_data WHERE meter_id = %s AND resolution = 'hour' ORDER BY recorded DESC LIMIT 1000", int(meter[0]))
        last_point = 0
        for data_point in cur.fetchall():
            val = data_point[0]
            if val == None:  # very few data points are null so just fill in the ones that are
                val = last_point
            instances[meter[0]].append(val)
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

def build_train_and_test_data(data, window_size, training_pct):
    test_set = []
    training_set = []
    actual_labels = []
    meter_id = data[0]
    meter_array = data[1]
    for i in range(len(meter_array) - window_size):
        if random.randint(0, 100) < training_pct:
            for tmp in meter_array[i:(i + window_size - 1)]:
                training_set.append(tmp)
        else:
            for tmp in meter_array[i:(i + window_size - 1)]:
                test_set.append(tmp)
        actual_labels.append(meter_array[i + window_size])
    training_set = normalize_data(training_set)
    test_set = normalize_data(test_set)
    x_train, y_train = make_batches(window_size, training_set)
    x_test, y_test = make_batches(window_size, test_set)

    x_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    x_test = np.array(x_test, dtype=float)
    y_test = np.array(y_test, dtype=float)

    return x_train, y_train, x_test, y_test


def main():
    epochs = 1
    window_size = 24
    db = pymysql.connect(host="67.205.179.187", port=3306, user=config.username, password=config.password, db="csci374", autocommit=True)
    cur = db.cursor()
    instances = query_db(cur)
    # print(len(instances), len(instances[1]))
    path = os.getcwd()
    for meter in instances.items():
        print("Processing meter", meter[0])
        if len(meter[1]) == 0:
            print(meter[0], "has no data")
            continue
        x_train, y_train, x_test, y_test = build_train_and_test_data(meter, window_size, 90)

        model = create_model(1, window_size, 100, 1)

        model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_split=0.1, shuffle=True)
        predictions = predict_sequences_multiple(model, x_test, window_size, window_size)
        # print(len(x_test), len(y_test), len(predictions))
        plot_results_multiple(predictions, y_test, window_size)
        print('Accuracy/Mean Squared Error: ', model.evaluate(x_test, y_test))
        model_json = model.to_json()
        model.save_weights(path + "/model.h5") # serialize weights to HDF5 to read from later
        cur.execute("INSERT INTO models (meter_id, model, weights) VALUES (%s, %s, %s)", (meter[0], model_json, open(path + "/model.h5", "rb").read()))
    os.remove(path + "/model.h5")
    db.close()

if __name__ == '__main__':
    main()
