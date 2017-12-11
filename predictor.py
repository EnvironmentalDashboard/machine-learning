import config
import sys
import os
import pymysql.cursors
import tensorflow as tf
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dense, Dropout, LSTM
import buildingNN

def windowSize(resolution):
    if resolution == 'day':
        return 7
    if resolution == 'hour':
        return 24
    else:
        return 10

#argv[1] = meter_id, argv[2] = resolution
def main():
    if len(sys.argv) != 3:
        print("Please provide a meter ID and resolution as command line arguments")
        sys.exit(0)
    path = os.getcwd()
    meter_id = sys.argv[1]
    res = sys.argv[2]
    db = pymysql.connect(host="67.205.179.187", port=3306, user=config.username, password=config.password, db="csci374", autocommit=True)
    cur = db.cursor()
    cur.execute("SELECT model, weights FROM models WHERE meter_id = %s AND res = %s LIMIT 1", (meter_id, res));
    result = cur.fetchone()
    with open(path + "/tmp.h5", 'wb') as weights_file:
        weights_file.write(result[1])
    # See https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    loaded_model = model_from_json(result[0])
    loaded_model.load_weights(path + "/tmp.h5")
    loaded_model.compile(loss="mse", optimizer="adam")
    # grab most recent data
    window_size = windowSize(res)
    cur.execute("SELECT value FROM meter_data WHERE meter_id = %s AND resolution = %s ORDER BY recorded DESC LIMIT %s", (meter_id, res, window_size))
    # print("SELECT value FROM meter_data WHERE meter_id = %s AND resolution = %s ORDER BY recorded DESC LIMIT %s" %(meter_id, res, window_size))
    window = [[]]
    last_point = 0
    for data_point in cur.fetchall():
        val = data_point[0]
        if val == None:
            val = last_point
        window[0].append(val)
        last_point = val
    print(window)
    window[0] = buildingNN.normalize_data(window[0])
    window = np.array(window, dtype=float)
    prediction = loaded_model.predict(window)
    # predictions = buildingNN.predict_sequences_multiple(loaded_model, window, window_size)
    print(len(window[0]), window, prediction)

if __name__ == "__main__":
    main()