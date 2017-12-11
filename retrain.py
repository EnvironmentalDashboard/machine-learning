import config
import sys
import os
import pymysql.cursors
import tensorflow as tf
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dense, Dropout, LSTM
import buildingNN

#argv[1] = meter_id, argv[2] = resolution
def main():
    if len(sys.argv) != 3:
        print("Please provide a meter ID and resolution as command line arguments")
        sys.exit(0)
    epochs = 1
    path = os.getcwd()
    meter_id = sys.argv[1]
    res = sys.argv[2]
    db = pymysql.connect(host="67.205.179.187", port=3306, user=config.username, password=config.password, db="csci374", autocommit=True)
    cur = db.cursor()
    cur.execute("SELECT id, res, model, weights, UNIX_TIMESTAMP(updated) AS updated FROM models WHERE meter_id = %s AND res = %s LIMIT 1", (meter_id, res));
    result = cur.fetchone()
    model_id = result[0]
    res = result[1]
    updated = result[4]
    window_size = buildingNN.windowSize(res)
    data = [meter_id]
    data.append([])
    # grab most recent data
    cur.execute("SELECT value FROM meter_data WHERE meter_id = %s AND resolution = %s AND recorded > %s ORDER BY recorded DESC", (meter_id, res, updated))
    last_point = 0
    for row in cur.fetchall():
        val = row[0]
        if val == None:
            val = last_point
        data[1].append(val)
        last_point = val
    if len(data[1]) == 0:
        print("No new data to train on")
        sys.exit(0)
    with open(path + "/tmp.h5", 'wb') as weights_file:
        weights_file.write(result[3])
    loaded_model = model_from_json(result[2])
    loaded_model.load_weights(path + "/tmp.h5")
    loaded_model.compile(loss="mse", optimizer="adam")
    x_train, y_train, x_test, y_test = buildingNN.build_train_and_test_data(data, window_size, 90)
    loaded_model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_split=0.1, shuffle=True)
    # predictions = buildingNN.predict_sequences_multiple(loaded_model, x_test, window_size)
    # buildingNN.plot_results_multiple(predictions, y_test, window_size)
    # print('Accuracy/Mean Squared Error: ', loaded_model.evaluate(x_test, y_test))
    model.save_weights(path + "/model.h5")
    cur.execute("UPDATE models SET weights = %s WHERE id = %s", (open(path + "/model.h5", "rb").read(), model_id))


if __name__ == "__main__":
    main()