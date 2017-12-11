import config
import sys
import os
import pymysql.cursors
import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dense, Dropout, LSTM

def main():
    if len(sys.argv) != 3:
        print("Please provide a meter ID and resolution as command line arguments")
        sys.exit(0)
    path = os.getcwd()
    meter_id = sys.argv[1]
    res = sys.argv[2]
    db = pymysql.connect(host="67.205.179.187", port=3306, user=config.username, password=config.password, db="csci374", autocommit=True)
    cur = db.cursor()
    cur.execute("SELECT model, weights FROM models WHERE meter_id = %s LIMIT 1", meter_id);
    result = cur.fetchone()
    with open(path + "/tmp.h5", 'wb') as weights_file:
        weights_file.write(result[1])
    # See https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    loaded_model = model_from_json(result[0])
    loaded_model.load_weights(path + "/tmp.h5")
    loaded_model.compile(loss="mse", optimizer="adam")
    # grab most recent data
    cur.execute("SELECT value FROM meter_data WHERE meter_id = %s AND resolution = %s ORDER BY recorded DESC LIMIT %s", (meter_id, res, choose_res(res)))
    window = []
    for data_point in cur.fetchall():
        window.append(data_point[0])

if __name__ == "__main__":
    main()