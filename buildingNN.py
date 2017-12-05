import datetime
import sys
import config
import pymysql.cursors # install with `pip install PyMySQL`
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

def create_time_list():
	timelist = []
	for month in range(1,13):
		for day in range(1, 32):
			if month < 10:
				if day < 10:
					time = '0'+str(month)+'-'+'0'+str(day)
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

def create_model(layer1 = 1, layer2 = 50, layer3 = 100, layer4 = 1): # do neural net stuff
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
  model.compile(loss="mse", optimizer="rmsprop")
  return model

def main():
	epochs = 1
	instances = {}
	test_set = []
	training_set = []
	timelist = create_time_list()
	db = pymysql.connect(host="67.205.179.187", port=3306, user=config.username, password=config.password, db="csci374")
	cur = db.cursor()
	cur.execute("SELECT id, resource FROM meters ORDER BY resource ASC") # we're going to build a seperate network for each resource type (e.g. electricity, water, gas, etc.)
	current_resource = None
	meter_index = 0
	for meter in cur.fetchall():
		if meter[1] != current_resource:
			current_resource = meter[1]
			instances[current_resource] = []
		cur.execute("SELECT value, recorded FROM meter_data WHERE meter_id = %s ORDER BY recorded DESC", int(meter[0]))
		last_point = 0
		for data_point in cur.fetchall():
			val = data_point[0]
			if val == None: # very few data points are null so just fill in the ones that are
				val = last_point
			instances[current_resource][meter_index].append(val)
			last_point = val
		meter_index += 1
	db.close()
	for similar_meters in instances.items():
		print(similar_meters)
		sys.exit(0)
		num_instances = len(instances)
		training_size = int(num_instances/2)
		window_size = int(num_instances/20)
		for i in range(num_instances-window_size):
			if i < training_size:
				training_set.append(instances[i:i+window_size])
			else:
				test_set.append(instances[i:i+window_size])
		training_set = np.array(training_set)
		test_set = np.array(test_set)
		training_set = np.reshape(training_set, (len(training_set), len(training_set[0]), 1))
		test_set = np.reshape(test_set, (len(test_set), len(test_set[0]), 1))
		print(training_set[0])
		model = create_model()
		model.fit(
	    X_train,
	    y_train,
	    batch_size=512,
	    nb_epoch=epochs,
	    validation_split=0.05,
	    shuffle=False)
	# predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)

main()