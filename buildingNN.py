import datetime
import sys
import random
import config
import pymysql.cursors # install with `pip install PyMySQL`
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
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

def normalize_windows(window_data):
	normalised_data = []
	for window in window_data:
		try:
			normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
		except ZeroDivisionError:
			normalised_window = 0 # lol
		normalised_data.append(normalised_window)
	return normalised_data

def make_batches(batch_size, data_set):
	x = [data_set[i:batch_size+i] for i in range(0, len(data_set) - batch_size)]
	y = [data_set[batch_size+i] for i in range(0, len(data_set) - batch_size)]
	return x, y

def main():
	epochs = 1
	instances = {}
	timelist = create_time_list()
	db = pymysql.connect(host="67.205.179.187", port=3306, user=config.username, password=config.password, db="csci374")
	cur = db.cursor()
	cur.execute("SELECT id, resource FROM meters ORDER BY resource DESC LIMIT 3") # we're going to build a seperate network for each resource type (e.g. electricity, water, gas, etc.)
	current_resource = None
	for meter in cur.fetchall():
		if meter[1] != current_resource:
			current_resource = meter[1]
			instances[current_resource] = {}
		instances[current_resource][meter[0]] = []
		cur.execute("SELECT value FROM meter_data WHERE meter_id = %s ORDER BY recorded DESC", int(meter[0]))
		last_point = 0
		for data_point in cur.fetchall():
			val = data_point[0]
			if val == None: # very few data points are null so just fill in the ones that are
				val = last_point
			instances[current_resource][meter[0]].append(val)
			last_point = val
	db.close()
	training_size = 90 # 90% training data
	window_size = 7
	for similar_meters in instances.items():
		test_set = []
		training_set = []
		current_resource = similar_meters[0]
		actual_labels = []
		for meter in similar_meters[1].items():
			meter_id = meter[0]
			meter_array = meter[1]
			if len(meter_array) <= window_size:
				continue
			for i in range(len(meter_array)-window_size):
				if random.randint(0, 100) < 90:
					for tmp in meter_array[i:(i+window_size-1)]:
						training_set.append([tmp])
				else:
					for tmp in meter_array[i:(i+window_size-1)]:
						test_set.append([tmp])
				actual_labels.append(meter_array[i+window_size])
			# training_set = normalize_windows(training_set)
			# test_set = normalize_windows(test_set)
			x_train, y_train = make_batches(7, training_set)
			print(len(x_train), len(y_train), x_train[0], y_train[0])
			training_set = np.array(training_set, dtype=float)
			test_set = np.array(test_set, dtype=float)
			training_set = np.reshape(training_set, (len(training_set), window_size, 1))
			# test_set = np.reshape(test_set, (len(test_set), window_size, 1))
			model = create_model()
			model.fit(x_train, y_train, batch_size=7, nb_epoch=epochs, validation_split=0.05, shuffle=False)

main()