import datetime
import config
import pymysql.cursors # install with `pip install PyMySQL`
import tensorflow as tf


def network(): # do neural net stuff

def main():
	test_set = []
	training_set = []
	db = pymysql.connect(host="67.205.179.187", port=3306, user=config.username, password=config.password, db="csci374")
	cur = db.cursor()
	cur.execute("SELECT value, recorded FROM meter_data WHERE meter_id = 313 AND DAYOFWEEK(FROM_UNIXTIME(recorded)) IN (1, 2, 3) ORDER BY recorded DESC")
	for row in cur.fetchall():
		print("%f recorded on" % float(row[0]), end=' ')
		print(datetime.datetime.fromtimestamp(int(row[1])).strftime('%Y-%m-%d'))
	db.close()
	network(test_set, training_set)

def Build_and_run_Network(training, testing, num_attr, num_labels, learning_rate, NUM_NEURONS, numIteration):
	x = tf.placeholder(tf.float32, shape=[None, num_attr])  # input matrix
	# weights and biases of hidden layer
	W_hidden = tf.Variable(tf.truncated_normal([num_attr, NUM_NEURONS], stddev=0.1))
	b_hidden = tf.Variable(tf.constant(0.1, shape=[NUM_NEURONS]))
	net_hidden = tf.matmul(x, W_hidden) + b_hidden
	out_hidden = tf.sigmoid(net_hidden)
	# output layer
	W_output = tf.Variable(tf.truncated_normal([NUM_NEURONS, num_labels], stddev=0.1))
	b_output = tf.Variable(tf.constant(0.1, shape=[num_labels]))

	net_output = tf.matmul(out_hidden, W_output) + b_output
	# actual labels
	y = tf.placeholder(tf.float32, shape=[None, num_labels])

	# train
	prediction = tf.nn.softmax(net_output)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=training['label']))

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	hm_epoch = numIteration
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(1, hm_epoch + 1):
			epoch_loss = 0
			_, c = sess.run([optimizer, cost], feed_dict={x: training['attr'], y: training['label']})
			epoch_loss += c
			if epoch % 50 == 0:
				# print('Epoch', epoch, "comleted out of", hm_epoch, '| loss: ', epoch_loss)
				correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
				accuracy_float = round(float(sess.run(accuracy, feed_dict={x: training['attr'], y: training['label']})), 5)
				print('Accuracy:', accuracy_float)

		# evaluate
		num_test = len(testing['attr'])
		num_train = len(training['attr'])
		print('The size of training set: %s\nThe size of test set: %s'%(num_train, num_test))

		correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		accuracy_float = round(float(sess.run(accuracy, feed_dict={x: testing['attr'], y: testing['label']})), 5)
		print('Accuracy:', accuracy_float)
		# print('95% confidence interval: ', cf(accuracy_float, num_test, 1.96))

main()