import datetime
import config
import pymysql.cursors # install with `pip install PyMySQL`
import random


def network(): # do neural net stuff
	pass

def main():
	test_set = []
	training_set = []
	db = pymysql.connect(host="67.205.179.187", port=3306, user=config.username, password=config.password, db="csci374")
	cur = db.cursor()
	cur.execute("SELECT id, name, area, occupancy, floors, type FROM buildings ORDER BY id ASC LIMIT 3")
	for building in cur.fetchall():
		building_id = int(building[0])
		# print(building_id)
		cur.execute("SELECT id, name, resource, units FROM meters WHERE building_id = %s", building_id)
		for meter in cur.fetchall():
			cur.execute("SELECT value, recorded FROM meter_data WHERE meter_id = %s ORDER BY recorded DESC", int(meter[0]))
			for data_point in cur.fetchall():
				if random.randint(0, 100) > 50:
					test_set.append([(building[1] + ' ' + meter[1]), data_point[0], datetime.datetime.fromtimestamp(int(data_point[1])), meter[3], building[2], building[3], building[4], building[5]])
				else:
					training_set.append([(building[1] + ' ' + meter[1]), data_point[0], datetime.datetime.fromtimestamp(int(data_point[1])), meter[3], building[2], building[3], building[4], building[5]])
	db.close()
	print(training_set[0])
	# network(test_set, training_set)

main()