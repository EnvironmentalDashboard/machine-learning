import matplotlib.pyplot as plt
import pymysql.cursors
import config
import datetime

db = pymysql.connect(host="67.205.179.187", port=3306, user=config.username, password=config.password, db="csci374")
cur = db.cursor()
cur.execute("SELECT value, recorded, meter_id FROM `meter_data` WHERE meter_id = 266 OR meter_id = 300 ORDER BY `recorded` ASC") # Barrows electricity use and total campus steam flow
x = []
y = []
x2 = []
y2 = []
for row in cur.fetchall():
	if row[2] == 266:
		x.append(datetime.datetime.fromtimestamp(int(row[1])))
		y.append(row[0])
	else:
		x2.append(datetime.datetime.fromtimestamp(int(row[1])))
		y2.append(row[0])

plt.plot(x,y)
plt.plot(x2,y2)
plt.show()