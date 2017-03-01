import csv 
import sys
import os

filename = open(sys.argv[1], 'rb')

file = csv.reader(filename)

destination = 'test/'

i = 1
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']

for row in file:
    if row[0] == 'Usage':
        continue

    if max(row[1:]) in ['8', '7', '6', '5']:
        from_path = row[0] + "/"
        destine_path = os.path.join(destination, row[0] + "/", emotions[row[1:].index(max(row[1:]))] + "/")

        os.rename(from_path + str(i) + ".jpg", destine_path + str(i) + ".jpg")

    i += 1

filename.close()
