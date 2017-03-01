import csv 
import sys
import numpy as np
from PIL import Image
import cv2
 
filename = open(sys.argv[1], 'rb')

file = csv.reader(filename)

i = 1

for row in file:
    if row[0] == 'emotion':
        continue

    image = np.fromstring(row[1], dtype=int, sep=' ').reshape(48, 48)

    location = row[2] + "/"

    cv2.imwrite(location + str(i) + ".jpg", image)

    i += 1

filename.close()
