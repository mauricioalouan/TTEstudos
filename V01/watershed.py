# -*- coding:utf-8 -*-
#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt
#dilatação e depois erosão
img = cv2.imread('arvorees.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#rgb2 = img_rgb[:,:,2]
cv2.imwrite("rgb.png", img_rgb)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imwrite("thresh.png",thresh)


# noise removal
kernel = np.ones((3,3),np.uint8)#como se fosse um apagador ( quanto menor mais preciso)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)#iterations (quantas vezes limpa)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
#dist_transform= cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.001*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imwrite("saida2.png",unknown )
cv2.imwrite("sure_fg.png",sure_fg )
cv2.imwrite("sure_bg.png",sure_bg )
 # Marker labelling
#print(dir(cv2))
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
cv2.imwrite("markers.png",markers)
markers = cv2.watershed(img,markers)
img[markers == -1] = [0,255,0]
cv2.imwrite("markers2.png",markers)