import numpy as np
import cv2
from glob import glob
from natsort import natsorted

vals = [0.061809040663856424, -0.17987288070762983]

imgs = ["80.jpg", "227.jpg"]

st_wheel = cv2.imread("steering_wheel_image.jpg")
rows, cols, _ = st_wheel.shape

for id, (img, val) in enumerate(zip(imgs, vals)):
    img = cv2.imread(img)
    rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), val * -520, 1)
    dst = cv2.warpAffine(st_wheel, rot, (cols, rows))

    img[5:5+cols, 5:5+rows] = dst

    cv2.imshow("img", img)  # [-300:]
    cv2.waitKey(0)
    cv2.imwrite(str(id) + ".jpg", img)
