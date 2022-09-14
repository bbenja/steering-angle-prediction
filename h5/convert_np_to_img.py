import numpy as np
import cv2
from glob import glob
from natsort import natsorted

folder = "D:\\bbenja\\datasets\\carla_dataset\\town04_clear_noon_0"
numpys = natsorted(glob(f"{folder}/dataset*.npy"))
images = [np.load(n) for n in numpys[0:5]]
images = np.concatenate(images, axis=0)

st_wheel = cv2.imread("steering_wheel_image.jpg")
rows, cols, _ = st_wheel.shape

with open(f"{folder}/data.txt") as datafile:
    y = []
    for line in datafile:
        y.append(float(line.split(",")[1]))
    for index in range(1, len(y) - 1):
        if y[index] == 0.0:
            y[index] = (y[index - 1] + y[index + 1]) / 2.0
#y = np.asarray(y, dtype="float32")


#cv2.imshow("img", images[0])
for i, img in enumerate(images):
    # img = cv2.circle(img, (int(640*y[i]+640), 700), radius=10, color=(0, 255, 0), thickness=-1)
    # img = cv2.circle(img, (640, 700), radius=13, color=(0, 0, 255), thickness=1)
    rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), y[i] * -520, 1)
    dst = cv2.warpAffine(st_wheel, rot, (cols, rows))

    # img[5:5+cols, 5:5+rows] = dst

    cv2.imshow("300", img) #[-300:]
    # cv2.imwrite(folder.split("\\", )[-1] + ".jpg", img)
    print(y[i])
    #cv2.imshow("350", img[-350:])
    #cv2.imshow("400", images[i][-400:])
    #print(y[i])
    cv2.waitKey(0)
cv2.destroyAllWindows()
