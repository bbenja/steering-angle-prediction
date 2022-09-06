from glob import glob
from natsort import natsorted
import os

folders = natsorted([folder for folder in glob("C:\\Users\\bbenja\\Desktop\\MetaDrive\\new_ds\\15_2_*")
                     if not os.path.basename(folder).endswith(".jpg")])

for folder in folders:
    print(folder)
    new_file = []
    with open(f"{folder}/data.txt", 'r') as file:
        for line in file:
            img, angle, throttle, speed = line.split(",")
            if img == 1:
                print(img, angle, throttle, speed)
            new_file.append(f"{img},{-float(angle)},{throttle},{speed}")
    with open(f"{folder}/data.txt", "w+") as f:
        for i in new_file:
            f.write(i)
print("Done.")
