from swin_transformer import *
import h5py
from natsort import natsorted
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.data_utils import Sequence
from albumentations import (Compose, Affine, Blur, ColorJitter, GaussianBlur, GaussNoise,
                            HueSaturationValue, MotionBlur,
                            MultiplicativeNoise,
                            RandomBrightnessContrast, RandomFog, RandomRain,
                            RandomShadow, RandomSnow, Sharpen, RGBShift)
from random import randint
from scipy.stats import norm
from math import ceil

folders = natsorted([folder for folder in glob("D:\\bbenja\\datasets\\metadrive_dataset\\*")
                  if not os.path.basename(folder).endswith(".jpg")])


def make_h5_file(path):
    with h5py.File(path, "a") as f:
        for folder in folders[:]:
            grp = f.create_group(folder.split("\\")[-1])
            print(f"Created group: {grp.name}")

            with open(f"{folder}/data.txt") as datafile:
                y = []
                x = []
                for line in datafile:
                    x.append(f"{folder}/{line.split(',')[0]}.jpg")
                    y.append(float(line.split(",")[1]))
                for index in range(1, len(y) - 1):
                    if y[index] == 0.0:
                        y[index] = (y[index - 1] + y[index + 1]) / 2.0
            y = np.asarray(y, dtype="float32")

            grp.create_dataset("y", data=y, dtype="float32")
            grp.create_dataset("X", (len(x), 66, 200, 3), dtype="float32")  # 66, 200

            for i, img in enumerate(x):
                X = cv2.imread(img)
                X = np.divide(X, 255.0, dtype=np.float32)
                X = X[-300:, :, :]
                X = cv2.resize(X, (200, 66), interpolation=cv2.INTER_AREA)  # 200, 66
                f[folder.split("\\")[-1]]['X'][i] = X
                print(f"Done {i+1} of {len(x)}", end="")
                print("\r", end="")


def filter_and_translate_32(dataset_path):
    with h5py.File(dataset_path, "a") as f:
        try:
            del f["filtered"], f["filtered_augmented"]
        except:
            pass
        keys = [ds for ds in list(f.keys())
                if not ds.startswith("0_")
                and not ds.startswith("filtered")
                ]
        indices = []
        to_remove = []
        for ds in keys:
            for idx, angle in enumerate(f[ds]["y"]):
                if abs(angle) <= 0.2:
                    indices.append([ds, idx])
                    to_remove.append([ds, idx])
                else:
                    indices.append([ds, idx])
        print(len(indices))
        gen = H5DataGenerator(f, indices, 32, None)
        bins = plt.hist(gen.get_angles(), bins=np.arange(-1, 1.05, 0.05))
        plt.show()
        for i, x in enumerate(to_remove):
            if i % 3:
                indices.remove(x)
        print(len(indices))

        gen = H5DataGenerator(f, indices, 32, None)
        bins = plt.hist(gen.get_angles(), bins=np.arange(-1, 1.05, 0.05))
        plt.show()
        # bins[0] = koliko brojeva je u binu
        # bins[1] = rubovi binova
        print(np.round(bins[1], 2), max(bins[0]), bins[1][np.where(bins[0] == max(bins[0]))])
        indices = [[ds, idx, abs(f[ds]["y"][idx]) < 0.5] for ds, idx in indices]

        grp = f.create_group("filtered")
        grp.create_dataset("X", (len(indices), 32, 32, 3), dtype=np.float32)
        grp.create_dataset("y", (len(indices), 1), dtype=np.float32)

        grp = f.create_group("filtered_augmented")
        grp.create_dataset("X", (2 * len(indices), 32, 32, 3), dtype=np.float32)
        grp.create_dataset("y", (2 * len(indices), 1), dtype=np.float32)

        x = np.arange(-10, 11, 1)
        y = norm.pdf(x, 0, 5)
        y = [y if x not in np.arange(-4, 5, 1) else 0 for x, y in zip(x, y)]

        for idx, tup in enumerate(indices):
            if tup[2]:
                temp = int(random.choices(x, weights=y, k=1)[0])
                translate = Affine(translate_px=dict(x=temp, y=0), p=1,
                                   mode=cv2.BORDER_REPLICATE)
                f["filtered_augmented"]["X"][idx] = translate(image=f[tup[0]]["X"][tup[1]])["image"]
                delta = translate.translate_px["x"][0] / 10 * 0.25
                f["filtered_augmented"]["y"][idx] = f[tup[0]]["y"][tup[1]] + delta
            else:
                f["filtered_augmented"]["X"][idx] = f[tup[0]]["X"][tup[1]]
                f["filtered_augmented"]["y"][idx] = f[tup[0]]["y"][tup[1]]
            f["filtered"]["X"][idx] = f[tup[0]]["X"][tup[1]]
            f["filtered"]["y"][idx] = f[tup[0]]["y"][tup[1]]

        for idx, tup in enumerate(indices):
            f["filtered_augmented"]["X"][idx + len(indices)] = f[tup[0]]["X"][tup[1]]
            f["filtered_augmented"]["y"][idx + len(indices)] = f[tup[0]]["y"][tup[1]]


        y = [y[0] for y in f["filtered"]["y"][:]]
        del f["filtered"]["y"]
        f["filtered"].create_dataset("y", data=y, dtype=np.float32)

        y = [y[0] for y in f["filtered_augmented"]["y"][:]]
        del f["filtered_augmented"]["y"]
        f["filtered_augmented"].create_dataset("y", data=y, dtype=np.float32)

        print(f["filtered_augmented"]["X"].shape, f["filtered"]["X"].shape)
        print(f["filtered_augmented"]["y"].shape, f["filtered"]["y"].shape)

        plt.hist(y, bins=np.arange(-1, 1.05, 0.05))
        plt.show()


def filter_and_viewpoint_transform(dataset_path, width, height):
    with h5py.File(dataset_path, "a") as f:
        try:
            del f["filtered_view"]
        except:
            pass
        try:
            del f["filtered"]
        except:
            pass
        keys = [ds for ds in list(f.keys())
                if not ds.startswith("0_")
                and not ds.startswith("filtered")
                ]
        indices = []
        to_remove = []
        for ds in keys:
            for idx, angle in enumerate(f[ds]["y"]):
                if abs(angle) <= 0.2:
                    indices.append([ds, idx])
                    to_remove.append([ds, idx])
                else:
                    indices.append([ds, idx])
        print(len(indices))
        gen = H5DataGenerator(f, indices, 32, None)
        bins = plt.hist(gen.get_angles(), bins=np.arange(-1, 1.05, 0.05))
        plt.show()
        for i, x in enumerate(to_remove):
            if i % 3:
                indices.remove(x)
        print(len(indices))

        gen = H5DataGenerator(f, indices, 32, None)
        bins = plt.hist(gen.get_angles(), bins=np.arange(-1, 1.05, 0.05))
        plt.show()
        # bins[0] = koliko brojeva je u binu
        # bins[1] = rubovi binova
        print(np.round(bins[1], 2), max(bins[0]), bins[1][np.where(bins[0] == max(bins[0]))])
        indices = [[ds, idx, abs(f[ds]["y"][idx]) < 0.5] for ds, idx in indices]

        grp = f.create_group("filtered")
        grp.create_dataset("X", (len(indices), height, width, 3), dtype=np.float32)
        grp.create_dataset("y", (len(indices), 1), dtype=np.float32)

        grp = f.create_group("filtered_view")
        grp.create_dataset("X", (2 * len(indices), height, width, 3), dtype=np.float32)
        grp.create_dataset("y", (2 * len(indices), 1), dtype=np.float32)
        shift = ceil(0.3 * width)
        x = np.arange(-shift, shift + 1, 1)
        y = norm.pdf(x, 0, int(shift / 2))
        y = [y if x not in np.arange(-int(shift / 2) - 1, int(shift / 2), 1) else 0 for x, y in zip(x, y)]

        for idx, tup in enumerate(indices):
            if tup[2]:
                temp = int(random.choices(x, weights=y, k=1)[0])
                if temp > 0:
                    f["filtered_view"]["X"][idx] = cv2.resize(f[tup[0]]["X"][tup[1]][:, temp:, :], (width, height), interpolation=cv2.INTER_AREA)
                else:
                    # print(f[tup[0]]["X"][tup[1]][:, :temp, :].shape)
                    f["filtered_view"]["X"][idx] = cv2.resize(f[tup[0]]["X"][tup[1]][:, :temp, :], (width, height), interpolation=cv2.INTER_AREA)
                delta = temp / shift * 0.25
                f["filtered_view"]["y"][idx] = f[tup[0]]["y"][tup[1]] - delta
            else:
                f["filtered_view"]["X"][idx] = f[tup[0]]["X"][tup[1]]
                f["filtered_view"]["y"][idx] = f[tup[0]]["y"][tup[1]]
            f["filtered"]["X"][idx] = f[tup[0]]["X"][tup[1]]
            f["filtered"]["y"][idx] = f[tup[0]]["y"][tup[1]]

        for idx, tup in enumerate(indices):
            f["filtered_view"]["X"][idx + len(indices)] = f[tup[0]]["X"][tup[1]]
            f["filtered_view"]["y"][idx + len(indices)] = f[tup[0]]["y"][tup[1]]
        print(f["filtered_view"]["X"].shape)
        print(f["filtered_view"]["y"].shape)

        y = [y[0] for y in f["filtered_view"]["y"][:]]
        del f["filtered_view"]["y"]
        f["filtered_view"].create_dataset("y", data=y, dtype=np.float32)
        plt.hist(y, bins=np.arange(-1, 1.05, 0.05))
        plt.show()

        y = [y[0] for y in f["filtered"]["y"][:]]
        del f["filtered"]["y"]
        f["filtered"].create_dataset("y", data=y, dtype=np.float32)
        plt.hist(y, bins=np.arange(-1, 1.05, 0.05))
        plt.show()

        # for img, angle in zip(f["filtered_view"]["X"], f["filtered_view"]["y"]):
        #     cv2.imshow("frame", img)
        #     print(f"{angle}\r")
        #     cv2.waitKey(0)


def mirror_images_and_angles(dataset_path):
    with h5py.File(dataset_path, "a") as f:

        for ds in list(f.keys()):
            grp = f.create_group(f"{ds}_mirrored")
            grp.create_dataset("X", f[ds]["X"].shape, dtype=np.float32)
            grp.create_dataset("y", f[ds]["y"].shape, dtype=np.float32)

            for idx, (img, angle) in enumerate(zip(f[ds]["X"], f[ds]["y"])):
                f[f"{ds}_mirrored"]["X"][idx] = np.fliplr(img)
                f[f"{ds}_mirrored"]["y"][idx] = angle * (-1)

        # cv2.imshow("original", f[ds]["X"][0])
        # cv2.imshow("flip", f[f"{ds}_mirrored"]["X"][0])
        # print(f[ds]["y"][0], f[f"{ds}_mirrored"]["y"][0])
        # cv2.waitKey(10)
    # cv2.destroyAllWindows()


dataset_path = "D:\\bbenja\\datasets\\metadrive_32x32.hdf5"
if __name__ == "__main__":
    #make_h5_file(dataset_path)
    #mirror_images_and_angles(dataset_path)
    #filter_and_viewpoint_transform(dataset_path, 200, 66)


    with h5py.File(dataset_path, "r") as f:
        print(list(f.keys()))
    #
    #     train_ds = [ds for ds in list(f.keys()) if ds.startswith("1_")]
    #     val_ds = [ds for ds in list(f.keys()) if ds.startswith("0_")]
    #     train_i, val_i = get_indices(f, train_ds, val_ds)
    #
    #     train_gen = H5DataGenerator(f, val_i, 1, None)
    #
    #     for item in train_gen:
    #         print(item[1])
    #         cv2.imshow("frame", np.squeeze(item[0]))
    #         cv2.waitKey()


        train_ds = [ds for ds in list(f.keys()) if ds.startswith("0_1_")]

        y = [f[ds]["y"][:].tolist() for ds in train_ds]
        y = np.concatenate(y)
        print(len(y))
        plt.xticks(np.arange(-1, 1.2, 0.2))
        plt.grid()
        #plt.title(ds)
        plt.hist(y, bins=np.arange(-1, 1.05, 0.05))
        plt.xlabel("Vrijednost zakreta upravljača vozila")
        plt.ylabel("Broj slika")
        # plt.savefig("md_sve.svg")
        plt.show()
        print("Done.")
