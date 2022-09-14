import h5py
from glob import glob
import os
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import choice
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.data_utils import Sequence
from scipy.stats import norm
import tensorflow as tf
from albumentations import Affine
from loaders_generators import H5DataGenerator, AUGMENTATIONS, RandomBrightnessContrast
from math import ceil
import random


def make_h5_file(path):  # "D:\\bbenja\\datasets\\carla_datasets\\town0*"
    folders = [folder for folder in glob(path)
               if not os.path.basename(folder).endswith(".npy")]

    numpys = [n for folder in folders for n in glob(f"{folder}/dataset*.npy")]
    numpys = natsorted(numpys)

    with h5py.File("D:\\bbenja\\datasets\\carla_dataset_32x32.hdf5", "a") as f:
        for folder in folders[:]:
            grp = f.create_group(folder.split("\\")[-1])
            print(f"Created group: {grp.name}")

            files = [n for n in numpys if n.startswith(f"{folder}\\")]
            grp.create_dataset("X", (len(files) * 100, 32, 32, 3), dtype="float32")

            for i, n in enumerate(files):
                X = np.load(n)
                X = np.divide(X, 255.0)
                X = X[:, -300:, :, :]
                X = [cv2.resize(x, (32, 32), interpolation=cv2.INTER_AREA) for x in X]  # (width, height)
                f[folder.split("\\")[-1]]['X'][i * 100:i * 100 + 100, :, :, :] = X
                print(f"Done {i + 1} of {len(files)}", end="")
                print("\r", end="")

            with open(f"{folder}/data.txt") as datafile:
                y = []
                for line in datafile:
                    y.append(float(line.split(",")[1]))
                for index in range(1, len(y) - 1):
                    if y[index] == 0.0:
                        y[index] = (y[index - 1] + y[index + 1]) / 2.0
            y = np.asarray(y, dtype="float32")

            grp.create_dataset("y", data=y, dtype="float32")


def add_angles(path):
    folders = [folder for folder in glob(path)
               if not os.path.basename(folder).endswith(".npy")]

    # numpys = [n for folder in folders for n in glob(f"{folder}/dataset*.npy")]
    # numpys = natsorted(numpys)
    images = []
    angles = []
    for folder in folders:
        for i, n in enumerate(natsorted(glob(f"{folder}/dataset*.npy"))):
            X = np.load(n)
            X = np.divide(X, 255.0)
            X = X[:, -300:, :, :]
            X = [cv2.resize(x, (200, 66), interpolation=cv2.INTER_AREA) for x in X]  # (width, height)
            images.append(X)
            print(f"Done {i + 1}", end="")
            print("\r", end="")
        with open(f"{folder}/data.txt") as datafile:
            y = []
            for line in datafile:
                y.append(float(line.split(",")[1]))
            for index in range(1, len(y) - 1):
                if y[index] == 0.0:
                    y[index] = (y[index - 1] + y[index + 1]) / 2.0
            angles.append(y)
    images = np.concatenate(images)
    angles = np.concatenate(angles)

    with h5py.File(dataset_path, "a") as f:
        try:
            del f["additional_angles"]
        except:
            pass
        grp = f.create_group("additional_angles")
        grp.create_dataset("X", data=images, dtype=np.float32)
        grp.create_dataset("y", data=angles, dtype=np.float32)


def mirror_images_and_angles(dataset_path):
    with h5py.File(dataset_path, "a") as f:

        for ds in list(f.keys()):
            if ds == "additional_angles":
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
    cv2.destroyAllWindows()


def filter_and_translate_32(dataset_path):
    with h5py.File(dataset_path, "a") as f:
        try:
            del f["filtered"]
        except:
            pass
        keys = [ds for ds in list(f.keys())
                if not ds.startswith("town07_clear_sunset_0")
                and not ds.startswith("town07_cloudy_night_0")
                and not ds.startswith("filtered")
                ]

        indices = []
        to_remove = []
        for ds in keys:
            for idx, angle in enumerate(f[ds]["y"]):
                if abs(angle) <= 0.01:
                    indices.append([ds, idx])
                    to_remove.append([ds, idx])
                else:
                    indices.append([ds, idx])
        print(len(indices))
        for i, x in enumerate(to_remove):
            indices.remove(x)
        print(len(indices))
        to_remove = []
        for ds, i in indices:
            if abs(f[ds]["y"][i]) < 0.05:
                to_remove.append([ds, i])
        for i, x in enumerate(to_remove):
            if i % 10:
                indices.remove(x)
        print(len(indices))
        to_remove = []
        for ds, i in indices:
            if abs(f[ds]["y"][i]) > 0.05 and abs(f[ds]["y"][i]) < 0.1:
                to_remove.append([ds, i])
        for i, x in enumerate(to_remove):
            if i % 5:
                indices.remove(x)
        print(len(indices))

        gen = H5DataGenerator(f, indices, 32, None)
        bins = plt.hist(gen.get_angles(), bins=np.arange(-1, 1.05, 0.05))
        # bins[0] = koliko brojeva je u binu
        # bins[1] = rubovi binova
        print(np.round(bins[1], 2), max(bins[0]), bins[1][np.where(bins[0] == max(bins[0]))])
        plt.grid()
        plt.show()
        indices = [[ds, idx, abs(f[ds]["y"][idx]) < 0.5] for ds, idx in indices]

        grp = f.create_group("filtered")
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
                f["filtered"]["X"][idx] = translate(image=f[tup[0]]["X"][tup[1]])["image"]
                delta = translate.translate_px["x"][0] / 10 * 0.25
                f["filtered"]["y"][idx] = f[tup[0]]["y"][tup[1]] + delta
            else:
                f["filtered"]["X"][idx] = f[tup[0]]["X"][tup[1]]
                f["filtered"]["y"][idx] = f[tup[0]]["y"][tup[1]]

        for idx, tup in enumerate(indices):
            f["filtered"]["X"][idx + len(indices)] = f[tup[0]]["X"][tup[1]]
            f["filtered"]["y"][idx + len(indices)] = f[tup[0]]["y"][tup[1]]
        print(f["filtered"]["X"].shape)
        print(f["filtered"]["y"].shape)

        y = [y[0] for y in f["filtered"]["y"][:]]
        del f["filtered"]["y"]
        f["filtered"].create_dataset("y", data=y, dtype=np.float32)
        plt.hist(y, bins=np.arange(-1, 1.05, 0.05))
        plt.show()


def filter_and_translate_200(dataset_path):
    with h5py.File(dataset_path, "a") as f:
        try:
            del f["filtered_2"]
        except:
            pass
        keys = [ds for ds in list(f.keys())
                if not ds.startswith("town07_clear_sunset_0")
                and not ds.startswith("town07_cloudy_night_0")
                and not ds.startswith("filtered")
                ]

        indices = []
        to_remove = []
        for ds in keys:
            for idx, angle in enumerate(f[ds]["y"]):
                if abs(angle) <= 0.01:
                    indices.append([ds, idx])
                    to_remove.append([ds, idx])
                else:
                    indices.append([ds, idx])
        print(len(indices))
        for i, x in enumerate(to_remove):
            indices.remove(x)
        print(len(indices))
        to_remove = []
        for ds, i in indices:
            if abs(f[ds]["y"][i]) <= 0.10:
                to_remove.append([ds, i])
        for i, x in enumerate(to_remove):
            if i % 5:
                indices.remove(x)
        print(len(indices))

        gen = H5DataGenerator(f, indices, 32, None)
        bins = plt.hist(gen.get_angles(), bins=np.arange(-1, 1.05, 0.05))
        # bins[0] = koliko brojeva je u binu
        # bins[1] = rubovi binova
        print(np.round(bins[1], 2), max(bins[0]), bins[1][np.where(bins[0] == max(bins[0]))])
        plt.show()
        indices = [[ds, idx, abs(f[ds]["y"][idx]) < 0.5] for ds, idx in indices]

        grp = f.create_group("filtered_2")
        grp.create_dataset("X", (2 * len(indices), 66, 200, 3), dtype=np.float32)
        grp.create_dataset("y", (2 * len(indices), 1), dtype=np.float32)

        x = np.arange(-60, 61, 1)
        y = norm.pdf(x, 0, 30)
        y = [y if x not in np.arange(-30, 31, 1) else 0 for x, y in zip(x, y)]

        for idx, tup in enumerate(indices):
            if tup[2]:
                temp = int(random.choices(x, weights=y, k=1)[0])
                translate = Affine(translate_px=dict(x=temp, y=0), p=1,
                                   mode=cv2.BORDER_REPLICATE)
                f["filtered_2"]["X"][idx] = translate(image=f[tup[0]]["X"][tup[1]])["image"]
                delta = translate.translate_px["x"][0] / 60 * 0.25
                f["filtered_2"]["y"][idx] = f[tup[0]]["y"][tup[1]] + delta
            else:
                f["filtered_2"]["X"][idx] = f[tup[0]]["X"][tup[1]]
                f["filtered_2"]["y"][idx] = f[tup[0]]["y"][tup[1]]

        for idx, tup in enumerate(indices):
            f["filtered_2"]["X"][idx + len(indices)] = f[tup[0]]["X"][tup[1]]
            f["filtered_2"]["y"][idx + len(indices)] = f[tup[0]]["y"][tup[1]]
        print(f["filtered_2"]["X"].shape)
        print(f["filtered_2"]["y"].shape)

        y = [y[0] for y in f["filtered_2"]["y"][:]]
        del f["filtered_2"]["y"]
        f["filtered_2"].create_dataset("y", data=y, dtype=np.float32)
        plt.hist(y, bins=np.arange(-1, 1.05, 0.05))
        plt.show()
        # for img, angle in zip(f["filtered_2"]["X"], f["filtered_2"]["y"]):
        #     cv2.imshow("frame", img)
        #     print(angle)
        #     cv2.waitKey(1)
        # cv2.destroyAllWindows()


def aug_angles(dataset_path):
    with h5py.File(dataset_path, "a") as f:
        try:
            del f["filtered_3"]
        except:
            pass
        keys = ["filtered"]

        indices = []
        to_aug = []
        for ds in keys:
            for idx, angle in enumerate(f[ds]["y"]):
                indices.append([ds, idx])
                if abs(angle) >= 0.15:
                    to_aug.append([ds, idx])
                if abs(angle) >= 0.25:
                    to_aug.append([ds, idx])
                if abs(angle) >= 0.3:
                    to_aug.append([ds, idx])
        print(len(indices))
        gen = H5DataGenerator(f, indices, 32, None)
        bins = plt.hist(gen.get_angles(), bins=np.arange(-1, 1.05, 0.05))
        # bins[0] = koliko brojeva je u binu
        # bins[1] = rubovi binova
        print(np.round(bins[1], 2), max(bins[0]), bins[1][np.where(bins[0] == max(bins[0]))])
        plt.show()

        grp = f.create_group("filtered_3")
        grp.create_dataset("X", (len(to_aug) + len(indices), 32, 32, 3), dtype=np.float32)
        grp.create_dataset("y", (len(to_aug) + len(indices), 1), dtype=np.float32)

        for idx, tup in enumerate(indices):
            f["filtered_3"]["X"][idx] = f[tup[0]]["X"][tup[1]]
            f["filtered_3"]["y"][idx] = f[tup[0]]["y"][tup[1]]

        aug = RandomBrightnessContrast(p=1, brightness_limit=0.2)
        for idx, tup in enumerate(to_aug):
            f["filtered_3"]["X"][idx + len(indices)] = aug(image=f[tup[0]]["X"][tup[1]])["image"]
            f["filtered_3"]["y"][idx + len(indices)] = f[tup[0]]["y"][tup[1]]
        print(f["filtered_3"]["X"].shape)
        print(f["filtered_3"]["y"].shape)

        y = [y[0] for y in f["filtered_3"]["y"][:]]
        del f["filtered_3"]["y"]
        f["filtered_3"].create_dataset("y", data=y, dtype=np.float32)
        plt.hist(y, bins=np.arange(-1, 1.05, 0.05))
        plt.show()


def filter_and_viewpoint_32(dataset_path):
    with h5py.File(dataset_path, "a") as f:
        try:
            del f["filtered_view"]
        except:
            pass
        keys = [ds for ds in list(f.keys())
                if not ds.startswith("town07_clear_sunset_0")
                and not ds.startswith("town07_cloudy_night_0")
                and not ds.startswith("filtered")
                ]

        indices = []
        to_remove = []
        for ds in keys:
            for idx, angle in enumerate(f[ds]["y"]):
                if abs(angle) <= 0.01:
                    indices.append([ds, idx])
                    to_remove.append([ds, idx])
                else:
                    indices.append([ds, idx])
        print(len(indices))
        for i, x in enumerate(to_remove):
            indices.remove(x)
        print(len(indices))
        to_remove = []
        for ds, i in indices:
            if abs(f[ds]["y"][i]) < 0.05:
                to_remove.append([ds, i])
        for i, x in enumerate(to_remove):
            if i % 10:
                indices.remove(x)
        print(len(indices))
        to_remove = []
        for ds, i in indices:
            if abs(f[ds]["y"][i]) > 0.05 and abs(f[ds]["y"][i]) < 0.1:
                to_remove.append([ds, i])
        for i, x in enumerate(to_remove):
            if i % 5:
                indices.remove(x)
        print(len(indices))

        gen = H5DataGenerator(f, indices, 32, None)
        bins = plt.hist(gen.get_angles(), bins=np.arange(-1, 1.05, 0.05))
        # bins[0] = koliko brojeva je u binu
        # bins[1] = rubovi binova
        print(np.round(bins[1], 2), max(bins[0]), bins[1][np.where(bins[0] == max(bins[0]))])
        plt.grid()
        plt.show()
        indices = [[ds, idx, abs(f[ds]["y"][idx]) < 0.5] for ds, idx in indices]

        grp = f.create_group("filtered_view")
        grp.create_dataset("X", (2 * len(indices), 32, 32, 3), dtype=np.float32)
        grp.create_dataset("y", (2 * len(indices), 1), dtype=np.float32)

        x = np.arange(-10, 11, 1)
        y = norm.pdf(x, 0, 5)
        y = [y if x not in np.arange(-4, 5, 1) else 0 for x, y in zip(x, y)]

        for idx, tup in enumerate(indices):
            if tup[2]:
                temp = int(random.choices(x, weights=y, k=1)[0])
                if temp > 0:
                    f["filtered_view"]["X"][idx] = cv2.resize(f[tup[0]]["X"][tup[1]][:, temp:, :],
                                                              (32, 32),
                                                              interpolation=cv2.INTER_AREA)
                else:
                    f["filtered_view"]["X"][idx] = cv2.resize(f[tup[0]]["X"][tup[1]][:, :temp, :],
                                                              (32, 32),
                                                              interpolation=cv2.INTER_AREA)
                delta = temp / 10 * 0.25
                f["filtered_view"]["y"][idx] = f[tup[0]]["y"][tup[1]] - delta
            else:
                f["filtered_view"]["X"][idx] = f[tup[0]]["X"][tup[1]]
                f["filtered_view"]["y"][idx] = f[tup[0]]["y"][tup[1]]

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

        # for img, angle in zip(f["filtered_view"]["X"], f["filtered_view"]["y"]):
        #     cv2.imshow("frame", img)
        #     print(f"{angle}\r")
        #     cv2.waitKey(0)



def filter_and_viewpoint_transform(dataset_path, width, height):
    with h5py.File(dataset_path, "a") as f:
        try:
            del f["filtered_view"], f["filtered"]
        except:
            pass
        keys = [ds for ds in list(f.keys())
                if not ds.startswith("town07_clear_sunset_0")
                and not ds.startswith("town07_cloudy_night_0")
                and not ds.startswith("filtered")
                ]

        indices = []
        to_remove = []
        for ds in keys:
            for idx, angle in enumerate(f[ds]["y"]):
                if abs(angle) <= 0.01:
                    indices.append([ds, idx])
                    to_remove.append([ds, idx])
                else:
                    indices.append([ds, idx])
        print(len(indices))
        for i, x in enumerate(to_remove):
            indices.remove(x)
        print(len(indices))
        to_remove = []
        for ds, i in indices:
            if abs(f[ds]["y"][i]) < 0.05:
                to_remove.append([ds, i])
        for i, x in enumerate(to_remove):
            if i % 10:
                indices.remove(x)
        print(len(indices))
        to_remove = []
        for ds, i in indices:
            if abs(f[ds]["y"][i]) > 0.05 and abs(f[ds]["y"][i]) < 0.1:
                to_remove.append([ds, i])
        for i, x in enumerate(to_remove):
            if i % 5:
                indices.remove(x)
        print(len(indices))

        gen = H5DataGenerator(f, indices, 32, None)
        bins = plt.hist(gen.get_angles(), bins=np.arange(-1, 1.05, 0.05))
        # bins[0] = koliko brojeva je u binu
        # bins[1] = rubovi binova
        print(np.round(bins[1], 2), max(bins[0]), bins[1][np.where(bins[0] == max(bins[0]))])
        plt.grid()
        plt.show()
        indices = [[ds, idx, abs(f[ds]["y"][idx]) < 0.5] for ds, idx in indices]

        grp = f.create_group("filtered_view")
        grp.create_dataset("X", (2 * len(indices), height, width, 3), dtype=np.float32)
        grp.create_dataset("y", (2 * len(indices), 1), dtype=np.float32)
        shift = ceil(0.3 * width)
        x = np.arange(-shift, shift + 1, 1)
        y = norm.pdf(x, 0, int(shift/2))
        y = [y if x not in np.arange(-int(shift/2)-1, int(shift/2), 1) else 0 for x, y in zip(x, y)]

        for idx, tup in enumerate(indices):
            if tup[2]:
                temp = int(random.choices(x, weights=y, k=1)[0])
                if temp > 0:
                    f["filtered_view"]["X"][idx] = cv2.resize(f[tup[0]]["X"][tup[1]][:, temp:, :],
                                                              (width, height),
                                                              interpolation=cv2.INTER_AREA)
                else:
                    f["filtered_view"]["X"][idx] = cv2.resize(f[tup[0]]["X"][tup[1]][:, :temp, :],
                                                              (width, height),
                                                              interpolation=cv2.INTER_AREA)
                delta = temp / shift * 0.25
                f["filtered_view"]["y"][idx] = f[tup[0]]["y"][tup[1]] - delta
            else:
                f["filtered_view"]["X"][idx] = f[tup[0]]["X"][tup[1]]
                f["filtered_view"]["y"][idx] = f[tup[0]]["y"][tup[1]]

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

        # for img, angle in zip(f["filtered_view"]["X"], f["filtered_view"]["y"]):
        #     cv2.imshow("frame", img)
        #     print(f"{angle}\r")
        #     cv2.waitKey(0)



dataset_path = "D:\\bbenja\\datasets\\carla_32x32.hdf5"
if __name__ == "__main__":
    #filter_and_viewpoint_transform(dataset_path, 200, 66)
    #add_angles("C:\\Users\\bbenja\\Desktop\\CARLA_0.9.13\\WindowsNoEditor\\PythonAPI\\examples\\angles\\*")
    #mirror_images_and_angles(dataset_path)

    #filter_and_translate_32(dataset_path)
    #aug_angles(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        #del f["filtered_2"]
        print([f.keys()])
    #     with h5py.File("D:\\bbenja\\datasets\\carla_200x66.hdf5", "r") as g:
    #         y = f["filtered_2"]["y"][:]
    #         plt.hist(y, bins=np.arange(-1, 1.05, 0.05), alpha=0.5, label="200")
    #
    #         y1 = g["filtered_2"]["y"][:]
    #         plt.hist(y1, bins=np.arange(-1, 1.05, 0.05), alpha=0.5, label="32")
    #         plt.legend()
    #         plt.show()
    #         print(len(y), len(y1))

        train_ds = [ds for ds in list(f.keys()) if ds == "filtered"]
        y = [f[ds]["y"][:].tolist() for ds in train_ds]
        y = np.concatenate(y)
        print(len(y))
        # for ds in train_ds:
        #     y = f[ds]["y"][:].tolist()
            # plt.figure(figsize=[6, 6])
            # plt.rcParams.update({'font.size': 14})
            # plt.rc('xtick', labelsize=12)
        plt.xticks(np.arange(-1, 1.2, 0.2))
        plt.grid()
        #plt.title(ds)
        plt.hist(y, bins=np.arange(-1, 1.05, 0.05))
        plt.xlabel("Vrijednost zakreta upravljača vozila")
        plt.ylabel("Broj slika")
        # plt.show()
        plt.savefig("sve.svg")
        print("Done.")

