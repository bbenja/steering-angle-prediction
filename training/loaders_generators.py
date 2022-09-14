from tensorflow.keras.utils import Sequence
import numpy as np
from random import choice
from albumentations import (Blur, ColorJitter, GaussianBlur, GaussNoise,
                            HueSaturationValue, MotionBlur,
                            MultiplicativeNoise,
                            RandomBrightnessContrast, RandomFog, RandomRain,
                            RandomShadow, RandomSnow, Sharpen, RGBShift)
from natsort import natsorted
from glob import glob
import cv2

p = 0.8
AUGMENTATIONS = [Blur(p=p, blur_limit=(3, 5)),
                 ColorJitter(p=p),
                 GaussianBlur(p=p),
                 GaussNoise(p=p, var_limit=0.001),
                 HueSaturationValue(p=p, sat_shift_limit=0.1, hue_shift_limit=0.1, val_shift_limit=0.1),
                 MotionBlur(p=p),
                 MultiplicativeNoise(p=p),
                 RandomBrightnessContrast(p=p, brightness_limit=0.15),
                 RandomFog(p=p),
                 RandomRain(p=p, blur_value=3),
                 RandomShadow(p=p),
                 RandomSnow(p=p, brightness_coeff=2),
                 Sharpen(p=p),
                 RGBShift(p=p, r_shift_limit=0.1, b_shift_limit=0.1, g_shift_limit=0.1)]


def import_multiple_simulator_data(folders):
    dataset = np.asarray([import_simulator_data(folder) for folder in folders], dtype=object)
    X = np.concatenate(dataset[:, 0], dtype="float32", axis=0)
    Y = np.concatenate(dataset[:, 1], dtype="float32", axis=0)
    return X, Y


def import_simulator_data(folder):
    try:
        X = np.load(f"{folder}_x.npy")
        ys = np.load(f"{folder}_y.npy")

    except:
        file_names = glob(f"{folder}/dataset*.npy")
        file_names = natsorted(file_names)
        X = []
        for f in file_names:
            images = np.load(f)
            for i, img in enumerate(images):
                img = cv2.resize(img[-300:], (200, 66), interpolation=cv2.INTER_AREA) / 255.0
                X.append(img)

        X = np.asarray(X, dtype="float32")
        np.save(f"{folder}_x", X)

        with open(f"{folder}/data.txt") as datafile:
            ys = []
            for line in datafile:
                ys.append(float(line.split(",")[1]))

            for index in range(1, len(ys) - 1):
                if ys[index] == 0.0:
                    ys[index] = (ys[index - 1] + ys[index + 1]) / 2.0
        ys = np.asarray(ys, dtype="float32")
        np.save(f"{folder}_y", ys)
    print(f"Loaded {folder}")

    return X, ys


def import_mk_data():
    file_names = glob("C:\\Users\\mk\\resized\\data-2022-02-16*x.npy")
    file_names = natsorted(file_names)
    arrays = [np.load(f) for f in file_names]
    images = np.concatenate(arrays)
    X = []
    for img in images:
        X.append(cv2.resize(img, (200, 66), interpolation=cv2.INTER_AREA) / 255.0)
    X = np.asarray(np.divide(images, 255.0))
    del images

    file_names = glob("C:\\Users\\mk\\resized\\data-2022-02-16*y.npy")
    file_names = natsorted(file_names)
    arrays = [np.load(f) for f in file_names]
    Y = np.concatenate(arrays)
    return X, Y


def get_indices(f, train_ds, val_ds):
    train_idx = []
    val_idx = []
    for ds in train_ds:
        indices = range(f[ds]["X"].shape[0])
        train_idx.append([[ds, i] for i in indices])
    for ds in val_ds:
        indices = range(f[ds]["X"].shape[0])
        val_idx.append([[ds, i] for i in indices])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)

    return train_idx, val_idx


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


class LSTMGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, n_steps):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.n_steps = n_steps

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        images = self.x[idx * self.batch_size:(idx + 1) * self.batch_size + self.n_steps]
        angles = self.y[idx * self.batch_size:(idx + 1) * self.batch_size + self.n_steps]

        batch_x, batch_y = list(), list()
        for x in range(self.n_steps, len(angles)):
            batch_x.append(images[x - self.n_steps:x])
            batch_y.append(angles[x])
        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)


class H5DataGenerator(Sequence):
    def __init__(self, file, indices, batch_size, augmentations, shuffle=False, threshold=0):
        self.file = file
        self.threshold = threshold
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.indices = [[ds, i] for ds, i in indices if abs(self.file[ds]["y"][int(i)]) > self.threshold]

    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = [self.file[d]['X'][int(i)] for d, i in batch] #cv2.cvtColor
        batch_y = [self.file[d]['y'][int(i)] for d, i in batch]

        if self.augmentations:
            aug = choice(self.augmentations)
            return np.stack([aug(image=x)["image"] for x in batch_x], axis=0), np.array(batch_y)
        else:
            return np.array(batch_x), np.array(batch_y)

    def get_angles(self):
        y = [self.file[ds]["y"][int(i)] for ds, i in self.indices]
        return np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class H5LSTMGenerator(Sequence):
    def __init__(self, file, indices, batch_size, n_steps=5, augmentations=None, shuffle=False, threshold=0):
        self.file = file
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.n_steps = n_steps
        self.shuffle = shuffle
        self.threshold = threshold
        self.indices = [[ds, i] for ds, i in indices if abs(self.file[ds]["y"][int(i)]) > self.threshold]

    def __len__(self):
        length = int(np.floor((len(self.indices)) / float(self.batch_size)))
        if len(self.indices) % float(self.batch_size) < self.n_steps:
            length -= 1
        if length < 1:
            raise Exception("Invalid number of indices (too short)")
        return length

    def __getitem__(self, idx):
        batch = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size + self.n_steps - 1]
        batch_x, batch_y = [], []
        for index in range(self.n_steps, self.batch_size + self.n_steps):
            seq = batch[index - self.n_steps:index]
            batch_x.append([self.file[d]['X'][int(i)] for d, i in seq])
            batch_y.append(self.file[batch[index - 1][0]]['y'][int(batch[index - 1][1])])
        if self.augmentations:
            aug = choice(self.augmentations)
            return np.stack([np.array([aug(image=x)["image"] for x in seq]) for seq in batch_x], axis=0), np.array(
                batch_y, dtype=np.float32)
        else:
            return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    def get_angles(self):
        y = [self.file[ds]["y"][int(i)] for ds, i in
             self.indices[self.n_steps - 1:len(self) * self.batch_size + self.n_steps - 1]]
        return y

    def on_epoch_end(self):
        if self.shuffle:
            n_indices = int(np.floor(len(self.indices) / self.n_steps) * self.n_steps)
            temp = np.array(self.indices[:n_indices])
            temp = np.reshape(temp, (-1, self.n_steps, 2))
            np.random.shuffle(temp)
            temp = np.reshape(temp, (-1, 2))
            self.indices = temp.tolist() + self.indices[n_indices:]
