import os
from glob import glob
import cv2
import h5py
import keras.models
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.metrics import mse
import tensorflow as tf
import cv2 as cv
import time
from models import *
from loaders_generators import H5DataGenerator, AUGMENTATIONS, get_indices, H5LSTMGenerator

pysical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(pysical_devices[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

batch_size = 512
dataset_files = [h5py.File("carla_32x32.hdf5", "r"), h5py.File("md_32x32.hdf5", "r")]

def plot_history(history):
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Broj epoha")
    plt.ylabel("Vrijednost kriterijske funkcije")
    plt.legend()
    plt.grid()
    # plt.savefig()
    plt.show()


def evaluate_model(model, gen):
    global batch_size
    test_predictions = model.predict(gen)
    test_predictions = np.asarray([t[0] for t in test_predictions])
    y_test = np.asarray(gen.get_angles())
    #print(f"NP MSE: {round(np.mean(np.square(y_test - test_predictions)), 2)}")
    print(f"Keras MSE: {mse(y_test, test_predictions)}")
    loss, mae, MSE = model.evaluate(x=gen, verbose=0)
    print(f"EVAL: loss={loss}\n\tmae={mae}\n\tmse={MSE}\n")

    t = np.array(range(0, len(y_test)))
    fig, ax = plt.subplots()
    ax.plot(t, y_test, label="Stvarne vrijednosti")
    ax.plot(t, test_predictions, label="Izlaz modela", alpha=0.7)

    plt.legend()
    plt.xlabel('Broj ulaznog podatka')
    plt.ylabel('Vrijednost zakreta upravljača')
    plt.title(f"{model.name}")
    plt.savefig(f"{model.name}.svg")


    # out = []
    # i = 0
    # w = 10
    # while i < len(test_predictions) - w + 1:
    #     out.append(sum(test_predictions[i: i + w]) / w)
    #     i += 1
    # out.insert(0, test_predictions[8])
    # out.insert(0, test_predictions[7])
    # out.insert(0, test_predictions[6])
    # out.insert(0, test_predictions[5])
    # out.insert(0, test_predictions[4])
    # out.insert(0, test_predictions[3])
    # out.insert(0, test_predictions[2])
    # out.insert(0, test_predictions[1])
    # out.insert(0, test_predictions[0])
    # ax.plot(t, out, label="Smooth")
    # print(f"Smooth MSE: {mse(y_test, out)}")
    # plt.show()
    return mse(y_test, test_predictions)


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr=1e-7, verbose=1, mode='min')
tb = TensorBoard(log_dir='./logs/{model.name}')
checkpoint = ModelCheckpoint('saves/model{epoch:02d}_{val_loss:.2f}.h5', save_freq='epoch')


if __name__ == "__main__":
    for f in dataset_files:
        train_ds = [ds for ds in list(f.keys()) if ds == "filtered"]
        if f.filename.startswith("carla"):
            val_ds = [ds for ds in list(f.keys()) if ds.startswith("town07_clear_sunset_0")]
        else:
            val_ds = [ds for ds in list(f.keys()) if ds.startswith("0_1_")]
        train_i, val_i = get_indices(f, train_ds, val_ds)
        val_gen = H5DataGenerator(f, val_i, batch_size, None)
        models = [build_swin_model_2_64_32_128_256(), build_swin_model_2_512_256_128(), build_swin_model_2_256(),
                  build_swin_model_2_128_128()]

        for model in models:
            for i in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]:
                train_gen = H5DataGenerator(f, train_i, batch_size, AUGMENTATIONS, threshold=i, shuffle=True)
                history = model.fit(x=train_gen, epochs=1, validation_data=val_gen)

            history = model.fit(x=train_gen, epochs=189,
                                validation_data=val_gen,
                                callbacks=[reduce_lr, stop_early, tb])

            evaluate_model(model, val_gen)
            plot_history(history)
            model.save(f"{f.filename.split('_')[0]}/{model.name}")

