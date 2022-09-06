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
import keras_tuner as kt
from models import *
from loaders_generators import H5DataGenerator, AUGMENTATIONS, \
    import_simulator_data, import_multiple_simulator_data, import_kelemen_data, get_indices, H5LSTMGenerator

pysical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(pysical_devices[0], True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

input_shape = (32, 32, 3)
batch_size = 512

def plot_history(history):
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Broj epoha")
    plt.ylabel("Vrijednost kriterijske funkcije")
    #plt.title("Train and Validation Losses Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()


    filename, extension = os.path.splitext("loss.svg")
    counter = 1
    path = ""
    while os.path.exists("loss.svg"):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    plt.savefig(path)
    plt.show()


def evaluate_model(model, gen):
    global batch_size
    test_predictions = model.predict(gen)
    test_predictions = np.asarray([t[0] for t in test_predictions])
    y_test = np.asarray(gen.get_angles())

    #print(f"NP MSE: {round(np.mean(np.square(y_test - test_predictions)), 2)}, "
    #      f"{round(np.sqrt(np.mean(np.square(y_test - test_predictions))), 2)}")
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
    #plt.title(f"{model.name}")
    plt.savefig(f"{model.name}.svg")


    out = []
    i = 0
    w = 10
    while i < len(test_predictions) - w + 1:
        out.append(sum(test_predictions[i: i + w]) / w)
        i += 1
    out.insert(0, test_predictions[8])
    out.insert(0, test_predictions[7])
    out.insert(0, test_predictions[6])
    out.insert(0, test_predictions[5])
    out.insert(0, test_predictions[4])
    out.insert(0, test_predictions[3])
    out.insert(0, test_predictions[2])
    out.insert(0, test_predictions[1])
    out.insert(0, test_predictions[0])
    ax.plot(t, out, label="Smooth")
    print(f"Smooth MSE: {mse(y_test, out)}")
    plt.show()

    # results = []
    # for i in range(len(y_test)):
    #     results.append((test_predictions[i], y_test[i]))
    # i = 0
    # while os.path.exists(f"saves/{model.name}_val_fold_{i}.npy"):
    #     i += 1
    # np.save(f"saves/{model.name}_val_fold_{i}", results)
    return mse(y_test, test_predictions)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr=1e-7, verbose=1, mode='min')
tb = TensorBoard(log_dir="./logs/{model.name}")
checkpoint = ModelCheckpoint('saves/swin_model{epoch:02d}_{val_loss:.2f}.h5', save_freq='epoch')
checkpoint_best = ModelCheckpoint(
    filepath='saves/best_swin__{epoch:02d}_{val_loss:.2f}.h5',
    save_weights_only=False,
    monitor='val_mean_squared_error',
    mode='min',
    verbose=0,
    save_best_only=True
)

def build_swin_v2():
    model = Sequential(name="swin_v2")
    model.add(PatchExtract(patch_size, input_shape=input_shape))
    model.add(PatchEmbedding(num_patch_x * num_patch_y, embed_dim))
    model.add(SwinBlock(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    ))
    model.add(SwinBlock(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=4,
        shift_size=2,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    ))
    model.add(SwinBlock(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=8,
        shift_size=4,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    ))
    model.add(PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim))
    model.add(keras.layers.LayerNormalization(epsilon=1e-5, name="norm10"))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(512, activation="elu", kernel_regularizer=regularizer))
    model.add(Dense(256, activation="elu", kernel_regularizer=regularizer))
    model.add(Dense(128, activation="elu", kernel_regularizer=regularizer))
    model.add(Dense(1, dtype='float32'))  # activation=tf.atan
    model.compile(loss=weighted_loss,
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    model.summary()
    return model


if __name__ == "__main__":
    with h5py.File("D:\\bbenja\\datasets\\carla_32x32.hdf5") as f:
        train_ds = [ds for ds in list(f.keys()) if ds == "filtered"]
        val_ds = [ds for ds in list(f.keys()) if ds.startswith("town07_clear_sunset_0")]
        train_i, val_i = get_indices(f, train_ds, val_ds)
        train_gen = H5DataGenerator(f, train_i, batch_size, AUGMENTATIONS, shuffle=True)
        val_gen = H5DataGenerator(f, val_i, batch_size, None)
        model = build_swin_v2()

        hist1, hist2 = [], []

        for i in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]:
            train_gen = H5DataGenerator(f, train_i, batch_size, AUGMENTATIONS, threshold=i, shuffle=True)
            history = model.fit(x=train_gen, epochs=1,
                                validation_data=val_gen,
                                callbacks=[reduce_lr, stop_early])
            hist1.append(history.history["loss"])
            hist2.append(history.history["val_loss"])
        history = model.fit(x=train_gen, epochs=39,
                            validation_data=val_gen,
                            callbacks=[reduce_lr, stop_early, tb])
        hist1.append(history.history["loss"])
        hist2.append(history.history["val_loss"])
        evaluate_model(model, val_gen)
        plot_history(history)
        hist1 = np.concatenate(hist1)
        hist2 = np.concatenate(hist2)
        plt.plot(hist1)
        plt.plot(hist2)
        plt.show()
        model.save("test")


    # with h5py.File("D:\\bbenja\\datasets\\metadrive_32x32.hdf5") as f:
    #     train_ds = [ds for ds in list(f.keys()) if ds == "filtered"]
    #     val_ds = [ds for ds in list(f.keys()) if ds.startswith("0_1_")]
    #     train_i, val_i = get_indices(f, train_ds, val_ds)
    #     train_gen = H5DataGenerator(f, train_i, batch_size, AUGMENTATIONS, shuffle=True)
    #     val_gen = H5DataGenerator(f, val_i, batch_size, None)
    #     model = build_swin_v2()
    #
    #     for i in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]:
    #             train_gen = H5DataGenerator(f, train_i, batch_size, AUGMENTATIONS, threshold=i, shuffle=True)
    #             history = model.fit(x=train_gen, epochs=1,
    #                                 validation_data=val_gen,
    #                                 callbacks=[reduce_lr, stop_early])
    #     train_gen = H5DataGenerator(f, train_i, batch_size, AUGMENTATIONS, shuffle=True)
    #     history = model.fit(x=train_gen, epochs=130,
    #                         validation_data=val_gen,
    #                         callbacks=[reduce_lr])  # , stop_early, tb])
    #     evaluate_model(model, val_gen)
    #     plot_history(history)


