import numpy as np
import tensorflow as tf
import tensorflow.keras.backend
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, GlobalAveragePooling1D, \
    GlobalAveragePooling2D, TimeDistributed, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow import tanh
from tensorflow import abs as tf_abs

mae_func = tf.keras.losses.MeanAbsoluteError()
optimizer = Adam(learning_rate=1e-3, clipnorm=1)
regularizer = l2(0.005)


def weighted_loss(y_true, y_pred):
    return mae_func(y_true, y_pred, tanh(tf_abs(y_true)) * 100 + 1)


input_shape = (32, 32, 3)
patch_size = (2, 2)  # 2-by-2 sized patches
dropout_rate = 0.05  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 256  # MLP layer size
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = input_shape[0]  # Initial image size
num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]



def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows


@tf.function
def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def get_config(self):
        config = super(DropPath, self).get_config().copy()
        config.update({
            "drop_prob": self.drop_prob
        })
        return config

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output


class WindowAttention(layers.Layer):
    def __init__(
            self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def get_config(self):
        config = super(WindowAttention, self).get_config().copy()
        config.update({
            "dim": self.dim,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "qkv_bias": self.qkv,
            "dropout_rate": self.dropout
        })
        return config

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
                2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
            name="rpbt"
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                    tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                    + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv


class SwinBlock(layers.Layer):
    def __init__(
            self,
            dim,
            num_patch,
            num_heads,
            window_size=7,
            shift_size=0,
            num_mlp=1024,
            qkv_bias=True,
            dropout_rate=0.0,
            **kwargs,
    ):
        super(SwinBlock, self).__init__(**kwargs)
        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")

        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ],
            name="mlp"
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x

    def get_config(self):
        config = super(SwinBlock, self).get_config().copy()
        config.update({
            "name": self.name,
            "dim": self.dim,
            "num_patch": self.num_patch,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "num_mlp": self.num_mlp,
            "qkv_bias": qkv_bias,
            "dropout_rate": dropout_rate
        })
        return config


class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]

    def get_config(self):
        config = super(PatchExtract, self).get_config()
        config.update({
            "patch_size": patch_size
        })
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)

    def get_config(self):
        config = super(PatchEmbedding, self).get_config().copy()
        config.update({
            "num_patch": self.num_patch,
            "embed_dim": embed_dim
        })
        return config


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchMerging, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)

    def get_config(self):
        config = super(PatchMerging, self).get_config().copy()
        config.update({
            "num_patch": self.num_patch,
            "embed_dim": embed_dim
        })
        return config


def build_swin_model_2_256():
    model = Sequential(name="2_256")
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
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    ))
    model.add(PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(256, activation="elu", kernel_regularizer=keras.regularizers.l2(0.005)))
    model.add(Dense(1, dtype='float32'))  # activation=tf.atan
    model.compile(loss=weighted_loss,
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def build_swin_model_2_128_128():
    model = Sequential(name="2_128_128")
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
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    ))
    model.add(PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation="elu", kernel_regularizer=keras.regularizers.l2(0.005)))
    model.add(Dense(128, activation="elu", kernel_regularizer=keras.regularizers.l2(0.005)))
    model.add(Dense(1, dtype='float32'))  # activation=tf.atan
    model.compile(loss=weighted_loss,
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def build_swin_model_2_64_32_128_256():
    model = Sequential(name="2_64_32_128_256")
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
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    ))
    model.add(PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation="elu", kernel_regularizer=keras.regularizers.l2(0.005)))
    model.add(Dense(32, activation="elu", kernel_regularizer=keras.regularizers.l2(0.005)))
    model.add(Dense(128, activation="elu", kernel_regularizer=keras.regularizers.l2(0.005)))
    model.add(Dense(256, activation="elu", kernel_regularizer=keras.regularizers.l2(0.005)))
    model.add(Dense(1, dtype='float32'))  # activation=tf.atan
    model.compile(loss=weighted_loss,
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def build_swin_model_2_512_256_128():
    model = Sequential(name="2_512_256_128")
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
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    ))
    model.add(PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(512, activation="elu", kernel_regularizer=regularizer))
    model.add(Dense(256, activation="elu", kernel_regularizer=regularizer))
    model.add(Dense(128, activation="elu", kernel_regularizer=regularizer))
    model.add(Dense(1, dtype='float32'))  # activation=tf.atan
    model.compile(loss=weighted_loss,
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def build_LSTM_multiple_input_model():
    model = Sequential(name="LSTM_multiple_input")

    model.add(TimeDistributed(Conv2D(24, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(2, 2),
                                     padding='valid'), input_shape=(5, 66, 200, 3)))
    model.add(TimeDistributed(Conv2D(36, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(2, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(48, (5, 5),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     strides=(2, 2),
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     padding='valid')))
    model.add(TimeDistributed(Conv2D(64, (3, 3),
                                     kernel_initializer='he_normal',
                                     activation='relu',
                                     padding='valid')))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=True, implementation=2))
    model.add(LSTM(64, return_sequences=True, implementation=2))
    model.add(LSTM(64, implementation=2))
    model.add(Dropout(0.2))
    model.add(Dense(128,
                    kernel_initializer='he_normal',
                    activation='relu',
                    kernel_regularizer=l2(1e-3)))
    model.add(Dropout(0.2))
    model.add(Dense(1,
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-3)))
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def build_pilotnet_model():
    model = Sequential(name="PilotNet")
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_regularizer=l2(1e-3)))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(1e-3)))
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(1e-3)))
    model.add(Dense(1))
    # model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

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
