import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential

k = 16
filter_size = 3
NUMBER_OF_UNITS_PER_BLOCK = 4
utilize_bias = False
# w_init = tf.keras.initializers.HeUniform()
w_init = tf.keras.initializers.GlorotUniform()


def dense_unit(no_of_filters=k, f_size=filter_size):
    unit = Sequential([
        layers.BatchNormalization(),
        layers.ELU(),
        layers.Conv3D(no_of_filters, f_size, kernel_initializer=w_init, use_bias=utilize_bias, padding='same', dtype=tf.float64)
    ])
    return unit

def dense_block(inputs, num_units=NUMBER_OF_UNITS_PER_BLOCK):

    for i in range(num_units):
        dense_unit_output = dense_unit(k, filter_size)(inputs)
        inputs = tf.keras.layers.Concatenate(dtype='float64')([inputs, dense_unit_output])

    return inputs


def Generator(patch_size=64):

    inputs = tf.keras.layers.Input(shape=[patch_size, patch_size, patch_size, 1], dtype='float64')
    conv0 = layers.Conv3D(2 * k, filter_size, kernel_initializer=w_init, use_bias=utilize_bias,
                          padding='same', dtype=tf.float64)(inputs)
    dense0 = dense_block(conv0)
    concat = layers.Concatenate(dtype=tf.float64)([conv0, dense0])
    compress0 = layers.Conv3D(2 * k, 1, padding='same', dtype=tf.float64)(concat)

    dense1 = dense_block(compress0)
    concat = layers.Concatenate(dtype=tf.float64)([concat, dense1])
    compress1 = layers.Conv3D(2 * k, 1, padding='same', dtype=tf.float64)(concat)

    dense2 = dense_block(compress1)
    concat = layers.Concatenate(dtype=tf.float64)([concat, dense2])
    compress2 = layers.Conv3D(2 * k, 1, padding='same', dtype=tf.float64)(concat)

    dense3 = dense_block(compress2)
    concat = layers.Concatenate(dtype=tf.float64)([concat, dense3])

    reconstruction = layers.Conv3D(1, 1, padding='same', dtype=tf.float64)(concat)
    return keras.Model(inputs=inputs, outputs=reconstruction)




if __name__ == '__main__':
    g = Generator(64)
    g.summary()
    x = tf.random.normal([1, 64, 64, 64, 1])
    out = g(x)
    print(out.shape)
