import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras import models
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import os
import logging
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from calibrator import calibrate, undistort
import cv2
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(funcName)s - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.info(f"done importing... tf version:{tf.__version__}")
logging.info(f"{tf.config.list_physical_devices('GPU')}")

img_size = (360, 640)
# Define diretório onde se encontram as imagens
left_image_path = './Final_pics/left'
right_image_path = './Final_pics/right'

# Escolhe tipos de arquivos desejados
glob_left_imgs = os.path.join(left_image_path, '*.png')
glob_right_imgs = os.path.join(right_image_path, '*.png')

def load_img_dir(dir:str,ret, mtx, dist, rvecs, tvecs):
    return [tf.image.resize(undistort(ret, mtx, dist, rvecs, tvecs,(640, 360),img_to_array(load_img(file)))[:, :, :3], img_size)/255. for file in glob.glob(dir)]

def load_calib_dir(dir:str):
    return [cv2.imread(file) for file in glob.glob(dir)]

def call_calibrate():
    left_calib_imgs = load_calib_dir('./calib/left_cal/*.jpeg')
    right_calib_imgs = load_calib_dir('./calib/right_cal/*.jpeg')
    return calibrate(left_calib_imgs),calibrate(right_calib_imgs)
# Cria lista dos nomes dos arquivos
calib_l,calib_r= call_calibrate()
left_imgs = load_img_dir(glob_left_imgs,*calib_l)
right_imgs = load_img_dir(glob_right_imgs,*calib_r)

left_imgs, right_imgs = shuffle(left_imgs, right_imgs)
logging.info("loaded DS!")
# Imprime nomes e pat
split = 3744
# Conjunto de dados de treinamento
train_left_imgs = left_imgs[:split]
train_right_imgs = right_imgs[:split]

# Conjunto de dados de validação
val_left_imgs = left_imgs[split:]
val_right_imgs = right_imgs[split:]

m_train = len(train_left_imgs)
m_val = len(val_left_imgs)

logging.info("splitted DS!")

# Classe para reconstrutor
def batch_generator(left_imgs, right_imgs, img_size, m_exemplos, batchsize):
    # Inicializa loop infinito que termina no final do treinamento
    while True:

        # Loop para selecionar imagens de cada lote
        for start in range(0, m_exemplos, batchsize):

            # Inicializa lista de imagens com máscara e sem máscara
            batch_left_img, batch_right_img, batch_disp_img, batch_out_img = [], [], [], []

            end = min(start + batchsize, m_exemplos)
            for i in range(start, end):
                left_imagem = left_imgs[i]
                right_imagem = right_imgs[i]
                disp_imagem = np.zeros(img_size)
                # Adiciona imagem original e segmentada aos lotes
                batch_left_img.append(left_imagem)
                batch_right_img.append(right_imagem)
                batch_out_img.append(right_imagem)
                batch_disp_img.append(disp_imagem)

            yield [np.stack(batch_left_img, axis=0), np.stack(batch_right_img, axis=0)], [np.stack(batch_out_img, axis=0), np.stack(batch_disp_img, axis=0)]
            # Não tenho certeza se deixa o [] na saída

class Reconstructor(tf.keras.layers.Layer):
    def __init__(self, height=40, width=40, num_channels=3, name="rec_img"):
        super(Reconstructor, self).__init__(name=name)
        self.height = height
        self.width = width
        self.num_channels = num_channels

    def compute_output_shape(self, input_shape):
        return [None, self.height, self.width, 1]

    def get_config(self):
        return {
            'height': self.height,
            'width': self.width,
        }

    def build(self, input_shape):
        logging.info(
            f"Building Reconstruction Layer with input shape: {input_shape}")

    def repeat(self, x, n_repeats):
        # with tf.variable_scope('_repeat'):
        rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
        return tf.reshape(rep, [-1])

    def interpolate(self, im, x, y, _num_batch):
        _height_f = tf.cast(self.height, tf.float32)
        _width_f = tf.cast(self.width,  tf.float32)

        # handle both texture border types
        _edge_size = 1
        im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        x = x + _edge_size
        y = y + _edge_size

        x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

        x0_f = tf.floor(x)
        y0_f = tf.floor(y)
        x1_f = x0_f + 1

        x0 = tf.cast(x0_f, tf.int32)
        y0 = tf.cast(y0_f, tf.int32)
        x1 = tf.cast(tf.minimum(x1_f,  _width_f -
                     1 + 2 * _edge_size), tf.int32)

        dim2 = (self.width + 2 * _edge_size)
        dim1 = (self.width + 2 * _edge_size) * (self.height + 2 * _edge_size)
        base = self.repeat(tf.range(_num_batch) * dim1,
                           self.height * self.width)
        base_y0 = base + y0 * dim2
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        im_flat = tf.reshape(im, tf.stack([-1, self.num_channels]))

        pix_l = tf.gather(im_flat, idx_l)
        pix_r = tf.gather(im_flat, idx_r)

        weight_l = tf.expand_dims(x1_f - x, 1)
        weight_r = tf.expand_dims(x - x0_f, 1)

        return weight_l * pix_l + weight_r * pix_r

    def transform(self, input_images, x_offset, _num_batch):
        _height_f = tf.cast(self.height, tf.float32)
        _width_f = tf.cast(self.width,  tf.float32)

        x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  self.width),
                               tf.linspace(0.0, _height_f - 1.0, self.height))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
        y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

        x_t_flat = tf.reshape(x_t_flat, [-1])
        y_t_flat = tf.reshape(y_t_flat, [-1])

        x_t_flat = x_t_flat + tf.reshape(x_offset, [-1])

        input_transformed = self.interpolate(
            input_images, x_t_flat, y_t_flat, _num_batch)

        output = tf.reshape(input_transformed, tf.stack(
            [_num_batch, self.height, self.width, self.num_channels]))

        return output

    def call(self, input_images,  x_offset):
        _num_batch = tf.shape(input_images)[0]

        output = self.transform(input_images, x_offset, _num_batch)
        return output


def loss_erro_rec(y_true, y_pred):
    mask = tf.where(y_pred < 1e-08,  0., 1.)
    erro = 255*tf.reduce_mean(tf.square(mask*(y_true - y_pred)))
    return erro


rec = Reconstructor(
    height=img_size[0], width=img_size[1])


# Número de filtros básico
nF = 32

# Rede convolucional de extração de características


def rna_carac(input_shape, nF):
    x0 = layers.Input(shape=input_shape)

    # Calculo das características das imagens
    x1 = layers.Conv2D(nF, (5, 5), padding='same',
                       activation=layers.LeakyReLU())(x0)
    x2 = layers.Conv2D(nF, (5, 5), padding='same', use_bias=False)(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU()(x2)
    x2 = layers.MaxPool2D(2, 2)(x2)

    x3 = layers.Conv2D(nF*2, (5, 5), padding='same',
                       activation=layers.LeakyReLU())(x2)
    x4 = layers.Conv2D(nF*2, (5, 5), padding='same', use_bias=False)(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.LeakyReLU()(x4)
    x4 = layers.MaxPool2D(2, 2)(x4)

    x5 = layers.Conv2D(nF*4, (5, 5), padding='same',
                       activation=layers.LeakyReLU())(x4)
    x6 = layers.Conv2D(nF*4, (5, 5), padding='same', use_bias=False)(x5)
    x6 = layers.BatchNormalization()(x6)
    x6 = layers.LeakyReLU()(x6)
    x6 = layers.MaxPool2D(2, 2)(x6)

    x7 = layers.Conv2D(nF*4, (5, 5), padding='same',
                       activation=layers.LeakyReLU())(x6)
    x8 = layers.Conv2D(nF*4, (5, 5), padding='same', use_bias=False)(x7)
    x8 = layers.BatchNormalization()(x8)
    x8 = layers.LeakyReLU()(x8)

    # Cria modelo
    rna_carac = models.Model(x0, x8)

    return rna_carac


input_shape = (img_size[0], img_size[1], 3)
rnaCV = rna_carac(input_shape, nF)

rnaCV.summary()


"""## Rede completa"""


# Define entradas
input_left = tf.keras.layers.Input(shape=input_shape)
input_right = tf.keras.layers.Input(shape=input_shape)

# Extrai características
xL = rnaCV(input_left)
xR = rnaCV(input_right)

# Cocatena com características
ac = layers.Concatenate(axis=-1)([xL, xR])

# Aplica convolução
x2 = layers.Conv2DTranspose(
    64, (3, 3), padding='same', activation='relu', strides=(2, 2))(ac)
x2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x2)

x3 = layers.Conv2DTranspose(
    32, (3, 3), padding='same', activation='relu', strides=(2, 2))(x2)
x3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x3)

# Calcula da disparidade
x4 = layers.Conv2DTranspose(
    16, (3, 3), padding='same', activation='relu', strides=(2, 2))(x3)
x4 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x4)

x5 = layers.Conv2D(1, (1, 1), padding='same', activation='relu')(x4)

disp_layer = layers.Lambda(lambda x: tf.squeeze(x), name='disp')
disp = disp_layer(x5)

# Imagem da direita reconstruída usando a imagem esquerda e a disparidade
img_rec = rec(input_left, disp)

# Create the Keras model.
rna_stereo = keras.Model(
    inputs=[input_left, input_right], outputs=[img_rec, disp])


# Define otimizador Adam
adam = optimizers.Adam(learning_rate=0.001, decay=1e-03)

# Compilação do autoencoder
rna_stereo.compile(optimizer=adam,
                   loss={
                       'rec_img': loss_erro_rec,
                       'disp': 'mse'},
                   loss_weights={
                       'rec_img': 1.0,
                       'disp': 0.0},
                   metrics={
                       'rec_img': 'mae',
                       'disp': 'mae'})


# Importa callback para salvar modelo durante treinamento

# Define o callback para salvar os parâmetros
checkpointer = ModelCheckpoint(
    'rna_stereo_CVN_REC_weigths', verbose=1, save_best_only=True, save_weights_only=True)
batch_size = 16
train_steps = len(train_left_imgs) // batch_size
val_steps = len(val_left_imgs) // batch_size


results = rna_stereo.fit(batch_generator(train_left_imgs, train_right_imgs,img_size, m_train, batch_size),
    steps_per_epoch=train_steps,
    epochs=500,
    validation_data=batch_generator(val_left_imgs, val_right_imgs, img_size, m_val, batchsize=batch_size),
    callbacks=[checkpointer],
    validation_steps=val_steps,
    verbose=1)


# Restore the weights
# 'my_checkpoint_classic')
rna_stereo.load_weights('rna_stereo_CVN_REC_weigths')
rna_stereo.save("rna_stereo.h5")

# Obtém dicionário com os resultados do treinamento
results_dict = results.history
df_hist = pd.DataFrame(results_dict)
hist_json_file = 'history.json'
with open(hist_json_file, mode='w') as f:
    df_hist.to_json(f)
