import json
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.utils import shuffle
results = json.loads(pd.read_csv(r"C:\Users\Julio\OneDrive\Área de Trabalho\depth-sensing-algo\results_1809 (1).csv").to_json())

# Obtém dicionário com os resultados do treinamento
results_dict = results

# Salva custos, métricas e epocas em vetores 
custo = results_dict['loss'].values()
rec_mae = results_dict['rec_img_mae'].values()
#disp_mae = results_dict['disp_mae']
val_custo = results_dict['val_loss'].values()
val_rec_mae = results_dict['val_rec_img_mae'].values()
#val_disp_mae = results_dict['val_disp_mae']
#val_mae = results_dict['val_mae']

# Cria vetor de épocas
epocas = range(1, len(custo) + 1)

# Gráfico dos valores de custo
plt.plot(epocas, custo, 'b', label='Custo - treinamento')
plt.plot(epocas, val_custo, 'r', label='Custo - validação')
plt.title('Valor da função de custo – treinamento e validação')
plt.xlabel('Épocas')
plt.ylabel('Custo')
plt.legend()
plt.show()

# Gráfico dos valores da métrica
plt.plot(epocas, rec_mae, 'b', label='MAE - treinamento')
plt.plot(epocas, val_rec_mae, 'r', label='MAE - validação')
plt.title('Valor da métrica de reconstrução – treinamento e validação')
plt.xlabel('Épocas')
plt.ylabel('REC_MAE')
plt.legend()
plt.show()


import numpy as np
import tensorflow as tf
from tensorflow import keras
from glob import glob
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from calibrator import calibrate, undistort
import keras
import cv2
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

left_image_path = './pics/Final_pics/Final_pics/left'
right_image_path = './pics/Final_pics/Final_pics/right'

# Escolhe tipos de arquivos desejados
glob_left_imgs = os.path.join(left_image_path, '*.png')
glob_right_imgs = os.path.join(right_image_path, '*.png')

def load_img_dir(dir:str,ret, mtx, dist, rvecs, tvecs):
    return [tf.image.resize(undistort(ret, mtx, dist, rvecs, tvecs,(640, 360),img_to_array(load_img(file)))[:, :, :3], (360,640))/255. for file in glob(dir)[:20]]

def load_calib_dir(dir:str):
    return [cv2.imread(file) for file in glob(dir)]

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
import os
from glob import glob
import numpy as np
glob_left_imgs = os.path.join('./calib/left_cal',"*.jpeg")
glob_right_imgs = os.path.join(r"./calib/right_cal/","*.jpeg")

# Define tamanho do lote
batch_size = 20

# Dimensão desejada para as imagen
rec = Reconstructor(
    height=360, width=640)
def loss_erro_rec(y_true, y_pred):
    mask = tf.where(y_pred < 1e-08,  0., 1.)
    erro = 255*tf.reduce_mean(tf.square(mask*(y_true - y_pred)))
    return erro
reconstructed_model=keras.models.load_model("rna_stereo.h5", custom_objects={'Reconstructor': rec,'loss_erro_rec':loss_erro_rec})
# Calcula previsão da rede para o lote de validação gerado
datagen=batch_generator(left_imgs,right_imgs,(360,640),200,batch_size)
[left_img_batch, right_img_batch], [out_img_batch,__] = next(datagen) 
img_prev_batch,disp_prev_batch = reconstructed_model.predict([left_img_batch, right_img_batch])

# Seleciona exemplo para análise
index = 2
left_img = left_img_batch[index]
right_img = right_img_batch[index]
img_prev = img_prev_batch[index]
disp_prev = disp_prev_batch[index]

# Retira eixos dos exemplos
left_img = np.squeeze(left_img)
right_img = np.squeeze(right_img)
disp_prev = np.squeeze(disp_prev)

# Mostra resultados do exemplos slecionado
f, pos = plt.subplots(1, 3, figsize=(12, 12))
pos[0].imshow(right_img)
pos[1].imshow(img_prev)
pos[2].imshow(disp_prev, cmap='gray')
plt.show()