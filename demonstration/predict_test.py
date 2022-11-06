import cv2
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import logging
from keras.applications.xception import Xception
import numpy as np
import glob
import matplotlib.pyplot as plt
import pdb
from datetime import datetime
import _thread
import os


def calibrate(frame_size: tuple = (640, 360), chessboard_size: tuple = (8, 6), image_dir: str = ''):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # os.chdir(dir)

    images = glob.glob(image_dir)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # show_image(img)
        # show_image(gray)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # cv2.imshow('img',gray)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            #cv2.drawChessboardCorners(img, chessboard_size, corners,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)
        else:
            raise "ImageError: Chessboard not found."

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs
    # cv2.destroyAllWindows()


def undistort(ret, mtx, dist, rvecs, tvecs, img_size, img):
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, img_size, 1, img_size)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst



cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = calibrate(
    image_dir='./calib/right_cal/*.jpeg')
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = calibrate(
    image_dir='./calib/left_cal/*.jpeg')


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
    height=360, width=640)


# Número de filtros básico
nF = 32


input_shape = (360, 640, 3)

base_model = Xception(include_top=False, input_shape=input_shape)
rnaCV = Model(inputs=base_model.input, outputs=base_model.layers[25].output)

# mark loaded layers as not trainable
for layer in rnaCV.layers:
    layer.trainable = False

input_left = tf.keras.layers.Input(shape=input_shape)
input_right = tf.keras.layers.Input(shape=input_shape)

# Extrai características
xL = rnaCV(inputs=input_left)
xR = rnaCV(inputs=input_right)

# Cocatena com características
ac = layers.Concatenate(axis=-1)([xL, xR])

# Aplica convolução
conv_size = 3
x2 = layers.Conv2DTranspose(
    64, (conv_size, conv_size), padding='same', activation='relu', strides=(2, 2))(ac)
x2 = layers.Conv2D(64, (conv_size, conv_size),
                   padding='same', activation='relu')(x2)
x3 = layers.Conv2DTranspose(
    32, (conv_size, conv_size), padding='same', activation='relu', strides=(2, 2))(x2)
x3 = layers.Conv2D(32, (conv_size, conv_size),
                   padding='same', activation='relu')(x3)

# Calcula da disparidade
x4 = layers.Conv2DTranspose(
    16, (conv_size, conv_size), padding='same', activation='relu', strides=(2, 2))(x3)
x4 = layers.Conv2D(16, (conv_size, conv_size),
                   padding='same', activation='relu')(x4)

x5 = layers.Conv2D(1, (1, 1), padding='same', activation='relu')(x4)

disp_layer = layers.Lambda(lambda x: tf.squeeze(x), name='disp')
disp = disp_layer(x5)

# Imagem da direita reconstruída usando a imagem esquerda e a disparidade
img_rec = rec(input_left, disp)

# Create the Keras model.
rna_stereo = keras.Model(
    inputs=[input_left, input_right], outputs=[img_rec, disp])

rna_stereo.load_weights('demonstration/rna_stereo_CVN_REC_weigths_close')
y = [*range(640)]
q= [[0],[0],[0]]
filters = np.ones((1, 4, 7, 1))/27.0
init = tf.constant_initializer(filters)
conv_mean = tf.keras.layers.Conv2D(
    1, (4, 7), padding='same', kernel_initializer=init, bias_initializer='zeros', trainable=False)

# Display Usage
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
thickness = 2


while (True):

    img = np.zeros((255, 255, 3), np.uint8)
    ret, frame = cap1.read()
    ret2, frame2 = cap2.read()
    right_imagem = undistort(
        ret_r, mtx_r, dist_r, rvecs_r, tvecs_r, (640, 360), frame2)
    left_imagem = undistort(
        ret_l, mtx_l, dist_l, rvecs_l, tvecs_l, (640, 360), frame)
    left_imagem = tf.image.resize(left_imagem, (360, 640))/255.
    right_imagem = tf.image.resize(right_imagem, (360, 640))/255.
    img_prev_batch, disp_prev_batch = rna_stereo.predict(
        [left_imagem[tf.newaxis, :, :, :], right_imagem[tf.newaxis, :, :, :]])

    # Realiza filtragem
    disp_mean = conv_mean(disp_prev_batch[tf.newaxis, :, :, tf.newaxis])
    cv2.imshow('left', frame)
    cv2.imshow('right', frame2)
    colormapped_image = cv2.applyColorMap(
        disp_mean[0, :, :, :].numpy().astype(np.uint8), cv2.COLORMAP_INFERNO)
    #disp_mean[0, :, :, 0]
    cv2.imshow('disp', colormapped_image)
    #cv2.imshow('rec', img_prev_batch[0, :, :, :])
    m = disp_mean[0, :, :, 0].numpy().max(axis=0, keepdims=True)
    q.insert(0, m[0])
    q.pop()
    curve = np.column_stack((y, np.mean(q, axis=0)))
    x = cv2.polylines(img, [curve.astype(np.int32)], False, (0, 255, 0))
    x = cv2.flip(x, 0)
    cv2.imshow("graph", x.astype(np.uint8))

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
