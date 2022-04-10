
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img
import PIL
from tensorflow.keras import backend as K
plt.switch_backend('WebAgg')

f = np.array([[[0,1,0],[1,-4,1],[0,1,0]]],dtype=np.float32)
f = f.reshape(3,3,1,1)
f = tf.Variable(np.array(f,dtype=np.float32))


def _loss(y_true,y_pred):
  diff = y_true - y_pred
  
  _laplacian = K.conv2d(y_true, f, padding = 'same')
  _laplacian = K.abs(_laplacian)
  _max = K.max(_laplacian)
  _laplacian = _laplacian//_max
  diff = tf.math.multiply(diff,_laplacian)

  mse = (K.mean(K.square(diff)))
  return mse


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths):
        self.batch_size = batch_size
        self.img_size_out = img_size
        self.img_size_in = (img_size[0] // 2, img_size[1] // 2)
        self.input_img_paths = input_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size_in + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size_out + (3,), dtype="float32")

        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size_out)
            img_out = np.array(img)/255
            img_in = get_lowres_image(img,2)
            x[j] = img_in
            y[j] = img_out
        return x, y

def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,

    )


def feature(str_,network):
  subnet = keras.models.Sequential()
  for l in network.layers:
    subnet.add(l)
    if l.name == str_:
      break
  return subnet

def model():

    SRCNN = keras.models.Sequential()
    SRCNN.add(keras.layers.Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                         activation='relu', padding='same', use_bias=True, input_shape=(None, None, 1))) 
    SRCNN.add(keras.layers.Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                         activation='relu', padding='same', use_bias=True))
    SRCNN.add(keras.layers.Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                         activation='linear', padding='same', use_bias=True))

    SRCNN.compile(optimizer=keras.optimizers.Adam(), loss=_loss)
    return SRCNN

#_______________________training___________________

SR_Path = '/media/aro/New Volume/Super resolution dataset/set5/Train/'
Valid_path = '/media/aro/New Volume/Super resolution dataset/set5/Valid/'

input_SR_paths = [
        os.path.join(SR_Path, fname)
        for fname in os.listdir(SR_Path)
        if fname.endswith(".png")
    ]

Valid_SR_paths = [
        os.path.join(Valid_path, fname)
        for fname in os.listdir(Valid_path)
        if fname.endswith(".png")
    ]


img_size = (512,512)
train_bsize = 2

train_generator = OxfordPets(batch_size = train_bsize, img_size = img_size, input_img_paths = input_SR_paths)
Valid_generator = OxfordPets(batch_size = train_bsize, img_size = img_size, input_img_paths = Valid_SR_paths)

model=model()

save = ModelCheckpoint('/media/aro/New Volume/Super resolution dataset/set5/laplacian.hdf5', save_best_only=True)

if os.path.exists('/media/aro/New Volume/Super resolution dataset/set5/laplacian.hdf5'):
  model.load_weights('/media/aro/New Volume/Super resolution dataset/set5/laplacian.hdf5')

lr_rate = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate= 0.001,
    decay_steps= 10,
    decay_rate= 0.09)

model.compile(optimizer=keras.optimizers.Adam(), loss=_loss)

history = model.fit(train_generator,epochs=50,callbacks=[save], validation_data=Valid_generator)

plt.plot(history.history['loss'],label = "Training")
plt.plot(history.history['val_loss'],label = "Validation")
plt.title("weighted MSE loss trend")
plt.ylabel("MSE Value")
plt.xlabel("No. epoch")
plt.legend(loc = "upper left")
plt.show()

s = load_img("/media/aro/New Volume/Super resolution dataset/set5/results/laplacian/baboon.png", target_size=(512,512))

s = get_lowres_image(s,2)

s = np.expand_dims(s,0)
a = model.predict(s)
a = np.squeeze(a)
a1 = np.array(a,np.uint8)
a2 = np.clip(a,0,1)
a2 = a2 * 255.0
a2 = np.uint8(a2)

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(a1)
plt.subplot(122)
plt.imshow(a2)
plt.imsave('/media/aro/New Volume/Super resolution dataset/set5/results/laplacian/baboon_out.png',a2)
plt.show()