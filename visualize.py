import tensorflow as tf
import matplotlib.pyplot as plt
from mobilenetv3.model import MobileNetV3
import imgviz
from PIL import Image

def display_sample(display_list):
   '''
   display_list - a list of length of 3, consist of with original image, true mask and mask prediction
   '''
   plt.figure(figsize=(18, 18))
 
   title = ['Input Image', 'True Mask', 'Predicted Mask']
 
   for i in range(len(display_list)):
       plt.subplot(1, len(display_list), i+1)
       plt.title(title[i])
       plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
       plt.axis('off')
   plt.show()
 
def show_predictions(model : MobileNetV3, dataset: tf.data.Dataset, num : int):
   '''
   model - MobileNetV3 model class
   dataset - a Tensorflow Dataset
   num - a integer value how many images function would take from datasets
   '''
   palette = imgviz.label_colormap(model.n_classes)[0:]

   for x, y in dataset.take(num):
       y_p = model.predict(x)
       y_p = output_conversion(y_p)
       y_p = tf.squeeze(y_p).numpy().astype('uint8')

       y_p = Image.fromarray(y_p).convert('P')
       y_p.putpalette(palette)
       y_p = y_p.convert("RGB")

       y_t = tf.squeeze(y[0]).numpy().astype('uint8')
       y_t = Image.fromarray(y_t).convert('P')
       y_t.putpalette(palette)
       y_t = y_t.convert('RGB')

       display_sample([x[0], y_t, y_p])
 
def output_conversion(output):
   '''
   output - a result of predict()
 
   this function converts predict() result to mask data
   '''
 
   output = tf.argmax(output, axis=-1)
   output = tf.expand_dims(output, axis=-1)
   return output[0]

