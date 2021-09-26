import tensorflow as tf
from mobilenetv3.model import MobileNetV3, model_exception_check_weights

def predict(
        model : MobileNetV3,
        img_path : str
    ):
    '''
    Get Mask Precdiction from given model and images
    
    model = pre-compiled model
    img_path = path to image file
    '''
    try:
        img = tf.keras.preprocessing.image.load_img(img_path)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.resize(img, model.shape)
        img = tf.reshape(tensor=img, shape=(1, model.shape[0], model.shape[1], 3))

        model_exception_check_weights(model)
        pred = model.predict(img)

        pred = tf.argmax(pred, axis=-1)
        pred = tf.expand_dims(pred, axis=-1)
    except Exception as e:
        raise Exception('Failed to start prediction. ', e)

    return pred[0]


   


