import tensorflow as tf
import time
import shutil
import video_stream
from label import label_pixel, label_name_n_code
from mask_beautifier import colorize_mask
from probability import write_probability_table_xml, get_full_probability
import signal

def mem_allocation():
    phy = tf.config.list_physical_devices('GPU')
    if phy:
        tf.config.experimental.set_virtual_device_configuration(
            phy[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=128)]
        )
    else:
        assert 'GPU Device not found'

def load_model(model_path: str):
    #mem_allocation()
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = tuple(input_details[0]['shape'])

    return {"input_shape" : input_shape, 
            "input_details" : input_details, 
            "output_details" : output_details, 
            "interpreter" : interpreter}

def predict_tflite(
        img,
        interpreter_dict=None
    ):
    '''
    Predict a image using Tensorflow lite model.

    interpreter_dict = A dictionary from load_model_tflite()
    img = input image

    returns, mask output with input shape (height, width, 1)
    '''
    if interpreter_dict is None:
        assert "Model is not specified"
    else:
        interpreter = interpreter_dict['interpreter']
        input_details = interpreter_dict['input_details']
        output_details = interpreter_dict['output_details']
        input_shape = interpreter_dict['input_shape']

    if img.shape != input_shape[1:3]:
        img = tf.image.resize(img, input_shape[1:3])
    if img.shape[-1] != input_shape[3]:
        assert 'Image channel(s) is not compatible'

    img = tf.reshape(img, (1, input_shape[1], input_shape[2], input_shape[3]))
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    img = interpreter.get_tensor(output_details[0]['index'])
    img = tf.argmax(img, axis=-1)
    img = tf.expand_dims(img, axis=-1)
    return img[0]

def predict_file_tflite(
        img: str,
        interpreter_dict=None
    ):
    '''
    Predict a image using Tensorflow lite model.

    interpreter_dict = A dictionary from load_model_tflite()
    img = input image

    returns, mask output with input shape (height, width, 1)
    '''
    img = tf.io.read_file(img)
    img = tf.io.decode_jpeg(img, channels=3)

    return predict_tflite(img, interpreter_dict)

if __name__ == "__main__":
    mem_allocation()
    model = load_model('tflite.model')

    pipe = video_stream.start_gst(video_stream.LAUNCH_PIPELINE)

    img = None

    i = 0
    total = 0

    while True:
        try:
            t = time.time()

            img = video_stream.get_frame(pipe)
            #tf.keras.preprocessing.image.save_img('/var/www/html/taken.jpg', img)
            shutil.copy('taken.jpg', '/var/www/html/taken.jpg')

            conv = time.time() - t
            print('Capture Time:', conv)
            
            t = time.time()
            ret = predict_tflite(img, model)
            pred = time.time() - t
            print('Predict Time: ', pred)

            t = time.time()
            probability = get_full_probability(ret, 32)
            write_probability_table_xml(probability, label_name_n_code)
            shutil.copy('probability.xml', '/var/www/html/probability.xml')
            
            ret = colorize_mask(ret, label_pixel)
            tf.keras.preprocessing.image.save_img('/var/www/html/mask.png', ret)
            prep = time.time() - t
            print('Processing Time: ', prep)

            total = total + pred + conv + prep
            i = i + 1
            print('Average: ', total / i)
        except KeyboardInterrupt:
            video_stream.release_pipe(pipe)
        except Exception as e:
            video_stream.release_pipe(pipe)
            raise Exception(e)
