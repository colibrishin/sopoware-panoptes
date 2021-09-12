import tensorflow as tf
import time
import shutil
import video_stream
from label import label_pixel, label_name_n_code
from mask_beautifier import colorize_mask
from probability import write_probability_table_xml, get_full_probability
import signal
import os

def mem_allocation():
    phy = tf.config.list_physical_devices('GPU')
    if phy:
        tf.config.experimental.set_virtual_device_configuration(
            phy[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)]
        )
    else:
        assert 'GPU Device not found'

def load_model(model_path: str):
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

def main():
    '''
    Start the Capture-and-determine loop

    DEBUG=[1 or 0] to see the process of loop.
    '''

    mem_allocation()
    model = load_model('tflite.model')
    pipe = video_stream.start_gst(video_stream.LAUNCH_PIPELINE)

    DEBUG = True if os.getenv('IS_DEBUG_MODE') == '1' else False
    print(DEBUG)

    capture_time = 0
    output_process_time = 0
    prediction_time = 0

    img = None
    i = 0
    total = 0

    while True:
        try:
            if DEBUG is True:
                t = time.time()
                img = video_stream.get_frame(pipe)
                capture_time = time.time() - t
                print('Capture Time:', capture_time)
                shutil.copy('taken.jpg', '/var/www/html/taken.jpg')
                
                t = time.time()
                ret = predict_tflite(img, model)
                prediction_time = time.time() - t
                print('Predict Time: ', prediction_time)

                t = time.time()
                probability = get_full_probability(ret, len(label_name_n_code)) 
                write_probability_table_xml(probability, label_name_n_code)
                shutil.copy('probability.xml', '/var/www/html/probability.xml')

                ret = colorize_mask(ret, label_pixel)
                tf.keras.preprocessing.image.save_img('/var/www/html/mask.png', ret)
                output_process_time = time.time() - t
                print('Processing Time: ', output_process_time)

                total = total + capture_time + prediction_time + output_process_time
                i = i + 1
                print('Average: ', total / i)
            else:
                img = video_stream.get_frame(pipe)
                ret = predict_tflite(img, model)
            
            # TODO : Get a section of prediction, calculate the percentage of sidewalk and call buzzer function if its higher than .7

        except KeyboardInterrupt as e:
            video_stream.release_pipe(pipe)
            raise Exception(e)
        except Exception as e:
            video_stream.release_pipe(pipe)
            raise Exception(e)

if __name__ == "__main__":
    main()
