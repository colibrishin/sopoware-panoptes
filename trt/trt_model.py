import video_stream
from mask_beautifier import colorize_mask
from percentage import write_percentage_table_xml, get_full_percentage
from trt_engine import TrtModel

import numpy as np
from PIL import Image

import time
import shutil
import signal, sys
import os
import threading

# Initialize the labels and colors. 
COLOR_SHIFT = 0
with open('./labels.txt', 'r') as f:
    labels = f.read().splitlines()

colors = np.load('./color_code.npy').tolist()
def rgb_to_hex(rgb: list):
    code = []
    for i in rgb:
        for j in i:
            if j < 0 or j > 255:
                assert 'Color code out of range'
        code.append('#%02x%02x%02x' % i)
    return code
colors = [tuple(color) for color in colors]
hex_colors = rgb_to_hex(colors)

# Initialize Camera
pipe = video_stream.start_gst(video_stream.LAUNCH_PIPELINE)

def load_model(model_path: str):
    model = TrtModel(model_path)
    shape = model.engine.get_binding_shape(0)

    return model, shape

def predict_trt(
        img,
        input_shape,
        engine: TrtModel=None,
    ):
    '''
    Predict a image using Tensorflow lite model.

    interpreter_dict = A dictionary from load_model_tflite()
    img = input image

    returns, mask output with input shape (height, width, 1)
    '''

    if img.shape != input_shape[1:3]:
        img = Image.fromarray(img).resize((input_shape[1], input_shape[2]))

    img = np.reshape(img, (1, input_shape[1], input_shape[2], input_shape[3]))
    img = engine(img, 1)
    img = np.reshape(img, (1, input_shape[1], input_shape[2], len(labels)))
    img = np.argmax(img, axis=-1)
    img = np.reshape(img, (1, img.shape[1], img.shape[2], 1))
    return img[0]

def print_time(cap, pred, out, avg):
    try:
        interval = 2
        while True:
            print('Updating in ', interval, ' seconds')
            print('Capture Time: ', cap())
            print('Prediction Time: ', pred())
            print('Processing Time: ', out())
            print('Average Time: ', avg())
            time.sleep(interval)
            os.system('clear')
    except KeyboardInterrupt:
        raise e

def main():
    global pipe

    '''
    Start the Capture-and-determine loop

    DEBUG=[1 or 0] to see the process of loop.
    '''

    model, shape = load_model('./trt_model.engine')
    DEBUG = True if os.environ['IS_DEBUG_MODE'] == '1' else False

    print('Model has been loaded...')
    print('Initialized as ', 'Demo' if DEBUG else 'Batch')
    
    capture_time = 0
    output_process_time = 0
    prediction_time = 0
    i = 0
    total = 0
    average_time = 0

    t_p = threading.Thread(target=print_time, args=(
        lambda : capture_time,
        lambda : prediction_time,
        lambda : output_process_time,
        lambda : average_time))
    
    t_p.setDaemon(True)
    t_p.start()

    warm = True
    img = None

    while True:
        try:
            if DEBUG is True:
                t = time.time()
                img = video_stream.get_frame(pipe)
                
                img = Image.fromarray(img)
                width, height = img.size
                img = img.crop((0, (width/2) + 75, width, height))
                img = np.array(img)

                capture_time = time.time() - t
                shutil.copy('taken.jpg', '/var/www/html/taken.jpg')
                
                t = time.time()
                ret = predict_trt(img, shape, model)
                print(ret)
                prediction_time = time.time() - t

                t = time.time()
                percentage = get_full_percentage(ret, len(labels))
                write_percentage_table_xml(percentage, labels, hex_colors)
                shutil.copy('percentage.xml', '/var/www/html/percentage.xml')

                ret = colorize_mask(ret, colors)
                ret = Image.fromarray(ret)
                ret = ret.resize((width, height))
                ret.save('/var/www/html/mask.png')
                output_process_time = time.time() - t

                total = total + capture_time + prediction_time + output_process_time
                i = i + 1

                average_time = total / i
            else:
                img = video_stream.get_frame(pipe)
                ret = predict_trt(img, shape, model)
        except Exception as e:
            video_stream.release_pipe(pipe)
            raise e
            
        # TODO : Get a section of prediction, calculate the percentage of sidewalk and call buzzer function if its higher than .7
if __name__ == "__main__":
    def camera_killer(signal, frame):
        global pipe
        print('Got interrupted')

        video_stream.release_pipe(pipe)
        sys.exit(0)

    signal.signal(signal.SIGINT, camera_killer)
    main()
