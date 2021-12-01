import video_stream
from util.mask_beautifier import colorize_mask
from util.percentage import write_percentage_table_xml, get_full_percentage
from trt_engine import TrtModel
from decision import decision
import bl

import numpy as np
from PIL import Image
from buzzer import GPIO as JGPIO

from datetime import datetime
import time
import shutil
import signal
import sys
import os
import threading

SAVE_DIR = './save/'
DATA_DIR = './data'

# Requested Inference State from Environment Variables
DEBUG = True if os.environ['MODE'] == '1' or os.environ['MODE'] == '3' else False
SAVE = True if os.environ['MODE'] == '3' else False

if SAVE and not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

if DEBUG:
    import faulthandler
    faulthandler.enable()

# Initialize the labels and colors.
COLOR_SHIFT = os.environ['COLOR_SHIFT']
GPIO_PIN = os.environ['GPIO_PIN']
SIDEWALK_CLASS = os.environ['SIDEWALK_CLASS']

with open(os.path.join(DATA_DIR, 'labels.txt'), 'r') as f:
    labels = f.read().splitlines()

if DEBUG:
    colors = np.load(os.path.join(DATA_DIR, 'color_code.npy')).tolist()
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

# Buzzer
GPIO = JGPIO(31)

# Initialize Camera
LAUNCH_PIPELINE = video_stream.LAUNCH_PIPELINE if DEBUG else video_stream.LAUNCH_PIPELINE_WO_FILE
pipe = video_stream.start_gst(LAUNCH_PIPELINE)

# Bluetooth listener
bluetooth_sock = bl.BL()
BL_TOGGLE = (lambda : bluetooth_sock.state)
BL_DEV_OFF = (lambda : bluetooth_sock.close)

T_B = threading.Thread(target=bluetooth_sock.cycle, args=())
T_B.setDaemon(True)
T_B.start()

def load_model(model_path: str):
    model = TrtModel(model_path)
    shape = model.engine.get_binding_shape(0)

    return model, shape

def predict_trt(
        img,
        input_shape: tuple,
        engine: TrtModel=None,
    ):
    '''
    Predict a image using Tensorflow lite model.

    interpreter_dict = A dictionary from load_model_tflite()
    img = input image

    returns, mask output with input shape (height, width, 1)
    '''

    if img.shape != input_shape[1:3]:
        img = Image.fromarray(np.uint8(img)).resize((input_shape[2], input_shape[1]))
    
    img = np.asarray(img).astype(np.uint8)
    img = np.reshape(img, (1, input_shape[1], input_shape[2], input_shape[3]))
    img = engine(img / 255.0, 1)
    img = np.reshape(img, (1, input_shape[1], input_shape[2], len(labels)))
    img = np.argmax(img, axis=-1)
    img = np.reshape(img, (1, img.shape[1], img.shape[2], 1))
    return img[0]

def print_time(start_time, cap, pred, out, avg, ratio, sidewalk, bl_state):
    try:
        interval = 2
        while True:
            print('Start time : ', start_time())
            print('Updating in ', interval, ' seconds')
            print('Capture Time: ', cap())
            print('Prediction Time: ', pred())
            print('Post-Processing Time from Debugging: ', out())
            print('Average Time: ', avg())
            print('Sidewalk Ratio: ', ratio())
            print('Bluetooth Connected :', bl_state()())
            if sidewalk():
                print('SIDEWALK DETECTED')
            time.sleep(interval)
            os.system('clear')
    except KeyboardInterrupt:
        raise e

def main():
    global DEBUG
    global SAVE

    global pipe
    global BL_TOGGLE
    global BL_DEV_OFF

    # Initialization
    img = None
    model, shape = load_model(os.path.join(DATA_DIR, 'trt_model.engine'))

    print('Model has been loaded...')
        
    print('Initialized as ', 'Demo' if DEBUG else 'Batch')
    
    # DEBUG Variables
    capture_time = 0
    output_process_time = 0
    prediction_time = 0
    i = 0
    total = 0
    average_time = 0
    sidewalk_ratio = 0
    start_times = datetime.today()

    # State Variable (Bluetooth, Device state is defined globally.)
    is_sidewalk = False

    if DEBUG:
        t_p = threading.Thread(target=print_time, args=(
                        lambda : start_times,
                        lambda : capture_time,
                        lambda : prediction_time,
                        lambda : output_process_time,
                        lambda : average_time,
                        lambda : sidewalk_ratio,
                        lambda : is_sidewalk,
                        lambda : BL_TOGGLE))
        t_p.setDaemon(True)
        t_p.start()

    # Main Routine
    while not BL_DEV_OFF():
        while not BL_TOGGLE():
            GPIO.off()
            time.sleep(2)
        try:
            if DEBUG:
                # CAPTURE
                t = time.time()
                img = video_stream.get_frame(pipe).astype(np.uint8)
                shutil.copy('taken.jpg', '/var/www/html/taken.jpg')
                
                img = Image.fromarray(img)
                width, height = img.size
                if SAVE:
                    shutil.copy('taken.jpg', './save/cam_' + str(i) + '.jpg')

                img = img.crop((0, (height/2) + 75, width, height))
                width, height = img.size
                img = np.asarray(img)
                capture_time = time.time() - t
                
                # INFERENCE
                t = time.time()
                ret = predict_trt(img, shape, model)

                # DECISION
                sidewalk_ratio, is_sidewalk = decision(ret, SIDEWALK_CLASS)
                if is_sidewalk:
                    GPIO.on()
                else:
                    GPIO.off()
                prediction_time = time.time() - t

                # META-DATA POST-PROCESSING
                t = time.time()
                percentage = get_full_percentage(ret, len(labels))
                write_percentage_table_xml(percentage, labels, hex_colors)
                shutil.copy('percentage.xml', '/var/www/html/percentage.xml')

                # INFERENCE RESULT POST-PROCESSING
                ret = colorize_mask(ret, colors)
                ret = Image.fromarray(ret)
                ret = ret.resize((width, height), resample=Image.NEAREST)
                ret.save('/var/www/html/mask.png')

                if SAVE:
                    ret.save('./save/mask_' + str(i) + ('b-on' if is_sidewalk else 'b-off') + '.png')
                
                output_process_time = time.time() - t

                total = total + capture_time + prediction_time + output_process_time
                i = i + 1

                average_time = total / i
            else:
                img = video_stream.get_frame(pipe).astype(np.uint8)
                
                img = Image.fromarray(img)
                width, height = img.size
                img = img.crop((0, (height/2) + 75, width, height))
                width, height = img.size
                img = np.asarray(img)

                ret = predict_trt(img, shape, model)

                is_sidewalk = decision(ret, SIDEWALK_CLASS)
                if is_sidewalk:
                    GPIO.on()
                else:
                    GPIO.off()

        except Exception as e:
            print(e)
            interrupt(None, None)
            
if __name__ == "__main__":
    def interrupt(signal, frame):
        global pipe
        global bluetooth_sock

        print('Got interrupted')

        bluetooth_sock.close = True
        bluetooth_sock.cleanup()
        
        GPIO.off()
        GPIO.release()
        video_stream.release_pipe(pipe)
        sys.exit(0)

    signal.signal(signal.SIGINT, interrupt)
    main()
    interrupt(None, None)
