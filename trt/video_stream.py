import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np

LAUNCH_PIPELINE = 'nvarguscamerasrc sensor-id=0 name=cam0  ! \
    video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)60/1 ! \
    nvvidconv flip-method=0 ! video/x-raw,width=640,height=480,format=BGRx ! videoconvert ! \
    video/x-raw,width=640,height=480,format=RGB ! videoconvert ! \
    tee name=t ! queue leaky=downstream max-size-buffers=1 ! appsink max-buffers=1 \
    t. ! queue leaky=downstream max-size-buffers=1 ! videoconvert ! jpegenc ! multifilesink location=taken.jpg'

LAUNCH_PIPELINE_WO_FILE = 'nvarguscamerasrc sensor-id=0 name=cam0  ! \
    video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)60/1 ! \
    nvvidconv flip-method=0 ! video/x-raw,width=640,height=480,format=BGRx ! videoconvert ! \
    video/x-raw,width=640,height=480,format=RGB ! videoconvert ! \
    queue leaky=downstream max-size-buffers=1 ! appsink max-buffers=1'

INDEX_CAPTURE_PIPELINE = 'nvarguscamerasrc sensor-id=0 name=cam0  ! \
    video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)60/1 ! \
    nvvidconv flip-method=0 ! video/x-raw,width=640,height=480,format=BGRx ! videoconvert ! \
    video/x-raw,width=640,height=480,format=RGB ! videoconvert ! \
    tee name=t ! queue leaky=downstream max-size-buffers=1 ! appsink max-buffers=1 \
    t. ! queue leaky=downstream max-size-buffers=1 ! videoconvert ! jpegenc ! multifilesink location=capture/taken-%00005d.jpg'

# This can be improved by using Gst.ElementFactory.

def start_gst(pipeline:str):
    Gst.init(None)
    pipe = Gst.parse_launch(pipeline)
    pipe.set_state(Gst.State.PLAYING)
    return pipe

def get_frame(pipe):
    sink = pipe.get_by_name('appsink0')
    sample = sink.emit('pull-sample')
    buffer = sample.get_buffer()
    cap = sample.get_caps()
    ret = np.ndarray((cap.get_structure(0).get_value('height'), cap.get_structure(0).get_value('width'), 3), \
                        buffer=buffer.extract_dup(0, buffer.get_size()), \
                        dtype=np.uint8)
    return ret

def release_pipe(pipe):
    pipe.set_state(Gst.State.NULL)
