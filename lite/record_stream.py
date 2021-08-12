import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import tensorflow as tf


LAUNCH_PIPELINE = 'nvarguscamerasrc sensor-id=0 name=cam0 aelock=true awblock=true wbmode=0 ! \
    video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)5/1 ! \
    nvvidconv flip-method=0 ! video/x-raw,width=640,height=480,format=BGRx ! nvvidconv ! \
    video/x-raw,width=640,height=480 ! videoconvert ! x265enc ! filesink location=record.mp4'

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
    return np.ndarray((cap.get_structure(0).get_value('height'), cap.get_structure(0).get_value('width'), 3), \
    buffer=buffer.extract_dup(0, buffer.get_size()), \
    dtype=np.uint8)

def release_pipe(pipe):
    pipe.set_state(Gst.State.NULL)

def flush_start(pipe):
    Gst.Element.send_event(pipe, Gst.Event.new_flush_start())

def flush_stop(pipe):
    Gst.Element.send_event(pipe, Gst.Event.new_flush_stop(True))
