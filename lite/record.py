import tensorflow as tf
import time
import shutil
import record_stream 
from label import label_pixel, label_name_n_code
from mask_beautifier import colorize_mask
from probability import write_probability_table_xml, get_full_probability
import signal

if __name__ == "__main__":
    pipe = record_stream.start_gst(record_stream.LAUNCH_PIPELINE)

    while True:
        try:
            print("Recording...")
            time.sleep(1)
        except KeyboardInterrupt:
            video_stream.release_pipe(pipe)
        except Exception as e:
            video_stream.release_pipe(pipe)
            raise Exception(e)
