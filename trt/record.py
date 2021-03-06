import time
import shutil
import video_stream 
import signal
import os

def main():
    pipe = video_stream.start_gst(video_stream.INDEX_CAPTURE_PIPELINE)
    
    if not os.path.exists('./capture'):
        os.mkdir('./capture')

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt as e:
            video_stream.release_pipe(pipe)
            raise Exception(e)
        except Exception as e:
            video_stream.release_pipe(pipe)
            raise Exception(e)

if __name__ == "__main__":
    main()
