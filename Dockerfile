# Arguments from docker-compose
ARG MODE
ARG COLOR_SHIFT
ARG GPIO_PIN
ARG SIDEWALK_CLASS

FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3 AS base
ARG MODE
ARG COLOR_SHIFT
ARG GPIO_PIN
ARG SIDEWALK_CLASS

ENV MODE=$MODE
ENV COLOR_SHIFT=$COLOR_SHIFT
ENV GPIO_PIN=$GPIO_PIN
ENV SIDEWALK_CLASS=$SIDEWALK_CLASS
RUN echo "MODE : $MODE"
RUN echo "COLOR_SHIFT : $COLOR_SHIFT"
RUN echo "GPIO PIN : $GPIO_PIN"
RUN echo "SIDEWALK_CLASS : $SIDEWALK_CLASS"

RUN apt-get update
RUN apt-get purge -y gnome* ubuntu-desktop
RUN apt-get autoremove
RUN apt-get clean
RUN apt-get upgrade -y 

# Pre-setting sopoware-panoptes
ENV SOPOWARE_DIR=/root/sopoware-panoptes/
RUN mkdir $SOPOWARE_DIR
WORKDIR ${SOPOWARE_DIR}
COPY ./ ./
RUN chmod +x ./trt/start.sh

# Install requirements
RUN apt-get install -y gstreamer1.0-tools gstreamer1.0-alsa gstreamer1.0-plugins-base gstreamer1.0-plugins-good \ 
gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev \
python3-gi python3-gst-1.0 python3-pil libglib2.0-dev git libbluetooth-dev

RUN pip3 install pybluez git+https://github.com/Heerpa/jetson-gpio 

FROM base AS runtype_0
# BATCH MODE
WORKDIR ${SOPOWARE_DIR}

FROM base AS runtype_1
# DEBUG MODE
WORKDIR ${SOPOWARE_DIR}
RUN apt-get install -y lighttpd iptables
RUN cp ./demo/index.html /var/www/html/index.lighttpd.html

FROM base AS runtype_2
# RECORD MODE
RUN mkdir $SOPOWARE_DIR/capture

FROM base AS runtype_3
# DEBUG + SAVE CAMERA & INFERENCE IMAGES MODE
WORKDIR ${SOPOWARE_DIR}
RUN apt-get install -y lighttpd iptables
RUN cp ./demo/index.html /var/www/html/index.lighttpd.html

FROM runtype_${MODE} AS final
ENTRYPOINT $SOPOWARE_DIR/trt/start.sh
