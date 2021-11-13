ARG DEBUG
ARG COLOR_SHIFT
ARG GPIO_PIN
ARG SIDEWALK_CLASS

FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3 AS base
ARG DEBUG
ARG COLOR_SHIFT
ARG GPIO_PIN
ARG SIDEWALK_CLASS

ENV IS_DEBUG_MODE=$DEBUG
ENV COLOR_SHIFT=$COLOR_SHIFT
ENV GPIO_PIN=$GPIO_PIN
ENV SIDEWALK_CLASS=$SIDEWALK_CLASS
RUN echo "MODE : $DEBUG"
RUN echo "COLOR_SHIFT : $COLOR_SHIFT"
RUN echo "GPIO PIN : $GPIO_PIN"
RUN echo "SIDEWALK_CLASS : $SIDEWALK_CLASS"

RUN apt-get update
RUN apt-get purge -y gnome* ubuntu-desktop
RUN apt-get autoremove
RUN apt-get clean
RUN apt-get upgrade -y 
ENV SOPOWARE_DIR=/root/sopoware-panoptes
RUN mkdir $SOPOWARE_DIR
WORKDIR ${SOPOWARE_DIR}
COPY ./ ./
RUN chmod +x ./trt/start.sh
RUN apt-get install -y gstreamer1.0-tools gstreamer1.0-alsa gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev
RUN apt-get install -y python3-gi python3-gst-1.0 python3-pil

RUN apt-get install -y libglib2.0-dev git
RUN apt-get install libbluetooth-dev
RUN pip3 install pybluez

RUN pip3 install git+https://github.com/Heerpa/jetson-gpio 

FROM base AS runtype_1
WORKDIR ${SOPOWARE_DIR}
RUN apt-get install -y lighttpd iptables
RUN cp ./demo/index.html /var/www/html/index.lighttpd.html

FROM base AS runtype_0

FROM base AS runtype_2
RUN mkdir $SOPOWARE_DIR/capture

FROM runtype_${DEBUG} AS final
ENTRYPOINT $SOPOWARE_DIR/trt/start.sh
