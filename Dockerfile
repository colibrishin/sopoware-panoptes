ARG DEBUG

FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3 AS base
ARG DEBUG
ENV IS_DEBUG_MODE=$DEBUG
RUN echo "MODE : $DEBUG"
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
RUN pip3 install bluepy

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
