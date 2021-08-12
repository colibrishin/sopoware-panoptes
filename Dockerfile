ARG DEBUG

FROM nvcr.io/nvidia/l4t-tensorflow:r32.5.0-tf2.3-py3 AS base
ARG DEBUG
ENV IS_DEBUG_MODE=$DEBUG
RUN echo $DEBUG
RUN apt-get update
RUN apt-get purge -y gnome* ubuntu-desktop
RUN apt-get autoremove
RUN apt-get clean
RUN apt-get upgrade -y 
ENV SOPOWARE_DIR=${HOME}/sopoware-panoptes
RUN mkdir $SOPOWARE_DIR
WORKDIR ${SOPOWARE_DIR}
COPY ./ ./
RUN chmod +x ./start.sh
RUN apt-get install -y gstreamer1.0-tools gstreamer1.0-alsa gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev
RUN apt-get install -y python3-gi python3-gst-1.0

FROM base AS runtype_1
RUN echo "Building Debug mode..."
# These are hardcoded as Pillow 8.3.1
RUN wget https://files.pythonhosted.org/packages/8f/7d/1e9c2d8989c209edfd10f878da1af956059a1caab498e5bc34fa11b83f71/Pillow-8.3.1.tar.gz
RUN tar -xvzf Pillow-8.3.1.tar.gz
WORKDIR ./Pillow-8.3.1
RUN pip3 install . 
WORKDIR ${SOPOWARE_DIR}
RUN rm -rf ./Pillow-8.3.1 Pillow-8.3.1.tar.gz
RUN apt-get install -y lighttpd iptables
# iptable setting. TODO
RUN cp ./demo/index.html /var/www/html/index.lighttpd.html

FROM base AS runtype_0
RUN echo "Building Batch mode..."

FROM runtype_${DEBUG} AS final
ENTRYPOINT $SOPOWARE_DIR/start.sh
