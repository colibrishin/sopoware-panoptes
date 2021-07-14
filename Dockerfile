ARG DEBUG

FROM nvcr.io/nvidia/l4t-tensorflow:r32.5.0-tf2.3-py3 AS base

RUN apt-get update
RUN apt-get purge -y gnome* ubuntu-desktop
RUN apt-get autoremove
RUN apt-get clean
RUN apt-get upgrade -y

ENV DIR=/root/sopoware-panoptes
RUN mkdir $DIR
WORKDIR ${DIR}
COPY ./ ./
RUN chmod +x ./start.sh

RUN apt-get install -y gstreamer1.0-tools gstreamer1.0-alsa gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev;
RUN apt-get install -y python3-gi python3-gst-1.0

FROM base AS RUNTYPE-1
RUN echo "Building Debug mode..."
RUN apt-get install -y lighttpd iptables
# iptables setting. TODO
RUN service lighttpd start
RUN cp ./demo/index.html /var/www/html/index.lighttpd.html

FROM base AS RUNTYPE-0
RUN echo "Building Batch mode..."

FROM RUNTYPE-${DEBUG} AS final
ENTRYPOINT $DIR/start.sh