#!/bin/bash

git clone https://github.com/Heerpa/jetson-gpio
cd ./jetson-gpio
sudo groupadd -f -r gpio
sudo usermod -a -G gpio ${USER}

sudo cp lib/python/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

cd ../
rm -rf jetson-gpio

sudo docker-compose build
