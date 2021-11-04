
# sopoware-panoptes
#### Team Sopoware
###### Yeong-chul Park, Beomsik Shin, Jewoong Yoon
###### Computer Enginnering, Daejeon University

## What is this?
This project is aiming to prevent a shared kickboard driving on a sidewalk by alerting a driver and using the semantic segmentation to determine if the shared kickboard is on the sidewalk.

This repository is dedicated to our project, '_A sidewalk warning system for shared kickboard using single board computer and deep learning_', includes scripts for training a deep learning model and deploying and running it in Jetson Nano.

## Requirements
### Necessary hardwares
* Nvidia Jetson Nano
* IMX219-77
* Buzzer (3.3V - 5V)
* Bluetooth Module
* microSD (At least 16GB ~)
### Softwares
* Tensorflow 2.5.0
* GStreamer 1.14.5
* Python 3.6.9
    * Pillow
    * python3-gi
    * python3-gst-1.0
* docker-compose 1.17.1 or later (See docker-compose.yml for more information)

## Documentation
* [Training](https://colab.research.google.com/drive/1rTYmXW5S9tPD-pBJHBiGGSeT_n3EQumC?usp=sharing)
* [Convert from Tenorflow Model to TensorRT](https://colab.research.google.com/drive/1Ow65KbqCK4A6_Znghwe02rgTau4tImsX?usp=sharing)

## Installation
Clone this repository
```
git clone https://github.com/colibrishin/sopoware-panoptes.git
```
Place the converted TensorRT engine into the trt/
```
cp [converted TensorRT engine] [cloned repository directory]/trt/trt_model.engine
```
Build the image (On default, image will be built as debug mode.)
```
chmod +x build.sh
sh build.sh
```
Start the container
```
sudo docker run -it -d --runtime nvidia -p 80:80 -v /tmp/argus_socket:/tmp/argus_socket -v /etc/udev/rules.d/:/etc/udev/rules.d -v /dev:/dev -v /sys/class/gpio:/sys/class/gpio -v /dev/gpiochip0:/dev/gpiochip0 -v /dev/gpiochip1:/dev/gpiochip1 sopoware-panoptes
```
If it's working correctly and if it is built as debug mode, you can retrieve the prediction monitoring by accessing the device IP address on port 80 in HTML format.

TODO : Monitoring Process

## Demo
TODO

## Cite

Labelme :
```
@misc{labelme2016,
  author =       {Kentaro Wada},
  title =        {{labelme: Image Polygonal Annotation with Python}},
  howpublished = {\url{https://github.com/wkentaro/labelme}},
  year =         {2016}
}
```

MobileNetV3 :
```
A. Howard, M. Sandler, G. Chu, L. Chen, Bo Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Quoc V. Le, H. Ada, "Searching for MobileNetV3," arXiv:1905.02244 [cs.CV] 
```
