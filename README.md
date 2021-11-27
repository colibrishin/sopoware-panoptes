
# sopoware-panoptes
#### Team Sopoware
###### Yeong-chul Park, Beomsik Shin, Jewoong Yoon
###### Dept of Computer Engineering, Daejeon University

## What is this?
This repository is dedicated to our project, '_Edge AI applied sidewalk warning system for public scooter_', includes scripts for training a deep learning model and deploying and running it in Jetson Nano. the project is aimed to prevent a shared kickboard driving on a sidewalk by alerting a driver and using the semantic segmentation to determine if the shared kickboard is on the sidewalk.

## Requirements
### Hardware
* Nvidia Jetson Nano
* CSI Camera
* Buzzer
* Bluetooth Module
* microSD (64GB Recommended)

## Documentation
* [Labelme to VOC](https://colab.research.google.com/drive/1-gydoon3ROho8mKwXy_VkbxQ-SlxjN1J?usp=sharing)
* [Training](https://colab.research.google.com/drive/1rTYmXW5S9tPD-pBJHBiGGSeT_n3EQumC?usp=sharing)
* [Convert from Tenorflow Model to TensorRT](https://colab.research.google.com/drive/1Ow65KbqCK4A6_Znghwe02rgTau4tImsX?usp=sharing)

## Installation
Clone this repository
```
git clone https://github.com/colibrishin/sopoware-panoptes.git
```
Place the converted TensorRT engine into the trt/
```
cp [converted TensorRT engine] [cloned repository directory]/trt/data/trt_model.engine
```
If image will be running in debug mode, label of Dataset and palette color code is required. Check Labelme To VOC for more detail.
```
cp [Dataset label] [cloned repository directory]/trt/data/labels.txt
cp [Color code npy] [cloned repository directory]/trt/data/color_codes.npy
```
Build the image (On default, image will be built as debug mode.)
```
chmod +x build.sh
sh build.sh
```
Start the container
```
sudo docker run --ipc host --privileged --rm -it -d \
                --runtime nvidia --net=host -v /tmp/argus_socket:/tmp/argus_socket \
                -v /sys:/sys -v /dev/bus/usb:/dev/bus/usb -v /var/run/dbus:/var/run/dbus \
                -v /var/lib/bluetooth:/var/lib/bluetooth \
                --device /dev/gpiochip0:/dev/gpiochip0 --device /dev/gpiochip1:/dev/gpiochip1 \
                --cap-add=SYS_ADMIN --group-add $(cut -d: -f3 < <(getent group gpio)) \
                sopoware-panoptes
```
If it's working correctly and built as debug mode, you can monitor the model prediction by accessing the device IP address on port 80.

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
> A. Howard, M. Sandler, G. Chu, L. Chen, Bo Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Quoc V. Le, H. Ada, "Searching for MobileNetV3," arXiv:1905.02244 [cs.CV] 

