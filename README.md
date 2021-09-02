
# sopoware-panoptes
#### Team Sopoware
###### Yeong-chul Park, Beomsik Shin, Jewoong Yoon
###### Computer Enginnering, Daejeon University

## What is this?
This repository is dedicated to our project, '_단일 보드 컴퓨터와 인공지능을 연계한 공유 킥보드 인도주행 경고 시스템_', includes scripts for training a deep learning model and deploying and running it in Jetson Nano. As a semantic segmentation model, MobileNetV3-Small is used.

## Requirments
* Tensorflow 2.4.1
* GStreamer 1.14.5
* Python 3.8.10
    * Pillow
    * python3-gi
    * python3-gst-1.0
* docker-compose 1.17.1 or later (See docker-compose.yml for more information)

## Documentation
TODO

## Installation
Clone this repository
```
git clone https://github.com/colibrishin/sopoware-panoptes.git
```
Place the quantized pre-trained Tensorflow lite model into the lite/
```
cp [quantized pre-trained Tensorflow lite model] [cloned repository directory]/lite/tflite.model
```
Build the image (On default, image will be built as debug mode.)
```
chmod +x build.sh
sh build.sh
```
Start the container
```
sudo docker run -it -d --runtime nvidia -p 80:80 -v /tmp/argus_socket:/tmp/argus_socket sopoware-panoptes
```
If it's working correctly and if it is built as debug mode, you can retrieve the prediction monitoring by accessing the device IP address on port 80 in HTML format.

TODO : Monitoring Process

## Demo
TODO

## Cite

CamVid:
> Gabriel, Jourdan & Shotton, Jamie & Fauqueur, Julien & Cipolla, Roberto. (2008). “Segmentation and Recognition Using Structure from Motion Point Clouds”. 5302. 44-57. 10.1007/978-3-540-88682-2_5. 

> Brostow, G., J. Fauqueur and R. Cipolla. “Semantic object classes in video: A high-definition ground truth database.” Pattern Recognit. Lett. 30 (2009): 88-97.

MobileNetV3 :
> A. Howard, M. Sandler, G. Chu, L. Chen, Bo Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Quoc V. Le, H. Ada, "Searching for MobileNetV3," arXiv:1905.02244 [cs.CV] 
