[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmheriyanto%2Fplay-with-tflite&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mheriyanto/play-with-coral/issues)
![GitHub contributors](https://img.shields.io/github/contributors/mheriyanto/play-with-coral)
![GitHub last commit](https://img.shields.io/github/last-commit/mheriyanto/play-with-coral)
![GitHub top language](https://img.shields.io/github/languages/top/mheriyanto/play-with-coral)
![GitHub language count](https://img.shields.io/github/languages/count/mheriyanto/play-with-coral)
![GitHub repo size](https://img.shields.io/github/repo-size/mheriyanto/play-with-coral)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/mheriyanto/play-with-coral)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat&logo=linkedin&colorB=555)](https://id.linkedin.com/in/mheriyanto)

# play-with-tflite
Repository for implementation Raspberry Pi + TensorFlow Lite to develop AI apps: **Vehicle detection & classification**.

## Tools
### Tested Hardware
+ RasberryPi 4 Model B [here](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/), RAM: **4 GB** and Processor **4-core @ 1.5 GHz** 
+ microSD Card **64 GB**
+ 5M USB Retractable Clip 120 Degrees WebCam Web Wide-angle Camera Laptop U7 Mini or [Raspi Camera](https://www.raspberrypi.org/documentation/hardware/camera/)

###  Tested Software
+ OS Raspbian 10 (Buster) 32 bit [**armv7l**](https://downloads.raspberrypi.org/raspios_armhf/images/raspios_armhf-2020-12-04/2020-12-02-raspios-buster-armhf.zip), install on RasberriPi 4
+ TensorFlow Lite library
+ Python min. ver. 3.5 (**3.7** recommended)

## Getting Started

+ Install TensorFlow Lite library (TensorFlow Lite APIs Python)

```console
$ pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
```

## Usage

## Image Classification
```console
$ git clone https://github.com/mheriyanto/play-with-tflite.git
$ cd play-with-tflite
$ cd examples
$ python3 classify.py --source /dev/video0 --model ../saved/models/mobilenet_v1_1.0_224_quant.tflite --labels ../saved/models/labels_mobilenet_quant_v1_224.txt

# Open on your browser and check http://0.0.0.0:5000/
```

## Object Detection
```console
$ python3 detection.py --source /dev/video0 --model ../saved/models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite --labels ../saved/models/coco_labels.txt

# Open on your browser and check http://0.0.0.0:5000/
```

<img src="https://github.com/mheriyanto/play-with-tflite/blob/main/docs/output.gif" width="640px" height="480px">

## Reference
+ TensorFlow Lite Python classification example with Pi Camera: [TensorFlow Lite example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi)
+ TensorFlow Lite Python object detection example with Pi Camera: [TensorFlow Lite example](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi)
+ Paper: [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)
+ Face Mask Detector (Tensorflow Lite): [GitHub - tanhouren](https://github.com/tanhouren/Face_mask_detector)
