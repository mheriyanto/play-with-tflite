[![HitCount](http://hits.dwyl.com/mheriyanto/play-with-coral.svg)](http://hits.dwyl.com/mheriyanto/play-with-coral)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mheriyanto/play-with-coral/issues)
![GitHub contributors](https://img.shields.io/github/contributors/mheriyanto/play-with-coral)
![GitHub last commit](https://img.shields.io/github/last-commit/mheriyanto/play-with-coral)
![GitHub top language](https://img.shields.io/github/languages/top/mheriyanto/play-with-coral)
![GitHub language count](https://img.shields.io/github/languages/count/mheriyanto/play-with-coral)
![GitHub repo size](https://img.shields.io/github/repo-size/mheriyanto/play-with-coral)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/mheriyanto/play-with-coral)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat&logo=linkedin&colorB=555)](https://id.linkedin.com/in/mheriyanto)

# play-with-coral
Repository for implementation Raspberry Pi + Google Edge TPU USB Accelerator to develop AI apps: **Face recognition**.

## Tools
### Tested Hardware
+ RasberryPi 4 Model B [here](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/), RAM: **4 GB** and Processor **4-core @ 1.5 GHz** 
+ microSD Card **64 GB**
+ Google Edge TPU USB Accelerator [here](https://coral.ai/products/accelerator)
+ 5M USB Retractable Clip 120 Degrees WebCam Web Wide-angle Camera Laptop U7 Mini or [Raspi Camera](https://www.raspberrypi.org/documentation/hardware/camera/)

###  Tested Software
+ OS Raspbian 10 (Buster) 32 bit [**armv7l**](https://downloads.raspberrypi.org/raspios_armhf/images/raspios_armhf-2020-12-04/2020-12-02-raspios-buster-armhf.zip), install on RaserriPi 4
+ Edge TPU runtime
+ TensorFlow Lite library
+ Python min. ver. 3.5 (**3.7** recommended)

## Getting Started

+ Install TensorFlow Lite library (TensorFlow Lite APIs Python)

```console
$ pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
```

## Usage

```console
$ git clone https://github.com/mheriyanto/play-with-coral.git
$ cd play-with-coral
$ python3 inference.py --source /dev/video0 --model weight/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite --labels weight/coco_labels.txt
```

## Credit to
+ Face recognition with Coral EdgeTPU Support based on MobileFacenet by
zye1996: https://github.com/zye1996/Mobilefacenet-TF2-coral_tpu
+ License Place Recognition with Google Coral TPU by zye1996: https://github.com/zye1996/edgetpu_ssd_lpr

## Reference
+ Coral's GitHub: https://github.com/google-coral
+ Paper: [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)
