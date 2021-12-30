# Auto Licence Plate Recognition in Deepstream

This repository implement an application to detect and recognition licence plate in NVIDIA Deepstream C++ SDK. Models used: WPOD + CRNN.
Tested on platform: Jetson NX + Deepstream 5.0.

## Requirements
-   CUDA 10.2
-   TensorRT 7.1.3
-   Deepstream 5.0
-   OpenCV
-   Gstreamer

## Config
- Edit CONF_THRESH, NMS_THRESH, batch size in nvdsinfer_wpod/nvds_parse_bbox_wpod.h
- Edit path to output file in main.cpp

## Build
    make all 
Note that you have to edit path to libaries in Makefile.

## Run

### Get engine 

-   WPOD engine can be get from my repository: [Wpod_TensorRT](https://github.com/LongDang2212/Wpod_TensorRT)
-   CRNN engine can be get from Tensorrtx repository: [Tensorrtx/CRNN](https://github.com/wang-xinyu/tensorrtx/tree/master/crnn)
-   Weight file: [Google Drive](https://drive.google.com/drive/folders/1aKAN6_3TYhAlivKjvX962NdS1uKKqyOu?usp=sharing)

### Config

-   Edit config in 2 file: config_pgie.txt & config.sgie.txt

### Run application
    ./deepstream-ALPR <uri>
<uri> likes file://, rtsp://,...
