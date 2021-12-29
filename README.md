/******************** Auto Licence Plate Recognition ******************/


/**** Build ****/

    - Edit directory to libs in Makefile: Gstreamer, opencv, nvinfer,...
    - Command: make clean && make all

/**** Config ****/
    - Edit 2 config file: pgie & sgie.
    - Edit confidence, NMS conf, batch size in nvdsinfer_wpod/nvds_parse_bbox_wpod.h
    - Edit directory to config file in main.cpp

/**** Run ****/
    - Source: uri like file://, rtsp://
    - Command: ./deepstream-ALPR <uri>
    - Output will be save in output/out.mp4
