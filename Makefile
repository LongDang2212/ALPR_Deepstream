################################################################################
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

APP:= deepstream-ALPR
CC = g++
TARGET_DEVICE = $(shell g++ -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=5.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

SRCS:= main.cpp
SRCS+= nvdsinfer_wpod/nvds_parse_bbox_wpod.cpp

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 opencv4

OBJS:= $(SRCS:.cpp=.o)

CFLAGS+= -I./includes -I../../apps-common/includes -I../deepstream-app/ -DDS_VERSION_MINOR=1 -DDS_VERSION_MAJOR=5
CFLAGS+= -I/usr/local/cuda-10.2/include 
CFLAGS+= -I/opt/nvidia/deepstream/deepstream-5.0/sources/alpr_ds/nvdsinfer_wpod -I/usr/include/opencv2/
CFLAGS+= -I/usr/include/opencv4/ -I/usr/include/gstreamer-1.0/ -I/usr/include/glib-2.0/ -I/usr/lib/aarch64-linux-gnu/glib-2.0/include/
CFLAGS+= -I/opt/nvidia/deepstream/deepstream-5.0/sources/gst-plugins/gst-nvinfer -I/opt/nvidia/deepstream/deepstream-5.0/sources/alpr_ds/include
CFLAGS+= `pkg-config --cflags $(PKGS)`
CFLAGS+= -fPIC -std=c++11

LIBS:= `pkg-config --libs $(PKGS)`

LIBS+= -L$(LIB_INSTALL_DIR) -L/opt/nvidia/deepstream/deepstream-5.0/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_smartrecord  -lnvds_infercustomparser -lnvbufsurface -lnvds_utils -lnvds_msgbroker -lnvds_inferutils -lrt -lm \
       -lgstrtspserver-1.0 -ldl -Wl,-rpath,$(LIB_INSTALL_DIR) -lopencv_core -lopencv_highgui -lopencv_imgproc 
LIBS+= -L/usr/local/cuda-10.2/lib64/ -lcudart -lopencv_imgcodecs
LIBS+= -L/usr/local/lib/ 

all: $(APP)


%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)

