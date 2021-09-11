#!/bin/bash

if [ "$IS_DEBUG_MODE" = "1" ]
then
	service lighttpd start
fi

cd $SOPOWARE_DIR/trt
python3 ./trt_model.py