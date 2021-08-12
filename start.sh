#!/bin/bash

if [ "$IS_DEBUG_MODE" = "1" ]
then
	service lighttpd start
fi

cd $SOPOWARE_DIR/lite
python3 ./tflite_model.py
