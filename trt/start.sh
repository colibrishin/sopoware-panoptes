#!/bin/bash
if [ "$IS_DEBUG_MODE" = "1" ]
then
	service lighttpd start
fi

cd $SOPOWARE_DIR/trt

if [ "$IS_DEBUG_MODE" = '2 ' ]
then
	python3 ./record.py
else
	python3 ./trt_model.py
fi