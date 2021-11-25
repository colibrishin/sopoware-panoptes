#!/bin/bash
/etc/init.d/dbus start
bluetoothd -C &
sleep 1

hciconfig hci0 piscan
sdptool add SP

if [ "$MODE" = "1" ] || [ "$MODE" = "3" ]
then
	service lighttpd start
fi

cd $SOPOWARE_DIR/trt

if [ "$MODE" = "2" ]
then
	python3 ./record.py
else
	python3 ./trt_model.py
fi