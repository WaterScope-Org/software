#!/bin/bash

git pull origin

sudo killall python3
sleep 5
cd /home/pi/WaterScope-Autofocus/firmware/
sudo avrdude -D -V -F -c arduino -p m328p -P /dev/ttyS0 -U flash:w:rtc_dual_36.hex:i
cd /home/pi/WaterScope-Autofocus/
rm password.txt