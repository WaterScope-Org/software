import sys

sys.path.append("/home/pi/.local/lib/python3.7/site-packages")
import numpy as np
import cv2
from PIL import Image
import picamera
import time
import io
import yaml
from set_picamera_gain import set_analog_gain, set_digital_gain

#import thread
from ctypes import *
keyboard = CDLL('lib/libarducam_keyboard.so')
arducam_vcm =CDLL('lib/libarducam_vcm.so')

arducam_vcm.vcm_init()
arducam_vcm.vcm_write(1000);


def update_camera_setting():
        with open('config_picamera.yaml') as config_file:
            config = yaml.load(config_file)
        # consistent imaging condition
        
        # Richard's library to set analog and digital gains
        set_analog_gain(camera, config['analog_gain'])
        set_digital_gain(camera, config['digital_gain'])
        camera.shutter_speed = 11000
        camera.saturation = config['saturation']
       # camera.exposure_mode = 'off'
        #camera.iso = 500

       # camera.led = False
        image_resolution = config['image_resolution']
        camera.resolution = image_resolution
        camera.awb_mode = config['awb_mode']
        time.sleep(0.5)
        camera.awb_gains = (config['red_gain'], config['blue_gain'])
        time.sleep(0.5)
        print(camera.awb_gains)
# Function to extract ROI
def extract_roi(image, x, y, width, height):
    return image.crop((x, y, x + width, y + height))

# Function to calculate average LAB values
def average_lab_values(image):
    img_np = np.array(image)
    lab_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2Lab)
    average_l = np.mean(lab_image[:, :, 0])
    average_a = np.mean(lab_image[:, :, 1])
    average_b = np.mean(lab_image[:, :, 2])
    return average_l, average_a, average_b

# ROI specifications (update these as needed)
x, y, width, height = 750, 450, 840, 1400

# Set up the camera
with picamera.PiCamera() as camera:
    update_camera_setting()

    camera.start_preview()
    time.sleep(2)  # Allow the camera to warm up
    
    arducam_vcm.vcm_write(1000);
    time.sleep(1)
    filename = "timelapse_data/chlorine/concentration_0.25.jpg"
    camera.capture(filename, format = 'jpeg', quality=100, bayer = False)
    image = Image.open(filename)
    # Extract the ROI
    roi = extract_roi(image, x, y, width, height)
    
    # Calculate average LAB values
    average_l, average_a, average_b = average_lab_values(roi)
    
    # Print the magenta levels (a* channel in LAB space represents green-magenta)
    print(f"Magenta Level (a* value): {average_a}")
    
    time.sleep(1)  # Wait for a second before capturing the next image

