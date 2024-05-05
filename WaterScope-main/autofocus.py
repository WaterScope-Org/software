import sys

sys.path.append("/home/pi/.local/lib/python3.7/site-packages")
import logging
import argparse
import csv
import time
import random
import os 
import json
import hashlib
import cv2
import pydbus
import socket
import traceback
import subprocess 
import uart
import random
import string
import re
import uuid
import hashlib
import csv
import pandas as pd
import math 

from picamera import PiCamera
from picamera.array import PiRGBArray
from set_picamera_gain import set_analog_gain, set_digital_gain
import yaml
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

import sys
import threading

#import thread
from ctypes import *
keyboard = CDLL('lib/libarducam_keyboard.so')
arducam_vcm =CDLL('lib/libarducam_vcm.so')


bus = pydbus.SystemBus()

sender_id = 0

package_size = 0


sample_ID=0
image_seq =0
image = []
physical_parameters = {}
camera = PiCamera()
folder_path=''
camera.resolution = (640, 480)
camera.rotation = 180
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
camera.awb_mode = 'off'
ROI = []
focus_box_ratio = 0.2
stream_resolution = (824, 616)
outstanding_data = None

print(camera.awb_gains)

def update_camera_setting():
        with open('config_picamera.yaml') as config_file:
            config = yaml.load(config_file)
        # consistent imaging condition
        
        # Richard's library to set analog and digital gains
        set_analog_gain(camera, config['analog_gain'])
        set_digital_gain(camera, config['digital_gain'])
        camera.shutter_speed = config['shutter_speed']
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
        
def initialise_data_folder():
        global folder_path,sample_ID
        if not os.path.exists('timelapse_data'):
            os.mkdir('timelapse_data')
        folder_path = 'timelapse_data/{}'.format(sample_ID)
        print(sample_ID)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
def autofocus():
        arducam_vcm.vcm_write(1000)
        
def define_ROI(box_ratio):
        # do some modification
        # the opencv size is (y,x)
        global image,ROI
        image_y, image_x = image.shape[:2]

        # a square from the centre of image
        box_size = int(image_x*box_ratio)
        roi_box = {
            'x1': int(image_x/2-box_size/2), 'y1':int(image_y/2-box_size/2), 
            'x2': int(image_x/2+box_size/2), 'y2':int(image_y/2+box_size/2)}
        
        # the rectangle affects the laplacian, draw it outside the ROI
        # draw the rectangle
        cv2.rectangle(
            image, 
            pt1=(roi_box['x1']-5, roi_box['y1']-5),
            pt2=(roi_box['x2']+5, roi_box['y2']+5), 
            color=(0,0,255),
            thickness=2)
        
        # crop the image
        ROI = image[roi_box['y1']: roi_box['y2'], roi_box['x1']:roi_box['x2']]
def defogging(connection='',sample_ID='', sample_comment=''):
    """ Heats the cartridge to reduce condensation buildup 
        New version automatically checks to see if there is any fog.
        
        This is achieved by checking whether the system is focused too high onto
        the coverslip or if it is close to the system average focal point.
        
        For new systems, the system will defog for 90 s for the first 10 runs,
        before switching over to the automatic detection.
        
        """
    global new_sample
    if(sample_comment!='nofog'):
        uart.send_serial("temp=95") # Increased temperature close to max (100) for faster defogging
                                    # Tg for PLA is ~60 C, Tm is ~170 C
                                    # Therefore risk of plastic softening if plastic actually reaches 100
                                    # However, heat should dissipate fast enough to avoid issue...
        with open(log_name, "a") as myfile:
        # now = datetime.now() # current date and time
            myfile.write("Defogging started at "+datetime.now().strftime("%m/%d/%Y %H:%M:%S")+"\n")
            myfile.close()

            
        # Check if database either doesn't exist, or has less than 10 entries
        if not os.path.exists("database.csv") or sum(1 for line in open('database.csv')) < 11:
            # In this case, this is the first 10 times this system has been used
            # We should therefore use a regular 90 s defogging cycle
            time.sleep(30)
            time.sleep(30)
            time.sleep(30)
            
            with open(log_name, "a") as myfile:
                myfile.write("Defogging finished at "+datetime.now().strftime("%m/%d/%Y %H:%M:%S")+"\n")
                myfile.close()
        
        # With the database, we can speed up the defogging procedure
        else:
            # Set initial values
            camera.resolution = (640, 480)
            i = 900
            arducam_vcm.vcm_write(i)
            time.sleep(0.2)
            rawCapture.truncate(0)

            fs = []
            zs = []
            t0 = time.time()
            ts = []

            

            # Load current average system focus
            df = pd.read_csv('database.csv')
            focus_sys_avg = np.mean(df["focus"].tail(10))
            
            # Define delta, (max focus deviation)
            delta = 45

            # Start cycling through focus positions whilst taking images
            for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                # Take image
                time.sleep(0.01)
                global image
                image = frame.array

                define_ROI(focus_box_ratio)

                # Calculate focus
                focus = variance_of_laplacian()

                # Print current focus
                fs.append(focus)
                zs.append(i)
                tf = time.time()
                t = tf-t0
                ts.append(t)
                print(str(i) + ',' + str(focus) + ',' + str(t))

                # Step the motor
                if(i<1000):
                    i=i+20 # we can use a relatively coarse step
                           # 20 cycles 100-700 in ~2.5 s
                           # 2.5 cycles 100-700 in ~10 s
                    arducam_vcm.vcm_write(i)

                rawCapture.truncate(0)
                image = 0

                # After a full 900 to 1000 range, check results and continue
                if i >=1000:
                    i = 100

                    # # sometimes it doesn't focus properly...
                    # but seems to fix itself after a while?
                    s = int((1000-900)/20) # no. of values per cycle
                    if  0.0 in fs[len(fs)-s:]:
                        print('bad focus')
                        #rawCapture.truncate(0)
                        #image = 0
                        #continue

                    # Check focus score magnitude is valid
                    # If score over all positions < 20
                    # Then v. likely to be no fog + no features on filter
                    if np.max(fs) < 20:
                        print('Defogging complete after %i s - no features' %t)
                        with open(log_name, "a") as myfile:
                            myfile.write("Defogging finished at "+datetime.now().strftime("%m/%d/%Y %H:%M:%S")+"\n")
                            myfile.close()
                        break

                    # Check to see if focus is in correct region
                    zmaxs = []
                    s = int((1000-900)/20) # no. of values per cycle
                        # extract the focal position of each recorded cycle
                    for j in range(len(ts)):
                        if j % s == 0:
                            lo = j
                            hi = j+s
                            x,y = ts[lo:hi],fs[lo:hi]

                            imax = np.argmax(y)
                            zmax = zs[imax]
                            zmaxs.append(zmax)

                        # Compare zmax with sys_avg
                    if len(zmaxs)>3:
                        # avg focus over 3 cycles
                        av_zmax = np.mean(zmaxs[-3:])

                        # check if current focus is below sys_avg + delta
                        # if it is, then system isn't focused on coverslip
                        # so defogging can likely stop
                        if av_zmax - focus_sys_avg < delta:
                            print('Defogging complete after %i s - No fog detected' %t)
                            with open(log_name, "a") as myfile:
                                myfile.write("Defogging finished at "+datetime.now().strftime("%m/%d/%Y %H:%M:%S")+"\n")
                                myfile.close()
                            break

                # set maximum cutoff time
                if t > 90:
                    print('Defogging complete after %i s - Maxed out time' %t)
                    with open(log_name, "a") as myfile:
                            myfile.write("Defogging finished at "+datetime.now().strftime("%m/%d/%Y %H:%M:%S")+"\n")
                            myfile.close()
                    break
    
def variance_of_laplacian():
            ''' focus calculation ''' 
            global image, ROI
            
            # compute the Laplacian of the image and then return the focus
            # measure, which is simply the variance of the Laplacian
            focus_value = cv2.Laplacian(ROI, cv2.CV_64F).var()
            #print(focus_value)
            focus_text = 'f: {:.2f}'.format(focus_value)#
            
            # CV font
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                image, focus_text,
                (int(image.shape[0]*0.1), int(image.shape[1]*0.1)), 
                font, 2, (0, 0, 255))
            return focus_value
            
            
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

def Chlorine_analysis(filename):
    # ROI specifications
    x, y, width, height = 910, 50, 750, 1300

    # Open the image using PIL
    image = Image.open(filename)
    # Convert the image to numpy array for processing
    image_np = np.array(image)

    # Crop to the ROI
    roi = image_np[y:y+height, x:x+width]

    # Convert ROI to HSV (using OpenCV, as PIL does not support HSV conversion directly)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # Calculate the average color
    h_avg = np.mean(hsv_roi[:,:,0])
    s_avg = np.mean(hsv_roi[:,:,1])
    v_avg = np.mean(hsv_roi[:,:,2])


    cl = 0.0421*math.exp(0.0222*s_avg)
    cl = min(cl, 3)
    if((cl < 0.25) or (h_avg < 120)):
        cl = 0
        
        
    # Position and settings for the text
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", size=160)  # Adjust the font size and path to the font file
    text = f"Chlorine: \n {cl:.1f} mg/L"
    draw = ImageDraw.Draw(image)
    draw.text((x, y), text, fill="white", font=font)
    
    # Draw the bounding box around the ROI
    draw.rectangle([x, y, x + width, y + height], outline="red", width=3)
    
    # Save the modified image
    new_filename = filename.replace('.jpg', '_result.jpg')
    image.save(new_filename, quality=40, optimize=True)
    print("Chlorine level")
    print(cl)
    
    return cl
    
def analysis_result():
    global log_name,filename,new_sample,sample_ID,image_seq,chlorine,physical_parameters
    uart.send_serial("led_on")
    time.sleep(2)
    uart.send_serial("temp=0")
    time.sleep(0.5)
    uart.send_serial("incubator_off")
    time.sleep(0.5)
    print('Capture an image')
    result = {}
    result_db = {}
    flagged = 0
    chlorine_level = 0.0
    with open(log_name, "a") as myfile:
       # now = datetime.now() # current date and time
        myfile.write("Sample ID: "+str(sample_ID)+" at "+datetime.now().strftime("%m/%d/%Y %H:%M:%S")+"\n")
        myfile.close()

    [current_z, timestamp] = capture_image()

    with open(log_name, "a") as myfile:
        # now = datetime.now() # current date and time
        myfile.write("Image captured at " + datetime.now().strftime("%m/%d/%Y %H:%M:%S") + "\n")
        myfile.close()
    print(filename)
    #uart.send_serial("line1=Insert ")
    #time.sleep(0.1)
    #uart.send_serial("line2=next sample")
    
    if new_sample == 0:
        with open(log_name, "a") as myfile:
            # now = datetime.now() # current date and time
            myfile.write("Running ML at " + datetime.now().strftime("%m/%d/%Y %H:%M:%S") + "\n")
            myfile.close()
        try:
            if(chlorine==0):
                ML_analysis(filename)
                print("Running ML")
            else:
                chlorine_level = Chlorine_analysis(filename)
                print("Running Chlorine")
        except BaseException as e:
            with open(log_name, "a") as myfile:
                # now = datetime.now() # current date and time
                myfile.write("Exception occured with ML"+"\n")
                myfile.write(str(e)+"\n")
                myfile.close()
                
            uart.send_serial('line1=Analysis failed')
            time.sleep(0.5)
            uart.send_serial('line2=please restart')
            time.sleep(2)
            uart.send_serial('incubator_37')
    if(chlorine==0):
    
        while True:
    
        # keep checking whether result it out
            if os.path.exists(filename.replace('.jpg', '_result.txt')):
                with open(log_name, "a") as myfile:
                    # now = datetime.now() # current date and time
                    myfile.write("ML completed at " + datetime.now().strftime("%m/%d/%Y %H:%M:%S") + "\n")
                    myfile.close()
                args.preview = filename.replace('.jpg', '_result.jpg')

                with open(filename.replace('.jpg', '_result.txt')) as file:
                    lines = file.readlines()
                    result_db["unet_eco"] = re.findall("\d+", lines[0])[0]
                    result_db["unet_col"] = re.findall("\d+", lines[1])[0]
                    result_db["yolo_eco"] = re.findall("\d+", lines[2])[0]
                    result_db["yolo_col"] = re.findall("\d+", lines[3])[0]
                    result_db["flag_string"] = lines[4].strip()
                    result_db["uploaded"] = 'no'

                    ecoli_count = lines[5].split()[-1]
                    coliform_count = lines[6].split()[-1]
                    result['coliforms'] = coliform_count
                    result['E.coli'] = ecoli_count
                    flag_string = lines[4].split()[0]
                    print(flag_string)
                    if new_sample == 1:
                        flagged = 0
                        result['coliforms'] = 0
                        result['E.coli'] = 0
                        flag_string = 'new'
                    if (flag_string == 'anomalous' or flag_string == 'too_many' or flag_string == 'Result uncertain' or flag_string =='overgrown/smeared'):
                        #send_serial('LED_RGB=100,0,0')
                        flagged = 1
                        if flag_string == 'overgrown/smeared':
                            flag_string = 'overgrown'
                            #result['coliforms'] = 250
                            #result['E.coli'] = 250
                        flag_message = 'Sample ' + flag_string
                        if flag_string == 'Result uncertain':
                            flag_message = 'Result uncertain'

                    break

            else:
                if new_sample == 1:
                    flagged = 0
                    result['coliforms'] = 0
                    result['E.coli'] = 0
                    flag_string = 'new'

                    result_db["unet_eco"] = 0
                    result_db["unet_col"] = 0
                    result_db["yolo_eco"] = 0
                    result_db["yolo_col"] = 0
                    result_db["flag_string"] = 'new sample'
                    result_db["uploaded"] = 'no'
                    # new_im.save(filename.replace('.jpg', '_resized_preview.jpg'),quality=10, optimize=False)
                    
                    break
            time.sleep(2)
    else:
                args.preview = filename.replace('.jpg', '_result.jpg')
                flagged = 0
                result['coliforms'] = 0
                result['E.coli'] = 0
                result['chlorine_level'] = chlorine_level  
                flag_string = 'chlorine'
                result_db["unet_eco"] = 0
                result_db["unet_col"] = 0
                result_db["yolo_eco"] = 0
                result_db["yolo_col"] = 0
                result_db["chlorine_level"] = chlorine_level
                result_db["flag_string"] = 'chlorine'
                result_db["uploaded"] = 'no'

    if(new_sample==1):
        uart.send_serial('line1=Place sample')
        time.sleep(0.5)
        uart.send_serial('line2=in incubator')
        new_sample=0
    elif (chlorine==1):
        uart.send_serial('line1=Free Chlorine:')
        time.sleep(0.5)
        text = f"line2={result['chlorine_level']:.1f} mg/L"
        uart.send_serial(text)
        chlorine = 0
    else:
        uart.send_serial('results={},{}'.format(result['coliforms'], result['E.coli']))
        time.sleep(5)
        if(flagged==1):
            uart.send_serial('line1='+flag_message)
            time.sleep(0.5)
            uart.send_serial('line2=check manual')
    with open("database.csv", "a") as database_file:
        # UID, sample_ID, image_seq, timestamp, focus, yolo_eco, yolo_coli, unet_eco, unet_coli, comment, filepath\n".replace(' ',))
        # result["unet_eco"], result["unet_col"], result["yolo_eco"], result["yolo_eco"] = re.findall("\d*", lines[0:3])
        # result["flag_string"] = lines[4]
        letters = string.ascii_lowercase
        random_string = ''.join(random.choice(letters) for _ in range(10))
      #  print("writing to the dabase")
       # print(image_seq)
        #print(timestamp)
        #print(filename)
        default_value = 'NA'  # You can change this to any appropriate default value
        UID = "{}_{}".format(hex(uuid.getnode()), random_string)
        database_file.write(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                UID,
                sample_ID,
                image_seq,
                timestamp,
                current_z,
                result_db.get("unet_eco", default_value),
                result_db.get("unet_col", default_value),
                result_db.get("yolo_eco", default_value),
                result_db.get("yolo_col", default_value),
                result_db.get("flag_string", default_value),
                filename,
                'no',
                result_db.get("chlorine_level", default_value),
                physical_parameters.get('pH', default_value),
                physical_parameters.get('tds', default_value),
                physical_parameters.get('turbidity', default_value),
                physical_parameters.get('conductivity', default_value),
                physical_parameters.get('salinity', default_value),
                physical_parameters.get('orp', default_value),
                physical_parameters.get('specific_gravity', default_value)
            )
)            
    uart.send_serial("temp=60")
    time.sleep(0.5)
    uart.send_serial("incubator_37")
    time.sleep(0.5)
    print(result)
    return result, flag_string
                
def capture_image(resolution='high_res'):
            
        global filename, image, folder_path
        with open('config_picamera.yaml') as config_file:
            config = yaml.load(config_file)

        camera.resolution = (640, 480)
        if(chlorine == 1):
            camera.shutter_speed = 11000
        else:
            camera.shutter_speed = config['shutter_speed']
        i = 900
        arducam_vcm.vcm_write(i)
        time.sleep(0.2)
        focus_table = {}
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
           # print("in the loop")
            image = frame.array
            define_ROI(focus_box_ratio)
            focus = variance_of_laplacian()
            # show the frame
           # cv2.imshow("stream", image)
            print(str(i) + ',' + str(focus))
            
            if(i<1000):
                i=i+5
                arducam_vcm.vcm_write(i)
            focus_table.update({i:focus})
            key = cv2.waitKey(1) & 0xFF
            

            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)
            image = 0

            # if the `q` key was pressed, break from the loop
            if(i==1000):
                i = 1000
                optimal_focus_z = max(focus_table, key=focus_table.get) 
                
                # Double check focus score is in valid range:
                max_score = focus_table[optimal_focus_z]
                if max_score <10:
                    # Very little to focus on, likely blank
                    # Likely focused in wrong position
                    # use average position
                    print("No features to focus on - using average position")
                    try: # In the case that the database is big enough
                        df = pd.read_csv('database.csv')
                        optimal_focus_z = np.mean(df["focus"].tail(10))
                        optimal_focus_z = int(optimal_focus_z - optimal_focus_z%5)
                        
                    except: # for new systems, use global average
                        optimal_focus_z = 1000
                    
                print("Optimal focus " + str(optimal_focus_z))
                focus = optimal_focus_z
                
                arducam_vcm.vcm_write(optimal_focus_z)
                time.sleep(3)
                
                print("Optimal focus: ")
                image = frame.array
                define_ROI(focus_box_ratio)
                focus = variance_of_laplacian()
                #define_ROI(focus_box_ratio)
                #test = variance_of_laplacian()
                #arducam_vcm.vcm_write(530)
                break
       # print(focus_table)
        #with open('focus_table.txt') as table_log:
         #   table_log.write(focus_table)
          #  table_log.close()
        filename=''
        with open('config_picamera.yaml') as config_file:
                config = yaml.load(config_file)
        global image_seq
        initialise_data_folder()
        print(folder_path)
        if filename == '':
            filename = folder_path+'/{:04d}_{}.jpg'.format(image_seq, datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        else:
            filename = folder_path+'/{:04d}-{}.jpg'.format(image_seq, filename)
        
        
       # camera.start_preview()
        camera.resolution = config['image_resolution']
        time.sleep(0.5)
        camera.capture(filename, format = 'jpeg', quality=100, bayer = False)
        #camera.capture(filename)
        image_seq+=1
        image = Image.open(filename)
        if(new_sample==1):
            image.save(filename.replace('.jpg', '_preview.jpg'),quality=40,optimize=True)
            time.sleep(0.5)
            args.preview = filename.replace('.jpg', '_preview.jpg')
        if(chlorine==1):    
            image.save(filename.replace('.jpg', '_compressed.jpg'),quality=40,optimize=True)
        else:
            image.save(filename.replace('.jpg', '_compressed.jpg'),quality=75,optimize=True)
       # camera.stop_preview()
        args.raw = filename.replace('.jpg', '_compressed.jpg')
        timestamp = datetime.now().strftime('%Y%m%d-%H:%M:%S')
        print(args)
        print(focus_table)
        print("image captured")            
        uart.send_serial("led_off")
        return (optimal_focus_z, timestamp)
        
def package_offsets(size, package_size):
    return range(0, size, package_size)


def package_end(size, offset, package_size):
    end = offset + package_size
    return end if end < size else size


# noinspection PyShadowingBuiltins
def info_matches(response, id, type, part):
    try:
        return response['id'] == id and response['type'] == type and response['part'] == part
    except KeyError:
        return False


def calculate_progress(size, offset):
    return round((offset / size) * 100, 2)




# noinspection PyShadowingBuiltins
def send_file_response(connection, instruction, file_path, package_size, type, id=None):
    global outstanding_data
    logging.info(f'Sending file "{file_path}"')

    with open(file_path, 'rb') as file:
        data = file.read()

    size = len(data)
    hashes = []

    if package_size < 1:
        package_size = size

    for offset in package_offsets(size, package_size):
        hashes.append(hashlib.sha1(data[offset:package_end(size, offset, package_size)]).hexdigest())

    logging.debug(f'{len(hashes)} file hash{"es" if len(hashes) > 1 else ""} created for "{file_path}".')

    info = {
        'id': id,
        'type': type,
        'hashes': hashes,
        'packageSize': package_size,
        'size': len(data)
    }

    send_response(connection, instruction, payload=json.dumps(info).encode('utf-8'))

    with open(file_path, 'rb') as file:
        for offset in package_offsets(size, package_size):
            logging.debug(
                f'Sending file "{file_path}" from offset {offset}, '
                f'which is equal to {calculate_progress(size, offset)}%.'
            )

            try:
                connection.sendfile(file, offset, package_end(size, offset, package_size) - offset)

                status = None
                payload = {'part': None}
                part = offset // package_size
                connection.settimeout(30)

                while not (status == 'ok' and info_matches(payload, id, type, part)):
                    response_raw = connection.recv(1024).decode('utf-8')
                    try:
                        print(response_raw)
                        response1 = json.loads(response_raw)
                        outstanding_data = None
                    except:
                        print("trying to recover")
                        response1 = json.loads(response_raw.split("}}{")[0]+"}}")
                        outstanding_data = bytearray("{"+response_raw.split("}}{")[1],'utf-8')
                        print(outstanding_data)
                        
                    if "instruction" in response1:
                        logging.error(
                            f'Received new instruction before file "{file_path}" was fully submitted. '
                            f'Skipping instruction "{response_raw["instruction"]}" and aborting transmission.'
                        )
                        return
                        
                    payload = read_payload(connection, response1)
                    if payload is None:
                        logging.error('Could not read file part response payload. Aborting file transmission.')
                        return

                    payload = json.loads(payload.decode('utf-8'))

                    if info_matches(payload, id, type, part):
                        try:
                            status = response1['status']
                        except KeyError:
                            continue

                        if status == 'invalid':
                            logging.warning(
                                f'File "{file_path}" got corrupted starting at offset {offset}, '
                                f'which is equal to {calculate_progress(size, offset)}%. '
                                f'Resending file starting from this offset.'
                            )
                            connection.sendfile(file, offset, package_end(size, offset, package_size) - offset)
                        elif status == 'failed':
                            logging.error(
                                f'Sending file "{file_path}" failed '
                                f'after {calculate_progress(size, offset)}% has been send.'
                            )
                            return
            except socket.timeout:
                logging.error(
                    f'Sending file "{file_path}" timed out after {calculate_progress(size, offset)}% has been send.'
                )
                return
            except OSError:
                raise ConnectionError
            finally:
                connection.settimeout(None)

    logging.info(f'Successful send file "{file_path}".')


def send_sample_update(connection, sample_id, sample_status, result=None):
    logging.info(f'Sample #{sample_id} reached status "{sample_status}".')

    payload = json.dumps({
        'id': sample_id,
        'status': sample_status,
        'result': result,
    }).encode('utf-8')
    for i in range (3):
        try:
            send_instruction(connection, 'sample', payload=payload)
            break
        except:
            print("exception")

# noinspection PyShadowingBuiltins
def send_sample(connection, instruction, id):
    logging.info(f'Sample #{id} requested.')

    global samples

    try:
        send_response(connection, instruction, payload=json.dumps(samples[id]).encode('utf-8'))
    except KeyError:
        send_response(connection, instruction, status=f'Sample #{id} not found')

# noinspection PyShadowingBuiltins
def analyse_sample(connection, instruction, id, data):
    global sample_ID,new_sample,chlorine,physical_parameters
    sample_ID=id
    print(data)
    logging.info(f'''Sample #{id} submitted.\n{json.dumps({
        'location': data.get('location', 'Unknown'),
        'time': data.get('time', 'Unknown'),
        'comment': data.get('comment', 'No comment'),
        'coordinates': data.get('coordinates', 'Unknown'),
        'analysis_type': data.get('analysis_type', 'Unknown'),
        'pH': data.get('pH', 'N/A'),
        'TDS': data.get('TDS', 'N/A'),
        'Turbidity': data.get('Turbidity', 'N/A'),
        'Conductivity': data.get('Conductivity', 'N/A'),
        'Salinity': data.get('Salinity', 'N/A'),
        'ORP': data.get('ORP', 'N/A'),
        'Specific Gravity': data.get('Specific Gravity', 'N/A')
    }, indent=4)}''')

    # Setting physical parameters from data with default values
    default_value = 'N/A'  # Default value for missing keys
    physical_parameters['pH'] = data.get('pH', default_value)
    physical_parameters['tds'] = data.get('TDS', default_value)
    physical_parameters['turbidity'] = data.get('Turbidity', default_value)
    physical_parameters['conductivity'] = data.get('Conductivity', default_value)
    physical_parameters['salinity'] = data.get('Salinity', default_value)
    physical_parameters['orp'] = data.get('ORP', default_value)
    physical_parameters['specific_gravity'] = data.get('Specific Gravity', default_value)
    
    if(data['time']==0):
        new_sample=1
    if(data['analysis_type']=="Chlorine"):
        print("Analysing chlorine")
        chlorine=1
    else:
        chlorine = 0
    uart.send_serial("led_on")
    sample_comment = data['comment']
    send_response(connection, instruction)
    time.sleep(1)
    if (data['time']==0):
        new_sample=1
    if sample_comment != None and ('updatewaterscope2021') in sample_comment:
        with open('password.txt', 'w+') as file:
            file.write(sample_comment.split('=')[1])
        time.sleep(0.5)
        uart.send_serial('line1=Updating system')
        time.sleep(0.5)
        uart.send_serial('line2=please wait')
        subprocess.call(['expect','update.sh'])
    

    try:
        try:
            send_sample_update(connection, id, 'analysing')
        except:
            print("exception occured")   
        time.sleep(1)
        uart.send_serial("temp=60")
        time.sleep(0.5)
        uart.send_serial('line1=Analysing sample')
        time.sleep(0.5)
        uart.send_serial('line2=check app')
        try:
            send_sample_update(connection, sample_ID, 'defogging')
        except:
            print("exception occured")
        if(new_sample==0):
            print("skipped defogging because of 25mm")
            #defogging(connection,sample_ID, sample_comment)
        send_sample_update(connection, sample_ID, 'autofocusing')
        time.sleep(1)
        send_sample_update(connection, sample_ID, 'counting')
        results,flag = analysis_result()         
        time.sleep(1)
        
        result = {
            'eColiform': results.get('E.coli', 'N/A'),  # Use 'N/A' or any other default value
            'otherColiform': results.get('coliforms', 'N/A'),
            'flag': flag,
            'chlorine_level': results.get('chlorine_level', 'N/A')
        }

        send_sample_update(connection, id, 'result', result)

    except TimeoutError:
        return

def analyse_sample_buttons(id):
    global sample_ID
    sample_ID = id
    
def sample(connection, instruction):
    data = read_payload(connection, instruction)

    if data is None:
        logging.error('Could not read sample instruction payload.')
        return

    data = json.loads(data)

    try:
        action = data['action']
        # noinspection PyShadowingBuiltins
        id = data['id']
    except KeyError:
        send_response(connection, instruction, status=f'A sample action is missing')
        return

    global args

    if action == 'submit':
        analyse_sample(connection, instruction, id, data)
    elif action == 'get':
        send_sample(connection, instruction, id)
    elif action == 'get raw image':
        try:
            send_file_response(
                connection,
                instruction,
                args.raw,
                args.package_size * 1024,
                'raw image',
                id=id,
            )
        except TimeoutError:
            pass
    elif action == 'get preview image':
        try:
            send_file_response(
                connection,
                instruction,
                args.preview,
                args.package_size * 1024,
                'preview image',
                id=id,
            )
        except TimeoutError:
            pass
    else:
        send_response(connection, instruction, status=f'Sample action "{action}" not supported')


def diagnostics():
    return {
        'temperature': 36.5,
        'servo': 'OK',
        'defogger': 'OK',
        'incubator': 'Too hot alert 02/03/20',
        'deviceTestCount': 50,
        'batteryLevel': 69,
        'averageFocusingPosition': 150.0,
        'softwareVersion': '3.4',
        'firmwareVersion': '4.2',
        'location': 'Tanzania',
        'internet': 'IoT/No Wi',
    }


def sample_history(sample_file_path):
    global args

    entries = {}

    try:
        with open(sample_file_path, newline='') as file:
            reader = csv.reader(file)
            next(reader, None)

            i = 0

            for row in reader:
                i += 1

                try:
                    hours = row[10].split(':')[0]
                    minutes = row[10].split(':')[1].split(' ')[0]

                    entries[int(row[0])] = {
                        'id': int(row[0]),
                        'eColiform': int(row[1]),
                        'otherColiform': int(row[2]),
                        'location': row[8],
                        'time': int(hours) * 60 + int(minutes),
                        'comment': row[11],
                        'flag': row[3],
                    }
                except (IndexError, ValueError):
                    logging.error(f'Could not parse sample in row #{i + 1}')
                    continue
    except FileNotFoundError:
        pass

    logging.debug(f'Loaded samples from csv-file\n{json.dumps(entries, indent=4)}')

    return entries


def calculate_payload_validation(payload):
    return {
        'size': len(payload) if payload is not None else 0,
        'checksum': hashlib.sha1(payload).hexdigest() if payload is not None else None
    }


def send_instruction(connection, instruction_name, payload=None):
    global sender_id
    sender_id += 1

    # noinspection PyShadowingBuiltins
    id = 'w' + str(sender_id)  # adding an character so they don't clash with the IDs send by the app

    instruction = {
        'id': id,
        'instruction': instruction_name,
        'payload': calculate_payload_validation(payload)
    }

    logging.debug(f'Sending instruction.\n{json.dumps(instruction, indent=4)}')

    connection.sendall(json.dumps(instruction).encode("utf-8"))

    connection.settimeout(5)

    if payload is not None:
        connection.sendall(payload)

    try:
        response = {'status': None, 'id': None}

        while not (response['status'] == 'ok' and response['id'] == id):
            try:
                response = json.loads(connection.recv(1024).decode('utf-8'))

                if response['id'] == id:
                    if response['status'] == 'invalid':
                        logging.warning(f'Retrying instruction "{instruction_name}" with ID "{id}".')
                        send_instruction(connection, instruction_name, payload=payload)
                    elif response['status'] != 'ok':
                        logging.error(
                            f'Instruction "{instruction_name}" with ID "{id}" '
                            f'failed with status {response["status"]}.'
                        )
                        break

            except socket.timeout:
                logging.error(f'Instruction "{instruction_name}" with ID "{id}" timed out.')
                
                raise TimeoutError
    except (ValueError, json.decoder.JSONDecodeError, KeyError):
        logging.error(f'Reading response for instruction "{instruction_name}" with ID "{id}" failed.')
    finally:
        connection.settimeout(None)


def send_response(connection, instruction, status='ok', payload=None):
    response = {
        'id': instruction['id'],
        'instruction': instruction['instruction'],
        'status': status,
        'payload': calculate_payload_validation(payload)
    }

    logging.debug(f'Sending response.\n{json.dumps(response, indent=4)}')

    connection.sendall(json.dumps(response).encode("utf-8"))

    if payload is not None:
        connection.sendall(payload)


def read_payload(connection, instruction, retries=5):
    global outstanding_data
    size = instruction['payload']['size']
    checksum = instruction['payload']['checksum']

    if size < 1:
        logging.warning(f'Expected payload but none got transmitted. An instruction might got skipped.')
        return

    tries = 0
    data = bytearray()
    if(outstanding_data==None):
        while len(data) < size:
            data += connection.recv(size - len(data))
         #   print("getting more data:")
          #  print(data)
    else:
            data = outstanding_data
           # print(data)
            outstanding_data=None

    while tries < retries and hashlib.sha1(data).hexdigest() != checksum:
        tries += 1
        logging.info(f'Payload checksum does not match. Retry #{tries}.')
        send_response(connection, instruction, status='invalid')
        data = connection.recv(size)

    if tries == retries:
        logging.error(f'Payload checksum are not match after retry #{tries} still. Aborting.')
        send_response(connection, instruction, status='failed')
    else:
        return data


def update_wifi(connection, instruction):
    data = read_payload(connection, instruction)

    if data is None:
        logging.error('Could not read Wi-Fi change payload.')
        return

    data = json.loads(data)

    logging.debug(f'Received Wi-Fi change details.\n{json.dumps(data, indent=4)}')

    
    # TODO add wifi via network manager dbus
    print(data["password"])
    CreateWifiConfig(data["ssid"],data["password"])
    logging.info(f'Wi-Fi network "{data["ssid"]}" added.')
    send_response(connection, instruction)
    

def CreateWifiConfig(SSID, password):
  config_lines = [
    '\n',
    'network={',
    '\tssid="{}"'.format(SSID),
    '\tpsk="{}"'.format(password),
    '}'
  ]

  config = '\n'.join(config_lines)
  print(config)

  with open("/etc/wpa_supplicant/wpa_supplicant.conf", "a+") as wifi:
    wifi.write(config)

  print("Wifi config added")
  
def update(connection, instruction):
    data = read_payload(connection, instruction)

    if data is None:
        logging.error('Could not read update change payload.')
        return

    data = json.loads(data.decode('utf-8'))
    size = data["size"]
    package_size = data["packageSize"]
    buffer_size = package_size

    send_response(connection, instruction)

    with open(instruction['instruction'], 'w+b') as file:
        # noinspection PyShadowingBuiltins
        for part, hash in enumerate(data['hashes']):
            if part == (len(data['hashes']) - 1):
                buffer_size = size - part * buffer_size

            checksum = None

            while checksum != hash:
                print(part)

                buffer = bytearray()

                try:
                    connection.settimeout(45)

                    while buffer_size != len(buffer):
                        buffer += connection.recv(buffer_size - len(buffer))
                except socket.timeout:
                    logging.error(
                        f'Receiving update file timed out '
                        f'after {calculate_progress(size, part * package_size)}% has been received.'
                    )
                    return
                finally:
                    connection.settimeout(None)

                checksum = hashlib.sha1(buffer).hexdigest()

                if checksum == hash:
                    status = 'ok'
                    file.write(buffer)
                else:
                    status = 'invalid'
                    buffer.clear()

                    logging.warning(
                        f'Update file got corrupted at offset {part * package_size}, '
                        f'which is equals to {calculate_progress(size, part * package_size)}%. '
                        f'Retrying transmission from this offset.'
                    )

                send_response(connection, instruction, status=status, payload=json.dumps({
                    'type': data['type'],
                    'part': part,
                }).encode('utf-8'))

    # TODO implement update mechanism
    send_instruction(connection, 'update', json.dumps({
        'status': 'success'     # if status not 'success', the status will get shown in the app
    }).encode('utf-8'))


def bluetooth_loop():
    global outstanding_data
    adapter = bus.get('org.bluez', '/org/bluez/hci0').Address
    s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    s.bind((adapter, 1))

    s.listen(1)

    while True:
        connection, address = s.accept()
        address = address[0]

        logging.info(f'Established connection with {address}.')

        while True:
            try:
                connection.settimeout(None)
                data_raw = connection.recv(1024).decode('utf-8')

                try:
                    data = json.loads(data_raw)
                    outstanding_data = None
                except json.decoder.JSONDecodeError:
                    logging.warning(
                        f'Received invalid instructions "{data_raw.strip()}". '
                        f'The android app might to be updated.'
                    )
                    data = json.loads(data_raw.split("}}{")[0]+"}}")
                    outstanding_data = bytearray(("{"+data_raw.split("}}{")[1]),'utf-8')
                    

                logging.debug(f'Instruction received.\n{json.dumps(data, indent=4)}')
                try:
                    instruction = data['instruction']
                except KeyError:
                    logging.warning('Ignoring payload received as instruction. An instruction got skipped most likely.')
                    continue

                # noinspection PyShadowingBuiltins
                id = str(data['id'])

                if id[0] == 'w':
                    logging.warning(f'Received response for timed out instruction "{instruction}" with ID "{id}".')

                    payload_size = data['payload']['size']

                    if payload_size > 0:
                        connection.recv(payload_size)

                    continue

                if instruction == 'diagnostics':
                    logging.info('Diagnostics requested.')
                    send_response(connection, data, payload=json.dumps(diagnostics()).encode('utf-8'))
                elif instruction == 'history':
                    global samples

                    logging.info('Sample history requested.')
                    send_response(connection, data, payload=json.dumps(list(samples.values())).encode('utf-8'))
                elif instruction == 'sample':
                    sample(connection, data)
                elif instruction == 'wifi':
                    update_wifi(connection, data)
                elif instruction == 'update':
                    update(connection, data)
                else:
                    logging.warning('Unavailable instruction received.')
                    send_response(connection, data, status=f'Instruction "{instruction}" not supported')
            except UnicodeDecodeError:
                logging.warning('Received binary data as instruction.')
            except ConnectionError:
                logging.info(f'Connection with {address} lost.')
                break
            except TimeoutError:
                logging.info(f'Connection with {address} timed out.')
                break

        connection.close()
def uart_loop():
    global sample_ID, new_sample,chlorine
    while True:
        
        command = uart.read_serial()
        if 'ID=' in command:
            sample_ID = int(command.strip("ID="))
            chlorine = 0
           # if(new_sample==0):
               # defogging() <==skipped defogging because of 25 mm
            result,flag = analysis_result()
        elif 'new_sample' in command:
            new_sample = 1
        elif 'chlorine' in command:
            print("Chlorine measurement")
            chlorine = 1
        elif 'pi_off' in command:
            with open(log_name, "a") as myfile:
       # now = datetime.now() # current date and time
                myfile.write("Power off at: "+datetime.now().strftime("%m/%d/%Y %H:%M:%S")+"\n")
                myfile.close()
            print("power off")
            subprocess.call(['sudo','/usr/sbin/shutdown','-h','now'])
            
def ML_analysis(input_filename=''):
    start = time.time()
    print("loading the ML module, please wait")
    import count_colony_yolor
    print("imported the ML module")
    print("time it tooks: {}".format(time.time() - start))
    if(input_filename!=''):
        result = count_colony_yolor.analysis_image(input_filename, input_filename.replace('.jpg', '_result.jpg'))
        print(result)
        print("time it tooks: {}".format(time.time() - start))
    

if __name__ == '__main__':
    if not os.path.exists('temp_logs'):
        os.mkdir('temp_logs')
    uart.open_serial()
    uart.send_serial("led_on")
    time.sleep(0.1)
    uart.send_serial("led_off")
    time.sleep(0.1)
    uart.send_serial("led_on")
    time.sleep(0.1)
    uart.send_serial("led_off")
    time.sleep(0.1)
    uart.send_serial("led_on")
    time.sleep(0.1)
    uart.send_serial("led_off")
    time.sleep(0.1)
    uart.send_serial("led_on")
    time.sleep(1)
    x = threading.Thread(target = uart_loop)
    x.start()
    y = threading.Thread(target = ML_analysis)
    y.start()            


    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='show debug information', action='store_true')
    parser.add_argument('--samples', default='samples.csv', help='samples file path used as mock-up history source')
    parser.add_argument('--preview', default='preview.jpg', help='preview image file path used as mock-up')
    parser.add_argument('--raw', default='raw.jpg', help='raw image file path used as mock-up')
    parser.add_argument(
        '--package-size',
        type=int,
        default=50,
        help='Kilobyte size in which a file gets split up for transmission. '
             'If less than one, files will not get split up.'
    )

    args = parser.parse_args()
    print(args)
    log_level = getattr(logging, 'DEBUG' if args.verbose else 'INFO', None)
    # noinspection SpellCheckingInspection
    logging.basicConfig(level=log_level, format='[%(levelname)s]\t(%(asctime)s)\t\t%(message)s')
    update_camera_setting()
    samples = sample_history(args.samples)
    arducam_vcm.vcm_init()
    new_sample = 0
    log_name = "log.txt"
    #uart.send_serial("led_on")
    uart.send_serial("temp=60")
                # Define the headers
    headers = "UID,sample_ID,image_seq,timestamp,focus,yolo_eco,yolo_col,unet_eco,unet_col,flag_string,filepath,uploaded,chlorine,ph,tds,turbidity,salinity,conductivity,orp,specific_gravity"

    # Check if file exists
    if not os.path.exists("database.csv"):
        with open("database.csv", "w", newline='', encoding='utf-8') as database_file:
            database_file.write(headers + "\n")
    else:
        # Read the file in binary mode and replace NULL bytes
        with open("database.csv", 'rb') as file:
            content = file.read().decode('utf-8').replace('\x00', '')   

        # Write the cleaned content back to the file
        with open("database.csv", 'w', newline='', encoding='utf-8') as file:
            file.write(content)

        # Now, read the file with csv.reader
        with open("database.csv", "r", newline='', encoding='utf-8') as database_file:
            reader = csv.reader(database_file)
            existing_data = list(reader)

        # Check and rewrite the file if necessary
        if not existing_data or existing_data[0] != headers.split(","):
            with open("database.csv", "w", newline='', encoding='utf-8') as database_file:
                writer = csv.writer(database_file)
                writer.writerow(headers.split(","))
                if existing_data:
                    writer.writerows(existing_data[1:])   # capture_image()
   # Chlorine_analysis("timelapse_data/666/0000_20231206-22:08:12.jpg")
    
    with open('image_to_analyse.txt', 'w+') as file:
        pass
   # analysis_result()
    while True:
        try:
            bluetooth_loop()
        except (Exception, json.decoder.JSONDecodeError, ValueError) as e:
            logging.error(f'Unexpected exception. Restarting.\n{e}\n\n{traceback.format_exc()}')
