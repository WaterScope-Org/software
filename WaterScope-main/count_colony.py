import numpy as np
from keras.models import load_model
from tensorflow.keras import backend as K
import cv2
import random as rng
from skimage.feature import peak_local_max
import os
import time
import tensorflow as tf
from PIL import Image
from integrate_folder.yolo import YOLO
from tflite_runtime.interpreter import Interpreter  # on Pi, uncomment this; otherwise, use tf.lite.Interpreter


def raw_to_cropped_old(raw_image, dim, color_check=False, print_log=False):
    '''
    test
    :parameter: raw_image
                dim -> output dimension
                color_check -> if True, crop 1.5*radius to focus only on the center color [since if highly contaminated,
                the center will already have enough information and avoid taking the edge color into account]
    :return: cropped 256x256 pixel image centered on the ROI
    '''

    def checkradius(x, y, radius):
        if x < 450 or x > 750 or y < 400 or radius < 150:  # May 6th, 560, 700, 400, 150
            x = 640
            y = 450
            radius = 200
            if print_log == True:
                print('adjust x,y,radius to 640,450,200 due to anomalous value(s) of %d, %d, %d.' % (x, y, radius))
            return x, y, radius
        else:
            return x, y, radius

    if color_check == True:
        def masking(mask, image, x, y, radius, dim):
            cropimg = cv2.subtract(mask, image)
            cropimg = cv2.subtract(mask, cropimg)
            cropimg = cropimg[y - radius + int(radius * 0.5):y + radius - int(radius * 0.5),
                      x - radius + int(radius * 0.5):x + radius - int(radius * 0.5)]
            cropimg = cv2.resize(cropimg, dim, interpolation=cv2.INTER_AREA)
            return cropimg

        img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        img_color = raw_image
        r = 1200.0 / img.shape[1]
        dimension = (1200, int(img.shape[0] * r))
        img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        img_color = cv2.resize(img_color, dimension, interpolation=cv2.INTER_AREA)

        # BGR channel
        blue, green, red = cv2.split(img_color)
        img_combine_1 = red
        img_combine_2 = blue
        img_combine_3 = green

        # Detecting ROI
        mask = np.zeros((901, 1200), dtype=np.uint8)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                   param1=60, param2=40, minRadius=200, maxRadius=238)
        if type(circles) != type(None):
            if len(circles[0, :, :]) > 1:
                x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))

                return crop_color
            else:
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                return crop_color
        else:
            if print_log == True:
                print('no circle found... try changing parameter (1)')
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                       param1=50, param2=30, minRadius=200,
                                       maxRadius=238)
            if type(circles) != type(None):
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                return crop_color
            else:
                if print_log == True:
                    print('no circles found... try changing parameter (2)')
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                           param1=40, param2=20, minRadius=200,
                                           maxRadius=238)
                if type(circles) != type(None):
                    if len(circles[0, :, :]) > 1:
                        x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        return crop_color
                    else:
                        x, y, radius = np.uint16(circles[0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        return crop_color
                else:
                    if print_log == True:
                        print('no circle found...:c... resorting to empirically estimated x,y,r=640,450,200')
                    x, y, radius = 640, 450, 200
                    cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    return crop_color
    else:  # if color_check=False, crop as usual
        t_0 = time.time()

        def masking(mask, image, x, y, radius, dim):
            cropimg = cv2.subtract(mask, image)
            cropimg = cv2.subtract(mask, cropimg)
            cropimg = cropimg[y - radius:y + radius, x - radius:x + radius]
            cropimg = cv2.resize(cropimg, dim, interpolation=cv2.INTER_AREA)
            return cropimg

        img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        img_color = raw_image
        r = 1200.0 / img.shape[1]
        dimension = (1200, int(img.shape[0] * r))
        img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        img_color = cv2.resize(img_color, dimension, interpolation=cv2.INTER_AREA)

        # BGR channel
        blue, green, red = cv2.split(img_color)
        img_combine_1 = red
        img_combine_2 = blue
        img_combine_3 = green

        # Detecting ROI
        mask = np.zeros((901, 1200), dtype=np.uint8)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                   param1=60, param2=40, minRadius=200, maxRadius=238)
        if type(circles) != type(None):
            if len(circles[0, :, :]) > 1:
                x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color
            else:
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color
        else:
            if print_log == True:
                print('no circle found... try changing parameter (1)')
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                       param1=50, param2=30, minRadius=200,
                                       maxRadius=238)
            if type(circles) != type(None):
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color
            else:
                if print_log == True:
                    print('no circles found... try changing parameter (2)')
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                           param1=40, param2=20, minRadius=200,
                                           maxRadius=238)
                if type(circles) != type(None):
                    if len(circles[0, :, :]) > 1:
                        x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        # if print_log == True:
                        #   print(x, y,radius)

                        return crop_color
                    else:
                        x, y, radius = np.uint16(circles[0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        # if print_log == True:
                        #   print(x, y,radius)

                        return crop_color
                else:
                    if print_log == True:
                        print('no circle found...:c... resorting to empirically estimated x,y,radius of 640,450,200')
                    x = 640
                    y = 450
                    radius = 200
                    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    # if print_log == True:
                    #   print(x, y,radius)
                    return crop_color


def raw_to_cropped(raw_image, dim, color_check=False, print_log=False, multiplier=0.05):
    '''
    :parameter: raw_image
                dim -> output dimension
                color_check -> if True, crop 1.5*radius to focus only on the center color [since if highly contaminated,
                the center will already have enough information and avoid taking the edge color into account]
    :return: cropped 256x256 pixel image centered on the ROI
    '''

    def checkradius(x, y, radius):
        if x < 450 or x > 750 or y < 400 or radius < 150:  # May 6th, 560, 700, 400, 150
            x = 640
            y = 450
            radius = 200
            if print_log == True:
                print('adjust x,y,radius to 640,450,200 due to anomalous value(s) of %d, %d, %d.' % (x, y, radius))
            return x, y, radius
        else:
            return x, y, radius

    if color_check == True:
        def masking(mask, image, x, y, radius, dim):
            cropimg = cv2.subtract(mask, image)
            cropimg = cv2.subtract(mask, cropimg)
            cropimg = cropimg[y - radius + int(radius * 0.5):y + radius - int(radius * 0.5),
                      x - radius + int(radius * 0.5):x + radius - int(radius * 0.5)]
            cropimg = cv2.resize(cropimg, dim, interpolation=cv2.INTER_AREA)
            return cropimg

        img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        img_color = raw_image
        r = 1200.0 / img.shape[1]
        dimension = (1200, int(img.shape[0] * r))
        img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        img_color = cv2.resize(img_color, dimension, interpolation=cv2.INTER_AREA)

        # BGR channel
        blue, green, red = cv2.split(img_color)
        img_combine_1 = red
        img_combine_2 = blue
        img_combine_3 = green

        # Detecting ROI
        mask = np.zeros((901, 1200), dtype=np.uint8)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                   param1=60, param2=40, minRadius=200, maxRadius=238)
        if type(circles) != type(None):
            if len(circles[0, :, :]) > 1:
                x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))

                return crop_color
            else:
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                return crop_color
        else:
            if print_log == True:
                print('no circle found... try changing parameter (1)')
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                       param1=50, param2=30, minRadius=200,
                                       maxRadius=238)
            if type(circles) != type(None):
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                return crop_color
            else:
                if print_log == True:
                    print('no circles found... try changing parameter (2)')
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                           param1=40, param2=20, minRadius=200,
                                           maxRadius=238)
                if type(circles) != type(None):
                    if len(circles[0, :, :]) > 1:
                        x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        return crop_color
                    else:
                        x, y, radius = np.uint16(circles[0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        return crop_color
                else:
                    if print_log == True:
                        print('no circle found...:c... resorting to empirically estimated x,y,r=640,450,200')
                    x, y, radius = 640, 450, 200
                    cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    return crop_color
    else:  # if color_check=False, crop as usual
        t_0 = time.time()

        def masking(mask, image, x, y, radius, dim):
            cropimg = cv2.subtract(mask, image)
            cropimg = cv2.subtract(mask, cropimg)
            cropimg = cropimg[y - radius:y + radius, x - radius:x + radius]
            cropimg = cv2.resize(cropimg, dim, interpolation=cv2.INTER_AREA)
            return cropimg

        img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        img_color = raw_image
        r = 1200.0 / img.shape[1]
        dimension = (1200, int(img.shape[0] * r))
        img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        img_color = cv2.resize(img_color, dimension, interpolation=cv2.INTER_AREA)

        # BGR channel
        blue, green, red = cv2.split(img_color)
        img_combine_1 = red
        img_combine_2 = blue
        img_combine_3 = green

        # Detecting ROI
        mask = np.zeros((901, 1200), dtype=np.uint8)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                   param1=70, param2=60, minRadius=230, maxRadius=270)
        if type(circles) != type(None):
            if len(circles[0, :, :]) > 1:
                x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                x, y, radius = checkradius(x, y, radius)
                radius += int(radius * multiplier)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color
            else:
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                radius += int(radius * multiplier)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color
        else:
            if print_log == True:
                print('no circle found... try changing parameter (1)')
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                       param1=60, param2=40, minRadius=230, maxRadius=270)
            if type(circles) != type(None):
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                radius += int(radius * multiplier)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color
            else:
                if print_log == True:
                    print('no circles found... try changing parameter (2)')
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                           param1=50, param2=30, minRadius=230, maxRadius=270)
                if type(circles) != type(None):
                    if len(circles[0, :, :]) > 1:
                        x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                        x, y, radius = checkradius(x, y, radius)
                        radius += int(radius * multiplier)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        # if print_log == True:
                        #   print(x, y,radius)

                        return crop_color
                    else:
                        x, y, radius = np.uint16(circles[0][0])
                        x, y, radius = checkradius(x, y, radius)
                        radius += int(radius * multiplier)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        # if print_log == True:
                        #   print(x, y,radius)

                        return crop_color
                else:
                    if print_log == True:
                        print('no circle found...:c... resorting to empirically estimated x,y,radius of 640,450,200')
                    x = 640
                    y = 450
                    radius = 200
                    radius += int(radius * multiplier)
                    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    # if print_log == True:
                    #   print(x, y,radius)
                    return crop_color

def raw_to_cropped_color_check(raw_image, dim, print_log=False, multiplier=0.05):
    original_dim = raw_image.shape

    def checkradius(x, y, radius):
        if x < 450 or x > 750 or y < 400 or radius < 150:  # May 6th, 560, 700, 400, 150
            x = 640
            y = 450
            radius = 200
            if print_log == True:
                print('adjust x,y,radius to 640,450,200 due to anomalous value(s) of %d, %d, %d.' % (x, y, radius))
            return x, y, radius
        else:
            return x, y, radius

    def masking(mask, image, x, y, radius, dim):
        cropimg = cv2.subtract(mask, image)
        cropimg = cv2.subtract(mask, cropimg)
        cropimg = cropimg[y - radius + int(radius * 0.5):y + radius - int(radius * 0.5),
                  x - radius + int(radius * 0.5):x + radius - int(radius * 0.5)]
        # cropimg = cv2.resize(cropimg, dim, interpolation=cv2.INTER_AREA)
        return cropimg

    img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    img_color = raw_image
    r = 1200.0 / img.shape[1]
    dimension = (1200, int(img.shape[0] * r))
    img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
    # img_color = cv2.resize(img_color, dimension, interpolation=cv2.INTER_AREA)

    # BGR channel
    blue, green, red = cv2.split(img_color)
    img_combine_1 = red
    img_combine_2 = blue
    img_combine_3 = green

    # Detecting ROI

    mask = np.zeros((original_dim[0], original_dim[1]), dtype=np.uint8)
    scale_to_ori = original_dim[1] / 1200
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                               param1=70, param2=40, minRadius=230, maxRadius=250)
    if type(circles) != type(None):
        if len(circles[0, :, :]) > 1:
            x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
            x, y, radius = checkradius(x, y, radius)
            radius += int(radius * multiplier)
            x = int(x * scale_to_ori)
            y = int(y * scale_to_ori)
            radius = int(radius * scale_to_ori)
            cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
            crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
            crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
            crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
            crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
            # if print_log == True:
            #   print(x, y, radius)

            return crop_color
        else:
            x, y, radius = np.uint16(circles[0][0])
            x, y, radius = checkradius(x, y, radius)

            radius += int(radius * multiplier)
            x = int(x * scale_to_ori)
            y = int(y * scale_to_ori)
            radius = int(radius * scale_to_ori)
            cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
            crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
            crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
            crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
            crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
            # if print_log == True:
            #   print(x, y, radius)

            return crop_color
    else:
        if print_log == True:
            print('no circle found... try changing parameter (1)')
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                   param1=60, param2=30, minRadius=230, maxRadius=250)
        if type(circles) != type(None):
            x, y, radius = np.uint16(circles[0][0])
            x, y, radius = checkradius(x, y, radius)
            radius += int(radius * multiplier)
            x = int(x * scale_to_ori)
            y = int(y * scale_to_ori)
            radius = int(radius * scale_to_ori)
            cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
            print(mask.shape)
            print(img_combine_1.shape)
            crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
            crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
            crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
            crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
            # if print_log == True:
            #   print(x, y, radius)

            return crop_color
        else:
            if print_log == True:
                print('no circles found... try changing parameter (2)')
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                       param1=50, param2=20, minRadius=230, maxRadius=250)
            if type(circles) != type(None):
                if len(circles[0, :, :]) > 1:
                    x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                    x, y, radius = checkradius(x, y, radius)
                    radius += int(radius * multiplier)
                    x = int(x * scale_to_ori)
                    y = int(y * scale_to_ori)
                    radius = int(radius * scale_to_ori)
                    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    # if print_log == True:
                    #   print(x, y,radius)

                    return crop_color
                else:
                    x, y, radius = np.uint16(circles[0][0])
                    x, y, radius = checkradius(x, y, radius)
                    radius += int(radius * multiplier)
                    x = int(x * scale_to_ori)
                    y = int(y * scale_to_ori)
                    radius = int(radius * scale_to_ori)
                    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    # if print_log == True:
                    #   print(x, y,radius)

                    return crop_color
            else:
                if print_log == True:
                    print('no circles found... try changing parameter (3)')
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                           param1=40, param2=20, minRadius=230, maxRadius=250)
                if type(circles) != type(None):
                    if len(circles[0, :, :]) > 1:
                        x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                        x, y, radius = checkradius(x, y, radius)
                        radius += int(radius * multiplier)
                        x = int(x * scale_to_ori)
                        y = int(y * scale_to_ori)
                        radius = int(radius * scale_to_ori)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        # if print_log == True:
                        #   print(x, y,radius)

                        return crop_color
                    else:
                        x, y, radius = np.uint16(circles[0][0])
                        x, y, radius = checkradius(x, y, radius)
                        radius += int(radius * multiplier)
                        x = int(x * scale_to_ori)
                        y = int(y * scale_to_ori)
                        radius = int(radius * scale_to_ori)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        # if print_log == True:
                        #   print(x, y,radius)

                        return crop_color
                else:
                    if print_log == True:
                        print(
                            'no circle found...:c... resorting to empirically estimated x,y,radius of 640,450,200')
                    x = 640
                    y = 450
                    radius = 200
                    radius += int(radius * multiplier)
                    x = int(x * scale_to_ori)
                    y = int(y * scale_to_ori)
                    radius = int(radius * scale_to_ori)
                    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    # if print_log == True:
                    #   print(x, y,radius)
                    return crop_color


def raw_to_cropped_ori_resol(raw_image, dim, color_check=False, print_log=False, multiplier=0.05):
    original_dim = raw_image.shape
    '''
    :parameter: raw_image
                dim -> output dimension
                color_check -> if True, crop 1.5*radius to focus only on the center color [since if highly contaminated,
                the center will already have enough information and avoid taking the edge color into account]
    :return: cropped 256x256 pixel image centered on the ROI
    '''
    '''
    December
    '''

    def checkradius(x, y, radius):
        if x < 450 or x > 750 or y < 400 or radius < 150:  # May 6th, 560, 700, 400, 150
            x = 640
            y = 450
            radius = 200
            if print_log == True:
                print('adjust x,y,radius to 640,450,200 due to anomalous value(s) of %d, %d, %d.' % (x, y, radius))
            return x, y, radius
        else:
            return x, y, radius

    if color_check == True:
        # def masking(mask, image, x, y, radius, dim):
        #     cropimg = cv2.subtract(mask, image)
        #     cropimg = cv2.subtract(mask, cropimg)
        #     cropimg = cropimg[y - radius + int(radius * 0.5):y + radius - int(radius * 0.5),
        #               x - radius + int(radius * 0.5):x + radius - int(radius * 0.5)]
        #     cropimg = cv2.resize(cropimg, dim, interpolation=cv2.INTER_AREA)
        #     return cropimg
        def masking(mask, image, x, y, radius, dim):
            real_y, real_x = mask.shape
            new_y_upper = np.clip(y + radius - int(radius * 0.5), 0, real_y)
            new_y_lower = np.clip(y - radius + int(radius * 0.5), 0, real_y)
            new_x_upper = np.clip(x + radius - int(radius * 0.5), 0, real_x)
            new_x_lower = np.clip(x - radius + int(radius * 0.5), 0, real_x)
            old_y_upper = y + radius - int(radius * 0.5)
            old_y_lower = y - radius + int(radius * 0.5)
            old_x_upper = x + radius - int(radius * 0.5)
            old_x_lower = x - radius + int(radius * 0.5)
            left_patch = abs(new_x_lower - old_x_lower)
            bottom_patch = abs(new_y_lower - old_y_lower)
            right_patch = abs(new_x_upper - old_x_upper)
            top_patch = abs(new_y_upper - old_y_upper)
            # print('left : {}\nright : {}\ntop : {}\nbottom : {}'.format(left_patch, right_patch, top_patch, bottom_patch))
            mask_to_patch = np.zeros((2*radius,2*radius), dtype=np.uint8)


            cropimg = cv2.subtract(mask, image)
            cropimg = cv2.subtract(mask, cropimg)
            # print('shape of cropimg is {}'.format(cropimg.shape))
            # print('shape of mask_to_patch is {}'.format(mask_to_patch.shape))

            mask_to_patch[0 + bottom_patch: 2*radius - top_patch,
            0 + left_patch: 2*radius - right_patch] = cropimg[new_y_lower:new_y_upper, new_x_lower:new_x_upper]
            cropimg = cv2.resize(mask_to_patch, dim, interpolation=cv2.INTER_AREA)
            return cropimg

        img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        img_color = raw_image
        r = 1200.0 / img.shape[1]
        dimension = (1200, int(img.shape[0] * r))
        img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        img_color = cv2.resize(img_color, dimension, interpolation=cv2.INTER_AREA)

        # BGR channel
        blue, green, red = cv2.split(img_color)
        img_combine_1 = red
        img_combine_2 = blue
        img_combine_3 = green

        # Detecting ROI
        mask = np.zeros((901, 1200), dtype=np.uint8)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                   param1=60, param2=30, minRadius=200, maxRadius=238)
        if type(circles) != type(None):
            if len(circles[0, :, :]) > 1:
                x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))

                return crop_color
            else:
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                return crop_color
        else:
            if print_log == True:
                print('no circle found... try changing parameter (1)')
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                       param1=50, param2=20, minRadius=200,
                                       maxRadius=238)
            if type(circles) != type(None):
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                return crop_color
            else:
                if print_log == True:
                    print('no circles found... try changing parameter (2)')
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                           param1=40, param2=10, minRadius=200,
                                           maxRadius=238)
                if type(circles) != type(None):
                    if len(circles[0, :, :]) > 1:
                        x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        return crop_color
                    else:
                        x, y, radius = np.uint16(circles[0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        return crop_color
                else:
                    if print_log == True:
                        print('no circle found...:c... resorting to empirically estimated x,y,r=640,450,200')
                    x, y, radius = 640, 450, 200
                    cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    return crop_color
    else:  # if color_check=False, crop as usual
        # def masking(mask, image, x, y, radius, dim):
        #     cropimg = cv2.subtract(mask, image)
        #     cropimg = cv2.subtract(mask, cropimg)
        #     cropimg = cropimg[y - radius:y + radius, x - radius:x + radius]
        #     # cropimg = cv2.resize(cropimg, dim, interpolation=cv2.INTER_AREA)
        #     return cropimg

        def masking(mask, image, x, y, radius, dim):
            real_y, real_x = mask.shape
            new_y_upper = np.clip(y + radius, 0, real_y)
            new_y_lower = np.clip(y - radius, 0, real_y)
            new_x_upper = np.clip(x + radius, 0, real_x)
            new_x_lower = np.clip(x - radius, 0, real_x)
            old_y_upper = y + radius
            old_y_lower = y - radius
            old_x_upper = x + radius
            old_x_lower = x - radius
            left_patch = abs(new_x_lower - old_x_lower)
            bottom_patch = abs(new_y_lower - old_y_lower)
            right_patch = abs(new_x_upper - old_x_upper)
            top_patch = abs(new_y_upper - old_y_upper)
            # print(
            #     'left : {}\nright : {}\ntop : {}\nbottom : {}'.format(left_patch, right_patch, top_patch, bottom_patch))
            mask_to_patch = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)

            cropimg = cv2.subtract(mask, image)
            cropimg = cv2.subtract(mask, cropimg)
            # print('shape of cropimg is {}'.format(cropimg.shape))
            # print('shape of mask_to_patch is {}'.format(mask_to_patch.shape))

            mask_to_patch[0 + bottom_patch: 2 * radius - top_patch,
            0 + left_patch: 2 * radius - right_patch] = cropimg[new_y_lower:new_y_upper, new_x_lower:new_x_upper]
            # cropimg = cv2.resize(cropimg, dim, interpolation=cv2.INTER_AREA)
            return mask_to_patch


        img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        img_color = raw_image
        r = 1200.0 / img.shape[1]
        dimension = (1200, int(img.shape[0] * r))
        img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        # img_color = cv2.resize(img_color, dimension, interpolation=cv2.INTER_AREA)

        # BGR channel
        blue, green, red = cv2.split(img_color)
        img_combine_1 = red
        img_combine_2 = blue
        img_combine_3 = green

        # Detecting ROI
        mask = np.zeros((901, 1200), dtype=np.uint8)

        mask = np.zeros((original_dim[0], original_dim[1]), dtype=np.uint8)
        scale_to_ori = original_dim[1] / 1200
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                   param1=70, param2=40, minRadius=230, maxRadius=250)
        if type(circles) != type(None):
            if len(circles[0, :, :]) > 1:
                x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                x, y, radius = checkradius(x, y, radius)
                radius += int(radius * multiplier)
                x = int(x * scale_to_ori)
                y = int(y * scale_to_ori)
                radius = int(radius * scale_to_ori)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color
            else:
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)

                radius += int(radius * multiplier)
                x = int(x * scale_to_ori)
                y = int(y * scale_to_ori)
                radius = int(radius * scale_to_ori)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color
        else:
            if print_log == True:
                print('no circle found... try changing parameter (1)')
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                       param1=60, param2=30, minRadius=230, maxRadius=250)
            if type(circles) != type(None):
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                radius += int(radius * multiplier)
                x = int(x * scale_to_ori)
                y = int(y * scale_to_ori)
                radius = int(radius * scale_to_ori)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                print(mask.shape)
                print(img_combine_1.shape)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color
            else:
                if print_log == True:
                    print('no circles found... try changing parameter (2)')
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                           param1=50, param2=20, minRadius=230, maxRadius=250)
                if type(circles) != type(None):
                    if len(circles[0, :, :]) > 1:
                        x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                        x, y, radius = checkradius(x, y, radius)
                        radius += int(radius * multiplier)
                        x = int(x * scale_to_ori)
                        y = int(y * scale_to_ori)
                        radius = int(radius * scale_to_ori)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        # if print_log == True:
                        #   print(x, y,radius)

                        return crop_color
                    else:
                        x, y, radius = np.uint16(circles[0][0])
                        x, y, radius = checkradius(x, y, radius)
                        radius += int(radius * multiplier)
                        x = int(x * scale_to_ori)
                        y = int(y * scale_to_ori)
                        radius = int(radius * scale_to_ori)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        # if print_log == True:
                        #   print(x, y,radius)

                        return crop_color
                else:
                    if print_log == True:
                        print('no circles found... try changing parameter (3)')
                    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                               param1=40, param2=20, minRadius=230, maxRadius=250)
                    if type(circles) != type(None):
                        if len(circles[0, :, :]) > 1:
                            x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                            x, y, radius = checkradius(x, y, radius)
                            radius += int(radius * multiplier)
                            x = int(x * scale_to_ori)
                            y = int(y * scale_to_ori)
                            radius = int(radius * scale_to_ori)
                            cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                            crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                            crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                            crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                            crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                            # if print_log == True:
                            #   print(x, y,radius)

                            return crop_color
                        else:
                            x, y, radius = np.uint16(circles[0][0])
                            x, y, radius = checkradius(x, y, radius)
                            radius += int(radius * multiplier)
                            x = int(x * scale_to_ori)
                            y = int(y * scale_to_ori)
                            radius = int(radius * scale_to_ori)
                            cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                            crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                            crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                            crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                            crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                            # if print_log == True:
                            #   print(x, y,radius)

                            return crop_color
                    else:
                        if print_log == True:
                            print(
                                'no circle found...:c... resorting to empirically estimated x,y,radius of 640,450,200')
                        x = 640
                        y = 450
                        radius = 200
                        radius += int(radius * multiplier)
                        x = int(x * scale_to_ori)
                        y = int(y * scale_to_ori)
                        radius = int(radius * scale_to_ori)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        # if print_log == True:
                        #   print(x, y,radius)
                        return crop_color



def raw_to_cropped_hi_res(raw_image, dim, color_check=False, print_log=False, multiplier=0.05):
    original_dim = raw_image.shape
    '''
    :parameter: raw_image
                dim -> output dimension
                color_check -> if True, crop 1.5*radius to focus only on the center color [since if highly contaminated,
                the center will already have enough information and avoid taking the edge color into account]
    :return: cropped 256x256 pixel image centered on the ROI
    '''

    def checkradius(x, y, radius):
        return x, y, radius

    scale_to_ori = original_dim[1] / 1200
    if color_check == True:
        # def masking(mask, image, x, y, radius, dim):
        #     cropimg = cv2.subtract(mask, image)
        #     cropimg = cv2.subtract(mask, cropimg)
        #     cropimg = cropimg[y - radius + int(radius * 0.5):y + radius - int(radius * 0.5),
        #               x - radius + int(radius * 0.5):x + radius - int(radius * 0.5)]
        #     cropimg = cv2.resize(cropimg, dim, interpolation=cv2.INTER_AREA)
        #     return cropimg

        def masking(mask, image, x, y, radius, dim):
            real_y, real_x = mask.shape
            new_y_upper = np.clip(y + radius - int(radius * 0.5), 0, real_y)
            new_y_lower = np.clip(y - radius + int(radius * 0.5), 0, real_y)
            new_x_upper = np.clip(x + radius - int(radius * 0.5), 0, real_x)
            new_x_lower = np.clip(x - radius + int(radius * 0.5), 0, real_x)
            old_y_upper = y + radius - int(radius * 0.5)
            old_y_lower = y - radius + int(radius * 0.5)
            old_x_upper = x + radius - int(radius * 0.5)
            old_x_lower = x - radius + int(radius * 0.5)
            left_patch = abs(new_x_lower - old_x_lower)
            bottom_patch = abs(new_y_lower - old_y_lower)
            right_patch = abs(new_x_upper - old_x_upper)
            top_patch = abs(new_y_upper - old_y_upper)
            # print('left : {}\nright : {}\ntop : {}\nbottom : {}'.format(left_patch, right_patch, top_patch, bottom_patch))
            mask_to_patch = np.zeros((2*radius,2*radius), dtype=np.uint8)


            cropimg = cv2.subtract(mask, image)
            cropimg = cv2.subtract(mask, cropimg)
            # print('shape of cropimg is {}'.format(cropimg.shape))
            # print('shape of mask_to_patch is {}'.format(mask_to_patch.shape))

            mask_to_patch[0 + bottom_patch: 2*radius - top_patch,
            0 + left_patch: 2*radius - right_patch] = cropimg[new_y_lower:new_y_upper, new_x_lower:new_x_upper]
            cropimg = cv2.resize(mask_to_patch, dim, interpolation=cv2.INTER_AREA)
            return cropimg

        img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        img_color = raw_image
        r = 1200.0 / img.shape[1]
        dimension = (1200, int(img.shape[0] * r))
        img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        # img_color = cv2.resize(img_color, dimension, interpolation=cv2.INTER_AREA)

        # BGR channel
        blue, green, red = cv2.split(img_color)
        img_combine_1 = red
        img_combine_2 = blue
        img_combine_3 = green

        # Detecting ROI
        mask = np.zeros((899, 1200), dtype=np.uint8)
        mask = np.zeros((original_dim[0], original_dim[1]), dtype=np.uint8)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                   param1=60, param2=30, minRadius=350, maxRadius=380)
        if type(circles) != type(None):
            if len(circles[0, :, :]) > 1:
                x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))

                return crop_color
            else:
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                return crop_color
        else:
            if print_log == True:
                print('no circle found... try changing parameter (1) ...')
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                       param1=50, param2=20, minRadius=350, maxRadius=380)
            if type(circles) != type(None):
                print('theres circle (1)')
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                return crop_color
            else:
                if print_log == True:
                    print('no circles found... try changing parameter (2)')
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                           param1=40, param2=20, minRadius=350, maxRadius=380)
                if type(circles) != type(None):
                    if len(circles[0, :, :]) > 1:
                        x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        return crop_color
                    else:
                        x, y, radius = np.uint16(circles[0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        return crop_color
                else:
                    if print_log == True:
                        print('no circle found...:c... resorting to empirically estimated x,y,r=640,450,200')
                    x, y, radius = 505, 450, 330
                    cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    return crop_color
    else:  # if color_check=False, crop as usual
        def masking(mask, image, x, y, radius, dim):
            real_y, real_x = mask.shape
            new_y_upper = np.clip(y + radius, 0, real_y)
            new_y_lower = np.clip(y - radius, 0, real_y)
            new_x_upper = np.clip(x + radius, 0, real_x)
            new_x_lower = np.clip(x - radius, 0, real_x)
            old_y_upper = y + radius
            old_y_lower = y - radius
            old_x_upper = x + radius
            old_x_lower = x - radius
            left_patch = abs(new_x_lower - old_x_lower)
            bottom_patch = abs(new_y_lower - old_y_lower)
            right_patch = abs(new_x_upper - old_x_upper)
            top_patch = abs(new_y_upper - old_y_upper)
            # print(
            #     'left : {}\nright : {}\ntop : {}\nbottom : {}'.format(left_patch, right_patch, top_patch, bottom_patch))
            mask_to_patch = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)

            cropimg = cv2.subtract(mask, image)
            cropimg = cv2.subtract(mask, cropimg)
            # print('shape of cropimg is {}'.format(cropimg.shape))
            # print('shape of mask_to_patch is {}'.format(mask_to_patch.shape))

            mask_to_patch[0 + bottom_patch: 2 * radius - top_patch,
            0 + left_patch: 2 * radius - right_patch] = cropimg[new_y_lower:new_y_upper, new_x_lower:new_x_upper]
            # cropimg = cv2.resize(cropimg, dim, interpolation=cv2.INTER_AREA)
            return mask_to_patch

        img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        img_color = raw_image
        r = 1200.0 / img.shape[1]
        dimension = (1200, int(img.shape[0] * r))
        img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        # img_color = cv2.resize(img_color, dimension, interpolation=cv2.INTER_AREA)

        # BGR channel
        blue, green, red = cv2.split(img_color)
        img_combine_1 = red
        img_combine_2 = blue
        img_combine_3 = green

        # Detecting ROI
        mask = np.zeros((899, 1200), dtype=np.uint8)
        mask = np.zeros((original_dim[0], original_dim[1]), dtype=np.uint8)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                   param1=70, param2=50, minRadius=350, maxRadius=380)
        if type(circles) != type(None):
            if len(circles[0, :, :]) > 1:
                x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                x, y, radius = checkradius(x, y, radius)
                radius += int(radius * multiplier)
                x = int(x * scale_to_ori)
                y = int(y * scale_to_ori)
                radius = int(radius * scale_to_ori)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color
            else:
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                radius += int(radius * multiplier)
                x = int(x * scale_to_ori)
                y = int(y * scale_to_ori)
                radius = int(radius * scale_to_ori)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color
        else:
            if print_log == True:
                print('no circle found... try changing parameter (1)')
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                       param1=60, param2=30, minRadius=350, maxRadius=380)
            if type(circles) != type(None):
                print('theres circle (1)')
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                radius += int(radius * multiplier)
                x = int(x * scale_to_ori)
                y = int(y * scale_to_ori)
                radius = int(radius * scale_to_ori)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color
            else:
                if print_log == True:
                    print('no circles found... try changing parameter (2)')
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                           param1=50, param2=20, minRadius=350, maxRadius=380)
                if type(circles) != type(None):
                    print('theres circle (2)')

                    if len(circles[0, :, :]) > 1:
                        x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                        x, y, radius = checkradius(x, y, radius)
                        radius += int(radius * multiplier)
                        x = int(x * scale_to_ori)
                        y = int(y * scale_to_ori)
                        radius = int(radius * scale_to_ori)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        # if print_log == True:
                        #   print(x, y,radius)

                        return crop_color
                    else:
                        x, y, radius = np.uint16(circles[0][0])
                        x, y, radius = checkradius(x, y, radius)
                        radius += int(radius * multiplier)
                        x = int(x * scale_to_ori)
                        y = int(y * scale_to_ori)
                        radius = int(radius * scale_to_ori)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        # if print_log == True:
                        #   print(x, y,radius)

                        return crop_color
                else:
                    if print_log == True:
                        print('no circle found...:c... resorting to empirically estimated x,y,radius of 640,450,200')
                    x = 505
                    y = 450
                    radius = 330
                    radius += int(radius * multiplier)
                    x = int(x * scale_to_ori)
                    y = int(y * scale_to_ori)
                    radius = int(radius * scale_to_ori)
                    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    # if print_log == True:
                    #   print(x, y,radius)
                    return crop_color



def RGB_comparator(cropped_img, n_colors=6, n_dominant=2):
    '''
    :parameter: cropped_img
                n_colors [1 + number of dominant colors to be found by k-means clustering (plus 1 due to black background]
                n_dominant [top n_dominant colors to use for calculation of average difference from white color]
    :return:    based on the average difference of the top n_dominant colors from white color
                if
                    first or second element (Red or Green channel) of the average difference is > 120
                then
                    return overgrown_flag = True
                else
                    return overgrown_flag = False
    '''
    # flatten the 3 color channels
    pixels = np.float32(cropped_img.reshape(-1, 3))

    # criteria defined such that whenever 200 iterations of k-means algorithm is ran, or an accuracy of epsilon = 0.1
    # is achieved, stop the algorithm and return output [compactness, labels, centers] where compactness is the
    # sum of squared distance from each point to their corresponding centers, labels with lowest compactness is returned
    # since you want the best clustered result
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)

    # specify how initial centers are taken
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    # sort the indices (note that argsort returns the indices that would sort the given list, counts, in reverse [::-1])
    indices_with_black = np.argsort(counts)[::-1]

    # remove the black background from the indices of dominant colors
    indices = indices_with_black
    for j, index in enumerate(indices_with_black):
        if np.array_equal(np.uint8(palette[index]), np.array([0, 0, 0])):
            index_black = index
            indices = np.delete(indices_with_black, j)
    # calculate average difference of top n dominant colors with white color
    diff = 0
    for i in range(n_dominant):
        diff += np.array([255, 255, 255]) - palette[indices[i]]
    avg_diff = np.int_(diff / n_dominant)

    overgrown_flag = False
    if avg_diff[0] > 120 or avg_diff[1] > 120:
        overgrown_flag = True
        return overgrown_flag
    else:
        return overgrown_flag


def predict_from_model(cropped_img, pred_thresh = 0.42):
    '''
    :parameter : cropped_img
    :return: a list containing binary images of blue and purple colonies (256x256 pixels)
    '''

    # Parameters
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    # Getting image into numpy array for prediction
    X_env_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    img = cropped_img
    X_env_test[0] = img

    # setting up interpreter and inputs

    # Predict colonies
    input_data2 = input_data
    input_data2[0] = cropped_img
    interpreter.set_tensor(input_details[0]['index'], input_data2)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    preds_env_test = output_data

    # Threshold predictions for blue and purple
    preds_env_test_t = preds_env_test
    preds_env_test_t[0, :, :, 0] = (preds_env_test_t[0, :, :, 0] < pred_thresh).astype(np.uint8)
    preds_env_test_t[0, :, :, 1] = (preds_env_test_t[0, :, :, 1] < pred_thresh).astype(np.uint8)

    thresholded_blue = (np.squeeze(preds_env_test_t[0, :, :, 0]))
    thresholded_blue = thresholded_blue * 255
    thresholded_blue = thresholded_blue.astype('uint8')
    thresholded_purple = (np.squeeze(preds_env_test_t[0, :, :, 1]))
    thresholded_purple = thresholded_purple * 255
    thresholded_purple = thresholded_purple.astype('uint8')

    return [thresholded_blue, thresholded_purple]


def segment_and_count(predicted_image, return_image='False'):
    '''
    :parameter: predicted_image [binary image from predict_from_model function]
    :parameter: return_image
    :return:    return_image == 'True'  : images [numpy.ndarray]
                return_image == 'False' : colonies count [int]
    '''

    img = predicted_image
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    # Prediction map was of size (256x256), enlarging gave uneven edge, gaussian blur then threshold
    # to smooth out the edges
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.GaussianBlur(bw, (7, 7), 3)
    ret, bw = cv2.threshold(bw, 30, 255, cv2.THRESH_BINARY)
    bw2 = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    bw2 = bw2.astype('uint8')

    # distance transform performed to obtain the distance map (output is a float of value between 0 to ~15, the further away a pixel is to the edge, the higher the value)
    dist2 = cv2.distanceTransform(bw, cv2.DIST_L2, 5)

    # Normalize the distance image for range = {0.0, 1.0}
    cv2.normalize(dist2, dist2, 0, 1.0, cv2.NORM_MINMAX)

    # Convert to unsigned 8-bit integer (compatible for cv2.imshow)
    dist2 = dist2 * 255
    dist2 = np.uint8(dist2)

    # Obtain coordinates of local maxima (output type = numpy array of maxima coordinates)
    coordinates = peak_local_max(dist2, min_distance=10)

    # Create empty mask
    markers2 = np.zeros(dist2.shape, dtype=np.int32)

    # Draw the colony markers as points
    for i in coordinates:
        cv2.circle(markers2, tuple((i[1], i[0])), 1, (255, 255, 255), -1)

    markers2show = markers2.astype("uint8")

    # Find total markers using findContours
    contours, _ = cv2.findContours(markers2show, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist2.shape, dtype=np.int32)

    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i + 1), -1)

    # Draw the background marker at (1,1) to reduce possibility of background segmentation
    cv2.rectangle(markers, (1, 1), (511, 511), (255, 255, 255), 1)

    # Using the markers, segment the binary image
    cv2.watershed(bw2, markers)

    # Generate random colors
    colors = []
    for contour in contours:
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(200, 256)))

    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i, j]
            if index > 0 and index <= len(contours):
                dst[i, j, :] = colors[index - 1]

    if return_image == 'True':
        return dst, len(contours)
    else:
        return len(contours)


def segment_and_count_boundary(input_img, return_image='True', color='Blue'):
    img = input_img

    bw2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bw2 = bw2.astype('uint8')

    # distance transform performed to obtain the distance map (output is a float of value between 0 to ~15, the further away a pixel is to the edge, the higher the value)
    dist2 = cv2.distanceTransform(img, cv2.DIST_L2, 5)

    # Normalize the distance image for range = {0.0, 1.0}
    cv2.normalize(dist2, dist2, 0, 1.0, cv2.NORM_MINMAX)

    # Convert to unsigned 8-bit integer (compatible for cv2.imshow)
    dist2 = dist2 * 255
    dist2 = np.uint8(dist2)

    # Obtain coordinates of local maxima (output type = numpy array of maxima coordinates)
    coordinates = peak_local_max(dist2, min_distance=2)

    # Create empty mask
    markers2 = np.zeros(dist2.shape, dtype=np.int32)

    # Draw the colony markers as points
    for i in coordinates:
        cv2.circle(markers2, tuple((i[1], i[0])), 0, (255, 255, 255), -1)

    markers2show = markers2.astype("uint8")

    # Find total markers using findContours
    contours, _ = cv2.findContours(markers2show, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist2.shape, dtype=np.int32)

    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i + 1), -1)

    # Draw the background marker at (1,1) to reduce possibility of background segmentation
    cv2.rectangle(markers, (1, 1), (511, 511), (255, 255, 255), 1)

    # Using the markers, segment the binary image
    watershed = cv2.watershed(bw2, markers)
    boundary = (watershed < 1).astype(np.uint8) * 255

    if color == 'Blue':
        boundary_colored = cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)
        boundary_colored[:, :, 1:] = boundary_colored[:, :, 1:] * 0
        if return_image == 'True':
            return boundary_colored, len(contours)
        else:
            return len(contours)
    else:
        boundary_colored = cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)
        boundary_colored[:, :, :2] = boundary_colored[:, :, :2] * 0
        if return_image == 'True':
            return boundary_colored, len(contours)
        else:
            return len(contours)


def get_count(inputimg, color):
    if color == 'Blue':
        return segment_and_count(predict_from_model(raw_to_cropped(inputimg, dim=(256, 256)))[0])
    else:
        return segment_and_count(predict_from_model(raw_to_cropped(inputimg, dim=(256, 256)))[1])


def get_image_and_count(inputimg):
    blue_predict, purple_predict = predict_from_model(raw_to_cropped(inputimg, dim=(256, 256)))
    return segment_and_count_boundary(blue_predict, return_image='True', color='Blue'), \
           segment_and_count_boundary(purple_predict, return_image='True', color='Purple')


# uncomment to save in sample folder format

def saving_result(img_directory='image', result_directory='result/general_test', wait_time=100, color_check=False,
                  save_name=['_ori_', '_e_coli_', '_coliform_'], print_log=False):
    def masking_boundary(ori_image, boundary_image):
        # Load two images
        img1 = ori_image
        img2 = boundary_image

        # ROI
        rows, cols, channels = img2.shape
        roi = img1[0:rows, 0:cols]

        # Now create a mask of boundary and its inverse mask
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of boundary in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # Take only region of boundary
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
        # Put boundary in ROI and modify the ori_image
        dst = cv2.add(img1_bg, img2_fg)
        img1[0:rows, 0:cols] = dst
        return img1

    if color_check == False:
        for img_file in os.listdir(img_directory):
            inputimg = cv2.imread(img_directory + '/' + img_file)
            inputimg2 = cv2.imread(img_directory + '/' + img_file)

            # Get colony counts ONLY (will be deprecated soon)
            # blue_count = get_count(inputimg, 'Blue')
            # purple_count = get_count(inputimg, 'Purple')

            # Get segmented images and counts
            blue_image, blue_count = get_image_and_count(inputimg, 'Blue')
            purple_image, purple_count = get_image_and_count(inputimg, 'Purple')

            # Get the cropped original image of the same dimension for ease of result viewing
            ori_image = raw_to_cropped(inputimg2, dim=(512, 512))
            cv2.rectangle(ori_image, (1, 1), (511, 511), (255, 255, 255), 2)
            if print_log == True:
                print('There are %d E. coli colonies and %d coliforms.' % (blue_count, purple_count))

            # Creating a directory to save the new images and count text file
            if os.path.exists(result_directory) == False:
                os.mkdir(result_directory)
            if os.path.exists(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result') == True:
                i = 1
                while (os.path.exists(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                        i) + ')') == True):
                    i += 1
                if print_log == True:
                    print("Folder exists, saving images and text file in a new folder called " + img_file[:(
                                len(img_file) - 4)] + '_result' + '(' + str(
                        i) + ')' + " to prevent overwriting files in the existing folder.")
                os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')')

                # Saving the counts in a text file
                f = open(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                    i) + ')/' + img_file[:(len(img_file) - 4)] + '_count', "w+")
                f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (blue_count, purple_count))
                f.close()

                # Writing title and count on images
                cv2.putText(blue_image, 'E.coli count : %d' % (blue_count),
                            (5, 500),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2)
                cv2.putText(purple_image, 'coliform count : %d' % (purple_count),
                            (5, 500),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2)
                cv2.rectangle(blue_image, (1, 1), (511, 511), (255, 255, 255), 2)
                cv2.rectangle(purple_image, (1, 1), (511, 511), (255, 255, 255), 2)
                # Saving the segmented images
                cv2.imwrite(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                    i) + ')/' + img_file[:(len(img_file) - 4)] + '_E.coli.png', blue_image)
                cv2.imwrite(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                    i) + ')/' + img_file[:(len(img_file) - 4)] + '_coliform.png', purple_image)
                cv2.imwrite(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                    i) + ')/' + img_file[:(len(img_file) - 4)] + '_c_original.png', ori_image)

                # Uncomment to show segmented images
                # cv2.imshow('E.coli colonies', blue_image)
                # cv2.imshow('coliforms', purple_image)
                if print_log == True:
                    print("Press any key to move on to the next image. (5 seconds)")
                # cv2.waitKey(wait_time)
            else:
                if print_log == True:
                    print("Saving images and text file in a new folder called " + img_file[
                                                                                  :(len(img_file) - 4)] + '_result')
                os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result')

                # Saving the counts in a text file
                f = open(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(
                            len(img_file) - 4)] + '_count', "w+")
                f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (blue_count, purple_count))
                f.close()

                # Writing title and count on images
                cv2.putText(blue_image, 'E.coli count : %d' % (blue_count),
                            (5, 500),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2)
                cv2.putText(purple_image, 'coliform count : %d' % (purple_count),
                            (5, 500),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2)
                cv2.rectangle(blue_image, (1, 1), (511, 511), (255, 255, 255), 2)
                cv2.rectangle(purple_image, (1, 1), (511, 511), (255, 255, 255), 2)
                # Saving the segmented images
                cv2.imwrite(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(
                            len(img_file) - 4)] + '_E.coli.png', blue_image)
                cv2.imwrite(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(
                            len(img_file) - 4)] + '_coliform.png', purple_image)
                cv2.imwrite(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(
                            len(img_file) - 4)] + '_c_original.png', ori_image)

                # Uncomment to show segmented images
                # cv2.imshow('E.coli colonies', blue_image)
                # cv2.imshow('coliforms', purple_image)
                if print_log == True:
                    print("Press any key to move on to the next image. (5 seconds)")
                # cv2.waitKey(wait_time)
    else:
        for img_file in os.listdir(img_directory):
            inputimg = cv2.imread(img_directory + '/' + img_file)
            inputimg2 = cv2.imread(img_directory + '/' + img_file)

            overgrown_flag = RGB_comparator(raw_to_cropped(inputimg, dim=(256, 256), color_check=True))

            if overgrown_flag == False:
                if print_log == True:
                    print('Sample is not overgrown, proceed with counting')
                # Get colony counts ONLY (will be deprecated soon)
                # blue_count = get_count(inputimg, 'Blue')
                # purple_count = get_count(inputimg, 'Purple')

                # Get segmented images and counts
                blue, purple = get_image_and_count(inputimg)
                blue_image, blue_count = blue[0], blue[1]
                purple_image, purple_count = purple[0], purple[1]

                # Get the cropped original image of the same dimension for ease of result viewing
                ori_image = raw_to_cropped(inputimg2, dim=(512, 512))
                cv2.rectangle(ori_image, (1, 1), (511, 511), (255, 255, 255), 2)

                # Creating a directory to save the new images and count text file
                if os.path.exists(result_directory) == False:
                    os.mkdir(result_directory)
                if os.path.exists(
                        result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result') == True:
                    i = 1
                    while (os.path.exists(
                            result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                                i) + ')') == True):
                        i += 1
                    if print_log == True:
                        print("Folder exists, saving images and text file in a new folder called " + img_file[:(
                                len(img_file) - 4)] + '_result' + '(' + str(
                            i) + ')' + " to prevent overwriting files in the existing folder.")
                    os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                        i) + ')')

                    # Saving the counts in a text file
                    f = open(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                        i) + ')/' + img_file[:(len(img_file) - 4)] + '_count', "w+")
                    f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (blue_count, purple_count))
                    f.close()

                    cv2.rectangle(blue_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    cv2.rectangle(purple_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    # Saving the segmented images
                    twoboundary = masking_boundary(masking_boundary(ori_image, blue_image), purple_image)
                    # Writing title and count on images
                    cv2.putText(twoboundary, 'E.coli count : %d' % (blue_count),
                                (5, 500),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2)
                    cv2.putText(twoboundary, 'coliform count : %d' % (purple_count),
                                (5, 450),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2)
                    cv2.imwrite(
                        result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                            i) + ')/' + img_file[:(len(img_file) - 4)] + save_name[0] + '.png', twoboundary)

                    # Uncomment to show segmented images
                    # cv2.imshow('E.coli colonies', blue_image)
                    # cv2.imshow('coliforms', purple_image)
                    if print_log == True:
                        print("Press any key to move on to the next image. (5 seconds)")
                    # cv2.waitKey(wait_time)
                else:
                    if print_log == True:
                        print(
                            "Saving images and text file in a new folder called " + img_file[
                                                                                    :(len(img_file) - 4)] + '_result')
                    os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result')

                    # Saving the counts in a text file
                    f = open(
                        result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(
                                len(img_file) - 4)] + '_count', "w+")
                    f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (blue_count, purple_count))
                    f.close()

                    cv2.rectangle(blue_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    cv2.rectangle(purple_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    # Saving the segmented images
                    twoboundary = masking_boundary(masking_boundary(ori_image, blue_image), purple_image)
                    # Writing title and count on images
                    cv2.putText(twoboundary, 'E.coli count : %d' % (blue_count),
                                (5, 500),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2)
                    cv2.putText(twoboundary, 'coliform count : %d' % (purple_count),
                                (5, 450),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2)
                    cv2.rectangle(blue_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    cv2.rectangle(purple_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    cv2.imwrite(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' +
                                img_file[:(len(img_file) - 4)] + save_name[0] + '.png', ori_image)

                    # Uncomment to show segmented images
                    # cv2.imshow('E.coli colonies', blue_image)
                    # cv2.imshow('coliforms', purple_image)
                    if print_log == True:
                        print("Press any key to move on to the next image. (5 seconds)")
                    # cv2.waitKey(wait_time)
            else:
                if print_log == True:
                    print('sample is overgrown, too many colonies to count')
                flag_string = '(flagged)'
                # Get colony counts ONLY (will be deprecated soon)
                # blue_count = get_count(inputimg, 'Blue')
                # purple_count = get_count(inputimg, 'Purple')

                # Get segmented images and counts
                blue, purple = get_image_and_count(inputimg)
                blue_image, blue_count = blue[0], blue[1]
                purple_image, purple_count = purple[0], purple[1]

                # Get the cropped original image of the same dimension for ease of result viewing
                ori_image = raw_to_cropped(inputimg2, dim=(512, 512))
                cv2.rectangle(ori_image, (1, 1), (511, 511), (255, 255, 255), 2)

                # Unif print_log == True:comment to
                #   print counts
                # if print_log == True:
                #   print('There are %d E. coli colonies and %d coliforms.' %(blue_count, purple_count))

                # Creating a directory to save the new images and count text file
                if os.path.exists(result_directory) == False:
                    os.mkdir(result_directory)
                if os.path.exists(
                        result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result') == True:
                    i = 1
                    while (os.path.exists(
                            result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                                i) + ')') == True):
                        i += 1
                    if print_log == True:
                        print("Folder exists, saving images and text file in a new folder called " + img_file[:(
                                len(img_file) - 4)] + '_result' + '(' + str(
                            i) + ')' + " to prevent overwriting files in the existing folder.")
                    os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                        i) + ')')

                    # Saving the counts in a text file
                    f = open(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                        i) + ')/' + img_file[:(len(img_file) - 4)] + '_count', "w+")
                    f.write('E. coli\t\t: %d %s \ncoliforms\t: %d %s' % (
                    blue_count, flag_string, purple_count, flag_string))
                    f.close()

                    cv2.rectangle(blue_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    cv2.rectangle(purple_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    # Saving the segmented images
                    twoboundary = masking_boundary(masking_boundary(ori_image, blue_image), purple_image)

                    # Writing title and count on images
                    if flag_string == "(flagged)":
                        cv2.putText(twoboundary, flag_string,
                                    (5, 231),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    2)


                    else:
                        cv2.putText(twoboundary, 'E.coli count : %d %s' % (blue_count, flag_string),
                                    (5, 500),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2)
                        cv2.putText(twoboundary, 'coliform count : %d %s' % (purple_count, flag_string),
                                    (5, 450),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2)

                    cv2.imwrite(
                        result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                            i) + ')/' + img_file[:(len(img_file) - 4)] + save_name[0] + '.png', twoboundary)

                    # Uncomment to show segmented images
                    # cv2.imshow('E.coli colonies', blue_image)
                    # cv2.imshow('coliforms', purple_image)
                    if print_log == True:
                        print("Press any key to move on to the next image. (5 seconds)")
                    # cv2.waitKey(wait_time)
                else:
                    if print_log == True:
                        print(
                            "Saving images and text file in a new folder called " + img_file[
                                                                                    :(len(img_file) - 4)] + '_result')
                    os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result')

                    # Saving the counts in a text file
                    f = open(
                        result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(
                                len(img_file) - 4)] + '_count', "w+")
                    f.write('E. coli\t\t: %d %s \ncoliforms\t: %d %s' % (
                    blue_count, flag_string, purple_count, flag_string))
                    f.close()

                    cv2.rectangle(blue_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    cv2.rectangle(purple_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    # Saving the segmented images
                    twoboundary = masking_boundary(masking_boundary(ori_image, blue_image), purple_image)

                    if flag_string == "(flagged)":
                        cv2.putText(twoboundary, flag_string,
                                    (5, 550),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    2)


                    else:
                        cv2.putText(twoboundary, 'E.coli count : %d %s' % (blue_count, flag_string),
                                    (5, 500),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2)
                        cv2.putText(twoboundary, 'coliform count : %d %s' % (purple_count, flag_string),
                                    (5, 450),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2)

                    cv2.imwrite(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' +
                                img_file[:(len(img_file) - 4)] + save_name[0] + '.png', ori_image)
                    # Uncomment to show segmented images
                    # cv2.imshow('E.coli colonies', blue_image)
                    # cv2.imshow('coliforms', purple_image)
                    if print_log == True:
                        print("Press any key to move on to the next image. (5 seconds)")
                    # cv2.waitKey(wait_time)


def saving_count_as_csv(img_directory='image', color_check=False, print_log=False):
    if color_check == False:
        img_name = []
        e_coli_count = []
        coliform_count = []
        for img_file in os.listdir(img_directory):
            inputimg = cv2.imread(img_directory + '/' + img_file)

            # Get colony counts ONLY (will be deprecated soon)
            blue_count = get_count(inputimg, 'Blue')
            purple_count = get_count(inputimg, 'Purple')
            img_name.append(img_file)
            e_coli_count.append(blue_count)
            coliform_count.append(purple_count)
        return img_name, e_coli_count, coliform_count

    else:
        img_name = []
        e_coli_count = []
        coliform_count = []
        for i, img_file in enumerate(os.listdir(img_directory)):
            inputimg = cv2.imread(img_directory + '/' + img_file)

            overgrown_flag = RGB_comparator(raw_to_cropped(inputimg, dim=(256, 256), color_check=True))

            if overgrown_flag == False:
                if print_log == True:
                    print('Sample is not overgrown, proceed with counting')
                # Get colony counts ONLY
                blue_count = get_count(inputimg, 'Blue')
                purple_count = get_count(inputimg, 'Purple')
                img_name.append(img_file)
                e_coli_count.append(blue_count)
                coliform_count.append(purple_count)
            else:
                if print_log == True:
                    print('sample is overgrown, too many colonies to count')
                blue_count = get_count(inputimg, 'Blue') + 0.1
                purple_count = get_count(inputimg, 'Purple') + 0.1
                img_name.append(img_file)
                e_coli_count.append(blue_count)
                coliform_count.append(purple_count)
        return img_name, e_coli_count, coliform_count


def model_param():
    # Defining metrics for evaluating prediction accuracy
    def tversky_loss(y_true, y_pred):
        alpha = 0.4
        beta = 0.6
        ones = K.ones(K.shape(y_true))
        p0 = y_pred  # proba that voxels are class i
        p1 = ones - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = ones - y_true
        num = K.sum(p0 * g0, (0, 1, 2))
        den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2))
        T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]
        Ncl = K.cast(K.shape(y_true)[-1], 'float32')
        return Ncl - T

    def generalized_dice_coeff(y_true, y_pred):
        Ncl = y_pred.shape[-1]
        w = K.zeros(shape=(Ncl,))
        w = K.sum(y_true, axis=(0, 1, 2))
        w = 1 / (w ** 2 + 0.000001)
        # Compute gen dice coef:
        numerator = y_true * y_pred
        numerator = w * K.sum(numerator, (0, 1, 2))
        numerator = K.sum(numerator)
        denominator = y_true + y_pred
        denominator = w * K.sum(denominator, (0, 1, 2))
        denominator = K.sum(denominator)
        gen_dice_coef = 2 * numerator / denominator
        return gen_dice_coef

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    return interpreter, input_details, input_data, output_details


def flagging_version2(unet_count_blue, yolo_count_blue, unet_count_purple, yolo_count_purple, flag, high_thresh=10,
                      too_many_thresh=30, yolo_edge_case_thresh=5):
    flag_type = {'anomalous': 0, 'too_many': 0, 'too_many_coliform': 0, 'anomalous_coliform': 0, 'trust_unet_blue': 0,
                 'trust_unet_purple': 0}
    '''
    using unet & yolo counts, establish the type of flag into three types : 
    1, 0 = anomalous
    0, 1 = too_many
    1, 1 = unsure
    note that :
    - anomalous can be due to intense background color, causing flagging even though count is accurate
    - too_many suggests count cannot be trusted, and count should be converted to an arbitrarily large number
    - unsure means exactly what it means

    last flag : trust_unet == 1 if we trust u-net count, otherwise 0
    '''

    if flag:
        if unet_count_blue == 0:
            if yolo_count_blue == 0:
                flag_type['trust_unet_blue'] = 1
                flag_type['anomalous'] = 1
            elif yolo_count_blue >= 1 and yolo_count_blue <= high_thresh:
                flag_type['anomalous'] = 1
                flag_type['too_many'] = 1
            elif yolo_count_blue > high_thresh and yolo_count_blue <= too_many_thresh:
                flag_type['anomalous'] = 1
            else:
                flag_type['too_many'] = 1
        elif unet_count_blue >= 1 and unet_count_blue <= high_thresh:
            if yolo_count_blue == 0:
                flag_type['trust_unet_blue'] = 1
                flag_type['anomalous'] = 1
            elif yolo_count_blue >= 1 and yolo_count_blue <= high_thresh:
                flag_type['trust_unet_blue'] = 1
                flag_type['anomalous'] = 1
            elif yolo_count_blue > high_thresh and yolo_count_blue <= too_many_thresh:
                flag_type['anomalous'] = 1
            else:
                flag_type['too_many'] = 1
        elif unet_count_blue > high_thresh and unet_count_blue <= too_many_thresh:
            if yolo_count_blue == 0:
                flag_type['trust_unet_blue'] = 1
                flag_type['anomalous'] = 1
            elif yolo_count_blue >= 1 and yolo_count_blue <= high_thresh:
                flag_type['trust_unet_blue'] = 1
                flag_type['anomalous'] = 1
            elif yolo_count_blue > high_thresh and yolo_count_blue <= too_many_thresh:
                flag_type['trust_unet_blue'] = 1
                flag_type['anomalous'] = 1
            else:
                flag_type['too_many'] = 1
        else:
            if yolo_count_blue == 0:
                flag_type['trust_unet_blue'] = 1
                flag_type['too_many'] = 1
            elif yolo_count_blue >= 1 and yolo_count_blue <= high_thresh:
                flag_type['trust_unet_blue'] = 1
                flag_type['too_many'] = 1
            elif yolo_count_blue > high_thresh and yolo_count_blue <= too_many_thresh:
                flag_type['trust_unet_blue'] = 1
                flag_type['too_many'] = 1
            else:
                flag_type['trust_unet_blue'] = 1
                flag_type['too_many'] = 1
        ############################ COLIFORM #######################
        if unet_count_purple == 0:
            if yolo_count_purple == 0:
                flag_type['trust_unet_purple'] = 1
                flag_type['anomalous_coliform'] = 1
            elif yolo_count_purple >= 1 and yolo_count_purple <= high_thresh:
                flag_type['anomalous_coliform'] = 1
                flag_type['too_many_coliform'] = 1
            elif yolo_count_purple > high_thresh and yolo_count_purple <= too_many_thresh:
                flag_type['anomalous_coliform'] = 1
            else:
                flag_type['too_many_coliform'] = 1
        elif unet_count_purple >= 1 and unet_count_purple <= high_thresh:
            if yolo_count_purple == 0:
                flag_type['trust_unet_purple'] = 1
                flag_type['anomalous_coliform'] = 1
            elif yolo_count_purple >= 1 and yolo_count_purple <= high_thresh:
                flag_type['trust_unet_purple'] = 1
                flag_type['anomalous_coliform'] = 1
            elif yolo_count_purple > high_thresh and yolo_count_purple <= too_many_thresh:
                flag_type['anomalous_coliform'] = 1
            else:
                flag_type['too_many_coliform'] = 1
        elif unet_count_purple > high_thresh and unet_count_purple <= too_many_thresh:
            if yolo_count_purple == 0:
                flag_type['trust_unet_purple'] = 1
                flag_type['anomalous_coliform'] = 1
            elif yolo_count_purple >= 1 and yolo_count_purple <= high_thresh:
                flag_type['trust_unet_purple'] = 1
                flag_type['anomalous_coliform'] = 1
            elif yolo_count_purple > high_thresh and yolo_count_purple <= too_many_thresh:
                flag_type['trust_unet_purple'] = 1
                flag_type['anomalous_coliform'] = 1
            else:
                flag_type['too_many_coliform'] = 1
        else:
            if yolo_count_purple == 0:
                flag_type['trust_unet_purple'] = 1
                flag_type['too_many_coliform'] = 1
            elif yolo_count_purple >= 1 and yolo_count_purple <= high_thresh:
                flag_type['trust_unet_purple'] = 1
                flag_type['too_many_coliform'] = 1
            elif yolo_count_purple > high_thresh and yolo_count_purple <= too_many_thresh:
                flag_type['trust_unet_purple'] = 1
                flag_type['too_many_coliform'] = 1
            else:
                flag_type['trust_unet_purple'] = 1
                flag_type['too_many_coliform'] = 1

    return flag_type


def analysis_image_old(img_name='image.jpg', result='result.jpg', print_log=False):
    def masking_boundary(ori_image, boundary_image):
        # Load two images
        img1 = ori_image
        img2 = boundary_image

        # ROI
        rows, cols, channels = img2.shape
        roi = img1[0:rows, 0:cols]

        # Now create a mask of boundary and its inverse mask
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of boundary in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # Take only region of boundary
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
        # Put boundary in ROI and modify the ori_image
        dst = cv2.add(img1_bg, img2_fg)
        img1[0:rows, 0:cols] = dst
        return img1

    if '.jpg' not in img_name[-4:] and '.png' not in img_name[:-4]:
        img_name = img_name + '.jpg'
    else:
        img_name = img_name
    inputimg = cv2.imread(img_name)
    inputimg2 = cv2.imread(img_name)
    overgrown_flag = RGB_comparator(raw_to_cropped(inputimg, dim=(256, 256), color_check=True))

    if '.jpg' not in result and '.png' not in result:
        result_name = result + '.png'
    else:
        result_name = result

    count_name = result_name[:-3] + 'txt'
    if overgrown_flag == False:
        if print_log == True:
            print('Sample is not overgrown, proceed with counting')
        # Get colony counts ONLY (will be deprecated soon)
        # blue_count = get_count(inputimg, 'Blue')
        # purple_count = get_count(inputimg, 'Purple')

        # Get segmented images and counts
        blue, purple = get_image_and_count(inputimg)
        blue_image, blue_count = blue[0], blue[1]
        purple_image, purple_count = purple[0], purple[1]

        # Get the cropped original image of the same dimension for ease of result viewing
        ori_image = raw_to_cropped(inputimg2, dim=(256, 256))
        cv2.rectangle(ori_image, (1, 1), (255, 255), (255, 255, 255), 2)

        # Saving the counts in a text file
        f = open(count_name, "w+")
        f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (blue_count, purple_count))
        f.close()

        cv2.rectangle(blue_image, (1, 1), (511, 511), (255, 255, 255), 2)
        cv2.rectangle(purple_image, (1, 1), (511, 511), (255, 255, 255), 2)
        # Saving the segmented images
        twoboundary = masking_boundary(masking_boundary(ori_image, blue_image), purple_image)
        # Writing title and count on images
        cv2.putText(twoboundary, 'E.coli count : %d' % (blue_count),
                    (5, 231),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2)
        cv2.putText(twoboundary, 'coliform count : %d' % (purple_count),
                    (5, 206),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2)
        cv2.imwrite(result_name, twoboundary)

        return {'e.coli': blue_count, 'coliform': purple_count}

    else:
        if print_log == True:
            print('sample is overgrown, too many colonies to count')
        flag_string = '(flagged)'
        # Get colony counts ONLY (will be deprecated soon)
        # blue_count = get_count(inputimg, 'Blue')
        # purple_count = get_count(inputimg, 'Purple')

        # Get segmented images and counts
        blue, purple = get_image_and_count(inputimg)
        blue_image, blue_count = blue[0], blue[1]
        purple_image, purple_count = purple[0], purple[1]

        # Get the cropped original image of the same dimension for ease of result viewing
        ori_image = raw_to_cropped(inputimg2, dim=(256, 256))
        cv2.rectangle(ori_image, (1, 1), (255, 255), (255, 255, 255), 2)

        # Saving the counts in a text file
        f = open(count_name, "w+")
        f.write('E. coli\t\t: %d %s \ncoliforms\t: %d %s' % (blue_count, flag_string, purple_count, flag_string))
        f.close()

        cv2.rectangle(blue_image, (1, 1), (255, 255), (255, 255, 255), 2)
        cv2.rectangle(purple_image, (1, 1), (255, 255), (255, 255, 255), 2)
        # Saving the segmented images
        twoboundary = masking_boundary(masking_boundary(ori_image, blue_image), purple_image)

        # Writing title and count on images
        if flag_string == "(flagged)":
            cv2.putText(twoboundary, 'too many to count or flagged',
                        (5, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2)
            cv2.putText(twoboundary, 'E.coli count : %d %s' % (blue_count, flag_string),
                        (5, 231),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2)
            cv2.putText(twoboundary, 'coliform count : %d %s' % (purple_count, flag_string),
                        (5, 206),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2)
            cv2.imwrite(result_name, twoboundary)
            return {'e.coli': blue_count, 'coliform': purple_count}

        else:
            cv2.putText(twoboundary, 'E.coli count : %d %s' % (blue_count, flag_string),
                        (5, 231),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2)
            cv2.putText(twoboundary, 'coliform count : %d %s' % (purple_count, flag_string),
                        (5, 206),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2)
            cv2.imwrite(result_name, twoboundary)

            return {'e.coli': blue_count, 'coliform': purple_count}


def analysis_image(img_name='image.jpg', result='result.jpg', predict_thresh = 0.42, print_log=False):
    def masking_boundary(ori_image, boundary_image):
        # Load two images
        img1 = ori_image
        img2 = boundary_image

        # ROI
        rows, cols, channels = img2.shape
        roi = img1[0:rows, 0:cols]

        # Now create a mask of boundary and its inverse mask
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of boundary in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # Take only region of boundary
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
        # Put boundary in ROI and modify the ori_image
        dst = cv2.add(img1_bg, img2_fg)
        img1[0:rows, 0:cols] = dst
        return img1

    if '.jpg' not in img_name[-4:] and '.png' not in img_name[:-4]:
        img_name = img_name + '.jpg'
    else:
        img_name = img_name
    inputimg = cv2.imread(img_name)

    if inputimg.shape[1] > 3500 :
        cropped = raw_to_cropped_hi_res(inputimg, (256, 256))
        print("high res image detected")
        print(inputimg.shape)
        
    else:
        cropped = raw_to_cropped_ori_resol(inputimg, (256,256))
    cv2.imwrite('cropped.png', cropped)
    
    cropped = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)
    overgrown_flag = RGB_comparator(raw_to_cropped_color_check(inputimg, dim=(256, 256)))

    # save and open in PIL image ... to do : find a better way?
    cv2.imwrite('cropped.png', cropped)
    large_crop_PIL = Image.open('cropped.png')

    if '.jpg' not in result and '.png' not in result:
        result_name = result + '.png'
    else:
        result_name = result

    count_name = result_name[:-3] + 'txt'
    if overgrown_flag == False:
        if print_log == True:
            print('Sample is not overgrown, proceed with counting')
        image_draw, out_label = yolo.detect_image(large_crop_PIL)

        image_draw_array = np.array(image_draw)
        image_draw_array = cv2.cvtColor(image_draw_array, cv2.COLOR_RGB2BGR)

        # Get segmented images and counts
        blue_predict, purple_predict = predict_from_model(cropped, pred_thresh=0.42)

        # get rid of false predictions outside of ROI
        mask_test = np.zeros((blue_predict.shape[0], blue_predict.shape[1]), dtype=np.uint8)

        cv2.circle(mask_test, (int(blue_predict.shape[0] / 2), int(blue_predict.shape[0] / 2)),
                   int(blue_predict.shape[0] / 2) - 5, (1, 1, 1), -1, 8, 0)

        blue_predict = np.multiply(mask_test, blue_predict)
        purple_predict = np.multiply(mask_test, purple_predict)

        blue, purple = segment_and_count_boundary(blue_predict, return_image='True', color='Blue'), \
                       segment_and_count_boundary(purple_predict, return_image='True', color='Purple')
        blue_image, blue_count = blue[0], blue[1]
        purple_image, purple_count = purple[0], purple[1]

        blue_count_new = blue_count
        purple_count_new = purple_count

        # Get the cropped original image of the same dimension for ease of result viewing
        cv2.rectangle(cropped, (1, 1), (255, 255), (255, 255, 255), 2)

        cv2.rectangle(blue_image, (1, 1), (255, 255), (255, 255, 255), 2)
        cv2.rectangle(purple_image, (1, 1), (255, 255), (255, 255, 255), 2)

        # Saving the counts in a text file
        f = open(count_name, "w+")
        f.write(
            '(u-net) E. coli\t\t: %d \n(u-net) coliforms\t: %d\n(yolo) E. coli\t\t: %d \n(yolo) coliforms\t: %d\n%s\n%d\n%d' % (
            blue_count, purple_count, len(out_label['blue']), len(out_label['coliform']), 'normal', blue_count_new,
            purple_count_new))
        f.close()

        # Saving the segmented images
        twoboundary = masking_boundary(masking_boundary(cropped, blue_image), purple_image)
        # Writing title and count on images
        cv2.putText(twoboundary, 'normal',
                    (5, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)
        cv2.putText(twoboundary, 'E.coli : %d' % (blue_count),
                    (5, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)
        cv2.putText(twoboundary, 'coliform : %d' % (purple_count),
                    (5, 225),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1)
        cv2.imwrite(result_name, twoboundary)
        os.remove('cropped.png')
        return {'e.coli': blue_count, 'coliform': purple_count}


    else:
        if print_log == True:
            print('sample is overgrown, too many colonies to count')
        image_draw, out_label = yolo.detect_image(large_crop_PIL)

        image_draw_array = np.array(image_draw)
        image_draw_array = cv2.cvtColor(image_draw_array, cv2.COLOR_RGB2BGR)
        # Get segmented images and counts
        blue_predict, purple_predict = predict_from_model(cropped)

        # get rid of false predictions outside of ROI
        mask_test = np.zeros((blue_predict.shape[0], blue_predict.shape[1]), dtype=np.uint8)

        cv2.circle(mask_test, (int(blue_predict.shape[0] / 2), int(blue_predict.shape[0] / 2)),
                   int(blue_predict.shape[0] / 2) - 5, (1, 1, 1), -1, 8, 0)

        blue_predict = np.multiply(mask_test, blue_predict)
        purple_predict = np.multiply(mask_test, purple_predict)

        blue, purple = segment_and_count_boundary(blue_predict, return_image='True', color='Blue'), \
                       segment_and_count_boundary(purple_predict, return_image='True', color='Purple')
        blue_image, blue_count = blue[0], blue[1]
        purple_image, purple_count = purple[0], purple[1]
        flag_type = flagging_version2(blue_count, len(out_label['blue']), purple_count, len(out_label['coliform']),
                                      overgrown_flag)
        if flag_type['anomalous'] and flag_type['too_many']:
            flag_string = 'unsure (e.coli)'
        elif flag_type['anomalous']:
            flag_string = 'anomalous (e.coli)'
        else:
            flag_string = 'too_many (e.coli)'
        if flag_type['anomalous_coliform'] and flag_type['too_many_coliform']:
            flag_string += ' unsure (coliform)'
        elif flag_type['anomalous_coliform']:
            flag_string += ' anomalous (coliform)'
        else:
            flag_string += ' too_many (coliform)'

        # if flag_type['trust_unet_blue'] == 0:
        #     blue_count_new = len(out_label['blue'])
        # else:
        #     blue_count_new = blue_count
        # if flag_type['trust_unet_purple'] == 0:
        #     purple_count_new = len(out_label['coliform'])
        # else:
        #     purple_count_new = purple_count
        #

        # if flag_type['too_many'] and flag_type['anomalous'] == 0:
        #     blue_count_new = 200
        # if flag_type['too_many_coliform'] and flag_type['anomalous_coliform'] == 0:
        #     purple_count_new = 200

        # Get the cropped original image of the same dimension for ease of result viewing
        # cv2.rectangle(image_draw_array, (1, 1), (255, 255), (255, 255, 255), 2)
        cv2.rectangle(cropped, (1, 1), (255, 255), (255, 255, 255), 2)

        blue_count_new = blue_count
        purple_count_new = purple_count

        # Saving the counts in a text file
        f = open(count_name, "w+")
        f.write(
            '(u-net) E. coli\t\t: %d \n(u-net) coliforms\t: %d\n(yolo) E. coli\t\t: %d \n(yolo) coliforms\t: %d\n%s\n%d\n%d' % (
                blue_count, purple_count, len(out_label['blue']), len(out_label['coliform']), flag_string,
                blue_count_new,
                purple_count_new))
        f.close()

        cv2.rectangle(blue_image, (1, 1), (255, 255), (255, 255, 255), 2)
        cv2.rectangle(purple_image, (1, 1), (255, 255), (255, 255, 255), 2)
        # Saving the segmented images
        twoboundary = masking_boundary(masking_boundary(cropped, blue_image), purple_image)

        # Writing title and count on images
        if flag_string:
            twoboundary = masking_boundary(masking_boundary(cropped, blue_image), purple_image)
            # Writing title and count on images
            if flag_string:
                cv2.putText(twoboundary, flag_string,
                            (5, 210),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 255, 255),
                            1)
                cv2.putText(twoboundary, 'E.coli : %d ' % (blue_count_new),
                            (5, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 255, 255),
                            1)
                cv2.putText(twoboundary, 'coliform : %d' % (purple_count_new),
                            (5, 225),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 255, 255),
                            1)
                cv2.imwrite(result_name, twoboundary)
                os.remove('cropped.png')
            return {'e.coli': blue_count_new, 'coliform': purple_count_new}


        else:
            cv2.putText(twoboundary, 'E.coli : %d' % (blue_count_new),
                        (5, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1)
            cv2.putText(twoboundary, 'coliform : %d' % (purple_count_new),
                        (5, 225),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1)
            cv2.imwrite(result_name, twoboundary)
            os.remove('cropped.png')
            return {'e.coli': blue_count_new, 'coliform': purple_count_new}


# if __name__ == "__main__":

# Defining metrics for evaluating prediction accuracy (not used in the 25nd_June_small3.tflite model)
def tversky_loss(y_true, y_pred):
    alpha = 0.4
    beta = 0.6
    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true
    num = K.sum(p0 * g0, (0, 1, 2))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2)) + beta * K.sum(p1 * g0, (0, 1, 2))
    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0, 1, 2))
    w = 1 / (w ** 2 + 0.000001)
    # Compute gen dice coef:
    numerator = y_true * y_pred
    numerator = w * K.sum(numerator, (0, 1, 2))
    numerator = K.sum(numerator)
    denominator = y_true + y_pred
    denominator = w * K.sum(denominator, (0, 1, 2))
    denominator = K.sum(denominator)
    gen_dice_coef = 2 * numerator / denominator
    return gen_dice_coef


# New Custom Metrics (specifically for the model used in training 25nd_June_small3.tflite
def dice_coef(y_true, y_pred):
    smooth = 0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def jacard(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection / union


# Load the TFLite model and allocate tensors.
interpreter = Interpreter(model_path='11_april_256_multi-res_filter_24_bs_16_last_epoch.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# # Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

FLAGS = {'image': True, 'input': 'integrate_folder/normal_yolo_evaluate/',
         'output': 'integrate_folder/',
         'model_path': 'integrate_folder/model_data/tiny-yolov3_coliform_500v300_2class_100img.h5',
         'classes_path': 'integrate_folder/_classes.txt'}

yolo = YOLO(**FLAGS)
