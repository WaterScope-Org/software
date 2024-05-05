import argparse
import os
import platform
import time
from pathlib import Path
import timeit
import json
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolor_pi.inference_script.utils.plots import plot_one_box
from yolor_pi.inference_script.utils.torch_utils import select_device, load_classifier, time_synchronized

from yolor_pi.inference_script.models.models import *
from yolor_pi.inference_script.utils.datasets import *
from yolor_pi.inference_script.utils.general import *

from tflite_runtime.interpreter import Interpreter  # on Pi, uncomment this; otherwise, use tf.lite.Interpreter

import numpy as np
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
import skimage


def rough_crop_estimate(im,zero=False, mm13=False, mm25=True):
    """ Apply a rough, square, 3*sigma crop to the input image
    (This should help speed up the next step)
    This is done using empirical values.
    Also returns some useful values.
    """
    
    # Define empirical measures for regular system
    if mm13 == True:
        # Measured empirical data from previous images
        # 13 mm
        original_shape = (1944,2592)
        ye, xe = (1057,1298)
        re = 500
        sd_x, sd_y, sd_r = 20, 60, 31
        
    if mm25 == True:
        # 25 mm
        original_shape = (1944,2592)
        ye, xe = (1024,1280)
        re = 900

        # Standard deviations of measurements
        sd_x, sd_y, sd_r = 50,50,50
    
    # Define empirical measures for zero system
    if zero==True:
        # Measured empirical data from previous images
        # Old WS Zero
#         original_shape = (1944,2592)
#         ye, xe = (980,1249)
#         re = 744
#         # Standard deviations of measurements
#         sd_x, sd_y, sd_r = 47, 36, 48

        # New WS Zero
        original_shape = (3496, 4656)
        ye,xe = (1307,2485)
        re = 620
        sd_x,sd_y,sd_r = 100,70,100
        
    # Check if the shape is the same
    if np.shape(im) == original_shape:
        factor=1

    # Otherwise, we can simply scale everything
    else:
        # f>1 --> input is bigger than original
        # f<1 --> input is smaller than original
        factor =  np.shape(im)[0]/original_shape[0]
        
    # Apply scale to get new empirical values
    xn,yn,rn = xe*factor,ye*factor,re*factor
       
    # Generous 3*sigma initial crop to speed hough step
    sig=3.5
    xl,xu = int(xn-sig*sd_x*factor - (rn+sig*sd_r*factor)),int(xn+sig*sd_x*factor + (rn+sig*sd_r*factor))
    yl,yu = int(yn-sig*sd_y*factor - (rn+sig*sd_r*factor)),int(yn+sig*sd_x*factor + (rn+sig*sd_r*factor))
    
    # Catch exceptions where limits might be beyond original image space
    if xl < 0:
        xl=0
    if yl < 0:
        yl=0
    if xu>np.shape(im)[1]:
        xu=np.shape(im)[1]
    if yu>np.shape(im)[0]:
        yu=np.shape(im)[0]
    
    # Crop the image
    im_roughcrop = im[yl:yu,xl:xu]
    
    if mm25 == True:
    
        ### Make the border more prominent
            # Change contrast
        im_roughcrop = cv2.equalizeHist(im)

            # Blur
        im_roughcrop = cv2.GaussianBlur(im_roughcrop,(111,111),10)

            # Edge
        im_roughcrop = cv2.Canny(im_roughcrop,20,1)

            # Hull mask
        hull = skimage.morphology.convex_hull_image(im_roughcrop)
        kernel = np.ones((310,310),'uint8')
        inner= cv2.dilate(1-hull.astype(np.uint8),kernel)
        inner = 255-(inner-hull)
        outer= cv2.erode(1-hull.astype(np.uint8),kernel)
        outer = -outer
        inner[inner<100]=0
        inner[inner>=100]=1
        outer[outer<100]=0
        outer[outer>=100]=1
        mask = inner-outer

            # Apply mask 
        im_roughcrop = im_roughcrop*mask
    
    # get new empirical centre of cropped image
    xn_rough = xn - int(xn-3*sd_x*factor - (rn+3*sd_r*factor))
    yn_rough = yn - int(yn-3*sd_y*factor - (rn+3*sd_r*factor))
    
    return im_roughcrop,xn_rough,yn_rough,rn,factor,xl,yl,sd_x, sd_y, sd_r 
    
def hough_estimate(im,zero=False, mm13=False, mm25=True):
    """" Try to improve on the empirical guess using the
    Hough circle transform"""
    
    # Get empirical guess
    im_rough,xre,yre,re,f,xl,yl,sd_x, sd_y, sd_r  = rough_crop_estimate(im,zero=zero, mm13=mm13, mm25=mm25)
    
    # dp = accumulator resolution ratio
    # mindist = min between circles
    # p1 = high canny threshold (low canny will be half)
    # p2 = accumulator threshold
    p1,p2 = 20,30
    if zero == True:
        # old zero
        #p1,p2 = 50, 50
        # New zero
        p1,p2 = 50, 100
    circles = cv2.HoughCircles(im_rough, cv2.HOUGH_GRADIENT, 1, 100,
                                               param1=p1, param2=p2, minRadius=int((re-3*sd_r)*f), maxRadius=int((re+3*sd_r)*f))
    
    # Standard deviation in x & y are 20 & 60
    # We also know the algorithm typically overestimates radius
    # So we will use the smallest radius at which x & y are within 2 sigma
        # circles is always in an extra list, and has smallest radii at the end, so let's flip it
    try:
        # Get biggest circle
        circles = circles[0]
        c=circles[0]
        xh, yh, rh = c[0],c[1],c[2]
        return (xh+xl)*1/f,(yh+yl)*1/f,rh*1/f
    except:
        # If none identified, return empirical
        return xre+xl,yre+yl,re

def create_circular_mask(h, w, center=None, radius=None):
    """ Create circular array of 0/1 of specified image height/width and circle centre/radius """
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def apply_crop(im,x,y,r):
    """ Apply crop to image given circle centre and radius """
    # get shape
    h,w = np.shape(im)
    
    # Create mask
    mask = create_circular_mask(h,w,center=(x,y),radius=r)
    
    # Apply mask
    masked_img = im
    masked_img[~mask] = 0
    
    # Crop around the edge
    yl=int(y-r)
    yu = int(y+r)
    xl=int(x-r)
    xu = int(x+r)

    if xl < 0:
        xl=0
    if yl < 0:
        yl=0
    if xu>np.shape(masked_img)[1]:
        xu=np.shape(masked_img)[1]
    if yu>np.shape(masked_img)[0]:
        yu=np.shape(masked_img)[0]

    masked_img = masked_img[yl:yu,xl:xu]
    
    return masked_img

def crop_colour_im(im,x,y,r,zero = False, mm13=False, mm25=True):
    """ Apply circular crop to color image """
    r=0.8*r
    if zero == True:
        # RGB channels need swapping
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    
    # Split image into colour channels
    red,g,b = cv2.split(im)
    
    # Crop each channel
    b = apply_crop(b,x,y,r)
    g = apply_crop(g,x,y,r)
    red = apply_crop(red,x,y,r)
    
    # Merge results back together
    cropped = cv2.merge((red,g,b))
    
    return cropped

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

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
            mask_to_patch = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)

            cropimg = cv2.subtract(mask, image)
            cropimg = cv2.subtract(mask, cropimg)
            # print('shape of cropimg is {}'.format(cropimg.shape))
            # print('shape of mask_to_patch is {}'.format(mask_to_patch.shape))

            mask_to_patch[0 + bottom_patch: 2 * radius - top_patch,
            0 + left_patch: 2 * radius - right_patch] = cropimg[new_y_lower:new_y_upper, new_x_lower:new_x_upper]
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

                return crop_color, x, y, radius
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

                return crop_color, x, y, radius
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
                # print(mask.shape)
                # print(img_combine_1.shape)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # if print_log == True:
                #   print(x, y, radius)

                return crop_color, x, y, radius
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

                        return crop_color, x, y, radius
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

                        return crop_color, x, y, radius
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

                            return crop_color, x, y, radius
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

                            return crop_color, x, y, radius
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
                        return crop_color, x, y, radius

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
        color_check_radius_multiplier = 3
        color_check_dimension = (64, 64)

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
            mask_to_patch = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)

            cropimg = cv2.subtract(mask, image)
            cropimg = cv2.subtract(mask, cropimg)

            mask_to_patch[0 + bottom_patch: 2 * radius - top_patch,
            0 + left_patch: 2 * radius - right_patch] = cropimg[new_y_lower:new_y_upper, new_x_lower:new_x_upper]
            return mask_to_patch

        img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        img_color = raw_image
        r = 1200.0 / img.shape[1]
        dimension = (1200, int(img.shape[0] * r))
        img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

        # BGR channel
        blue, green, red = cv2.split(img_color)
        img_combine_1 = red
        img_combine_2 = blue
        img_combine_3 = green

        # Detecting ROI
        mask = np.zeros((original_dim[0], original_dim[1]), dtype=np.uint8)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                   param1=70, param2=50, minRadius=350, maxRadius=380)
        if type(circles) != type(None):
            if len(circles[0, :, :]) > 1:
                x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                x, y, radius = checkradius(x, y, radius)
                x = int(x * scale_to_ori)
                y = int(y * scale_to_ori)
                radius += int(radius * multiplier)
                radius = int(radius * color_check_radius_multiplier)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                crop_color = cv2.resize(crop_color, color_check_dimension, interpolation=cv2.INTER_AREA)
                return crop_color, x, y, radius
            else:
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                x = int(x * scale_to_ori)
                y = int(y * scale_to_ori)
                radius += int(radius * multiplier)
                radius = int(radius * color_check_radius_multiplier)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                crop_color = cv2.resize(crop_color, color_check_dimension, interpolation=cv2.INTER_AREA)
                return crop_color, x, y, radius
        else:
            if print_log == True:
                print('no circle found... try changing parameter (1) ...')
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                       param1=60, param2=30, minRadius=350, maxRadius=380)
            if type(circles) != type(None):
                print('theres circle (1)')
                x, y, radius = np.uint16(circles[0][0])
                x, y, radius = checkradius(x, y, radius)
                x = int(x * scale_to_ori)
                y = int(y * scale_to_ori)
                radius += int(radius * multiplier)
                radius = int(radius * color_check_radius_multiplier)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                crop_color = cv2.resize(crop_color, color_check_dimension, interpolation=cv2.INTER_AREA)
                return crop_color, x, y, radius
            else:
                if print_log == True:
                    print('no circles found... try changing parameter (2)')
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                           param1=50, param2=20, minRadius=350, maxRadius=380)
                if type(circles) != type(None):
                    if len(circles[0, :, :]) > 1:
                        x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                        x, y, radius = checkradius(x, y, radius)
                        x = int(x * scale_to_ori)
                        y = int(y * scale_to_ori)
                        radius += int(radius * multiplier)
                        radius = int(radius * color_check_radius_multiplier)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        crop_color = cv2.resize(crop_color, color_check_dimension, interpolation=cv2.INTER_AREA)
                        return crop_color, x, y, radius
                    else:
                        x, y, radius = np.uint16(circles[0][0])
                        x, y, radius = checkradius(x, y, radius)
                        x = int(x * scale_to_ori)
                        y = int(y * scale_to_ori)
                        radius += int(radius * multiplier)
                        radius = int(radius * color_check_radius_multiplier)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        crop_color = cv2.resize(crop_color, color_check_dimension, interpolation=cv2.INTER_AREA)
                        return crop_color, x, y, radius
                else:
                    if print_log == True:
                        print('no circle found...:c... resorting to empirically estimated x,y,r=640,450,200')
                    x, y, radius = 505, 450, 330
                    x = int(x * scale_to_ori)
                    y = int(y * scale_to_ori)
                    radius += int(radius * multiplier)
                    radius = int(radius * color_check_radius_multiplier)
                    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    crop_color = cv2.resize(crop_color, color_check_dimension, interpolation=cv2.INTER_AREA)
                    return crop_color, x, y, radius
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

                return crop_color, x, y, radius
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

                return crop_color, x, y, radius
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

                return crop_color, x, y, radius
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

                        return crop_color, x, y, radius
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

                        return crop_color, x, y, radius
                else:
                    if print_log == True:
                        print(
                            'no circle found...:c... resorting to empirically estimated x,y,radius of 640,450,200')
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
                    return crop_color, x, y, radius

def raw_to_cropped_ori_resol_moving_avg(raw_image, dim, x, y, radius):
    '''
    14th October 2021 :
        - using moving average of 10 previous cropping locations, crop image
    '''

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

    original_dim = raw_image.shape

    # BGR channel
    blue, green, red = cv2.split(raw_image)
    img_combine_1 = red
    img_combine_2 = blue
    img_combine_3 = green

    # Detecting ROI
    mask = np.zeros((original_dim[0], original_dim[1]), dtype=np.uint8)

    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)

    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, original_dim)
    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, original_dim)
    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, original_dim)
    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))

    return crop_color, x, y, radius

def raw_to_cropped_hi_res_moving_avg(raw_image, dim, x, y, radius):
    '''
    14th October 2021 :
        - using moving average of 10 previous cropping locations, crop image
    '''

    original_dim = raw_image.shape

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

    # BGR channel
    blue, green, red = cv2.split(raw_image)
    img_combine_1 = red
    img_combine_2 = blue
    img_combine_3 = green

    # Detecting ROI
    mask = np.zeros((original_dim[0], original_dim[1]), dtype=np.uint8)

    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)

    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, original_dim)
    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, original_dim)
    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, original_dim)
    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))

    return crop_color, x, y, radius

def RGB_comparator(inputimg, unsure_low_thresh=0.79, unsure_gap_percent=0.01):
    IMG_WIDTH = 64
    IMG_HEIGHT = 64
    IMG_CHANNELS = 3

    # Getting image into numpy array for prediction
    X_env_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    X_env_test[0] = inputimg/255
    # setting up interpreter and inputs

    # Predict colonies
    input_data_RGB2 = input_data_RGB
    input_data_RGB2[0] = inputimg/255
    interpreter_RGB.set_tensor(input_details_RGB[0]['index'], input_data_RGB2)
    interpreter_RGB.invoke()

    output_data = interpreter_RGB.get_tensor(output_details_RGB[0]['index'])

    if output_data[0][0] > (unsure_low_thresh + unsure_gap_percent * (1 - unsure_low_thresh)):
        return 'overgrown/smeared'
    elif output_data[0][0] > unsure_low_thresh:
        return 'unsure'
    else:
        return 'normal'

def analysis_image(img_name='image.jpg', result='result.jpg', predict_thresh = 0.42, print_log=False, use_avg_cropping=False, analyse_time=False, check_color_dimension=(64,64), imgsz=640):

    # file path assertion
    if '.jpg' not in img_name[-4:] and '.png' not in img_name[-4:]:
        img_name = img_name + '.jpg'
    else:
        img_name = img_name

    if '.jpg' not in result[-4:] and '.png' not in result[-4:]:
        result_name = result + '.png'
    else:
        result_name = result

    count_name = result_name[:-3] + 'txt'

    def reduce_ROI_radius(inputimg, percent=0.7):
        lala = inputimg[int(((1 - percent) / 2) * inputimg.shape[0]):inputimg.shape[0] - int(
            ((1 - percent) / 2) * inputimg.shape[0]),
               int(((1 - percent) / 2) * inputimg.shape[0]):inputimg.shape[0] - int(
                   ((1 - percent) / 2) * inputimg.shape[0]), :]
        return lala

    def outlier_filter(data, m=2.):
        """
        :param data: the np.array with size >= 8, holding the crop location (x,y,r)
        :param m: tolerance threshold to determine outlier, smaller 'm' = smaller tolerance
        :return: outlier-filtered data array
        """
        d = np.abs(data - np.median(data))
        mdev = np.median(d)

        # mdev can sometimes be 0, so assert a small mdev to be 0.01, this prevents undefined 's' variable
        if abs(mdev) < 0.01:
            mdev = 0.01
        s = d / mdev
        return data[s < m]

    def find_missing_index(running_avg_filtered, running_avg_ori):
        index_list = []
        for index, element in enumerate(running_avg_ori):
            if element not in running_avg_filtered:
                index_list.append(index)
        return index_list

    if '.jpg' not in img_name[-4:] and '.png' not in img_name[-4:]:
        img_name = img_name + '.jpg'
    else:
        img_name = img_name
    inputimg = cv2.imread(img_name)

    # # cropping happens here

    ##############################################################
    ############## CROPPING ######################################
    ##############################################################
    if os.path.exists('cropped')==False:
        os.mkdir('cropped')
    time1 = timeit.default_timer()
    if use_avg_cropping == True:
        if os.path.exists('running_avg.txt') == True:
            running_avg = np.loadtxt('running_avg.txt', dtype=int)

            if len(running_avg) == 0:
                # previous filtering removed all crop location ...
                if print_log == True:
                    print('not enough cropping position data to deduce moving average, crop no. 0')
                if inputimg.shape[1] > 3500:
                    cropped_out, ROI_x, ROI_y, cropped_radius = raw_to_cropped_hi_res(inputimg, (imgsz, imgsz))
                else:
                    cropped_out, ROI_x, ROI_y, cropped_radius = raw_to_cropped_ori_resol(inputimg, (imgsz, imgsz))

                if print_log == True:
                    print('{} : time taken to crop'.format(str(timeit.default_timer() - time1)))
                if analyse_time == True:
                    time_analysis['crop'] = timeit.default_timer() - time1

                cropped_dimension = cropped_out.shape[0]
                cropped = cv2.resize(cropped_out, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
                cropped_out_reduced_radius = reduce_ROI_radius(cropped, percent=0.7)
                cropped_check_color = cv2.resize(cropped_out_reduced_radius, check_color_dimension,
                                                 interpolation=cv2.INTER_AREA)
                time2 = timeit.default_timer()
                overgrown_flag = RGB_comparator(cropped_check_color)
                if print_log == True:
                    print('{} : time taken to RGB compare'.format(str(timeit.default_timer() - time2)))
                if analyse_time == True:
                    time_analysis['RGB_compare'] = timeit.default_timer() - time2

                ROI_r = int(cropped_radius)

                running_avg = [[0], [0], [0]]

                running_avg[0].append(ROI_x)
                running_avg[1].append(ROI_y)
                running_avg[2].append(ROI_r)
                running_avg = np.array(running_avg)
                np.savetxt('running_avg.txt', running_avg, fmt='%d')

                cv2.imwrite('cropped/'+str(Path(img_name).name), cropped)

            # if the previous filtering reduced running_avg.txt to only single element, they will be read as a single list, the below code fixes that by reshaping
            elif type(running_avg[0]) == np.int32:
                running_avg = running_avg.reshape(3, 1)

                if print_log == True:
                    print('not enough cropping position data to deduce moving average, crop no. {}'.format(
                        str(len(running_avg[0]))))
                running_avg = [list(x) for x in running_avg]

                if inputimg.shape[1] > 3500:
                    cropped_out, ROI_x, ROI_y, cropped_radius = raw_to_cropped_hi_res(inputimg, (imgsz, imgsz))
                else:
                    cropped_out, ROI_x, ROI_y, cropped_radius = raw_to_cropped_ori_resol(inputimg, (imgsz, imgsz))

                if print_log == True:
                    print('{} : time taken to crop'.format(str(timeit.default_timer() - time1)))
                if analyse_time == True:
                    time_analysis['crop'] = timeit.default_timer() - time1

                cropped_dimension = cropped_out.shape[0]
                cropped = cv2.resize(cropped_out, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
                cropped_out_reduced_radius = reduce_ROI_radius(cropped, percent=0.7)
                cropped_check_color = cv2.resize(cropped_out_reduced_radius, check_color_dimension,
                                                 interpolation=cv2.INTER_AREA)

                time2 = timeit.default_timer()
                overgrown_flag = RGB_comparator(cropped_check_color)
                if print_log == True:
                    print('{} : time taken to RGB compare'.format(str(timeit.default_timer() - time2)))
                if analyse_time == True:
                    time_analysis['RGB_compare'] = timeit.default_timer() - time2

                ROI_r = int(cropped_radius)

                # after the 11th insert of x,y,radius, do outlier detection

                running_avg[0].append(ROI_x)
                running_avg[1].append(ROI_y)
                running_avg[2].append(ROI_r)

                running_avg = np.array(running_avg)
                np.savetxt('running_avg.txt', running_avg, fmt='%d')

                cv2.imwrite('cropped/'+str(Path(img_name).name), cropped)

            # if accumulated running avg crop between 2 to 10
            elif len(running_avg[0]) < 11:

                if print_log == True:
                    print('not enough cropping position data to deduce moving average, crop no. {}'.format(
                        str(len(running_avg[0]))))
                running_avg = [list(x) for x in running_avg]

                if inputimg.shape[1] > 3500:
                    cropped_out, ROI_x, ROI_y, cropped_radius = raw_to_cropped_hi_res(inputimg, (imgsz, imgsz))
                else:
                    cropped_out, ROI_x, ROI_y, cropped_radius = raw_to_cropped_ori_resol(inputimg, (imgsz, imgsz))

                if print_log == True:
                    print('{} : time taken to crop'.format(str(timeit.default_timer() - time1)))
                if analyse_time == True:
                    time_analysis['crop'] = timeit.default_timer() - time1

                cropped_dimension = cropped_out.shape[0]
                cropped = cv2.resize(cropped_out, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
                cropped_out_reduced_radius = reduce_ROI_radius(cropped, percent=0.7)
                cropped_check_color = cv2.resize(cropped_out_reduced_radius, check_color_dimension,
                                                 interpolation=cv2.INTER_AREA)

                time2 = timeit.default_timer()
                overgrown_flag = RGB_comparator(cropped_check_color)
                if print_log == True:
                    print('{} : time taken to RGB compare'.format(str(timeit.default_timer() - time2)))
                if analyse_time == True:
                    time_analysis['RGB_compare'] = timeit.default_timer() - time2

                ROI_r = int(cropped_radius)

                # after the 11th insert of x,y,radius, do outlier detection

                running_avg[0].append(ROI_x)
                running_avg[1].append(ROI_y)
                running_avg[2].append(ROI_r)

                cv2.imwrite('cropped/'+str(Path(img_name).name), cropped)


                if len(running_avg[0]) >= 8:
                    #################################### ROBUST ANOMALOUS CROPPING DETECTION STARTS ############################
                    # once you have collected 8 crop location data, start measuring median + detect anomalous cropping
                    # using m=4. is good enough to filter any outliers that are >70 from the biggest / <70 from the smallest..
                    # use smaller m to filter away more inconsistent crop, and hence get a more well calibrated average cropping
                    # however, this comes at the cost of more filter == more normal cropping before average cropping can be
                    # kickstarted (once it reaches 10 accumulated values). This means longer compute time.
                    running_avg_filter1 = outlier_filter(np.array(running_avg[0]), m=4.)
                    running_avg_filter2 = outlier_filter(np.array(running_avg[1]), m=4.)
                    running_avg_filter3 = outlier_filter(np.array(running_avg[2]), m=4.)

                    missing_index = find_missing_index(running_avg_filter1, running_avg[0]) + find_missing_index(
                        running_avg_filter2, running_avg[1]) + find_missing_index(running_avg_filter3, running_avg[2])
                    missing_index = list(set(missing_index))

                    missing_index = sorted(missing_index, reverse=True)

                    # print('missing index is : {}'.format(str(missing_index)))
                    # print('length of running average BEFORE removing = {}'.format(str(len(running_avg[0]))))

                    for remove_index in missing_index:
                        # print('removing anomalous cropping index : {} ######################'.format(remove_index))
                        del running_avg[0][remove_index]
                        del running_avg[1][remove_index]
                        del running_avg[2][remove_index]

                    # print('length of running average AFTER removing = {}'.format(str(len(running_avg[0]))))
                    # print('running_avg after removing : {} with type : {}'.format(str(running_avg[0]), str(type(running_avg[0]))))

                    #################################### ROBUST ANOMALOUS CROPPING DETECTION ENDS ############################

                running_avg = np.array(running_avg)
                np.savetxt('running_avg.txt', running_avg, fmt='%d')

            # have enough running avg crop values, e.g. 11, but only use the latest 10
            else:
                if print_log == True:
                    print('use moving average for cropping')
                moving_x, moving_y, moving_r = np.average(running_avg[:, 1:], axis=1)

                if inputimg.shape[1] > 3500:
                    cropped_out, ROI_x, ROI_y, cropped_radius = raw_to_cropped_hi_res_moving_avg(inputimg, (imgsz, imgsz),
                                                                                                 int(moving_x),
                                                                                                 int(moving_y),
                                                                                                 int(moving_r))
                else:
                    cropped_out, ROI_x, ROI_y, cropped_radius = raw_to_cropped_ori_resol_moving_avg(inputimg,
                                                                                                    (imgsz, imgsz),
                                                                                                    int(moving_x),
                                                                                                    int(moving_y),
                                                                                                    int(moving_r))
                if print_log == True:
                    print('{} : time taken to crop'.format(str(timeit.default_timer() - time1)))
                if analyse_time == True:
                    time_analysis['crop'] = timeit.default_timer() - time1

                cropped_dimension = cropped_out.shape[0]
                cropped = cv2.resize(cropped_out, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
                cropped_out_reduced_radius = reduce_ROI_radius(cropped, percent=0.7)
                cropped_check_color = cv2.resize(cropped_out_reduced_radius, check_color_dimension,
                                                 interpolation=cv2.INTER_AREA)
                time2 = timeit.default_timer()
                overgrown_flag = RGB_comparator(cropped_check_color)
                if print_log == True:
                    print('{} : time taken to RGB compare'.format(str(timeit.default_timer() - time2)))
                if analyse_time == True:
                    time_analysis['RGB_compare'] = timeit.default_timer() - time2

                ROI_r = int(cropped_radius)

                running_avg = running_avg[:, 1:]

                running_avg = [list(x) for x in running_avg]

                running_avg[0].append(ROI_x)
                running_avg[1].append(ROI_y)
                running_avg[2].append(ROI_r)

                cv2.imwrite('cropped/'+str(Path(img_name).name), cropped)

                #################################### ROBUST ANOMALOUS CROPPING DETECTION STARTS ############################
                # using m=4. is good enough to filter any outliers that are >70 from the biggest / <70 from the smallest..
                # use smaller m to filter away more inconsistent crop, and hence get a more well calibrated average cropping
                # however, this comes at the cost of more filter == more normal cropping before average cropping can be
                # kickstarted (once it reaches 10 accumulated values). This means longer compute time.
                running_avg_filter1 = outlier_filter(np.array(running_avg[0]), m=4.)
                running_avg_filter2 = outlier_filter(np.array(running_avg[1]), m=4.)
                running_avg_filter3 = outlier_filter(np.array(running_avg[2]), m=4.)

                missing_index = find_missing_index(running_avg_filter1, running_avg[0]) + find_missing_index(
                    running_avg_filter2, running_avg[1]) + find_missing_index(running_avg_filter3, running_avg[2])
                missing_index = list(set(missing_index))

                missing_index = sorted(missing_index, reverse=True)

                # print('missing index is : {}'.format(str(missing_index)))
                # print('length of running average BEFORE removing = {}'.format(str(len(running_avg[0]))))

                for remove_index in missing_index:
                    # print('removing anomalous cropping index : {} ######################'.format(remove_index))
                    del running_avg[0][remove_index]
                    del running_avg[1][remove_index]
                    del running_avg[2][remove_index]

                # print('length of running average AFTER removing = {}'.format(str(len(running_avg[0]))))
                # print('running_avg after removing : {} with type : {}'.format(str(running_avg[0]),
                #                                                               str(type(running_avg[0]))))

                #################################### ROBUST ANOMALOUS CROPPING DETECTION ENDS ############################

                running_avg = np.array(running_avg)
                np.savetxt('running_avg.txt', running_avg, fmt='%d')

        else:
            if print_log == True:
                print('not enough cropping position data to deduce moving average, crop no. 0')
            if inputimg.shape[1] > 3500:
                cropped_out, ROI_x, ROI_y, cropped_radius = raw_to_cropped_hi_res(inputimg, (imgsz, imgsz))
            else:
                cropped_out, ROI_x, ROI_y, cropped_radius = raw_to_cropped_ori_resol(inputimg, (imgsz, imgsz))

            if print_log == True:
                print('{} : time taken to crop'.format(str(timeit.default_timer() - time1)))
            if analyse_time == True:
                time_analysis['crop'] = timeit.default_timer() - time1

            cropped_dimension = cropped_out.shape[0]
            cropped = cv2.resize(cropped_out, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
            cropped_out_reduced_radius = reduce_ROI_radius(cropped, percent=0.7)
            cropped_check_color = cv2.resize(cropped_out_reduced_radius, check_color_dimension,
                                             interpolation=cv2.INTER_AREA)
            time2 = timeit.default_timer()
            overgrown_flag = RGB_comparator(cropped_check_color)
            if print_log == True:
                print('{} : time taken to RGB compare'.format(str(timeit.default_timer() - time2)))
            if analyse_time == True:
                time_analysis['RGB_compare'] = timeit.default_timer() - time2

            ROI_r = int(cropped_radius)

            running_avg = [[0], [0], [0]]

            running_avg[0].append(ROI_x)
            running_avg[1].append(ROI_y)
            running_avg[2].append(ROI_r)
            running_avg = np.array(running_avg)
            np.savetxt('running_avg.txt', running_avg, fmt='%d')

            cv2.imwrite('cropped/' + str(Path(img_name).name), cropped)

    else:
        # This code will run if 'use_avg_cropping' is False
        # i.e. this is the code that runs the hough-cropping method
        
        # Old cropping implementation
#         if inputimg.shape[1] > 3500:
#             cropped_out, ROI_x, ROI_y, cropped_radius = raw_to_cropped_hi_res(inputimg, (imgsz, imgsz))
#         else:
#             cropped_out, ROI_x, ROI_y, cropped_radius = raw_to_cropped_ori_resol(inputimg, (imgsz, imgsz))
        
        # New cropping implementation
        im_gray = np.asarray(cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY))
        ROI_x,ROI_y,cropped_radius = hough_estimate(im_gray,zero=False, mm13=False, mm25=True)
        cropped_out = crop_colour_im(inputimg,ROI_x,ROI_y,cropped_radius,zero=False) 
        

        if print_log == True:
            print('{} : time taken to crop'.format(str(timeit.default_timer() - time1)))
        if analyse_time == True:
            time_analysis['crop'] = timeit.default_timer() - time1

        cropped_dimension = cropped_out.shape[0]
        cropped = cv2.resize(cropped_out, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
        cropped_out_reduced_radius = reduce_ROI_radius(cropped, percent=0.7)
        cropped_check_color = cv2.resize(cropped_out_reduced_radius, check_color_dimension,
                                         interpolation=cv2.INTER_AREA)
        time2 = timeit.default_timer()
        overgrown_flag = RGB_comparator(cropped_check_color)
        if print_log == True:
            print('{} : time taken to RGB compare'.format(str(timeit.default_timer() - time2)))
        if analyse_time == True:
            time_analysis['RGB_compare'] = timeit.default_timer() - time2

        cv2.imwrite('cropped/' + str(Path(img_name).name), cropped)

    ##############################################################
    ############## CROPPING ENDS HER #############################
    ##############################################################

    # Settings
    agnostic_nms = True
    save_txt = False
    save_img = True
    view_img = False

    names = 'yolor_pi/inference_script/yolor.names'
    # out = 'result_yolor/'
    #
    # if os.path.exists(out)==False:
    #     os.mkdir(out)
    
    # imgsz = 640
    iou_thresh = 0.5
    conf_thresh = 0.17

    with torch.no_grad():
        dataset = LoadImages('cropped/' + str(Path(img_name).name), img_size=imgsz, auto_size=64)

        # inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=None, agnostic=agnostic_nms)
            t2 = time_synchronized()

            json_data = {}
            json_data['link_placeholder'] = {"filename": "filename_placeholder", "size": -1, "regions": [],
                                             "file_attributes": {"image_quality": {}}}

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s

                # save_path = str(Path(out) / Path(p).name)
                save_path = str(Path(result_name))
                # txt_path = str(Path(out) / Path(p).stem)
                txt_path = str(Path(result_name))[:-3] + 'txt'

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                ecoli_count = 0
                coliform_count = 0

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results

                    if len(det[:, -1].unique())==2:
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string
                            if int(c) == 0:
                                ecoli_count = n
                            else:
                                coliform_count = n
                    elif len(det[:, -1].unique())==1:
                        c = det[:, -1].unique()[0]
                        n = (det[:, -1] == c).sum()
                        s += '%g %ss, ' % (n, names[int(c)])
                        if int(c) == 0:
                            ecoli_count = n
                            coliform_count = 0
                        else:
                            coliform_count = n
                            ecoli_count = 0
                    else:
                        ecoli_count = 0
                        coliform_count = 0
                    # Saving the counts in a text file
                    f = open(count_name, "w+")
                    f.write(
                        '(u-net) E. coli\t\t: %d \n(u-net) coliforms\t: %d\n(yolo) E. coli\t\t: %d \n(yolo) coliforms\t: %d\n%s\n%d\n%d' % (
                            0, 0, ecoli_count, coliform_count, overgrown_flag,
                            ecoli_count,
                            coliform_count))
                    f.close()

                    # Write results
                    for *xyxy, conf, cls in det:
                        if save_txt:  # Write to file
                            # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xyxy))  # label format

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                        xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()

                        if int(cls)==1:
                            json_data['link_placeholder']["regions"].append({"shape_attributes": {"name": "rect",
                                                                                                  "x": int(((xywh[0]* cropped_dimension / imgsz) + ROI_x - cropped_dimension / 2)-(xywh[2] * cropped_dimension / imgsz)/2),
                                                                                                  "y": int(((xywh[1]* cropped_dimension / imgsz) + ROI_y - cropped_dimension / 2)-(xywh[3] * cropped_dimension / imgsz)/2),
                                                                                                  "width": int((xywh[2] * cropped_dimension / imgsz)),
                                                                                                  "height": int((xywh[3] * cropped_dimension / imgsz))},
                                                                             "region_attributes": {
                                                                                 "type": {"coliform": True}}})
                        else:
                            json_data['link_placeholder']["regions"].append({"shape_attributes": {"name": "rect",
                                                                                                  "x": int(((xywh[0]* cropped_dimension / imgsz) + ROI_x - cropped_dimension / 2)-(xywh[2] * cropped_dimension / imgsz)/2),
                                                                                                  "y": int(((xywh[1]* cropped_dimension / imgsz) + ROI_y - cropped_dimension / 2)-(xywh[3] * cropped_dimension / imgsz)/2),
                                                                                                  "width": int((xywh[2] * cropped_dimension / imgsz)),
                                                                                                  "height": int((xywh[3] * cropped_dimension / imgsz))},
                                                                             "region_attributes": {
                                                                                 "type": {"ecoli": True}}})

                        # det_index+=1
                else:
                    # Saving the counts in a text file
                    f = open(count_name, "w+")
                    f.write(
                        '(u-net) E. coli\t\t: %d \n(u-net) coliforms\t: %d\n(yolo) E. coli\t\t: %d \n(yolo) coliforms\t: %d\n%s\n%d\n%d' % (
                            0, 0, ecoli_count, coliform_count, overgrown_flag,
                            ecoli_count,
                            coliform_count))
                    f.close()
                
                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.putText(im0, overgrown_flag,
                                    (int(0.1*im0.shape[0]), int(0.9*im0.shape[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0,
                                    (255, 255, 255),
                                    2)
                        cv2.imwrite(save_path, im0)
                    else:
                        print('wrong dataset mode')

            with open('{}.json'.format(result_name[:-4]), 'w') as f:
                json.dump(json_data, f)

        if save_txt or save_img:
            print('Results saved to %s' %  str(Path(result_name)))

        print('Done. (%.3fs)' % (time.time() - t0))

    os.remove('cropped/' + str(Path(img_name).name))

    return {'e.coli': ecoli_count, 'coliform': coliform_count}

# Settings
cfg = 'yolor_pi/inference_script/yolor_p6small_filter'
weights = ['yolor_pi/inference_script/best_p6_small_filter.pt']
names = 'yolor_pi/inference_script/yolor.names'

imgsz = 640

time_analysis = {}

# MODEL SETTINGS
RGB_model_path = '2022_April_19_RGB.tflite' #'RGB_CNN_model.tflite'

# LOAD RGB ML MODEL
# define model
interpreter_RGB = Interpreter(model_path='{}'.format(RGB_model_path))
# interpreter = tf.lite.Interpreter(
#     model_path='{}.tflite'.format(model_to_check))
interpreter_RGB.allocate_tensors()

# Get input and output tensors.
input_details_RGB = interpreter_RGB.get_input_details()
output_details_RGB = interpreter_RGB.get_output_details()

# # Test the model on random input data.
input_shape_RGB = input_details_RGB[0]['shape']
input_data_RGB = np.array(np.random.random_sample(input_shape_RGB), dtype=np.float32)

with torch.no_grad():
    # Initialize
    device = select_device('cpu')
    half = device.type != 'cpu'

    # Load model
    model = Darknet(cfg, imgsz).cpu()
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    model.to(device).eval()
    if half:
        model.half()

    # Get names and box plotting colors
    names = load_classes(names)
    colors = [[0, 255, 0], [0, 255, 255]]
