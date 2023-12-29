import numpy as np
from keras.models import load_model
from keras import backend as K
import cv2
import random as rng
from skimage.feature import peak_local_max
import os
import time
import tensorflow as tf
from tflite_runtime.interpreter import Interpreter

def raw_to_cropped(raw_image, dim, color_check=False):

    '''
    :parameter: raw_image
                dim -> output dimension
                color_check -> if True, crop 1.5*radius to focus only on the center color [since if highly contaminated,
                the center will already have enough information and avoid taking the edge color into account]
    :return: cropped 256x256 pixel image centered on the ROI
    '''
    def checkradius(x,y,radius):
        if x < 450 or x > 750 or y < 400 or radius<150: # May 6th, 560, 700, 400, 150
            x=640
            y=450
            radius=200
            print('adjust x,y,radius to 640,450,200 due to anomalous value(s) of %d, %d, %d.' %(x,y,radius))
            return x,y,radius
        else:
            return x,y,radius
    if color_check==True:
        def masking(mask, image, x, y, radius, dim):
            cropimg = cv2.subtract(mask, image)
            cropimg = cv2.subtract(mask, cropimg)
            cropimg = cropimg[y - radius+int(radius*0.5):y + radius-int(radius*0.5), x - radius+int(radius*0.5):x + radius-int(radius*0.5)]
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
                x,y,radius=checkradius(x, y,radius)
                cv2.circle(mask, (x, y), radius-int(radius*0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))

                return crop_color
            else:
                x, y, radius = np.uint16(circles[0][0])
                x,y,radius=checkradius(x, y,radius)
                cv2.circle(mask, (x, y), radius-int(radius*0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                return crop_color
        else:
            print('no circle found... try changing parameter (1)')
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                       param1=50, param2=30, minRadius=200,
                                       maxRadius=238)
            if type(circles) != type(None):
                x, y, radius = np.uint16(circles[0][0])
                x,y,radius=checkradius(x, y,radius)
                cv2.circle(mask, (x, y), radius-int(radius*0.5), (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                return crop_color
            else:
                print('no circles found... try changing parameter (2)')
                circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                           param1=40, param2=20, minRadius=200,
                                           maxRadius=238)
                if type(circles) != type(None):
                    if len(circles[0, :, :]) > 1:
                        x, y, radius = np.uint16([[circles[0, 0, :]]][0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius-int(radius*0.5), (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        return crop_color
                    else:
                        x, y, radius = np.uint16(circles[0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius-int(radius*0.5), (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        return crop_color
                else:
                    print('no circle found...:c... resorting to empirically estimated x,y,r=640,450,200')
                    x,y,radius=640,450,200
                    cv2.circle(mask, (x, y), radius - int(radius * 0.5), (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    return crop_color
    else: # if color_check=False, crop as usual
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
                x,y,radius=checkradius(x, y,radius)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # print(x, y, radius)

                return crop_color
            else:
                x, y, radius = np.uint16(circles[0][0])
                x,y,radius=checkradius(x, y,radius)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # print(x, y, radius)

                return crop_color
        else:
            print('no circle found... try changing parameter (1)')
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300,
                                       param1=50, param2=30, minRadius=200,
                                       maxRadius=238)
            if type(circles) != type(None):
                x, y, radius = np.uint16(circles[0][0])
                x,y,radius=checkradius(x, y,radius)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                # print(x, y, radius)

                return crop_color
            else:
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
                        # print(x, y,radius)

                        return crop_color
                    else:
                        x, y, radius = np.uint16(circles[0][0])
                        x, y, radius = checkradius(x, y, radius)
                        cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                        crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                        crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                        crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                        crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                        # print(x, y,radius)

                        return crop_color
                else:
                    print('no circle found...:c... resorting to empirically estimated x,y,radius of 640,450,200')
                    x=640
                    y=450
                    radius=200
                    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1, 8, 0)
                    crop_img_combine_1 = masking(mask, img_combine_1, x, y, radius, dim)
                    crop_img_combine_2 = masking(mask, img_combine_2, x, y, radius, dim)
                    crop_img_combine_3 = masking(mask, img_combine_3, x, y, radius, dim)
                    crop_color = cv2.merge((crop_img_combine_2, crop_img_combine_3, crop_img_combine_1))
                    # print(x, y,radius)
                    return crop_color

def RGB_comparator(cropped_img, n_colors = 6, n_dominant = 2):

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
    if avg_diff[0]>120 or avg_diff[1]>120:
        overgrown_flag = True
        return overgrown_flag
    else:
        return overgrown_flag

def predict_from_model(cropped_img):
    t_0 = time.time()
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

    # Predict colonies
    input_data2 = input_data
    input_data2[0] = cropped_img
    interpreter.set_tensor(input_details[0]['index'], input_data2)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    preds_env_test = output_data

    # Threshold predictions for blue and purple
    preds_env_test_t = preds_env_test
    preds_env_test_t[0,:,:,0] = (preds_env_test_t[0, :, :, 0] < 0.2).astype(np.uint8)
    preds_env_test_t[0,:,:,2] = (preds_env_test_t[0, :, :, 2] < 0.01).astype(np.uint8)

    thresholded_blue = (np.squeeze(preds_env_test_t[0, :, :, 0]))
    thresholded_blue = thresholded_blue * 255
    thresholded_blue = thresholded_blue.astype('uint8')
    thresholded_purple = (np.squeeze(preds_env_test_t[0, :, :, 2]))
    thresholded_purple = thresholded_purple * 255
    thresholded_purple = thresholded_purple.astype('uint8')

    # print("predict time")
    # print(time.time() - t_0)
    return [thresholded_blue, thresholded_purple]

# def predict_from_model(cropped_img):
#     t_0 = time.time()
#     '''
#     :parameter : cropped_img
#     :return: a list containing binary images of blue and purple colonies (256x256 pixels)
#     '''
#
#     # Parameters
#     IMG_WIDTH = 256
#     IMG_HEIGHT = 256
#     IMG_CHANNELS = 3
#
#     # Getting image into numpy array for prediction
#     X_env_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
#     img = cropped_img
#     X_env_test[0] = img
#
#     # Predict colonies
#     preds_env_test = model.predict(X_env_test, verbose=1)
#
#     # Threshold predictions for blue and purple
#     preds_env_test_t = preds_env_test
#     preds_env_test_t[0,:,:,0] = (preds_env_test_t[0, :, :, 0] < 0.2).astype(np.uint8)
#     preds_env_test_t[0,:,:,2] = (preds_env_test_t[0, :, :, 2] < 0.01).astype(np.uint8)
#
#     thresholded_blue = (np.squeeze(preds_env_test_t[0, :, :, 0]))
#     thresholded_blue = thresholded_blue * 255
#     thresholded_blue = thresholded_blue.astype('uint8')
#     thresholded_purple = (np.squeeze(preds_env_test_t[0, :, :, 2]))
#     thresholded_purple = thresholded_purple * 255
#     thresholded_purple = thresholded_purple.astype('uint8')
#
#     # print("predict time")
#     # print(time.time() - t_0)
#     return [thresholded_blue, thresholded_purple]

def segment_and_count(predicted_image, return_image = 'False'):
    t_0 = time.time()
    '''
    :parameter: predicted_image [binary image from predict_from_model function]
    :parameter: return_image
    :return:    return_image == 'True'  : images [numpy.ndarray]
                return_image == 'False' : colonies count [int]
    '''

    img = predicted_image
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)

    # Prediction map was of size (256x256), enlarging gave uneven edge, gaussian blur then threshold
    # to smooth out the edges
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.GaussianBlur(bw,(7,7),3)
    ret, bw = cv2.threshold(bw, 30, 255, cv2.THRESH_BINARY)
    bw2 = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    bw2 = bw2.astype('uint8')

    # distance transform performed to obtain the distance map (output is a float of value between 0 to ~15, the further away a pixel is to the edge, the higher the value)
    dist2 = cv2.distanceTransform(bw, cv2.DIST_L2, 5)

    # Normalize the distance image for range = {0.0, 1.0}
    cv2.normalize(dist2, dist2, 0, 1.0, cv2.NORM_MINMAX)

    # Convert to unsigned 8-bit integer (compatible for cv2.imshow)
    dist2 = dist2*255
    dist2 = np.uint8(dist2)

    # Obtain coordinates of local maxima (output type = numpy array of maxima coordinates)
    coordinates = peak_local_max(dist2, min_distance=2)

    # Create empty mask
    markers2 = np.zeros(dist2.shape, dtype=np.int32)

    # Draw the colony markers as points
    for i in coordinates:
        cv2.circle(markers2, tuple((i[1],i[0])), 1, (255,255,255), -1)

    markers2show = markers2.astype("uint8")

    # Find total markers using findContours
    contours, _ = cv2.findContours(markers2show, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist2.shape, dtype=np.int32)


    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i + 1), -1)

    # Draw the background marker at (1,1) to reduce possibility of background segmentation
    cv2.rectangle(markers, (1, 1), (511,511), (255, 255, 255), 1)

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
                dst[i, j, :] = colors[index-1]

    if return_image == 'True':
        #print('There are %d cells based on countours' %len(contours))
        # print("segment_and_count time")
        # print(time.time() - t_0)
        return dst, len(contours)
    else:
        # print("segment_and_count time")
        # print(time.time() - t_0)
        return len(contours)

def segment_and_count_boundary(input_img, return_image='True', color='Blue'):
    img = input_img
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
    coordinates = peak_local_max(dist2, min_distance=2)

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
    watershed = cv2.watershed(bw2, markers)
    boundary = (watershed < 1).astype(np.uint8) * 255

    if color == 'Blue':
        boundary_colored = cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)
        boundary_colored[:, :, 1:] = boundary_colored[:, :, 1:] * 0
        if return_image == 'True':
            # print('There are %d cells based on countours' %len(contours))
            # print("segment_and_count time")
            # print(time.time() - t_0)
            return boundary_colored, len(contours)
        else:
            # print("segment_and_count time")
            # print(time.time() - t_0)
            return len(contours)
    else:
        boundary_colored = cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)
        boundary_colored[:, :, :2] = boundary_colored[:, :, :2] * 0
        if return_image == 'True':
            # print('There are %d cells based on countours' %len(contours))
            # print("segment_and_count time")
            # print(time.time() - t_0)
            return boundary_colored, len(contours)
        else:
            # print("segment_and_count time")
            # print(time.time() - t_0)
            return len(contours)

def get_count(inputimg, color):
    if color == 'Blue':
        return segment_and_count(predict_from_model(raw_to_cropped(inputimg, dim=(256, 256)))[0])
    else:
        return segment_and_count(predict_from_model(raw_to_cropped(inputimg, dim=(256, 256)))[1])

def get_image_and_count(inputimg):
    blue_predict, purple_predict = predict_from_model(raw_to_cropped(inputimg, dim=(256, 256)))
    return segment_and_count_boundary(blue_predict, return_image='True', color='Blue'),\
           segment_and_count_boundary(purple_predict, return_image='True', color='Purple')

# uncomment to revert back to saving in the timepoint folder format

# def saving_result(img_directory = 'image', result_directory = 'result/color-check', wait_time = 100, color_check=False, save_name = ['_ori_','_e_coli_', '_coliform_']):
#     if color_check == False:
#         for img_file in os.listdir(img_directory):
#             inputimg = cv2.imread(img_directory + '/' + img_file)
#             inputimg2= cv2.imread(img_directory + '/' + img_file)
#
#             # Get colony counts ONLY (will be deprecated soon)
#             #blue_count = get_count(inputimg, 'Blue')
#             #purple_count = get_count(inputimg, 'Purple')
#
#             # Get segmented images and counts
#             blue_image, blue_count = get_image_and_count(inputimg, 'Blue')
#             purple_image, purple_count = get_image_and_count(inputimg, 'Purple')
#
#             # Get the cropped original image of the same dimension for ease of result viewing
#             ori_image = raw_to_cropped(inputimg2, dim=(512, 512))
#             cv2.rectangle(ori_image, (1, 1), (511, 511), (255, 255, 255), 2)
#
#             # Uncomment to print counts
#             #print('There are %d E. coli colonies and %d coliforms.' %(blue_count, purple_count))
#
#             # Creating a directory to save the new images and count text file
#             if os.path.exists(result_directory) == False:
#                 os.mkdir(result_directory)
#             if os.path.exists(result_directory + '/' + img_file[:(len(img_file)-4)] + '_result') == True:
#                 i = 1
#                 while(os.path.exists(result_directory + '/' + img_file[:(len(img_file)-4)] + '_result' + '(' + str(i) + ')') == True):
#                     i += 1
#                 print("Folder exists, saving images and text file in a new folder called " + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')' + " to prevent overwriting files in the existing folder.")
#                 os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')')
#
#                 # Saving the counts in a text file
#                 f = open(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')/' + img_file[:(len(img_file) - 4)] + '_count', "w+")
#                 f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (blue_count, purple_count))
#                 f.close()
#
#                 # Writing title and count on images
#                 cv2.putText(blue_image, 'E.coli count : %d' % (blue_count),
#                             (5, 500),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             1,
#                             (255, 255, 255),
#                             2)
#                 cv2.putText(purple_image, 'coliform count : %d' % (purple_count),
#                             (5, 500),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             1,
#                             (255, 255, 255),
#                             2)
#                 cv2.rectangle(blue_image, (1,1), (511,511), (255, 255, 255), 2)
#                 cv2.rectangle(purple_image, (1,1), (511,511), (255, 255, 255), 2)
#                 # Saving the segmented images
#                 cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')/' + img_file[:(len(img_file) - 4)] + '_E.coli.png', blue_image)
#                 cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')/' + img_file[:(len(img_file) - 4)] + '_coliform.png', purple_image)
#                 cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')/' + img_file[:(len(img_file) - 4)] + '_c_original.png', ori_image)
#
#                 # Uncomment to show segmented images
#                 cv2.imshow('E.coli colonies', blue_image)
#                 cv2.imshow('coliforms', purple_image)
#                 print("Press any key to move on to the next image. (5 seconds)")
#                 cv2.waitKey(wait_time)
#             else:
#                 print("Saving images and text file in a new folder called " + img_file[:(len(img_file) - 4)] + '_result')
#                 os.mkdir(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result')
#
#                 # Saving the counts in a text file
#                 f = open(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(len(img_file) - 4)] + '_count', "w+")
#                 f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (blue_count, purple_count))
#                 f.close()
#
#                 # Writing title and count on images
#                 cv2.putText(blue_image, 'E.coli count : %d'%(blue_count),
#                             (5,500),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             1,
#                             (255,255,255),
#                             2)
#                 cv2.putText(purple_image, 'coliform count : %d' % (purple_count),
#                             (5, 500),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             1,
#                             (255, 255, 255),
#                             2)
#                 cv2.rectangle(blue_image, (1,1), (511,511), (255, 255, 255), 2)
#                 cv2.rectangle(purple_image, (1,1), (511,511), (255, 255, 255), 2)
#                 # Saving the segmented images
#                 cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(len(img_file) - 4)] + '_E.coli.png', blue_image)
#                 cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(len(img_file) - 4)] + '_coliform.png', purple_image)
#                 cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(len(img_file) - 4)] + '_c_original.png', ori_image)
#
#                 # Uncomment to show segmented images
#                 cv2.imshow('E.coli colonies', blue_image)
#                 cv2.imshow('coliforms', purple_image)
#                 print("Press any key to move on to the next image. (5 seconds)")
#                 cv2.waitKey(wait_time)
#     else:
#         for img_file in os.listdir(img_directory):
#             inputimg = cv2.imread(img_directory + '/' + img_file)
#             inputimg2= cv2.imread(img_directory + '/' + img_file)
#
#             overgrown_flag = RGB_comparator(raw_to_cropped(inputimg, dim=(256, 256), color_check=True))
#
#             if overgrown_flag == False:
#                 print('Sample is not overgrown, proceed with counting')
#                 # Get colony counts ONLY (will be deprecated soon)
#                 # blue_count = get_count(inputimg, 'Blue')
#                 # purple_count = get_count(inputimg, 'Purple')
#
#                 # Get segmented images and counts
#                 blue_image, blue_count = get_image_and_count(inputimg, 'Blue')
#                 purple_image, purple_count = get_image_and_count(inputimg, 'Purple')
#
#                 # Get the cropped original image of the same dimension for ease of result viewing
#                 ori_image = raw_to_cropped(inputimg2, dim=(512, 512))
#                 cv2.rectangle(ori_image, (1, 1), (511, 511), (255, 255, 255), 2)
#
#                 # Uncomment to print counts
#                 # print('There are %d E. coli colonies and %d coliforms.' %(blue_count, purple_count))
#
#                 # Creating a directory to save the new images and count text file
#                 if os.path.exists(result_directory) == False:
#                     os.mkdir(result_directory)
#                 if os.path.exists(
#                         result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result') == True:
#                     i = 1
#                     while (os.path.exists(
#                             result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
#                                     i) + ')') == True):
#                         i += 1
#                     print("Folder exists, saving images and text file in a new folder called " + img_file[:(
#                             len(img_file) - 4)] + '_result' + '(' + str(
#                         i) + ')' + " to prevent overwriting files in the existing folder.")
#                     os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
#                         i) + ')')
#
#                     # Saving the counts in a text file
#                     f = open(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
#                         i) + ')/' + img_file[:(len(img_file) - 4)] + '_count', "w+")
#                     f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (blue_count, purple_count))
#                     f.close()
#
#                     # Writing title and count on images
#                     cv2.putText(blue_image, 'E.coli count : %d' % (blue_count),
#                                 (5, 500),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 1,
#                                 (255, 255, 255),
#                                 2)
#                     cv2.putText(purple_image, 'coliform count : %d' % (purple_count),
#                                 (5, 500),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 1,
#                                 (255, 255, 255),
#                                 2)
#                     cv2.rectangle(blue_image, (1, 1), (511, 511), (255, 255, 255), 2)
#                     cv2.rectangle(purple_image, (1, 1), (511, 511), (255, 255, 255), 2)
#                     # Saving the segmented images
#                     cv2.imwrite(
#                         result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
#                             i) + ')/' + img_file[:(len(img_file) - 4)] + save_name[1] +'.png', blue_image)
#                     cv2.imwrite(
#                         result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
#                             i) + ')/' + img_file[:(len(img_file) - 4)] + save_name[2] +'.png', purple_image)
#                     cv2.imwrite(
#                         result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
#                             i) + ')/' + img_file[:(len(img_file) - 4)] + save_name[0]+'.png', ori_image)
#
#                     # Uncomment to show segmented images
#                     cv2.imshow('E.coli colonies', blue_image)
#                     cv2.imshow('coliforms', purple_image)
#                     print("Press any key to move on to the next image. (5 seconds)")
#                     cv2.waitKey(wait_time)
#                 else:
#                     print(
#                         "Saving images and text file in a new folder called " + img_file[:(len(img_file) - 4)] + '_result')
#                     os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result')
#
#                     # Saving the counts in a text file
#                     f = open(
#                         result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(
#                                 len(img_file) - 4)] + '_count', "w+")
#                     f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (blue_count, purple_count))
#                     f.close()
#
#                     # Writing title and count on images
#                     cv2.putText(blue_image, 'E.coli count : %d' % (blue_count),
#                                 (5, 500),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 1,
#                                 (255, 255, 255),
#                                 2)
#                     cv2.putText(purple_image, 'coliform count : %d' % (purple_count),
#                                 (5, 500),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 1,
#                                 (255, 255, 255),
#                                 2)
#                     cv2.rectangle(blue_image, (1, 1), (511, 511), (255, 255, 255), 2)
#                     cv2.rectangle(purple_image, (1, 1), (511, 511), (255, 255, 255), 2)
#                     # Saving the segmented images
#                     cv2.imwrite(
#                         result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(
#                                 len(img_file) - 4)] + save_name[1]+'.png', blue_image)
#                     cv2.imwrite(
#                         result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(
#                                 len(img_file) - 4)] + save_name[2]+'.png', purple_image)
#                     cv2.imwrite(
#                         result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(
#                                 len(img_file) - 4)] + save_name[0]+'.png', ori_image)
#
#                     # Uncomment to show segmented images
#                     cv2.imshow('E.coli colonies', blue_image)
#                     cv2.imshow('coliforms', purple_image)
#                     print("Press any key to move on to the next image. (5 seconds)")
#                     cv2.waitKey(wait_time)
#             else:
#                 print('sample is overgrown, too many colonies to count')
#
#                 # Get the cropped original image of the same dimension for ease of result viewing
#                 ori_image = raw_to_cropped(inputimg2, dim=(512, 512))
#                 cv2.rectangle(ori_image, (1, 1), (511, 511), (255, 255, 255), 2)
#
#                 # Creating a directory to save the new images and count text file
#                 if os.path.exists(result_directory) == False:
#                     os.mkdir(result_directory)
#                 if os.path.exists(result_directory +'/' + img_file[:(len(img_file) - 4)] + '_result') == True:
#                     i = 1
#                     while (os.path.exists(
#                             result_directory +'/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
#                                 i) + ')') == True):
#                         i += 1
#                     print("Folder exists, saving images and text file in a new folder called " + img_file[:(
#                             len(img_file) - 4)] + '_result' + '(' + str(
#                         i) + ')' + " to prevent overwriting files in the existing folder.")
#                     os.mkdir(
#                         result_directory +'/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')')
#
#                     # Saving the counts in a text file
#                     f = open(result_directory +'/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
#                         i) + ')/' + img_file[:(len(img_file) - 4)] + '_count', "w+")
#                     f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (1000, 1000))
#                     f.close()
#
#                     cv2.imwrite(result_directory +'/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
#                         i) + ')/' + img_file[:(len(img_file) - 4)] + save_name[0]+'.png', ori_image)
#                 else:
#                     print(
#                         "Saving text file in a new folder called " + img_file[
#                                                                                 :(len(img_file) - 4)] + '_result')
#                     os.mkdir(result_directory +'/' + img_file[:(len(img_file) - 4)] + '_result')
#
#                     # Saving the counts in a text file
#                     f = open(result_directory +'/' + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(
#                             len(img_file) - 4)] + '_count', "w+")
#                     f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (1000, 1000))
#                     f.close()
#
#                     cv2.imwrite(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '/' + img_file[:(len(img_file) - 4)] + save_name[0]+'.png', ori_image) #

# uncomment to save in sample folder format

def saving_result(img_directory = 'image', result_directory = 'result/general_test', wait_time = 100, color_check=False, save_name = ['_ori_','_e_coli_', '_coliform_']):
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
            inputimg2= cv2.imread(img_directory + '/' + img_file)

            # Get colony counts ONLY (will be deprecated soon)
            #blue_count = get_count(inputimg, 'Blue')
            #purple_count = get_count(inputimg, 'Purple')

            # Get segmented images and counts
            blue_image, blue_count = get_image_and_count(inputimg, 'Blue')
            purple_image, purple_count = get_image_and_count(inputimg, 'Purple')

            # Get the cropped original image of the same dimension for ease of result viewing
            ori_image = raw_to_cropped(inputimg2, dim=(512, 512))
            cv2.rectangle(ori_image, (1, 1), (511, 511), (255, 255, 255), 2)

            # Uncomment to print counts
            #print('There are %d E. coli colonies and %d coliforms.' %(blue_count, purple_count))

            # Creating a directory to save the new images and count text file
            if os.path.exists(result_directory) == False:
                os.mkdir(result_directory)
            if os.path.exists(result_directory + '/' + img_file[:(len(img_file)-4)] + '_result') == True:
                i = 1
                while(os.path.exists(result_directory + '/' + img_file[:(len(img_file)-4)] + '_result' + '(' + str(i) + ')') == True):
                    i += 1
                print("Folder exists, saving images and text file in a new folder called " + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')' + " to prevent overwriting files in the existing folder.")
                os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')')

                # Saving the counts in a text file
                f = open(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')/' + img_file[:(len(img_file) - 4)] + '_count', "w+")
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
                cv2.rectangle(blue_image, (1,1), (511,511), (255, 255, 255), 2)
                cv2.rectangle(purple_image, (1,1), (511,511), (255, 255, 255), 2)
                # Saving the segmented images
                cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')/' + img_file[:(len(img_file) - 4)] + '_E.coli.png', blue_image)
                cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')/' + img_file[:(len(img_file) - 4)] + '_coliform.png', purple_image)
                cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')/' + img_file[:(len(img_file) - 4)] + '_c_original.png', ori_image)

                # Uncomment to show segmented images
                cv2.imshow('E.coli colonies', blue_image)
                cv2.imshow('coliforms', purple_image)
                print("Press any key to move on to the next image. (5 seconds)")
                cv2.waitKey(wait_time)
            else:
                print("Saving images and text file in a new folder called " + img_file[:(len(img_file) - 4)] + '_result')
                os.mkdir(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result')

                # Saving the counts in a text file
                f = open(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(len(img_file) - 4)] + '_count', "w+")
                f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (blue_count, purple_count))
                f.close()

                # Writing title and count on images
                cv2.putText(blue_image, 'E.coli count : %d'%(blue_count),
                            (5,500),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,255,255),
                            2)
                cv2.putText(purple_image, 'coliform count : %d' % (purple_count),
                            (5, 500),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2)
                cv2.rectangle(blue_image, (1,1), (511,511), (255, 255, 255), 2)
                cv2.rectangle(purple_image, (1,1), (511,511), (255, 255, 255), 2)
                # Saving the segmented images
                cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(len(img_file) - 4)] + '_E.coli.png', blue_image)
                cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(len(img_file) - 4)] + '_coliform.png', purple_image)
                cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(len(img_file) - 4)] + '_c_original.png', ori_image)

                # Uncomment to show segmented images
                cv2.imshow('E.coli colonies', blue_image)
                cv2.imshow('coliforms', purple_image)
                print("Press any key to move on to the next image. (5 seconds)")
                cv2.waitKey(wait_time)
    else:
        for img_file in os.listdir(img_directory):
            inputimg = cv2.imread(img_directory + '/' + img_file)
            inputimg2= cv2.imread(img_directory + '/' + img_file)

            overgrown_flag = RGB_comparator(raw_to_cropped(inputimg, dim=(256, 256), color_check=True))

            if overgrown_flag == False:
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

                # Uncomment to print counts
                # print('There are %d E. coli colonies and %d coliforms.' %(blue_count, purple_count))

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
                            i) + ')/' + img_file[:(len(img_file) - 4)] + save_name[0]+'.png', twoboundary)

                    # Uncomment to show segmented images
                    cv2.imshow('E.coli colonies', blue_image)
                    cv2.imshow('coliforms', purple_image)
                    print("Press any key to move on to the next image. (5 seconds)")
                    cv2.waitKey(wait_time)
                else:
                    print(
                        "Saving images and text file in a new folder called " + img_file[:(len(img_file) - 4)] + '_result')
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
                                img_file[:(len(img_file) - 4)] + save_name[0]+'.png', ori_image)

                    # Uncomment to show segmented images
                    cv2.imshow('E.coli colonies', blue_image)
                    cv2.imshow('coliforms', purple_image)
                    print("Press any key to move on to the next image. (5 seconds)")
                    cv2.waitKey(wait_time)
            else:
                print('sample is overgrown, too many colonies to count')
                flag_string = 'flagged as too many to count'
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

                # Uncomment to print counts
                # print('There are %d E. coli colonies and %d coliforms.' %(blue_count, purple_count))

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
                    print("Folder exists, saving images and text file in a new folder called " + img_file[:(
                            len(img_file) - 4)] + '_result' + '(' + str(
                        i) + ')' + " to prevent overwriting files in the existing folder.")
                    os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                        i) + ')')

                    # Saving the counts in a text file
                    f = open(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(
                        i) + ')/' + img_file[:(len(img_file) - 4)] + '_count', "w+")
                    f.write('E. coli\t\t: %d %s \ncoliforms\t: %d %s' % (blue_count, flag_string, purple_count, flag_string))
                    f.close()

                    cv2.rectangle(blue_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    cv2.rectangle(purple_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    # Saving the segmented images
                    twoboundary = masking_boundary(masking_boundary(ori_image, blue_image), purple_image)

                    # Writing title and count on images
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
                            i) + ')/' + img_file[:(len(img_file) - 4)] + save_name[0]+'.png', twoboundary)

                    # Uncomment to show segmented images
                    cv2.imshow('E.coli colonies', blue_image)
                    cv2.imshow('coliforms', purple_image)
                    print("Press any key to move on to the next image. (5 seconds)")
                    cv2.waitKey(wait_time)
                else:
                    print(
                        "Saving images and text file in a new folder called " + img_file[:(len(img_file) - 4)] + '_result')
                    os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result')

                    # Saving the counts in a text file
                    f = open(
                        result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(
                                len(img_file) - 4)] + '_count', "w+")
                    f.write('E. coli\t\t: %d %s \ncoliforms\t: %d %s' % (blue_count, flag_string, purple_count, flag_string))
                    f.close()

                    cv2.rectangle(blue_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    cv2.rectangle(purple_image, (1, 1), (511, 511), (255, 255, 255), 2)
                    # Saving the segmented images
                    twoboundary = masking_boundary(masking_boundary(ori_image, blue_image), purple_image)

                    # Writing title and count on images
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
                                img_file[:(len(img_file) - 4)] + save_name[0]+'.png', ori_image)
                    # Uncomment to show segmented images
                    cv2.imshow('E.coli colonies', blue_image)
                    cv2.imshow('coliforms', purple_image)
                    print("Press any key to move on to the next image. (5 seconds)")
                    cv2.waitKey(wait_time)

def saving_result_timelapse(img_directory = 'image', result_directory = 'result/color-check', wait_time = 100, color_check=False, save_name = ['_ori_','_e_coli_', '_coliform_']):
    if color_check == False:
        for img_file in os.listdir(img_directory):
            inputimg = cv2.imread(img_directory + '/' + img_file)
            inputimg2= cv2.imread(img_directory + '/' + img_file)

            # Get colony counts ONLY (will be deprecated soon)
            #blue_count = get_count(inputimg, 'Blue')
            #purple_count = get_count(inputimg, 'Purple')

            # Get segmented images and counts
            blue_image, blue_count = get_image_and_count(inputimg, 'Blue')
            purple_image, purple_count = get_image_and_count(inputimg, 'Purple')

            # Get the cropped original image of the same dimension for ease of result viewing
            ori_image = raw_to_cropped(inputimg2, dim=(512, 512))
            cv2.rectangle(ori_image, (1, 1), (511, 511), (255, 255, 255), 2)

            # Uncomment to print counts
            #print('There are %d E. coli colonies and %d coliforms.' %(blue_count, purple_count))

            # Creating a directory to save the new images and count text file
            if os.path.exists(result_directory) == False:
                os.mkdir(result_directory)
            if os.path.exists(result_directory + '/' + img_file[:(len(img_file)-4)] + '_result') == True:
                i = 1
                while(os.path.exists(result_directory + '/' + img_file[:(len(img_file)-4)] + '_result' + '(' + str(i) + ')') == True):
                    i += 1
                print("Folder exists, saving images and text file in a new folder called " + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')' + " to prevent overwriting files in the existing folder.")
                os.mkdir(result_directory + '/' + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')')

                # Saving the counts in a text file
                f = open(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')/' + img_file[:(len(img_file) - 4)] + '_count', "w+")
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
                cv2.rectangle(blue_image, (1,1), (511,511), (255, 255, 255), 2)
                cv2.rectangle(purple_image, (1,1), (511,511), (255, 255, 255), 2)
                # Saving the segmented images
                cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')/' + img_file[:(len(img_file) - 4)] + '_E.coli.png', blue_image)
                cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')/' + img_file[:(len(img_file) - 4)] + '_coliform.png', purple_image)
                cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result' + '(' + str(i) + ')/' + img_file[:(len(img_file) - 4)] + '_c_original.png', ori_image)

                # Uncomment to show segmented images
                cv2.imshow('E.coli colonies', blue_image)
                cv2.imshow('coliforms', purple_image)
                print("Press any key to move on to the next image. (5 seconds)")
                cv2.waitKey(wait_time)
            else:
                print("Saving images and text file in a new folder called " + img_file[:(len(img_file) - 4)] + '_result')
                os.mkdir(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result')

                # Saving the counts in a text file
                f = open(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(len(img_file) - 4)] + '_count', "w+")
                f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (blue_count, purple_count))
                f.close()

                # Writing title and count on images
                cv2.putText(blue_image, 'E.coli count : %d'%(blue_count),
                            (5,500),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,255,255),
                            2)
                cv2.putText(purple_image, 'coliform count : %d' % (purple_count),
                            (5, 500),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2)
                cv2.rectangle(blue_image, (1,1), (511,511), (255, 255, 255), 2)
                cv2.rectangle(purple_image, (1,1), (511,511), (255, 255, 255), 2)
                # Saving the segmented images
                cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(len(img_file) - 4)] + '_E.coli.png', blue_image)
                cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(len(img_file) - 4)] + '_coliform.png', purple_image)
                cv2.imwrite(result_directory + '/'  + img_file[:(len(img_file) - 4)] + '_result/' + img_file[:(len(img_file) - 4)] + '_c_original.png', ori_image)

                # Uncomment to show segmented images
                cv2.imshow('E.coli colonies', blue_image)
                cv2.imshow('coliforms', purple_image)
                print("Press any key to move on to the next image. (5 seconds)")
                cv2.waitKey(wait_time)
    else:
        for i,img_file in enumerate(os.listdir(img_directory)):
            inputimg = cv2.imread(img_directory + '/' + img_file)
            inputimg2= cv2.imread(img_directory + '/' + img_file)

            overgrown_flag = RGB_comparator(raw_to_cropped(inputimg, dim=(256, 256), color_check=True))

            if overgrown_flag == False:
                print('Sample is not overgrown, proceed with counting')
                # Get colony counts ONLY (will be deprecated soon)
                # blue_count = get_count(inputimg, 'Blue')
                # purple_count = get_count(inputimg, 'Purple')

                # Get segmented images and counts
                blue_image, blue_count = get_image_and_count(inputimg, 'Blue')
                purple_image, purple_count = get_image_and_count(inputimg, 'Purple')

                # Get the cropped original image of the same dimension for ease of result viewing
                ori_image = raw_to_cropped(inputimg2, dim=(512, 512))
                cv2.rectangle(ori_image, (1, 1), (511, 511), (255, 255, 255), 2)

                # Uncomment to print counts
                # print('There are %d E. coli colonies and %d coliforms.' %(blue_count, purple_count))

                # Creating a directory to save the new images and count text file
                if os.path.exists(result_directory) == False:
                    os.mkdir(result_directory)
                if os.path.exists(
                        result_directory + '/' + str(i) + '_result') == False:
                    os.mkdir(result_directory + '/' + str(i) + '_result')

                # Saving the counts in a text file
                f = open(
                    result_directory + '/' + str(i) + '_result/' + str(i) +'_'+ img_file[:(
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
                cv2.imwrite(
                    result_directory + '/' + str(i) + '_result/' + str(i) +'_'+ img_file[:(
                            len(img_file) - 4)] + save_name[1]+'.png', blue_image)
                cv2.imwrite(
                    result_directory + '/' + str(i) + '_result/' + str(i) +'_'+ img_file[:(
                            len(img_file) - 4)] + save_name[2]+'.png', purple_image)
                cv2.imwrite(
                    result_directory + '/' + str(i) + '_result/'+ str(i) +'_'+ img_file[:(
                            len(img_file) - 4)] + save_name[0]+'.png', ori_image)

                # Uncomment to show segmented images
                cv2.imshow('E.coli colonies', blue_image)
                cv2.imshow('coliforms', purple_image)
                print("Press any key to move on to the next image. (5 seconds)")
                cv2.waitKey(wait_time)
            else:
                print('sample is overgrown, too many colonies to count')

                # Get the cropped original image of the same dimension for ease of result viewing
                ori_image = raw_to_cropped(inputimg2, dim=(512, 512))
                cv2.rectangle(ori_image, (1, 1), (511, 511), (255, 255, 255), 2)

                # Creating a directory to save the new images and count text file
                if os.path.exists(result_directory) == False:
                    os.mkdir(result_directory)
                if os.path.exists(result_directory +'/' + str(i) + '_result') == False:
                    os.mkdir(result_directory +'/'+ str(i) + '_result')

                # Saving the counts in a text file
                f = open(result_directory +'/' + str(i) + '_result/' + str(i) +'_' + img_file[:(
                        len(img_file) - 4)] + '_count', "w+")
                f.write('E. coli\t\t: %d \ncoliforms\t: %d' % (1000, 1000))
                f.close()

                cv2.imwrite(result_directory + '/' + str(i) + '_result' + '/' + str(i) +'_'+ img_file[:(len(img_file) - 4)] + save_name[0]+'.png', ori_image)

def saving_count_as_csv(img_directory = 'image', color_check=False):
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
        for i,img_file in enumerate(os.listdir(img_directory)):
            inputimg = cv2.imread(img_directory + '/' + img_file)

            overgrown_flag = RGB_comparator(raw_to_cropped(inputimg, dim=(256, 256), color_check=True))

            if overgrown_flag == False:
                print('Sample is not overgrown, proceed with counting')
                # Get colony counts ONLY
                blue_count = get_count(inputimg, 'Blue')
                purple_count = get_count(inputimg, 'Purple')
                img_name.append(img_file)
                e_coli_count.append(blue_count)
                coliform_count.append(purple_count)
            else:
                print('sample is overgrown, too many colonies to count')
                blue_count = get_count(inputimg, 'Blue') + 0.1
                purple_count = get_count(inputimg, 'Purple') + 0.1
                img_name.append(img_file)
                e_coli_count.append(blue_count)
                coliform_count.append(purple_count)
        return img_name, e_coli_count, coliform_count

if __name__ == "__main__":

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

    # testimg = cv2.imread('image/2020 03 16 02 R4 15_hour_15.jpg')
    # testimg_cropped = raw_to_cropped(testimg, dim=(256, 256))


    #test loading model as tflite format
    # Load the TFLite model and allocate tensors.
    #interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    
    interpreter = Interpreter(model_path="converted_model.tflite")
    
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    #
    # input_data2 = input_data
    # input_data2[0] = testimg_cropped
    # interpreter.set_tensor(input_details[0]['index'], input_data2)
    #
    # interpreter.invoke()
    #
    # # The function `get_tensor()` returns a copy of the tensor data.
    # # Use `tensor()` in order to get a pointer to the tensor.
    # output_data = interpreter.get_tensor(output_details[0]['index'])
    #
    # output_data2 = output_data[0]
    # cv2.imshow('test', output_data2)
    # cv2.imshow('allaa', testimg_cropped)
    # cv2.waitKey(0)

    # Loading trained model
    # model = load_model('XPS13_beta06_tensor1150_keras224_60epochs.h5',
    #                    custom_objects={'tversky_loss': tversky_loss, 'generalized_dice_coeff': generalized_dice_coeff})

    ############################################################

    # # Convert the model.
    # converter = tf.lite.TFLiteConverter.from_keras_model_file(model)
    # tflite_model = converter.convert()
    #
    # # Save the TF Lite model.
    # with tf.io.gfile.GFile('model_lite_test.tflite', 'wb') as f:
    #     f.write(tflite_model)
    # Converting a tf.keras model.

    # converter = tf.lite.TFLiteConverter.from_keras_model_file('XPS13_beta06_tensor1150_keras224_60epochs.h5',
    #                     custom_objects={'tversky_loss': tversky_loss, 'generalized_dice_coeff': generalized_dice_coeff})
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)
    ############################################################
    # # just for 1 timepoint
    # saving_result(img_directory='image/April_data/single_final', result_directory = 'result/Aprildata', wait_time = 10, color_check=True)
    #
    # # for timelapse and for saving count as csv
    # t = [i for i in range(16)]
    # growth_rate = {}
    # for timepoint in t[:1]:
    #     # #if want image, uncomment the line below
    #     saving_result(img_directory='image/time_sliced/'+'final_timepoint',result_directory='result/May_10_data_new/final_timepoint', wait_time=10, color_check=True)
    #
    #     #if just want count, uncomment the line below
    #     _, e, c = saving_count_as_csv(img_directory='image/time_sliced/'+'final_timepoint', color_check=True)
    #     if timepoint == 0:
    #         growth_rate['img_name'] = _
    #     growth_rate['e_coli_'+str(timepoint)]=e
    #     growth_rate['coliform_'+str(timepoint)]=c
    # df = pd.DataFrame(growth_rate)
    # df.to_csv('result/growth_rate_May_10_test.csv')
    # img_path = 'image/time_sliced/single_15/2020 04 24 03 5ml 10ul 1200s_15.jpg'
    # for just 1 folder
    saving_result(img_directory='image',result_directory='result', wait_time=10, color_check=True)


