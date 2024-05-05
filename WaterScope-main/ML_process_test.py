import sys
sys.path.append("/home/pi/.local/lib/python3.7/site-packages")
import time
import os

def analysis_image(input_filename):
    start = time.time()
    result = count_colony_yolor.analysis_image(input_filename, input_filename.replace('.jpg', '_result.jpg'))
    print(result)
    print("time it tooks: {}".format(time.time() - start))


if __name__ == "__main__":
    # wipe up the file for the first time
    with open('image_to_analyse.txt', 'w+') as file:
        pass
        
    start = time.time()
    print("loading the ML module, please wait")
    import count_colony_yolor
    print("imported the ML module")
    print("time it tooks: {}".format(time.time() - start))
    print("waiting for incoming image to process....")

    for img_name in os.listdir('yolor_pi/inference_script/images/'):
        if '.jpg' in img_name[-4:] or '.png' in img_name[-4:]:
            analysis_image('yolor_pi/inference_script/images/'+img_name)
            time.sleep(1)

