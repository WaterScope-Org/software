import sys
sys.path.append("/home/pi/.local/lib/python3.7/site-packages")
import time

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

    while True:
        with open('image_to_analyse.txt', 'r') as file:
            all_image_to_analyse = file.readlines()
            # print(all_image_to_analyse)

        with open('image_to_analyse.txt', 'w') as file:
            if len(all_image_to_analyse) > 0:
                print("incoming image! Processing it!")
                file.writelines(all_image_to_analyse[1:])
                analysis_image(all_image_to_analyse[0])
                print("processed {}".format(all_image_to_analyse[0]))
                print("waiting for incoming image to process....")
            
            time.sleep(1)

