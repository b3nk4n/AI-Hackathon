import os
import sys
import shutil
import glob
import sqlite3
import subprocess
from PIL import Image

def jaffe_structure():

    input_dir = os.path.join(os.getcwd(), 'raw', 'jaffe')
    output_dir = os.path.join(os.getcwd(), 'emotions')

    classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    print('INPUT DIR: %s' % input_dir)
    print('OUTPUT DIR: %s' % output_dir)

    # creates a new output directory
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    for _class in classes:
        if not os.path.exists(os.path.join(output_dir, _class)):
            os.makedirs(os.path.join(output_dir, _class))

    jaffe_images = [file for file in os.listdir(input_dir) if file.endswith(".tiff")]

    emotions_dict = {"AN": 0, "DI": 1, "FE": 2, "HA": 3, "SA": 4, "SU": 5, "NE": 6}

    size = 48, 48

    for n, img in enumerate(jaffe_images):
    	emotion = img.split('.')[1][0:2]
    	label = emotions_dict[emotion]
    	image = Image.open(os.path.join(input_dir, img)).crop((20, 20, 236, 236)).resize(size)
    	image.save(os.path.join(output_dir,  classes[label], "jaffe_image{}.png".format(n)))


if __name__ == '__main__':
	jaffe_structure()

# download data from: http://www.kasrl.org/jaffe_info.html
# Extract data into raw (folder should be called jaffe)
# run this script