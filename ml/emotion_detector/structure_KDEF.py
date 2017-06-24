import os
import sys
import shutil
import glob
import sqlite3
import subprocess
import random
from PIL import Image

def KDEF_structure(split=False):

    input_dir = os.path.join(os.getcwd(), 'raw', 'KDEF')
    output_dir = os.path.join(os.getcwd(), 'emotions') if not split else os.path.join(os.getcwd(), 'emotions_split')

    classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    size = 48, 48

    KDEF_folders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir,folder))]

    emotions_dict = {"AN": 0, "DI": 1, "AF": 2, "HA": 3, "SA": 4, "SU": 5, "NE": 6}

    print('INPUT DIR: %s' % input_dir)
    print('OUTPUT DIR: %s' % output_dir)

    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    if not split:
        for _class in classes:
            if not os.path.exists(os.path.join(output_dir, _class)):
                os.makedirs(os.path.join(output_dir, _class))

        for n_folder, folder in enumerate(KDEF_folders):
            images = [file for file in os.listdir(os.path.join(input_dir, folder)) if file.endswith(".JPG")]
            
            for n, img in enumerate(images):
                emotion = img[4:6]
                label = emotions_dict[emotion] if emotion in emotions_dict.keys() else len(emotions_dict)
                angle = img[6:8]
                if angle not in ["FL", "FR"] and label < len(emotions_dict):
                    img_path = os.path.join(input_dir, folder, img)
                    image = Image.open(img_path).crop((0, 150, 562, 712)).resize(size).convert('L')
                    image.save(os.path.join(output_dir,  classes[label], "KDFE_{}_image{}.png".format(folder,n)))

    else:
        if not os.path.exists(os.path.join(output_dir, "train")):
            os.makedirs(os.path.join(output_dir, "train"))
        if not os.path.exists(os.path.join(output_dir, "validation")):
            os.makedirs(os.path.join(output_dir, "validation"))
        for _class in classes:
            if not os.path.exists(os.path.join(output_dir, "train", _class)):
                os.makedirs(os.path.join(output_dir, "train", _class))
            if not os.path.exists(os.path.join(output_dir, "validation", _class)):
                os.makedirs(os.path.join(output_dir, "validation", _class))

        for n_folder, folder in enumerate(KDEF_folders):
            images = [file for file in os.listdir(os.path.join(input_dir,folder)) if file.endswith(".JPG")]
            
            for n, img in enumerate(images):
                randm = random.random()
                subfolder = "train" if randm >= 0.2 else "validation"
                emotion = img[4:6]
                label = emotions_dict[emotion] if emotion in emotions_dict.keys() else len(emotions_dict)
                angle = img[6:8]
                if angle not in ["FL", "FR"] and label < len(emotions_dict):
                    img_path = os.path.join(input_dir, folder,img)
                    image = Image.open(img_path).crop((0, 150, 562, 712)).resize(size).convert('L')
                    image.save(os.path.join(output_dir,  subfolder, classes[label], "KDFE_{}_image{}.png".format(folder,n)))

if __name__ == '__main__':
    KDEF_dir = os.path.join(os.getcwd(), 'raw', 'KDEF')
    if not os.path.exists(KDEF_dir):
        print("KDEF folder couldn't be found in raw/")
        exit(1)

    if len(sys.argv) == 1:
        KDEF_structure()
    elif sys.argv[1] == '--split':
        KDEF_structure(split=True)
    else:
        print("invalid command line argument")

# Extract data into raw/ (folder should be called KDEF)
# run this script