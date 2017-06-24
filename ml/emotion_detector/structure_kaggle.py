import os
import sys
import shutil
import glob
import sqlite3
import subprocess
import random
from PIL import Image

from custom_kaggle_dataset import KaggleFacialExpression

def install_kaggle_dataset(input_dir):
    kaggle = KaggleFacialExpression()
    kaggle.install(os.path.join(input_dir,'fer2013.tar'))
    return kaggle

def kaggle_structure(split=False):

    input_dir = os.path.join(os.getcwd(), 'raw')
    output_dir = os.path.join(os.getcwd(), 'emotions') if not split else os.path.join(os.getcwd(), 'emotions_split')

    classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    print('INPUT DIR: %s' % input_dir)
    print('OUTPUT DIR: %s' % output_dir)

    # creates a new output directory
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)


    if not split:
        for _class in classes:
            if not os.path.exists(os.path.join(output_dir, _class)):
                os.makedirs(os.path.join(output_dir, _class))

        kaggle = install_kaggle_dataset(input_dir)


        for n, image_entry in enumerate(kaggle.meta):
            image = image_entry['pixels']
            label = image_entry['label']
            pil_image = Image.fromarray(image)
            pil_image.save(os.path.join(output_dir,  classes[label], "kaggle_image{}.png".format(n)))

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

        kaggle = install_kaggle_dataset(input_dir)

        for i in range(len(classes)):
            label_meta = [meta for meta in kaggle.meta if meta['label']==i]

            for n, image_entry in enumerate(label_meta):
                randm = random.random()
                subfolder = "train" if randm >= 0.2 else "validation"
                image = image_entry['pixels']
                label = image_entry['label']
                pil_image = Image.fromarray(image)
                pil_image.save(os.path.join(output_dir, subfolder, classes[label], "kaggle_image_{}_{}.png".format(i,n)))

if __name__ == '__main__':

    kaggle_file = os.path.join(os.getcwd(), 'raw', 'fer2013.tar')

    if not os.path.exists(kaggle_file):
        print("kaggle file (fer2013.tar) couldn't be found in raw/")
        exit(1)

    if len(sys.argv) == 1:
        kaggle_structure()
    elif sys.argv[1] == '--split':
        kaggle_structure(split=True)
    else:
        print("invalid command line argument")

