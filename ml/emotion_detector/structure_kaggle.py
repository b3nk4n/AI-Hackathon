import os
import sys
import shutil
import glob
import sqlite3
import subprocess
from PIL import Image

from custom_kaggle_dataset import KaggleFacialExpression

def install_kaggle_dataset(input_dir):
    kaggle = KaggleFacialExpression()
    kaggle.install(os.path.join(input_dir,'fer2013.tar'))
    return kaggle

def kaggle_structure():

    input_dir = os.path.join(os.getcwd(), 'raw')
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

    kaggle = install_kaggle_dataset(input_dir)


    for n, image_entry in enumerate(kaggle.meta):
        image = image_entry['pixels']
        label = image_entry['label']
        pil_image = Image.fromarray(image)
        pil_image.save(os.path.join(output_dir,  classes[label], "kaggle_image{}.png".format(n)))

        # ie c2v.imwrite(image, os.path.join(output_dir,  _class, 'some image name'))

    #print('Training images: %d, Test images: %d' %(no_training_images,no_test_images))


if __name__ == '__main__':

    kaggle_file = os.path.join(os.getcwd(), 'raw', 'fer2013.tar')

    if os.path.exists(kaggle_file):
        kaggle_structure()
    else:
        print("kaggle file (fer2013.tar) couldn't be found in raw/")