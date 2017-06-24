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

    if len(sys.argv) < 2:
        kaggle_structure()
    else:
        mode = sys.argv[1]

        if len(sys.argv) > 2:
            base_dir = sys.argv[2]
            

        # if mode == 'kaggle':
            # classifier_structure()
    
        else:
            print('Unknown mode')