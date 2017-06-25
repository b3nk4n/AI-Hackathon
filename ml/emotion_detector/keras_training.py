from __future__ import print_function

import coremltools

from keras.preprocessing.image import ImageDataGenerator

from models.dexpression import DexpressionNetKeras #, model_keras
from models.dexpression.conf import dex_hyper_params as hyper_params
from datasets.face_expression_dataset import CLASSES

batch_size = 128
image_size = 48
input_shape = (image_size, image_size, 1)
n_classes = len(CLASSES)

dexpression_net_keras = DexpressionNetKeras(0.0005, input_shape=input_shape, hyper_params=hyper_params,
                                            n_classes=n_classes)
model = dexpression_net_keras.inference(None, None)
# model = model_keras.create_model(image_size)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,  # TODO what is this and is it useful?
        zoom_range=0.1,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# the subfolders of the given root, and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'emotions_split/train',
        target_size=(image_size, image_size),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'emotions_split/validation',
        target_size=(image_size, image_size),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=200000,  # TODO count number of training examples?
        nb_epoch=5,
        validation_data=validation_generator,
        nb_val_samples=800)  # TODO count number of valid examples?

# export model weights
model.save_weights('first_try.h5')
coreml_model = coremltools.converters.keras.convert(model)
coreml_model.save("keras_model.mlmodel")
