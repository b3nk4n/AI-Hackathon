from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator

from models.dexpression import model_keras

batch_size = 64
image_size = 48

model = model_keras.create_model(image_size)

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
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'emotions',  # this is the target directory
        target_size=(image_size, image_size),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'emotions',
        target_size=(image_size, image_size),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=20000 // batch_size,  # TODO count number of training examples?
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)  # TODO count number of valid examples?

# export model weights
model.save_weights('first_try.h5')
