import os
import cv2

import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def process_image(image):
    # Crop
    image = image[60:-25, :, :]
    # Resize
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    # RGB to YUV (As done by Nvidia Model)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

# Image Augmentation
def image_augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    # Random choice from 1-3
    choice = np.random.choice(3)
    if choice == 0: # Left Image
        image, steering_angle = load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:  # Right Image
        image, steering_angle = load_image(data_dir, right), steering_angle - 0.2
    else :  # Center Image
        image, steering_angle = load_image(data_dir, center), steering_angle

    # Flip Image
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle

    # Translate Image
    t_x, t_y = range_x * (np.random.rand() - 0.5), range_y * (np.random.rand() - 0.5)
    steering_angle += t_x * 0.002
    t_m = np.float32([[1, 0, t_x], [0, 1, t_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, t_m, (width, height))

    # Random Shadow in image
    # Credits - 'https://github.com/naokishibuya'
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    # Change Brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] =  hsv[:,:,2] * (1.0 + 0.4 * (np.random.rand() - 0.5))
    image =  cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    st_angles = np.empty(batch_size)

    # Infinite loop, So keeps on generating
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # augmentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = image_augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            # add the image and steering angle to the batch
            images[i] = process_image(image)
            st_angles[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, st_angles


# Load data from csv file and split in train and test data
def load_csv():
    data = pd.read_csv('data/driving_log.csv')
    X = data[['center', 'left', 'right']].values
    y = data['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20)
    return (X_train, X_valid, y_train, y_valid)


def train_model(X_train, X_valid, y_train, y_valid):
    model = Sequential()
    model.add(Lambda(lambda x: (x/127.5-1.0), input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu', subsample=(1, 1)))
    model.add(Conv2D(64, 3, 3, activation='elu', subsample=(1, 1)))
    model.add(Dropout(0.50))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    # See the sumary of the model
    model.summary()

    checkpoint = ModelCheckpoint('model_new.h5', monitor='val_loss', verbose=0, mode='auto')
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))

    model.fit_generator(batch_generator('data', X_train, y_train, 40, True),
                        20000, 1, max_q_size=1,
                        validation_data=batch_generator('data', X_valid, y_valid, 40, False),
                        nb_val_samples=len(X_valid), callbacks=[checkpoint], verbose=1)


def main():
    X_train, X_valid, y_train, y_valid = load_csv()
    train_model(X_train, X_valid, y_train, y_valid)


if __name__ == '__main__':
    main()
