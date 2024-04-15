import cv2
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
# import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split


##############################################################
# global variables
digit_path = "../data/digits"  # path to digit directories
test_ratio = 0.2  # test ratio
valid_ratio = 0.2  # validation ratio
img_dimen = (32, 32, 3)  # image dimensions
batch_size = 50
epochs = 10
shuffle = True
##############################################################

# declare variables
digit_imgs = list()  # digit images
digit_cat = list()  # digit categories

digit_dir = os.listdir(digit_path)  # digit directories
number_cat = len(digit_dir)  # number categories

# load, resize, and sort digit images into a list
for i in range(0, number_cat):
    digit_picts = os.listdir(digit_path+"/"+str(i))  # digit pictures
    for pict_name in digit_picts:  # picture name
        img_curr = cv2.imread(digit_path+'/'+str(i)+'/' + pict_name)  # image current
        img_curr = cv2.resize(img_curr, (img_dimen[0], img_dimen[1]))
        digit_imgs.append(img_curr)
        digit_cat.append(i)

# convert list into numpy array
digit_imgs = np.array(digit_imgs)
digit_cat = np.array(digit_cat)


# convert all images into grayscale and equalize histogram
def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


# preprocess images
digit_imgs = np.array(list(map(process_image, digit_imgs)))
digit_imgs = digit_imgs.reshape(digit_imgs.shape[0], digit_imgs.shape[1], digit_imgs.shape[2], 1)

# covert categorical
digit_cat = to_categorical(digit_cat, number_cat)

# splitting data into training and testing
x_train, x_test, y_train, y_test = train_test_split(digit_imgs, digit_cat, test_size=test_ratio)  # train and test datasets
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_ratio)  # validation datasets


# count the number of each type of image in training data
# nmb_smp = list()
# for i in range(0, nmb_cls):
#     nmb_smp.append(len(np.where(y_trn == i)[0]))
#
# plt.figure(figsize=(10, 5))
# plt.title("Number of Images for each Digit")
# plt.bar(range(0, nmb_cls), nmb_smp)
# plt.xlabel("Digits")
# plt.ylabel("Number of Images")
# plt.show()

# add variation to images through random shift, zoom, shear, rotation
data_gener = ImageDataGenerator(width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=0.2,
                                shear_range=0.1,
                                rotation_range=10)
data_gener.fit(x_train)

# setup model
def generate_model():
    number_flt = 60  # number of filters
    size_flt1 = (5, 5)  # size filter 1
    size_flt2 = (3, 3)  # size filter 2
    size_pool = (2, 2)  # size of pool
    number_nd1 = 500  # number of nodes 1

    model = Sequential()
    model.add((Conv2D(number_flt,
                      size_flt1,
                      input_shape=(img_dimen[0], img_dimen[1], 1),
                      activation='relu')))
    model.add((Conv2D(number_flt,
                      size_flt1,
                      activation='relu')))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add((Conv2D(number_flt//2,
                      size_flt2,
                      activation='relu')))
    model.add((Conv2D(number_flt//2,
                      size_flt2,
                      activation='relu')))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(number_nd1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_cat, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# create model
model0 = generate_model()  # model variable

# train model
steps_per_epoch = x_train.shape[0]//50
model_hst = model0.fit(data_gener.flow(x_train, y_train, batch_size=batch_size),
                  epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=(x_valid, y_valid),
                  shuffle=shuffle)


# test model
model_scr = model0.evaluate(x_test, y_test, verbose=0)  # model score
print("Test Score", model_scr[0])
print("Test Accuracy", model_scr[1])


# export model
# model_out = open("model_trained_10_2.p", "wb")  # model output
# pickle.dump(model0, model_out)
# model_out.close()
model0.save('../model/model_trained_10_1.keras')


