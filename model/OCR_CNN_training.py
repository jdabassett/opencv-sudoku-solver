import cv2
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split

##############################################################
# global variables
dgt_pth = "../data/digits"
test_ratio = 0.2
validation_ratio = 0.2
img_dim = (32, 32, 3)
##############################################################

# declare variables
dgt_img = list()
dgt_num = list()

my_list = os.listdir(dgt_pth)
num_classes = len(my_list)

# load, resize, and sort digit images into a list
for i in range(0, num_classes):
    pic_list = os.listdir(path+"/"+str(i))
    for pic_name in pic_list:
        curr_img = cv2.imread(path+'/'+str(i)+'/'+pic_name)
        curr_img = cv2.resize(curr_img, (img_dim[0], img_dim[1]))
        digits_img.append(curr_img)
        digits_num.append(i)

# convert list into numpy array
digits_img = np.array(digits_img)
digits_num = np.array(digits_num)

# splitting data into training and testing
x_train, x_test, y_train, y_test = train_test_split(digits_img, digits_num, test_size=test_ratio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_ratio)

print(digits_img.shape)
print(x_train.shape)
print(x_validation.shape)

num_samples = list()
for i in range(0, num_classes):
    num_samples.append(len(np.where(y_train==i)[0]))

plt.figure(figsize=(10, 5))
plt.title("Number of Images for each Digit")
plt.bar(range(0, num_classes), num_samples)
plt.xlabel("Digits")
plt.ylabel("Number of Images")
plt.show()


# process images
def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


x_train = np.array(list(map(process_image, x_train)))
x_test = np.array(list(map(process_image, x_test)))
x_validation = np.array(list(map(process_image, x_validation)))

# change shape before training
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

# add variation to images through random shift, zoom, shear, rotation
data_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
data_gen.fit(x_train)

# convert categorical data into binary for training
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_validation = to_categorical(y_validation, num_classes)


# setup model
def generate_model():
    num_filters = 60
    size_filter1 = (5, 5)
    size_filter2 = (3, 3)
    size_pool = (2, 2)
    num_node = 500

    model = Sequential()
    model.add((Conv2D(num_filters,
                      size_filter1,
                      input_shape=(img_dim[0], img_dim[1], 1),
                      activation='relu')))
    model.add((Conv2D(num_filters,
                      size_filter1,
                      activation='relu')))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add((Conv2D(num_filters//2,
                      size_filter2,
                      activation='relu')))
    model.add((Conv2D(num_filters//2,
                      size_filter2,
                      activation='relu')))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_node, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_node, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model0 = generate_model()
print(model0.summary())
