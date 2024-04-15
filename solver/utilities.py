import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pickle


# create model
def initialize_prediction_model():
    new_model = load_model('../model/model_trained_10_1.keras')
    # new_model.summary()
    return new_model


# preprocessing Image
def img_to_thr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    threshold = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return threshold


# convert all images into grayscale and equalize histogram
def img_to_equalized(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


# 3 - Reorder points for Warp Perspective
def reorder(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new


# 3 - FINDING THE BIGGEST CONTOUR ASSUMING THAT IS THE SUDOKU PUZZLE
def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


# 4 - TO SPLIT THE IMAGE INTO 81 DIFFERENT IMAGES
def split_boxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


# 4 - GET PREDICTIONS ON ALL IMAGES
def get_prediction(boxes, model):
    result = [[] for _ in range(9)]
    new_boxes = []
    probabilities = []
    for idx,image in enumerate(boxes):
        img = np.asarray(image)
        height, width = img.shape[0], img.shape[1]
        h_ten, w_ten = height//10, width//10
        img = img[h_ten:height - h_ten, w_ten:width-w_ten]
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
        img = cv2.resize(img, (32, 32))
        img = img / 255
        new_boxes.append(img)
        img = img.reshape(1, 32, 32, 1)
        pred = model.predict(img)
        prob_idx = np.argmax(pred, axis=1)
        prob_hgh = pred[0, prob_idx][0]
        row = idx // 9

        # print(idx, pred, prob_idx[0], prob_hgh)

        if prob_hgh > 0.5:
            result[row].append((prob_idx[0], round(prob_hgh, 1)))
            # result[row].append(prob_idx[0])
            probabilities.append(round(prob_hgh, 1))
        else:
            result[row].append(0)
    print(probabilities)
    return result, new_boxes


# display solution on puzzle
def display_numbers(img, numbers, color=(0, 255, 0)):
    sec_w = int(img.shape[1]/9)
    sec_h = int(img.shape[0]/9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y*9)+x] != 0:
                cv2.putText(img, str(numbers[(y*9)+x]), (x*sec_w+int(sec_w/2)-10, int((y+0.8)*sec_h)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv2.LINE_AA)
    return img


# draw grid to more effectively see puzzle
def draw_grid(img):
    sec_w = int(img.shape[1]/9)
    sec_h = int(img.shape[0]/9)
    for i in range(0, 9):
        pt1 = (0, sec_h*i)
        pt2 = (img.shape[1], sec_h*i)
        pt3 = (sec_w * i, 0)
        pt4 = (sec_w*i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)
    return img


# stack all images in one window
def stack_images(img_array, scale):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank]*rows
        hor_con = [image_blank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
            hor_con[x] = np.concatenate(img_array[x])
        ver = np.vstack(hor)
        # ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        # hor_con = np.concatenate(img_array)
        ver = hor
    return ver


def plot_cells(cells):
    plt.figure(figsize=(12, 12))

    for i in range(0, 81):
        plt.subplot(9, 9, i + 1)
        # plt.axis('off')
        plt.imshow(cells[i], cmap='gray')

    plt.show()


if __name__ == "__main__":
    model0 = initialize_prediction_model()