import copy
import cv2
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Any


# create model
def initialize_prediction_model() -> Model:
    new_model = load_model('../model/model_trained_10_3.keras')
    return new_model


# preprocessing Image
def img_to_thr(img: np.ndarray) -> np.ndarray:
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp = cv2.GaussianBlur(temp, (5, 5), 1)
    temp = cv2.adaptiveThreshold(temp, 255, 1, 1, 11, 2)
    return temp


# convert all images into grayscale and equalize histogram
def img_to_equalized(img: np.ndarray) -> np.ndarray:
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp = cv2.equalizeHist(img)
    temp = temp/255
    return temp


# reorder points for Warp Perspective
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


# find biggest contour
def biggest_contour(contours: List[np.ndarray]) -> np.ndarray:
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
    return biggest


# split image into 81 cells
def split_boxes(img: np.ndarray) -> List[np.ndarray]:
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


# predict value of each cell
def get_prediction(boxes: List[np.ndarray], model: Model):
    result_lst = [[] for _ in range(9)]
    new_boxes = []
    for idx,image in enumerate(boxes):
        img = np.asarray(image)
        height, width = img.shape[0], img.shape[1]
        h_ten, w_ten = height//7, width//7
        img = img[h_ten:height - h_ten, w_ten:width-w_ten]
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
        img = cv2.resize(img, (32, 32))
        img = img / 255
        new_boxes.append(img)
        img = img.reshape(1, 32, 32, 1)
        pred = model.predict(img)
        prob_idx = np.argmax(pred, axis=1)
        prob_hgh = pred[0, prob_idx][0]
        row = idx // 9
        if prob_hgh > 0.8:
            result_lst[row].append(int(prob_idx[0]+1))
        else:
            result_lst[row].append(0)
    return result_lst, new_boxes


# display solution on puzzle
def display_numbers(img: np.ndarray, puzzle: List[List[int]] , solution: List[List[int]], color: Tuple[int] = (0, 255, 0)) -> np.ndarray:
    temp = copy.deepcopy(img)
    width = int(temp.shape[1]/9)
    height = int(temp.shape[0]/9)
    for x in range(0, 9):
        for y in range(0, 9):
            prev = puzzle[y][x]
            if prev == 0:
                val = str(solution[y][x])
                cv2.putText(temp, val, (x*width+int(width/2)-10, int((y+0.8)*height)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv2.LINE_AA)
    return temp


# stack all images in one window
def stack_images(img_array: List[np.ndarray], scale: int):
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
    else:
        for x in range(0, rows):
            img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def plot_cells(cells: List[np.ndarray]) -> None:
    plt.figure(figsize=(12, 12))

    for i in range(0, 81):
        plt.subplot(9, 9, i + 1)
        plt.axis('off')
        plt.imshow(cells[i], cmap='gray')

    plt.show()


# if __name__ == "__main__":
#     model0 = initialize_prediction_model()