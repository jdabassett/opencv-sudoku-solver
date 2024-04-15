import cv2
import numpy as np
import os
from sudoku_solver import (
    Sudoku,
    convert_board,
    solve_board)
from utilities import (
    biggest_contour,
    display_numbers,
    draw_grid,
    get_prediction,
    plot_cells,
    initialize_prediction_model,
    img_to_thr,
    reorder,
    split_boxes,
    stack_images)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

########################################################################
# img_path = "../data/unsolved/1.jpg"
img_path = "../data/unsolved/3.jpg"
# img_path = "../data/sudokus/_0_926439.jpeg" # very bad
# img_path = "../data/sudokus/_16_5655697.jpeg" # not good
# img_path = "../data/sudokus/_70_71729.jpeg" # not good
# img_path = "../data/sudokus/_90_762455.jpeg" # few misses
# img_path = "../data/sudokus/_141_6782588.jpeg" # misses everything
# img_path = "../data/sudokus/_196_1844104.jpeg"
img_hgh = 450
img_wdt = 450
model = initialize_prediction_model()
test_puzzle = [[2, 1, 0, 0, 6, 0, 9, 0, 0], [0, 0, 0, 0, 0, 9, 1, 0, 0], [4, 0, 9, 3, 1, 0, 0, 5, 8], [0, 0, 1, 0, 0, 5, 0, 4, 0], [9, 0, 4, 0, 3, 0, 8, 0, 5], [0, 5, 0, 2, 0, 0, 6, 0, 0], [3, 8, 0, 0, 4, 0, 5, 0, 6], [0, 0, 6, 7, 0, 0, 0, 0, 2], [0, 0, 7, 0, 8, 0, 3, 0, 9]]
test_solved = [[2, 1, 5, 4, 6, 8, 9, 3, 7], [7, 3, 8, 5, 2, 9, 1, 6, 4], [4, 6, 9, 3, 1, 7, 2, 5, 8], [6, 2, 1, 8, 9, 5, 7, 4, 3], [9, 7, 4, 1, 3, 6, 8, 2, 5], [8, 5, 3, 2, 7, 4, 6, 9, 1], [3, 8, 2, 9, 4, 1, 5, 7, 6], [1, 9, 6, 7, 5, 3, 4, 8, 2], [5, 4, 7, 6, 8, 2, 3, 1, 9]]
########################################################################

print('Setting UP')

# prepare image
img = cv2.imread(img_path)
img = cv2.resize(img, (img_wdt, img_hgh))
img_bln = np.zeros((img_hgh, img_wdt, 3), np.uint8)
img_thr = img_to_thr(img)

# find all borders/contours
img_cnt, img_cnt_bgg = img.copy(), img.copy()
cntrs, _ = cv2.findContours(img_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_cnt, cntrs, -1, (0, 255, 0), 3)

# find border of sudoku
bgg, max_area = biggest_contour(cntrs)

# print("reached if statement")
if bgg.size != 0:
    # print('reached inside if statement')
    bgg = reorder(bgg)
    cv2.drawContours(img_cnt_bgg, bgg, -1, (0, 0, 255), 25)
    # make transformation matrix
    pts1 = np.float32(bgg)
    pts2 = np.float32([[0, 0], [img_wdt, 0], [0, img_hgh], [img_wdt, img_hgh]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # apply perspective shift
    img_wrp_clr = cv2.warpPerspective(img, matrix, (img_wdt, img_hgh))
    img_wrp_clr = cv2.cvtColor(img_wrp_clr, cv2.COLOR_BGR2GRAY)

    # split puzzle into individual cells
    img_slv_dgt = img_bln.copy()
    boxs = split_boxes(img_wrp_clr)

    unsolved, new_boxes = get_prediction(boxs, model)

    print(unsolved)
    plot_cells(new_boxes)
    # imgDetectedDigits = display_numbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    # numbers = np.asarray(numbers)
    # posArray = np.where(numbers > 0, 0, 1)
    #
    # # find solution for the board
    # board = np.array_split(numbers, 9)
    #
    # try:
    #     sudukoSolver.solve(board)
    # except:
    #     pass
    #
    # flatList = []
    # for sublist in board:
    #     for item in sublist:
    #         flatList.append(item)
    # solvedNumbers = flatList*posArray
    # imgSolvedDigits = display_numbers(imgSolvedDigits, solvedNumbers)
    #
    # # #### 6. OVERLAY SOLUTION
    # pts2 = np.float32(biggest)
    # pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # imgInvWarpColored = img.copy()
    # imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    # inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    # imgDetectedDigits = draw_grid(imgDetectedDigits)
    # imgSolvedDigits = draw_grid(imgSolvedDigits)

    # imageArray = ([img, img_thr, img_cnt, img_cnt_bgg],
    #                 [img_wrp_clr, img, img, img])
    #               # [imgWarpColored, imgBlank, imgBlank, imgBlank])
    # stacked_image = stack_images(imageArray, 1)
    # cv2.imshow('Stacked Images', stacked_image)

else:
    print("No Sudoku Found")
