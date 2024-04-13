import cv2
import numpy as np
import os
from sudoku_solver import (
    Sudoku,
    convert_board,
    solve_board)
from utlis import (
    biggest_contour,
    display_numbers,
    draw_grid,
    get_prediction,
    initialize_prediction_model,
    img_to_thr,
    reorder,
    split_boxes,
    stack_images)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

########################################################################
pth_img = "../data/unsolved/1.jpg"
hgh_img = 450
wdt_img = 450
# model = initialize_prediction_model()
########################################################################

print('Setting UP')

# prepare image
img = cv2.imread(pth_img)
img = cv2.resize(img, (wdt_img, hgh_img))
img_bln = np.zeros((hgh_img, wdt_img, 3), np.uint8)
img_thr = img_to_thr(img)

# find all borders/contours
img_cnt, img_cnt_bgg = img.copy(), img.copy()
cntrs, _ = cv2.findContours(img_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_cnt, cntrs, -1, (0, 255, 0), 3)

# find border of sudoku
bgg, max_area = biggest_contour(cntrs)

print("reached if statement")
if bgg.size != 0:
    print('reached inside if statement')
    bgg = reorder(bgg)
    cv2.drawContours(img_cnt_bgg, bgg, -1, (0, 0, 255), 25)
    # make transformation matrix
    pts1 = np.float32(bgg)
    pts2 = np.float32([[0, 0], [wdt_img, 0], [0, hgh_img], [wdt_img, hgh_img]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # apply perspective shift
    img_wrp_clr = cv2.warpPerspective(img, matrix, (wdt_img, hgh_img))
    img_wrp_clr = cv2.cvtColor(img_wrp_clr, cv2.COLOR_BGR2GRAY)

    # split puzzle into individual cells
    img_slv_dgt = img_bln.copy()
    bxs = split_boxes(img_wrp_clr)
    # cv2.imshow("Sample", bxs[65])

    # numbers = get_prediction(boxes, model)
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
