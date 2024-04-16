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
img_path = "../data/sudokus/_196_1844104.jpeg"
img_hgh = 450
img_wdt = 450
model = initialize_prediction_model()
puzzle_paths = ["_196_1844104.jpeg","_7_7661130.jpeg","_10_654435.jpeg","_12_126115.jpeg","_16_254551.jpeg","_18_589801.jpeg","_22_293703.jpeg","_25_133964.jpeg","_34_369826.jpeg","_41_2774946.jpeg"]
puzzle_unsolved = ["000060080007000004050803100006000800700010005008000400005609020100000300040070000",
                   "004169075100008030005000400006700080407000201010004900008000700090600008600287500",
                   "810003004005041000030008010009100470040050060081004500060900080000310700300400096",
                   "060020780592007041700004060046205000300040005000301890020800003930400218078030050",
                   "000230000067000920090007030004070008600402001700010600070600010018000370000051000",
                   "800010009050807010004090700060701020508060107010502090007040600080309040300050008",
                   "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
                   "496000030008006040050204690031672480800500306609000050962410873080000024000028165",
                   "000230000067000920090007030004070008600402001700010600070600010018000370000051000",
                   "042000005000632080080040200000000000715068340908350761091006000000020190006100050"]
test_puzzle = [[2, 1, 0, 0, 6, 0, 9, 0, 0], [0, 0, 0, 0, 0, 9, 1, 0, 0], [4, 0, 9, 3, 1, 0, 0, 5, 8], [0, 0, 1, 0, 0, 5, 0, 4, 0], [9, 0, 4, 0, 3, 0, 8, 0, 5], [0, 5, 0, 2, 0, 0, 6, 0, 0], [3, 8, 0, 0, 4, 0, 5, 0, 6], [0, 0, 6, 7, 0, 0, 0, 0, 2], [0, 0, 7, 0, 8, 0, 3, 0, 9]]
test_solved = [[2, 1, 5, 4, 6, 8, 9, 3, 7], [7, 3, 8, 5, 2, 9, 1, 6, 4], [4, 6, 9, 3, 1, 7, 2, 5, 8], [6, 2, 1, 8, 9, 5, 7, 4, 3], [9, 7, 4, 1, 3, 6, 8, 2, 5], [8, 5, 3, 2, 7, 4, 6, 9, 1], [3, 8, 2, 9, 4, 1, 5, 7, 6], [1, 9, 6, 7, 5, 3, 4, 8, 2], [5, 4, 7, 6, 8, 2, 3, 1, 9]]
########################################################################

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

    unsolved_lst, unsolved_str, new_boxes = get_prediction(boxs, model)

    print(unsolved_lst)
    print(unsolved_str)
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
