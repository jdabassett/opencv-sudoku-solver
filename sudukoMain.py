from utlis import *
import sudukoSolver

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

########################################################################
pathImage = "data/unsolved/1.jpg"
heightImg = 450
widthImg = 450
# model = initialize_prediction_model()
########################################################################

print('Setting UP')

# 1. PREPARE THE IMAGE
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
imgThreshold = process(img)

# 2. FIND ALL CONTOURS
imgContours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

# 3. FIND THE BIGGEST CONTOUR AND USE IT AS SUDOKU
biggest, maxArea = biggest_contour(contours)

if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

    # 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    # imgSolvedDigits = imgBlank.copy()
    # boxes = split_boxes(imgWarpColored)
    # cv2.imshow("Sample", boxes[65])
    # numbers = get_prediction(boxes, model)
    # imgDetectedDigits = display_numbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    # numbers = np.asarray(numbers)
    # posArray = np.where(numbers > 0, 0, 1)
    #
    # # 5. FIND SOLUTION OF THE BOARD
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
    #
imageArray = ([img, imgThreshold, imgContours, imgBigContour],
                [imgWarpColored, img, img, img])
              # [imgWarpColored, imgBlank, imgBlank, imgBlank])
stackedImage = stack_images(imageArray, 1)
cv2.imshow('Stacked Images', stackedImage)

# else:
#     print("No Sudoku Found")

cv2.waitKey(0)
