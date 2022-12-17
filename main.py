from pprint import pprint
import cv2
import numpy as np
import math
import time

vid = cv2.VideoCapture(0)

while (True):
    ret, img = vid.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    width = img.shape[1]
    height = img.shape[0]

    y_sobel = np.zeros((width, height), dtype=np.int8)

    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    def kernel_match(x, y):
        # Get the surrounding pixels
        surrounding_pixels = blur[x - 1:x + 2, y - 1:y + 2]

        if surrounding_pixels.shape != (3, 3):
            return 0

        # Multiply the kernel with the surrounding pixels
        result = np.multiply(surrounding_pixels, kernel)

        # Sum the result
        result = np.sum(result)

        return result

    for x in range(width):
        for y in range(height):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                y_sobel[x][y] = 0
            else:
                y_sobel[x][y] = kernel_match(x, y)

    # display the image
    cv2.imshow("y_sobel", y_sobel)

    x_sobel = np.zeros((width, height), dtype=np.int8)

    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    for x in range(width):
        for y in range(height):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                x_sobel[x][y] = 0
            else:
                x_sobel[x][y] = kernel_match(x, y)

    xy_sobel = np.zeros((width, height), dtype=np.int8)

    for x in range(width):
        for y in range(height):
            xy_sobel[x][y] = math.sqrt(x_sobel[x][y] ** 2 + y_sobel[x][y] ** 2)

    # convert xy_sobel to uint8
    xy_sobel = np.uint8(xy_sobel)

    # blue xy_sobel
    xy_sobel = cv2.GaussianBlur(xy_sobel, (5, 5), 0)

    # display the image
    cv2.imshow("xy_sobel", xy_sobel)

    # display the image
    cv2.imshow("x_sobel", x_sobel)

    # time.sleep(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
