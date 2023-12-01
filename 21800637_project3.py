import cv2
import numpy as np

# Global variables initialization
roi_start = None
roi_end = None
cropping = False
roi_cropped = None
warped_image = None
perspective_points = []
perspective_done = False

def draw_vertical_line(event, x, y, flags, param):
    global warped_image
    if event == cv2.EVENT_LBUTTONDOWN and warped_image is not None:
        line_image = warped_image.copy()
        cv2.line(line_image, (x, 0), (x, line_image.shape[0]), (0, 255, 0), 1)
        cv2.imshow("Warped ROI", line_image)
        warped_image = line_image  # Update the global variable

def click_and_crop(event, x, y, flags, param):
    global roi_start, roi_end, cropping, roi_cropped, perspective_points, perspective_done, warped_image

    # Record the starting point when the left mouse button is pressed
    if event == cv2.EVENT_LBUTTONDOWN and not cropping and not perspective_done:
        roi_start = (x, y)
        cropping = True
    
    # Record the end point when the left mouse button is dragged
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        roi_end = (x, y)

    # When the left mouse button is released, show the cropped ROI in a new window
    elif event == cv2.EVENT_LBUTTONUP and cropping:
        roi_end = (x, y)
        cropping = False

        if roi_start and roi_end and (roi_end[0] - roi_start[0] > 1) and (roi_end[1] - roi_start[1] > 1):
            roi_cropped = image[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]
            roi_cropped = cv2.resize(roi_cropped, None, fx=2, fy=2)
            cv2.imshow("ROI", roi_cropped)
            cv2.setMouseCallback("ROI", collect_perspective_points)

def collect_perspective_points(event, x, y, flags, param):
    global perspective_points, perspective_done, warped_image, roi_cropped

    if event == cv2.EVENT_LBUTTONDOWN and not perspective_done:
        perspective_points.append((x, y))

        if len(perspective_points) == 4:
            warped_image = four_point_transform(roi_cropped, np.array(perspective_points))
            cv2.imshow("Warped ROI", warped_image)
            cv2.setMouseCallback("Warped ROI", draw_vertical_line)
            perspective_done = True

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Perspective transformation target points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Calculate and apply the perspective transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# Load and resize the image
image = cv2.imread('test.png')
image = cv2.resize(image, (640, 480))

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# Display the image and wait for key input
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # Display a temporary ROI while dragging
    if cropping and roi_start and roi_end:
        temp_image = image.copy()
        cv2.rectangle(temp_image, roi_start, roi_end, (0, 255, 0), 2)
        cv2.imshow("image", temp_image)
    else:
        cv2.imshow("image", image)

    # Exit when 'q' key is pressed
    if key == ord("q"):
        break

cv2.destroyAllWindows()
