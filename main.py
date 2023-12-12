# import numpy as np
# import cv2
# import mediapipe as mp

from landmarker import *
from rectangle import *


def color_panel_rect():
    cpanel = list()
    color_bgr = {"0": (255, 255, 255), "1": (0, 0, 0), "2": (22, 45, 93), "3": (35, 51, 235), "4": (51, 134, 240),
                 "5": (85, 254, 255), "6": (76, 250, 117), "7": (61, 205, 105), "8": (253, 252, 116), "9": (245, 30, 0),
                 "10": (247, 61, 234), "11": (192, 49, 111)}
    for key, value in color_bgr.items():
        cpanel.append(Rect(0, int(key)*60, 60, 60, value, alpha=0.1))
    return cpanel


def pen_panel_rect():
    ppanel = list()
    for index in range(7):
        ppanel.append(Rect(1215, 20 + index * 60, 60, 60, (0, 0, 0), alpha=0.5))
    return ppanel


def pen_size_rect():
    psize = list()
    for index, size in enumerate(range(5, 25, 5)):
        psize.append(Rect(1215, 450+index*60, 60, 60, (0, 0, 0), str(size), alpha=0.3))
    return psize


def main_func():

    color_panel = color_panel_rect()
    pen_panel = pen_panel_rect()
    pen_size = pen_size_rect()

    pen_images = {'0': 'images/clear.png', '1': 'images/eraser.png', '2': 'images/drawing.png',
                  '3': 'images/line.png', '4': 'images/rectangle.png', '5': 'images/circle.png',
                  '6': 'images/ellipse.png'}

    hands = HandLandmarker()
    camera_live = cv2.VideoCapture(0)
    camera_live.set(3, 1280)
    camera_live.set(4, 720)
    cv2.namedWindow('Virtual Paint', cv2.WINDOW_NORMAL)


    while camera_live.isOpened():
        read, frame = camera_live.read()
        if not read:
            continue
        frame = cv2.flip(frame, 1)
        hands.detect_async(frame)
        frame = draw_marker(frame, hands.result)

        for index in range(len(color_panel)):
            color_panel[index].draw_rect(frame)

        for index in range(len(pen_size)):
            pen_size[index].draw_rect(frame)

        for key, value in pen_images.items():
            image = cv2.imread(value, cv2.IMREAD_UNCHANGED)
            frame = pen_panel[int(key)].add_image(frame, image)

        cv2.imshow('Virtual Paint', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    camera_live.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_func()