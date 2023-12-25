
from landmarker import *
from rectangle import *


def color_panel_rect():
    cpanel = list()
    color_bgr = {"0": (255, 255, 255), "1": (91, 183, 230), "2": (22, 45, 93), "3": (35, 51, 235), "4": (51, 134, 240),
                 "5": (85, 254, 255), "6": (76, 250, 117), "7": (61, 205, 105), "8": (253, 252, 116), "9": (245, 30, 0),
                 "10": (247, 61, 234), "11": (192, 49, 111)}
    for key, value in color_bgr.items():
        cpanel.append(Rect(0, int(key)*60, 60, 60, value, alpha=0.1))
    return cpanel


def pen_panel_rect():
    ppanel = list()
    for index in range(3):
        ppanel.append(Rect(1215, 20 + index * 60, 60, 60, (255, 255, 255), alpha=0.5))
    return ppanel


def pen_size_rect():
    psize = list()
    for index, size in enumerate(range(5, 35, 5)):
        psize.append(Rect(1215, 210+index*60, 60, 60, (0, 0, 0), str(size), alpha=0.3))
    return psize


def main_func():

    color_panel = color_panel_rect()
    pen_panel = pen_panel_rect()
    pen_size = pen_size_rect()

    pen_images = {'0': 'images/clear.png', '1': 'images/eraser.png', '2': 'images/drawing.png'}

    pen_types = {'0': 'clear', '1': 'eraser', '2': 'brush'}


    hands = HandLandmarker()
    camera_live = cv2.VideoCapture(0)
    camera_live.set(3, 1280)
    camera_live.set(4, 720)
    cv2.namedWindow('Virtual Paint', cv2.WINDOW_NORMAL)

    color = (0, 255, 0)
    pensize = 5
    drawing = False
    canvas = None
    pen_type = 'brush'

    while camera_live.isOpened():
        read, frame = camera_live.read()
        if not read:
            continue
        frame = cv2.flip(frame, 1)
        hands.detect_async(frame)
        frame = draw_marker(frame, hands.result)

        left_position, right_position = positions(frame, hands.result)
        finger_statue = up_fingers(hands.result)

        for index in range(len(color_panel)):
            color_panel[index].draw_rect(frame)
            if isinstance(finger_statue, dict) and 'LEFT_PINKY' in finger_statue and 'RIGHT_PINKY' in finger_statue:
                if finger_statue['LEFT_PINKY'] and color_panel[index].is_over(left_position[20][0],
                                                                              left_position[20][1]):
                    color = color_panel[index].color
                elif finger_statue['RIGHT_PINKY'] and color_panel[index].is_over(right_position[20][0],
                                                                                 right_position[20][1]):
                    color = color_panel[index].color

        for index in range(len(pen_size)):
            pen_size[index].draw_rect(frame)
            if isinstance(finger_statue, dict) and 'LEFT_PINKY' in finger_statue and 'RIGHT_PINKY' in finger_statue:
                if finger_statue['LEFT_PINKY'] and pen_size[index].is_over(left_position[20][0], left_position[20][1]):
                    pensize = int(pen_size[index].text)
                elif finger_statue['RIGHT_PINKY'] and pen_size[index].is_over(right_position[20][0],
                                                                              right_position[20][1]):
                    pensize = int(pen_size[index].text)

        for key, value in pen_images.items():
            image = cv2.imread(value, cv2.IMREAD_UNCHANGED)
            frame = pen_panel[int(key)].add_image(frame, image)

        for key, value in pen_types.items():
            if isinstance(finger_statue, dict) and 'LEFT_PINKY' in finger_statue and 'RIGHT_PINKY' in finger_statue:
                if int(key) == 0:
                    if finger_statue['LEFT_PINKY'] and pen_panel[int(key)].is_over(left_position[20][0],
                                                                                   left_position[20][1]):
                        canvas = np.zeros_like(frame)
                        drawing = False
                    elif finger_statue['RIGHT_PINKY'] and pen_panel[int(key)].is_over(right_position[20][0],
                                                                                      right_position[20][1]):
                        canvas = np.zeros_like(frame)
                        drawing = False
                elif int(key) > 0:
                    if finger_statue['LEFT_PINKY'] and pen_panel[int(key)].is_over(left_position[20][0],
                                                                                   left_position[20][1]):
                        pen_type = value
                    elif finger_statue['RIGHT_PINKY'] and pen_panel[int(key)].is_over(right_position[20][0],
                                                                                      right_position[20][1]):
                        pen_type = value

        if drawing:
            if canvas is None:
                canvas = np.zeros_like(frame)
            if isinstance(finger_statue, dict) and 'LEFT_INDEX' in finger_statue and 'RIGHT_INDEX' in finger_statue:
                if finger_statue['LEFT_INDEX'] and pen_type == 'brush':
                    cv2.circle(canvas, left_position[8], pensize, color, cv2.FILLED)
                elif finger_statue['RIGHT_INDEX'] and pen_type == 'brush':
                    cv2.circle(canvas, right_position[8], pensize, color, cv2.FILLED)
                elif finger_statue['LEFT_INDEX'] and pen_type == 'eraser':
                    cv2.circle(canvas, left_position[8], pensize, (0, 0, 0), cv2.FILLED)
                elif finger_statue['RIGHT_INDEX'] and pen_type == 'eraser':
                    cv2.circle(canvas, right_position[8], pensize, (0, 0, 0), cv2.FILLED)

        if canvas is not None:
            frame = cv2.add(frame, canvas)

        cv2.imshow('Virtual Paint', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == 100:
            drawing = not drawing

    camera_live.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_func()
