import cv2
import numpy as np


class Rect():
    def __init__(self, x, y, width, height, color, text='', alpha=0.5):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.text = text
        self.alpha = alpha

    def draw_rect(self, img, text_color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2):
        rec_roi = img[self.y: self.y + self.height, self.x: self.x + self.width]
        w_rect = np.ones(rec_roi.shape, dtype=np.uint8)
        w_rect[:] = self.color
        res = cv2.addWeighted(rec_roi, self.alpha, w_rect, 1 - self.alpha, 1.0)
        img[self.y: self.y + self.height, self.x: self.x + self.width] = res

        text_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.width / 2 - text_size[0][0] / 2),
                    int(self.y + self.height / 2 + text_size[0][1] / 2))
        cv2.putText(img, self.text, text_pos, fontFace, fontScale, text_color, thickness)

    def add_image(self, img, img2):
        rec_roi = img[self.y: self.y + self.height, self.x: self.x + self.width]
        alpha = img2[:, :, -1]
        img_bgr = img2[:, :, :-1]
        rec_roi[alpha == 255] = img_bgr[alpha == 255]
        img[self.y: self.y + self.height, self.x: self.x + self.width] = rec_roi
        return img

    def is_over(self, x, y):
        if (self.x + self.width > x > self.x) and (self.y + self.height > y > self.y):
            return True
        return False
