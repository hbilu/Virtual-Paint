import time
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np


class HandLandmarker():
    def __init__(self):
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.createLandmarker()

    def createLandmarker(self):
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_img: mp.Image, timestamp_ms: int):
            self.result = result

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.8,
            min_hand_presence_confidence=0.8,
            min_tracking_confidence=0.8,
            result_callback=update_result)
        self.landmarker = self.landmarker.create_from_options(options)

    def detect_async(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(mp_image, int(time.time() * 1000))

    def close(self):
        self.landmarker.close()


def draw_marker(rgb_img, result: mp.tasks.vision.HandLandmarkerResult):
    if not hasattr(result, 'hand_landmarks'):
        # print('result.hand_landmarks == []')
        return rgb_img
    else:
        hand_landmark_list = result.hand_landmarks
        if(hand_landmark_list == []):
            return rgb_img
        # print(hand_landmark_list)
        output_img = np.copy(rgb_img)
        for index in range(len(hand_landmark_list)):
            hand_landmarks = hand_landmark_list[index]
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in hand_landmarks])
            mp.solutions.drawing_utils.draw_landmarks(
                output_img,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())
        return output_img
