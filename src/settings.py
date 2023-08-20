from math import pow
MEDIAPIPE_MODEL_PATH = "./src/engagmentScore/models/face_landmarker.task"
ANGLE_SCORE = lambda x: -1/5*pow(x, 2)+2
DEPTH_CHANGE_SCORE = lambda x: -x/3000
DISCOUNT_FACTOR = 0.5