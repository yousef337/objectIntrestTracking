#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import mediapipe as mp
from sensor_msgs.msg import LaserScan, Image
from lasr_perception_server.srv import DetectImage
from fer import FER
from engagementScore.srv import engagementScore, engagementScoreRequest, engagementScoreResponse
from PIL import Image as PILImage
from cv_bridge import CvBridge
from math import sqrt, asin
from settings import MEDIAPIPE_MODEL_PATH, ANGLE_SCORE

def getData():
    img_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
    laser_scan = ""#rospy.wait_for_message('/scan_raw', LaserScan)

    return img_msg, laser_scan


def yawPitchRoll(faceMesh):
    mesh = faceMesh.face_landmarks[0]

    # yaw
    yaw = 0

    a = sqrt(pow(mesh[50].x - mesh[18].x, 2) + pow(mesh[50].y - mesh[18].y, 2))
    b = sqrt(pow(mesh[280].x - mesh[18].x, 2) + pow(mesh[280].y - mesh[18].y, 2))
    c = sqrt(pow(mesh[280].x - mesh[50].x, 2) + pow(mesh[280].y - mesh[50].y, 2))

    v_dist = np.linalg.norm(np.cross([mesh[280].x - mesh[50].x, mesh[280].y - mesh[50].y, mesh[280].z - mesh[50].z], [mesh[18].x - mesh[50].x, mesh[18].y - mesh[50].y, mesh[18].z - mesh[50].z]) / np.linalg.norm([mesh[280].x - mesh[50].x, mesh[280].y - mesh[50].y, mesh[280].z - mesh[50].z]))
    lR = sqrt(pow(a, 2) - pow(v_dist, 2))
    lL = sqrt(pow(b, 2) - pow(v_dist, 2))

    yaw = asin(1-lR/lL) if a < b else asin(1-lL/lR)


    a = sqrt(pow(mesh[50].x - mesh[4].x, 2) + pow(mesh[50].y - mesh[4].y, 2))
    b = sqrt(pow(mesh[280].x - mesh[4].x, 2) + pow(mesh[280].y - mesh[4].y, 2))
    
    v_dist = np.linalg.norm(np.cross([mesh[280].x - mesh[50].x, mesh[280].y - mesh[50].y, mesh[280].z - mesh[50].z], [mesh[4].x - mesh[50].x, mesh[4].y - mesh[50].y, mesh[4].z - mesh[50].z]) / np.linalg.norm([mesh[280].x - mesh[50].x, mesh[280].y - mesh[50].y, mesh[280].z - mesh[50].z]))

    # pitch

    pitchL = 0
    pitchR = 0

    if b > v_dist:
        pitchL = asin(v_dist/b)
    elif b < v_dist:
        pitchL = asin(b/v_dist)

    if a > v_dist:
        pitchR = asin(v_dist/a)
    elif a < v_dist:
        pitchR = asin(a/v_dist)

    pitch = (pitchL + pitchR) / 2

    # roll

    a = sqrt(pow(mesh[280].x - mesh[50].x, 2) + pow(mesh[280].y - mesh[50].y, 2))
    b = abs(mesh[280].y - mesh[50].y)

    roll = 0

    if b > a:
        roll = asin(a/b)
    elif b < a:
        roll = asin(b/a)

    return yaw, pitch, roll


def emotionScore(score):
    a = {
        "angry" : -5,
        "disgust": -4,
        "fear" : -3,
        "happy" : 5,
        "sad" : 1,
        "surprise" : 4,
        "neutral" : 2,
    }

    return a.get(score, 0)


def distractionScore(distraction):
    a = {
        "ipad" : -0.6,
        "tablet": -0.6,
        "phone" : -0.5,
        "mobile" : -0.5,
        "laptop" : -0.7,
        "computer": -0.75
    }

    return a.get(distraction, 0)

def angleScore(angle):
    return ANGLE_SCORE(angle)


def scoreEmotion(scores, r, i):
    detector = FER()
    emotion = detector.top_emotion(r)
    if emotion:
        scores[i]['score'] += emotionScore(emotion[0])


def scoreDistraction(scores, r, i, cvBridge, objectRecognitionService):
    r_imgmsg = cvBridge.cv2_to_imgmsg(r, encoding="passthrough")
    distractionResp = objectRecognitionService(
            [r_imgmsg], 'coco', 0.7, 0.3, ["ipad", "tablet", "phone", "mobile", "laptop", "computer"], 'yolo'
        ).detected_objects
    
    for d in distractionResp:
        scores[i]['score'] += distractionScore(d.name)

def scoreHeadPosition(scores, r, i):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE)
        
    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=r)
        face_landmarker_result = landmarker.detect(mp_image)

        if len(face_landmarker_result.face_landmarks):
            yaw, pitch, roll = yawPitchRoll(face_landmarker_result)
            scores[i]['score'] += angleScore(yaw)
            scores[i]['score'] += angleScore(pitch)


def locateEngagedObjects(req: engagementScoreRequest):

    img, laserReading = getData()
    scores = {}

    # detect ppl -> img
    rospy.wait_for_service('lasr_perception_server/detect_objects_image')
    objectRecognitionService = rospy.ServiceProxy(
        'lasr_perception_server/detect_objects_image', DetectImage
    )

    resp = objectRecognitionService(
            [img], 'coco', 0.7, 0.3, ["person"], 'yolo'
        ).detected_objects

    # assign them to score map
    c = 0
    for i in resp:            
        scores[c] = {"xywh": i.xywh, "score": 0}
        c += 1


    cvBridge = CvBridge()
    image2D = cvBridge.imgmsg_to_cv2(img)
    EncodedImage2D = cv2.cvtColor(image2D, cv2.COLOR_BGR2RGB)
    EncodedImage2D.flags.writeable = False

    # new_image = PILImage.fromarray(EncodedImage2D)
    # new_image.save(f'newMTr.png')

    for i in scores:
        r = EncodedImage2D[max(0, scores[i]["xywh"][1]):max(0, scores[i]["xywh"][1])+scores[i]["xywh"][3]+1, max(0, scores[i]["xywh"][0]):max(0, scores[i]["xywh"][0])+scores[i]["xywh"][2]+1]
        r = r.copy(order='c')

        scoreHeadPosition(scores, r, i)
        scoreEmotion(scores, r, i)
        scoreDistraction(scores, r, i, cvBridge, objectRecognitionService)


    res = engagementScoreResponse()
    
    if len(scores) != 0:
        res.dimensions = list(scores[max(scores, key=lambda x: scores[x]['score'])]['xywh'])
    
    return res



rospy.init_node("objectEngagementTracking")
rospy.Service('engagementScore', engagementScore, locateEngagedObjects)
rospy.spin()