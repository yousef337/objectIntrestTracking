#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan, Image
from lasr_perception_server.srv import DetectImage
import numpy as np
import cv2
import mediapipe as mp
from fer import FER
from engagementScore.srv import engagementScore, engagementScoreRequest, engagementScoreResponse, imgLstConverter, imgLstConverterRequest
from PIL import Image as PILImage
from cv_bridge import CvBridge
from math import sqrt, asin

def getData():
    # get image 
    img_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)

    # get laser reading
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


def angleScore(angle):
    return angle*-1/5 + 2

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


    # calculate face/eyes direction -> img

    image2D = CvBridge().imgmsg_to_cv2(img)
    EncodedImage2D = cv2.cvtColor(image2D, cv2.COLOR_BGR2RGB) # no need since _pnp will do it, here because will unuse _pnp
    EncodedImage2D.flags.writeable = False

    # new_image = PILImage.fromarray(EncodedImage2D)
    # new_image.save(f'newMTr.png')

    for i in scores:
        # detect face orientation
        r = EncodedImage2D[max(0, scores[i]["xywh"][1]):max(0, scores[i]["xywh"][1])+scores[i]["xywh"][3]+1, max(0, scores[i]["xywh"][0]):max(0, scores[i]["xywh"][0])+scores[i]["xywh"][2]+1]
        r = r.copy(order='c')
        # new_image = PILImage.fromarray(r)
        # new_image.save(f'ddq{i}.png')

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="./src/engagmentScore/models/face_landmarker.task"),
            running_mode=VisionRunningMode.IMAGE)
            
        with FaceLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=r)
            face_landmarker_result = landmarker.detect(mp_image)

            if len(face_landmarker_result.face_landmarks):
                yaw, pitch, roll = yawPitchRoll(face_landmarker_result)
                scores[i]['score'] += angleScore(yaw)
                scores[i]['score'] += angleScore(pitch)



        # emotions -> img

        detector = FER()
        emotion = detector.top_emotion(r)
        if emotion:
            scores[i]['score'] += emotionScore(emotion[0])

    # print(scores)
    # print(scores[max(scores, key=lambda x: scores[x]['score'])]['xywh'])
    res = engagementScoreResponse()
    
    if len(scores) != 0:
        res.dimensions = list(scores[max(scores, key=lambda x: scores[x]['score'])]['xywh'])
    
    # distraction detection -> img & laser

    return res



rospy.init_node("objectEngagementTracking")
rospy.Service('engagementScore', engagementScore, locateEngagedObjects)
rospy.spin()