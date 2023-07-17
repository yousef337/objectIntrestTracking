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

def getData():
    # get image 
    img_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)

    # get laser reading
    laser_scan = rospy.wait_for_message('/scan_raw', LaserScan)

    return img_msg, laser_scan

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

    rospy.wait_for_service('ImgLstConverter')
    imgConverterRes = rospy.ServiceProxy('ImgLstConverter', imgLstConverter)(img)
    image2D = np.array(imgConverterRes.data, dtype="uint8").reshape(imgConverterRes.dimensions) #bgr8
    EncodedImage2D = cv2.cvtColor(image2D, cv2.COLOR_BGR2RGB)
    EncodedImage2D.flags.writeable = False

    new_image = PILImage.fromarray(EncodedImage2D)
    new_image.save(f'newMT.png')

    for i in scores:
        # detect face orientation
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        r = EncodedImage2D[max(0, scores[i]["xywh"][1]):max(0, scores[i]["xywh"][1])+scores[i]["xywh"][3]+1, max(0, scores[i]["xywh"][0]):max(0, scores[i]["xywh"][0])+scores[i]["xywh"][2]+1]

        new_image = PILImage.fromarray(r)
        new_image.save(f'newMT{i}.png')
        continue
        results = face_mesh.process(r)


        img_h, img_w, img_c = EncodedImage2D.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       

                    # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

                    # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)


            # The camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]]) 
            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)
            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x = angles[0] * 360
            y = angles[1] * 360

            print(x)
            print(y)



            # emotions -> img

            detector = FER()
            emotion = detector.top_emotion(EncodedImage2D)

            if emotion == "angry":
                pass


    # distraction detection -> img & laser


rospy.init_node("objectEngagementTracking")
rospy.Service('engagementScore', engagementScore, locateEngagedObjects)
rospy.spin()