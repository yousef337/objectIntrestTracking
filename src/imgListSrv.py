#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge
from engagementScore.srv import imgLstConverter, imgLstConverterResponse

def imgConvert(img):
    res = imgLstConverterResponse()
    a = CvBridge().imgmsg_to_cv2(img.img, desired_encoding='bgr8')
    res.data = a.flatten()
    res.dimensions = a.shape
    return res

rospy.init_node("temp")
rospy.Service('ImgLstConverter', imgLstConverter, imgConvert)
rospy.spin()