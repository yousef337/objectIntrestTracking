# Description
This repo aims to provide a solution to who the robot should engage with in a human-like social environment in ROS. The package currently works only in Noetic. Due to mediapipe requirements of python > 3.8, the package won't run in melodic.

The current process takes an image frame, then detect people on the frame, then separate them to apply the scoring pipeline. The pipeline is outlines below on the scoring section. It will always output a person, even if the person has a low engagement score. The provided angle need not to be in degrees or radians, as it will output a result that has the same unit as provided. The weights for accounting past scores decreases exponentially from a base of 0.5.

# Required packages
### Face similarity (https://github.com/yousef337/face_comparison)

### A package that detects recognize objects from images

#

# Services
## engagementScore
## Input:

string id: a uuid for the episode, as the robot store the previous information to aid itself in future scoring with the same requester 

float32 viewAngle: The angle that the robot can observe

## Output:
string id

int32[] dimensions: A list that has the frame part of the person to engage with [xywh]

float32 score: The score obtained for this person

float32 depth: The depth of the perceived center of the person

float32 angle: The angle that the robot needs to move in order to align itself with the person


### Scoring

The current scoring takes into account emotions, head position, availability of distractions, previous scoring, and change in depth. It will be extended to include more criteria.

# References

A. M. Al-Nuimi and G. J. Mohammed, "Face Direction Estimation based on Mediapipe Landmarks," 2021 7th International Conference on Contemporary Information Technology and Mathematics (ICCITM), Mosul, Iraq, 2021, pp. 185-190, doi: 10.1109/ICCITM53167.2021.9677878.


#
## Tested only in simulation, yet to be tested in a robot
