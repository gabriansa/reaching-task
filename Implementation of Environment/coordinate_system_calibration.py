########################################################################
#
# Author:   Gabriele Ansaldo
# Date:     28 Dec 2021  
#
########################################################################

"""
The following code is used to perform calibration between the Sawyer and ZED coordinate system.
When running the code make sure the ZED camera is connected and the Sawyer turned on and connected 
to the computer.

Steps to perform calibration using this code:
1. Press 'n' to go to retrieve the current frame of the camera.
2. Left click at the center of the end effector to get a sample point
3. Press 'c' to confirm the selected point
4. Repeat step 1 to 3 for 4 times
5. The output will be a rotation matrix R and translation matrix T
6. Copy and paste R and T on line 302 and 306 of ReachingTask.py
"""

import cv2
import numpy as np
from time import sleep

# Packages for controlling the camera (ZED 2)
import pyzed.sl as sl

# Packages for controlling the robot (Sawyer)
import rospy
import intera_interface


def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img,(x,y),10,(255,0,0),-1)
        mouseX,mouseY = x,y

        # Show point cloud clicked (this does not save it)
        # Get ZED cloud point
        point_cloud = sl.Mat()
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        # Get the 3D point cloud values for pixel
        # point3D = point_cloud.get_value(mouseX,mouseY)
        # x = point3D[0][0]
        # y = point3D[0][1]
        # z = point3D[0][2]
        # print('Just Clicked Pixel: ', mouseX, mouseY)
        # print('Just Clicked Point Cloud: ', x, y, z)

def SAWYER_init():
    global limb
    # Initialize ROS node, registering it with the Master
    rospy.init_node('Calibration')
    # Create instances of limb
    limb = intera_interface.Limb('right')
    # Return to neutral joint positions
    limb.move_to_neutral(speed=0.2)

def ZED_init():
    global zed
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    # init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_DOWN

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

def start_calibration():
    global robot_points, camera_points, img
    # Retrieve ZED image
    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # A new image is available if grab() returns SUCCESS
        zed.retrieve_image(image, sl.VIEW.LEFT) # Retrieve the left image
        img = image.get_data()
    
    # Create window with image and start the callback function
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    count = 0
    camera_points = []
    robot_points = []
    while count < 4:
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

        # Press 'c' to confirm point
        elif k == ord('c'):
            count += 1
            # Get ZED cloud point
            point_cloud = sl.Mat()
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            # Get the 3D point cloud values for pixel
            point3D = point_cloud.get_value(mouseX,mouseY)
            x = point3D[1][0]
            y = point3D[1][1]
            z = point3D[1][2]
            if np.isnan(np.sum([x,y,z])):
                print("******** WARNING: Clicked pixel data not available! ********")
            else:
                camera_points.append([x, y, z])

            # Get Sawyer point
            robot_points.append(list(limb.endpoint_pose()["position"]))


        # Press 'n' to go to the next image
        elif k == ord('n'):
            # Get new image (Change to ZED snapshot)
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # A new image is available if grab() returns SUCCESS
                zed.retrieve_image(image, sl.VIEW.LEFT) # Retrieve the left image
                img = image.get_data()
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',draw_circle)

def get_calibration_matrix():
    x1_R = robot_points[0]
    x1_C = camera_points[0]

    x2_R = robot_points[1]
    x2_C = camera_points[1]

    x3_R = robot_points[2]
    x3_C = camera_points[2]

    x4_R = robot_points[3]
    x4_C = camera_points[3]

    print('Robot', x1_R, x2_R, x3_R, x4_R)
    print('Camera',x1_C, x2_C, x3_C, x4_C)


    A = np.array([

                [x1_C[0], x1_C[1], x1_C[2], 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, x1_C[0], x1_C[1], x1_C[2], 0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 0, x1_C[0], x1_C[1], x1_C[2], 0, 0, 1],

                [x2_C[0], x2_C[1], x2_C[2], 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, x2_C[0], x2_C[1], x2_C[2], 0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 0, x2_C[0], x2_C[1], x2_C[2], 0, 0, 1],

                [x3_C[0], x3_C[1], x3_C[2], 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, x3_C[0], x3_C[1], x3_C[2], 0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 0, x3_C[0], x3_C[1], x3_C[2], 0, 0, 1],

                [x4_C[0], x4_C[1], x4_C[2], 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, x4_C[0], x4_C[1], x4_C[2], 0, 0, 0, 0, 1, 0], 
                [0, 0, 0, 0, 0, 0, x4_C[0], x4_C[1], x4_C[2], 0, 0, 1],
                
                ])

    b = np.array(x1_R + x2_R + x3_R + x4_R)

    x = np.linalg.solve(A, b)

    R = x[0:9].reshape(3,3)
    T = x[9:]

    return R, T


if __name__ == '__main__':
    # Initialize Sawyer and ZED
    SAWYER_init()
    ZED_init()

    start_calibration()

    R, T = get_calibration_matrix()

    # Check if matrices are correct
    PART1 = R@camera_points[0]
    PART2 = np.eye(3)@T

    x1_R_CHECK = np.add(PART1, PART2)
    print('Correct Solution: ', robot_points[0])
    print('Computed Solution: ', x1_R_CHECK)

    print('Robot Points: ', robot_points)
    print('Camera Points: ', camera_points)

    # Print matrices
    print('')
    print('#####################################')
    print('R: ', R)
    print('')
    print('T: ', T)
    print('#####################################')