########################################################################
#
# Author:   Gabriele Ansaldo
# Date:     28 Dec 2021  
#
########################################################################

"""
The following code is used to transfer the Reaching Task trained in simulation to an actual Sawyer robot.
When running the code make sure the ZED camera is connected and the Sawyer turned on and connected 
to the computer.
"""

# Packages for rendering policy
import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
from ray.rllib.agents import ppo, sac
from ray.tune.logger import pretty_print
from numpngw import write_apng
import pybullet as p

# Packages for controlling the robot (Sawyer)
import rospy
import intera_interface
from time import sleep

# Packages for controlling the camera (ZED 2)
import cv2
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import queue
from threading import Thread
import random


# Functions for the Reaching Task
def STOCHASTIC_task():
    global TASK_IS_NOT_DONE
    render_policy()
    TASK_IS_NOT_DONE = False
    sleep(1)
    gripper_util(mode='open')
    # Return to neutral joint positions
    sleep(1.5)
    limb.move_to_neutral(speed=0.2)
    head_display.display_image('/home/gabrigoo/assistive-gym/RA Reaching Task/SawyerMessages/Done.png', display_in_loop=False, display_rate=100)


def DETERMINISTIC_task():
    eps = 0.018
    observation_range_steady = 25
    observation_range_pointing = 35
    is_pointing = False
    while not is_pointing:
        steady_count = 0
        pointing_count = 0
        wrist_pos_real_DIFF = []
        if len(wrist_pos_real_LIST) > observation_range_pointing:
            # Check if wrist is abobe certain limit
            for i in range(1,observation_range_pointing):
                if wrist_pos_real_LIST[-i][2] > (3*neck_pos_real_LIST[-i][2] + hip_pos_real_LIST[-i][2])/4:
                    pointing_count += 1
            # Check if wrist is steady
            for i in range(1,observation_range_steady):
                dx = wrist_pos_real_LIST[-i][0] - wrist_pos_real_LIST[-i-1][0]
                dy = wrist_pos_real_LIST[-i][1] - wrist_pos_real_LIST[-i-1][1]
                dz = wrist_pos_real_LIST[-i][2] - wrist_pos_real_LIST[-i-1][2]
                wrist_pos_real_DIFF.append([dx, dy, dz])
            for i in range(1,len(wrist_pos_real_DIFF)):
                if abs(wrist_pos_real_DIFF[-1-i][0]) < eps and abs(wrist_pos_real_DIFF[-1-i][1]) < eps and abs(wrist_pos_real_DIFF[-1-i][2]) < eps:
                    steady_count += 1

            head_display.display_image('/home/gabrigoo/assistive-gym/RA Reaching Task/SawyerMessages/Point.png', display_in_loop=False, display_rate=100)
            print('')
            sleep(0.5)
            if steady_count >= observation_range_steady-6 and pointing_count == observation_range_pointing-1:
                if wrist_pos_real_LIST[-1][0] > neck_pos_real_LIST[-1][0]*7/6 and wrist_pos_real_LIST[-1][0] > shoulder_pos_real_LIST[-1][0]*8/9:
                    marker_color = 'RED'
                    is_pointing = True
                elif wrist_pos_real_LIST[-1][0] < (4*neck_pos_real_LIST[-1][0] + shoulder_pos_real_LIST[-1][0])/5 and wrist_pos_real_LIST[-1][0] < shoulder_pos_real_LIST[-1][0]:
                    marker_color = 'BLUE'
                    is_pointing = True

    if marker_color == 'RED':
        head_display.display_image('/home/gabrigoo/assistive-gym/RA Reaching Task/SawyerMessages/BLUE.png', display_in_loop=False, display_rate=100)
        # Go to object location
        position1 = {'right_j0': 0.75248046875, 'right_j1': -1.261177734375, 'right_j2': 0.4293896484375, 'right_j3': 1.91543359375, 'right_j4': 2.4016376953125, 'right_j5': 0.7269013671875, 'right_j6': 3.82627734375}
        limb.move_to_joint_positions(position1, timeout=15.0, threshold=0.008726646, test=None)
        position2 = {'right_j0': 0.92429296875, 'right_j1': -0.9459560546875, 'right_j2': 0.445171875, 'right_j3': 1.4202275390625, 'right_j4': 2.357236328125, 'right_j5': 0.4633369140625, 'right_j6': 3.82669140625}
        limb.move_to_joint_positions(position2, timeout=15.0, threshold=0.008726646, test=None)
        # Grip object
        gripper.set_ee_signal_value('grip', True)
        limb.move_to_joint_positions(position1, timeout=15.0, threshold=0.008726646, test=None)
        position3 = {'right_j0': 0.788009765625, 'right_j1': -0.358619140625, 'right_j2': -0.7064140625, 'right_j3': 1.058984375, 'right_j4': 2.5238525390625, 'right_j5': 1.16690625, 'right_j6': 4.43057421875}
        limb.move_to_joint_positions(position3, timeout=15.0, threshold=0.008726646, test=None)

    elif marker_color == 'BLUE':
        head_display.display_image('/home/gabrigoo/assistive-gym/RA Reaching Task/SawyerMessages/RED.png', display_in_loop=False, display_rate=100)
        # Go to object location
        position1 = {'right_j0': 1.1987373046875, 'right_j1': -0.9116552734375, 'right_j2': 0.4263486328125, 'right_j3': 2.164966796875, 'right_j4': 2.976380859375, 'right_j5': 1.1658857421875, 'right_j6': 3.1977861328125}
        limb.move_to_joint_positions(position1, timeout=15.0, threshold=0.008726646, test=None)
        position2 = {'right_j0': 1.25677734375, 'right_j1': -0.7172744140625, 'right_j2': 0.4356162109375, 'right_j3': 1.7069384765625, 'right_j4': 2.9736826171875, 'right_j5': 0.85280859375, 'right_j6': 3.1977861328125}
        limb.move_to_joint_positions(position2, timeout=15.0, threshold=0.008726646, test=None)
        # Grip object
        gripper.set_ee_signal_value('grip', True)
        limb.move_to_joint_positions(position1, timeout=15.0, threshold=0.008726646, test=None)
        position3 = {'right_j0': 0.788009765625, 'right_j1': -0.358619140625, 'right_j2': -0.7064140625, 'right_j3': 1.058984375, 'right_j4': 2.5238525390625, 'right_j5': 1.16690625, 'right_j6': 4.43057421875}
        limb.move_to_joint_positions(position3, timeout=15.0, threshold=0.008726646, test=None)
    

# Main functions for Sawyer
def SAWYER_init():
    # Initialize ROS node, registering it with the Master
    rospy.init_node('Passing_Task')

    # Create instances of limb, gripper, head, head_display, and lights
    global limb, gripper, head, head_display, lights
    limb = intera_interface.Limb('right')
    gripper = intera_interface.get_current_gripper_interface()
    head = intera_interface.Head()
    head_display = intera_interface.HeadDisplay()
    lights = intera_interface.Lights()

    # Turn on blue light sawyer
    lights_util(light_color='blue')

    # Start with 'hello' from sawyer screen
    head_display.display_image('/home/gabrigoo/assistive-gym/RA Reaching Task/SawyerMessages/InitSawyer.png', display_in_loop=False, display_rate=100)
    sleep(1)

    # Return to neutral joint positions
    limb.move_to_neutral(speed=0.2)

    # Open the grip
    gripper_util(mode='open')

    # Set head to neutral position
    head.set_pan(-1.2)

def lights_util(light_color):
    if light_color == 'red':
        lights.set_light_state('head_red_light', on=True)
        lights.set_light_state('head_green_light', on=False)
        lights.set_light_state('head_blue_light', on=False)
        lights.set_light_state('right_hand_blue_light', on=False)
        lights.set_light_state('right_hand_green_light', on=False)
        lights.set_light_state('right_hand_red_light', on=True)
    elif light_color == 'green':
        lights.set_light_state('head_red_light', on=False)
        lights.set_light_state('head_green_light', on=True)
        lights.set_light_state('head_blue_light', on=False)
        lights.set_light_state('right_hand_blue_light', on=False)
        lights.set_light_state('right_hand_green_light', on=True)
        lights.set_light_state('right_hand_red_light', on=False)
    elif light_color == 'blue':
        lights.set_light_state('head_red_light', on=False)
        lights.set_light_state('head_green_light', on=False)
        lights.set_light_state('head_blue_light', on=True)
        lights.set_light_state('right_hand_blue_light', on=True)
        lights.set_light_state('right_hand_green_light', on=False)
        lights.set_light_state('right_hand_red_light', on=False)

def gripper_util(mode):
    if mode == 'open':
        gripper.set_ee_signal_value('grip', False)
    elif mode == 'close':
        gripper.set_ee_signal_value('grip', True)


# Main functions for the ZED camera
def ZED_init():
    global TASK_IS_NOT_DONE
    global wrist_pos_real_LIST, elbow_pos_real_LIST, shoulder_pos_real_LIST, hip_pos_real_LIST, neck_pos_real_LIST
    TASK_IS_NOT_DONE = True
    wrist_pos_real_LIST = []
    elbow_pos_real_LIST = []
    shoulder_pos_real_LIST = []
    hip_pos_real_LIST = []
    neck_pos_real_LIST = []
    view = True
    Thread(target=get_humans_loc, args=(view,)).start()

def get_humans_loc(view):
    print("Running Body Tracking sample ... Press 'q' to quit")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    # init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_DOWN
    
    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.svo_real_time_mode = True
        init_params.set_from_svo_file(filepath)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True            # Smooth skeleton move
    obj_param.enable_tracking = True                # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_MEDIUM
    obj_param.body_format = sl.BODY_FORMAT.POSE_18  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280), min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
                 , display_resolution.height / camera_info.camera_resolution.height]


    if view == True:
        # Create OpenGL viewer
        viewer = gl.GLViewer()
        viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking,obj_param.body_format)

    # Create ZED objects filled in the main loop
    bodies = sl.Objects() # Contains all detected bodies/objects
    image = sl.Mat()

    while TASK_IS_NOT_DONE:
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:

            # Retrieve 3D location of keypoints 
            if len(bodies.object_list) == 1:
                body = sl.ObjectData() 
                for object in bodies.object_list:
                    bodies.get_object_data_from_id(body, object.id)
                wrist_pos = body.keypoint[4]
                wrist_pos_real = ZED_to_Sawyer_coordinate(wrist_pos)
                wrist_pos_real_LIST.append(wrist_pos_real)

                elbow_pos = body.keypoint[3]
                elbow_pos_real = ZED_to_Sawyer_coordinate(elbow_pos)
                elbow_pos_real_LIST.append(elbow_pos_real)

                shoulder_pos = body.keypoint[2]
                shoulder_pos_real = ZED_to_Sawyer_coordinate(shoulder_pos)
                shoulder_pos_real_LIST.append(shoulder_pos_real)

                hip_pos = body.keypoint[8]
                hip_pos_real = ZED_to_Sawyer_coordinate(hip_pos)
                hip_pos_real_LIST.append(hip_pos_real)

                neck_pos = body.keypoint[1]
                neck_pos_real = ZED_to_Sawyer_coordinate(neck_pos)
                neck_pos_real_LIST.append(neck_pos_real)


            # Retrieve objects
            zed.retrieve_objects(bodies, obj_runtime_param)

            if view == True:
                # Retrieve left image
                zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)

                # Update GL view
                viewer.update_view(image, bodies) 
                # Update OCV view
                image_left_ocv = image.get_data()
                cv_viewer.render_2D(image_left_ocv,image_scale,bodies.object_list, obj_param.enable_tracking, obj_param.body_format)
                cv2.imshow("ZED | 2D View", image_left_ocv)
                cv2.waitKey(10)

    viewer.exit()

    image.free(sl.MEM.CPU)
    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()

def ZED_to_Sawyer_coordinate(ZED_point):
    # Rotation and Translation matrix computed with coordinate_system_calibration.py
    R = [[-0.99301132, -0.01937369,  0.02107001],
         [-0.02204311, -0.0190118,  -0.98562337],
         [ 0.01580896, -0.99111898, -0.00220738]]

    T = [0.53645284, 0.93436674, 0.80664281]

    Robot_Point = np.add(R@ZED_point, np.eye(3)@T)

    x_Sawyer = Robot_Point[0]
    y_Sawyer = Robot_Point[1]
    z_Sawyer = Robot_Point[2]

    return [x_Sawyer, y_Sawyer, z_Sawyer]


# Fuctions to render policy
def RENDER_init():
    global env, env_name, policy_path, seed, test_agent
    # Load ReachingSawyer Task
    env_name = 'ReachingSawyer-v1'
    seed = 0
    policy_path = '/home/gabrigoo/assistive-gym/trained_models'

    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env()
    test_agent, _ = load_policy()   

def setup_config():
    num_processes = multiprocessing.cpu_count()
    # Utilized algorithm is PPO
    config = ppo.DEFAULT_CONFIG.copy()
    config['train_batch_size'] = 19200
    config['num_sgd_iter'] = 50
    config['sgd_minibatch_size'] = 128
    config['lambda'] = 0.95
    config['model']['fcnet_hiddens'] = [100, 100]

    config['num_workers'] = num_processes
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'

    return {**config}

def load_policy():
    agent = ppo.PPOTrainer(setup_config(), 'assistive_gym:'+env_name)
    # Find the most recent policy in the directory
    directory = os.path.join(policy_path, 'ppo', env_name)
    files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
    files_ints = [int(f) for f in files]

    checkpoint_max = max(files_ints)
    checkpoint_num = files_ints.index(checkpoint_max)
    checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
    agent.restore(checkpoint_path)
    return agent, None

def make_env():
    env = gym.make('assistive_gym:'+env_name)
    env.seed(seed)
    return env

def render_policy():
    obs, done = _get_real_obs()
    while not done:
        # Compute the next action using the trained policy
        action = test_agent.compute_action(obs)
        # Step the real robot forward using the action from our trained policy
        obs, done = _step_rob(action)
    env.disconnect()

def _get_real_obs():
    observation_range = 30
    if len(wrist_pos_real_LIST) > observation_range:
        target_pos_real = np.mean(wrist_pos_real_LIST[-observation_range:], axis=0)
    else:
        target_pos_real = wrist_pos_real_LIST[-1]
    hand_pos_real = target_pos_real 

    end_effector_pos_real = list(limb.endpoint_pose()["position"])
    end_effector_orient_real = list(limb.endpoint_pose()["orientation"])
    gripper_pos = [0,0,0.45]
    end_effector_pos_real, _ = p.multiplyTransforms(end_effector_pos_real, end_effector_orient_real, gripper_pos, [0, 0, 0, 1])

    distance_to_target = [a_i - b_i for a_i, b_i in zip(end_effector_pos_real, target_pos_real)]
    robot_joint_angles = list(limb.joint_angles().values())
    # Fix joint angles to be in [-pi, pi]
    robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
    real_obs = np.concatenate([end_effector_pos_real, distance_to_target, robot_joint_angles, hand_pos_real]).ravel()

    # Change sawyer speed depending on distance to target
    dist = np.linalg.norm(distance_to_target)
    joint_speed = min(0.2, 0.1*dist+0.08)
    limb.set_joint_position_speed(speed = joint_speed)

    # Done if target reached within given threshold
    if dist < 0.2: 
        done = True
    else:
        done = False

    return real_obs, done

def _step_rob(action):
    joint_angles = list(limb.joint_angles().values())
    positions = [x + y for (x, y) in zip(action, joint_angles)]
    joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
    positions_rob = dict(zip(joint_names, positions))
    # limb.move_to_joint_positions(positions_rob, timeout=0, threshold=0.008726646, test=None)
    limb.set_joint_positions(positions_rob)
    return _get_real_obs()


if __name__ == '__main__':
    # Policy Rendering Initialization
    RENDER_init()

    # Sawyer and ZED Initialization
    SAWYER_init()
    ZED_init()

    # Start Reaching Task
    DETERMINISTIC_task()
    STOCHASTIC_task()
