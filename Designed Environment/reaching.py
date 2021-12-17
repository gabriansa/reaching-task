import numpy as np
import pybullet as p
from .agents.furniture import Furniture
from assistive_gym.envs.env import AssistiveEnv
import random

class ReachingEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(ReachingEnv, self).__init__(robot=robot, human=human, task='reaching', obs_robot_len=16, obs_human_len=20)

    def step(self, action):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])

        # Execute the action. Step the simulator forward 
        # Avoid jerky movement (suppress small joint angles)
        action_threshold = 0.2
        i = 0
        for a in action:
            if abs(a) < action_threshold:
                a = 0
            action[i] = a
            i += 1
        self.take_step(action)
        obs = self._get_obs()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.right_end_effector))
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=total_force_on_human)

        # Define our reward function
        end_effector_pos, end_effector_orient = self.robot.get_pos_orient(self.robot.right_end_effector)
        reward_distance_target = -np.linalg.norm(self.target_pos - end_effector_pos) # Penalize robot for distance between the end effector and human hand.
        reward_action = -np.linalg.norm(action) # Penalize actions
        # reward = self.config('distance_weight')*reward_distance_target + self.config('action_weight')*reward_action + preferences_score
        reward = 1*reward_distance_target + 0.007*reward_action + preferences_score
        info = {'total_force_on_human': total_force_on_human}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}
    
    def _get_obs(self, agent=None):
        end_effector_pos, end_effector_orient = self.robot.get_pos_orient(self.robot.right_end_effector)
        # Convert pos/orient from global coordinates to robot-centric coordinate frame
        end_effector_pos_real, end_effector_orient_real = self.robot.convert_to_realworld(end_effector_pos, end_effector_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi

        hand_pos, hand_orient = self.human.get_pos_orient(self.human.right_arm_joints[9])
        hand_pos_real, hand_orient_real = self.robot.convert_to_realworld(hand_pos, hand_orient)
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)
        # Define the robot observation and add Domain Randomization
        noise_level = 0.05
        end_effector_pos_real = end_effector_pos_real + np.random.uniform(low=-noise_level, high=noise_level , size=len(end_effector_pos_real))
        target_pos_real = target_pos_real + np.random.uniform(low=-noise_level, high=noise_level , size=len(target_pos_real))
        robot_joint_angles = robot_joint_angles + np.random.uniform(low=-noise_level, high=noise_level , size=len(robot_joint_angles))
        hand_pos_real = hand_pos_real + np.random.uniform(low=-noise_level, high=noise_level , size=len(hand_pos_real))
            
        robot_obs = np.concatenate([end_effector_pos_real, end_effector_pos_real - target_pos_real, robot_joint_angles, hand_pos_real]).ravel()

        return robot_obs
        
    def reset(self):
        super(ReachingEnv, self).reset()
        self.build_assistive_env()

        # Table and shelf dimensions
        table_width = 0.455     # meters
        table_length = 1.825    # meters
        shelf_width = 0.910     # meters
        shelf_length = 0.353    # meters

        # Create robot instance and set its base position and initial joint position
        robot_base_pos, robot_base_orient = self.robot.get_base_pos_orient()
        self.robot.set_base_pos_orient([-table_width/2-0.170,0, 0.985], robot_base_orient)
        #robot_joint_angles = [0.7433154296875, -0.045564453125, -0.511083984375, -0.1053876953125, 1.02415625, 0.388298828125, 3.311984375]
        robot_joint_angles = self.generate_robot_joint_angles()
        self.robot.set_joint_angles(self.robot.controllable_joint_indices, robot_joint_angles)

        # Create table instance and set its base position
        self.table = Furniture()
        self.table.init('lab_table', self.directory, self.id, self.np_random)
        self.table.set_base_pos_orient([0,0,0], [0, 0, np.pi])

        # Create shelf instance and set its base position
        self.shelf = Furniture()
        self.shelf.init('lab_shelf', self.directory, self.id, self.np_random)
        self.shelf.set_base_pos_orient([-shelf_width/2+table_width/2+0.038,table_length/2 + shelf_length/2,0], [0, 0, np.pi/2])

        # Create human instance and set its base position and initial joint position
        human_base_pos, human_base_orient = self.generate_human_base_pos_orient(table_width, table_length)
        self.human.set_base_pos_orient(human_base_pos, human_base_orient)

        joints_positions = self.generate_human_joints_positions()
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        # Set Camera Angle
        p.resetDebugVisualizerCamera(cameraDistance=2.50, cameraYaw=60, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        # Set Generate Target
        self.generate_target()

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True, mesh_scale=[0.001]*3)

        target_ee_pos = np.array([-0.15, -0.65, 1.15]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        # self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient), (self.target_pos, None)], [(self.target_pos, target_ee_orient)], arm='right', tools=[self.tool], collision_objects=[self.human])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        # Final things
        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()

        return self._get_obs()

    def generate_target(self):
        # Set target on hand
        self.hand_pos = [0, 0, -0.08]
        wrist_pos, wrist_orient = self.human.get_pos_orient(self.human.right_arm_joints[9])
        target_pos, target_orient = p.multiplyTransforms(wrist_pos, wrist_orient, self.hand_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, collision=False, rgba=[0, 1, 0, 1])
        self.update_targets()

    def update_targets(self):
        # Update target position on hand
        wrist_pos, wrist_orient = self.human.get_pos_orient(self.human.right_arm_joints[9])
        target_pos, target_orient = p.multiplyTransforms(wrist_pos, wrist_orient, self.hand_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])

    # Extra functions
    def generate_human_base_pos_orient(self, table_width=None, table_length=None):
        a = 0.2
        b = 0.7
        c = 0.3

        p1 = [table_width/2 + c, -table_length/2-a]
        p3 = [-table_width/2 - c, -table_length/2-b-a]

        done = False
        while not done:
            myP = [random.uniform(p3[0], p1[0]), random.uniform(p3[1], p1[1])]
            if (myP[0] > p3[0] and myP[0] < p1[0] and myP[1] > p3[1] and myP[1] < p1[1]):
                done = True

        human_base_orient = [0, 0, random.uniform(-np.pi/3 - np.pi/2, -2*np.pi/3 - np.pi/2)]
        human_base_pos, _ = self.human.get_base_pos_orient()
        human_base_pos = [myP[0], myP[1], human_base_pos[2]+0.3]

        return human_base_pos, human_base_orient

    def generate_human_joints_positions(self):
        j_right_elbow_angle = random.randint(-50, -20)
        j_right_shoulder_angle = random.randint(-70, -20)
        j_left_elbow_angle = random.randint(-10, 0)
        joints_positions = [(self.human.j_right_elbow, j_right_elbow_angle), (self.human.j_right_shoulder_y, j_right_shoulder_angle), (self.human.j_left_elbow, j_left_elbow_angle), (self.human.j_right_hip_x, 0),
                            (self.human.j_right_knee, 0), (self.human.j_left_hip_x, 0), (self.human.j_left_knee, 0)]
        return joints_positions

    def generate_robot_joint_angles(self):
        # Alternate between deterministic marker locations
        if random.randint(1,100) < 50:
            robot_joint_angles_red_marker = [0.7928525390625, -0.489884765625, -0.209728515625,  0.934181640625, 2.3889716796875,  0.702125,  4.1994560546875]
            robot_joint_angles = robot_joint_angles_red_marker
        else:
            robot_joint_angles_blue_marker = [0.8868984375, -0.4859677734375, 0.054611328125, 1.3401181640625, 2.975955078125, 0.9132578125, 3.46280859375]
            robot_joint_angles = robot_joint_angles_blue_marker
        return robot_joint_angles