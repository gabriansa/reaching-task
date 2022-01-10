# Reaching Task

This repository is based on [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym) which is a physics-based simulation framework for physical human-robot interaction and robotic assistance. 

This code accompanies the submission:  
["A Multi-Agent Simulation Environment for Human-Robot Collaboration in an Industrial Setting"]()  
Gabriele Ansaldo


## Reaching Task - Overview
The Reaching Task consists of two sub-tasks, a **deterministic** and subsequently a **stochastic** one. The first sub-task is called deterministic since the robot’s joint angles and movements are predetermined and hardcoded. The second sub-task is referred to as stochastic since the robot’s joint positions are not predetermined but are the result of a Reinforcement Learning policy learned in simulation using Assistive Gym. For the Reaching Task, a blue and a red marker are placed on a shelf out of reach from a human operator. The position of the markers is constant for all simulations. Firstly, in the deterministic sub-task, the human points at the desired marker, and through computer vision (ZED camera), the robot understands which of the markers was chosen. The chosen marker is then grabbed by the robot using predetermined joint positions. Secondly, the stochastic sub-task consists of the robot passing the marker to the human’s right hand. This sub-task is solely based on the environment designed in Assistive Gym which allows the robot to learn to reach a human’s hand.  
  
An example of the designed task can be seen in the following video: (add video of also ZED)  
  
<img src="https://github.com/gansaldo/reaching-task/blob/main/images/RT-Overview.gif" width="450"> <img src="https://github.com/gansaldo/reaching-task/blob/main/images/RT-Overview-ZED.gif" width="450">

***
## Repository Overview
The repository is composed of two main folders. One folder contains the files created for **designing** the simulation environment, and the other contains files related to **transferring** a learned policy onto a real [Sawyer Robot](https://www.rethinkrobotics.com/sawyer) with the help of a [ZED 2](https://www.stereolabs.com/zed-2/) stereo camera.  
  
Here is a side by side picture of the real environment and the designed environment.  
  
<img src="https://github.com/gansaldo/reaching-task/blob/main/images/real-env.jpg" width="300"> <img src="https://github.com/gansaldo/reaching-task/blob/main/images/sim-env.jpg" width="300">

#### Trained model in simulation
The designed environment was trained in simulation for approximately 10,000,000 timesteps. The following video shows how the trained model performs.  
  
<img src="https://github.com/gansaldo/reaching-task/blob/main/images/sim-trained-model.gif" width="600">


#### Trained model on real Sawyer robot 
The policy learned in simulation was then transfered to the real environment. The following video shows how the trained model performs in real life.  
![Trained model on real Sawyer robot](images/real-trained-model.gif "Trained model on real Sawyer robot") (add video of robot just moving to predetermined point)

# Installation Guide
## Desing of Environment (Folder 1)
For the design of the environment [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym) was utilized. For details on how to install Assistive Gym please check out the [installation guide for Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym/wiki/1.-Install).  
Once Assistive Gym is installed replace/add the following files and folders with the ones present in Folder 1 ([Designed Environment](https://github.com/gansaldo/reaching-task/tree/main/Designed%20Environment)):  

Replace/add the following files at given location:
File Name           | Location                                      | Actions
-------------       | -------------                                 | -------------
`config.ini`        | `/assistive-gym/assistive_gym/`               | Replace
`__init__.py`       | `/assistive-gym/assistive_gym/`               | Replace
`__init__ 2.py`     | `/assistive-gym/assistive_gym/envs/`          | Rename to `__init__.py` and Replace
`reaching.py`       | `/assistive-gym/assistive_gym/envs/`          | Add
`reaching_envs.py`  | `/assistive-gym/assistive_gym/envs/`          | Add
`furniture.py`      | `/assistive-gym/assistive_gym/envs/agents/`   | Replace
`sawyer.py`         | `/assistive-gym/assistive_gym/envs/agents/`   | Replace
`tool.py`           | `/assistive-gym/assistive_gym/envs/agents/`   | Replace

Add the following folders at given location:
Folder Name             | Location                                      | Actions
-------------           | -------------                                 | -------------
`ZED_camera`            | `/assistive-gym/assistive_gym/envs/assets/`   | Add
`lab_table`             | `/assistive-gym/assistive_gym/envs/assets/`   | Add
`lab_shelf`             | `/assistive-gym/assistive_gym/envs/assets/`   | Add
`lab_marker`            | `/assistive-gym/assistive_gym/envs/assets/`   | Add
`ReachingSawyer-v1`     | `/assistive-gym/trained_models/ppo/`          | Add


It will now be possible to render the designed environment in simuation. The below command will render a Sawyer robot taking random actions within the Reaching Task  with a static person.  
```bash 
python3 -m assistive_gym --env "ReachingSawyer-v1"
```  

In order to render a single rollout of the trained policy use the following command:
```bash 
python3 -m assistive_gym.learn --env "ReachingSawyer-v1" --algo ppo --render --seed 0 --load-policy-path ./trained_models/ --render-episodes 10
```  


## Implementation of Environment (Folder 2)
Refer to the [ZED 2 installation guide](https://www.stereolabs.com/docs/installation/linux/) and [Sawyer SDK installation guide](https://sdk.rethinkrobotics.com/intera/Main_Page) to set up both the camera and robot.  
The second folder ([Simulation to Real](https://github.com/gansaldo/reaching-task/tree/main/Simulation%20to%20Real)) contains two main codes:
- `ReachingTask.py` is the main code used to run the full task on the Sawyer and ZED camera
- `coordinate_system_calibration.py` is utilized to calibrate the ZED coordinate system with the Sawyer reference frame  
Move the entire Folder 2 to `/assistive-gym/`.
***
### `ReachingTask.py`
This code contains both the **deterministic** and **stochastic** sub-tasks. It is necessary to use both the Sawyer robot and ZED 2 camera in order to run this code. However, if different hardware is used, this code may be used as a template for further modification.  
  
### `coordinate_system_calibration.py`
A crucial step in assuring proper functioning of the `ReachingTask.py` code is to perform an accurate calibration of the ZED and Sawyer coordinate systems. This code is used to perform calibration. The following video shows how calibration is performed: (screenrecord calibration with two videos: one is screen one is robot)
  
<img src="https://github.com/gansaldo/reaching-task/blob/main/images/sim-trained-model.gif" width="600">
