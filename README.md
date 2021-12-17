# Reaching Task

This repository is based on [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym) which is a physics-based simulation framework for physical human-robot interaction and robotic assistance. 

This code accompanies the submission:\
"A Multi-Agent Simulation Environment for Human-Robot Collaboration in an Industrial Setting"\
Gabriele Ansaldo

The repository is composed of two main folders. One folder contains the files created for **designing** the simulation environment, and the other contains files related to **transferring** a learned policy onto a real [Sawyer Robot](https://www.rethinkrobotics.com/sawyer) with the help of a [ZED 2](https://www.stereolabs.com/zed-2/) stereo camera.

<img src="https://github.com/gansaldo/reaching-task/blob/main/images/real-env.jpg" width="300"> <img src="https://github.com/gansaldo/reaching-task/blob/main/images/sim-env.jpg" width="300">

#### Trained model in simulation
<img src="https://github.com/gansaldo/reaching-task/blob/main/images/sim-trained-model.gif" width="400">


#### Trained model on real Sawyer robot 
![Trained model on real Sawyer robot](images/real-trained-model.gif "Trained model on real Sawyer robot")

### Desing of Environment (Folder 1)
For the design of the environment [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym) was utilized. For details on how to install Assistive Gym please check out the [installation guide for Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym/wiki/1.-Install).\
Once Assistive Gym is installed replace/add the following files and folders with the ones present in Folder 1:\

Replace the following files:
File Name     | Location
------------- | -------------
config.ini    | /assistive-gym/assistive_gym/config.ini
__init__.py   | /assistive-gym/assistive_gym/envs/__ init __.py
reaching.py   | /assistive-gym/assistive_gym/envs/reaching.py
furniture.py  | /assistive-gym/assistive_gym/envs/agents/furniture.py
sawyer.py     | /assistive-gym/assistive_gym/envs/agents/sawyer.py
tool.py       | /assistive-gym/assistive_gym/envs/agents/tool.py

Add the following folders:
Folder Name   | Location
------------- | -------------
ZED_camera    | /assistive-gym/assistive_gym/envs/assets/
lab_table     | /assistive-gym/assistive_gym/envs/assets/
lab_shelf     | /assistive-gym/assistive_gym/envs/assets/
lab_marker    | /assistive-gym/assistive_gym/envs/assets/

(add trained model)
It will now be possible
### Implementation of Environment (Folder 2)
Implementing uses a ZED camera and Sawyer Robot the code is ... but it is crucial to calibrate the coordinate systems then use the code
