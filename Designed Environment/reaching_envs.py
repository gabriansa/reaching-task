from assistive_gym.envs.agents import pr2, baxter, sawyer, jaco, stretch, panda, human
from assistive_gym.envs.agents.sawyer import Sawyer
from assistive_gym.envs.agents.jaco import Jaco
from assistive_gym.envs.agents.human import Human
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
import assistive_gym.envs
from .reaching import ReachingEnv

robot_arm = 'right'
# human_controllable_joint_indices = human.right_arm_joints
human_controllable_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
class ReachingSawyerEnv(ReachingEnv):
    def __init__(self):
        super(ReachingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ReachingJacoEnv(ReachingEnv):
    def __init__(self):
        super(ReachingJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class ReachingSawyerHumanEnv(ReachingEnv, MultiAgentEnv):
    def __init__(self):
        super(ReachingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ReachingSawyerHuman-v1', lambda config: ReachingSawyerHumanEnv())

class ReachingJacoHumanEnv(ReachingEnv, MultiAgentEnv):
    def __init__(self):
        super(ReachingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:ReachingJacoHuman-v1', lambda config: ReachingJacoHumanEnv())
