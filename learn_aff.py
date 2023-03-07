#!/usr/bin/env python
import gym
from gym import spaces
import time
from out_PPO import PPO
from time import time
import pdb
from datetime import datetime
import yaml
import os
import argparse
import numpy as np
# from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
# from stable_baselines3.ppo.ppo import PPO as PPO2
# from stable_baselines3.ppo.policies import MlpPolicy

from matplotlib import pyplot as plt


import random
import numpy as np

import pdb
import sys

import yaml
import os


CUBE_EDGE = 0.5
RENDER_EACH = 1000

def position_to_map(pos):
    return ((np.array(pos) - np.array([-10,-10])) // 0.05).astype(int)

def map_to_position(indices):
    return indices * 0.05 + np.array([-10,-10])

def create_map(ws,map_num=1):
    ws_centers = [x.location for x in ws.values()]
    aff_centers = [x.affordance_center for x in ws.values()]
    map = np.zeros((320,320))
    if map_num == 1:
        map[298:305, 50:300] = 100
        map[150:305, 45:52] = 100
        map[150:157, 45:80] = 100
        map[80:150, 73:80] = 100
        map[73:80, 73:140] = 100
        map[73:150, 133:140] = 100
        map[143:150, 140:290] = 100
        map[150:195, 283:290] = 100
        map[188:195, 150:290] = 100
        map[188:225, 143:150] = 100
        map[218:225, 150:300] = 100
        map[225:300, 293:300] = 100
    if map_num == 2:
        map[298:305, 50:300] = 100
        map[230:305, 50:57] = 100
        map[230:237, 10:50] = 100
        map[200:237, 3:10] = 100
        map[193:200, 3:57] = 100
        map[150:200, 50:57] = 100
        map[150:157, 10:50] = 100
        map[120:157, 3:10] = 100
        map[113:120, 3:97] = 100
        map[50:120, 90:97] = 100
        map[50:57, 97:200] = 100
        map[50:250, 193:200] = 100
        map[243:250, 193:240] = 100
        map[50:250, 233:240] = 100
        map[43:50, 233:297] = 100
        map[43:300, 293:300] = 100

    if map_num == 3:
        import matplotlib
        matplotlib.use("TkAgg")
        map[298:305, 20:300] = 100
        map[230:305, 20:27] = 100
        map[230:237, 20:300] = 100
        map[230:300, 293:300] = 100


    for cen in ws_centers:
        top_left = position_to_map([cen[0]+CUBE_EDGE/2,cen[1]+CUBE_EDGE/2])
        top_right = position_to_map([cen[0] + CUBE_EDGE / 2, cen[1] - CUBE_EDGE / 2])
        bottom_right = position_to_map([cen[0] - CUBE_EDGE / 2, cen[1] - CUBE_EDGE / 2])
        bottom_left = position_to_map([cen[0] - CUBE_EDGE / 2, cen[1] + CUBE_EDGE / 2])
        map[top_right[0],top_right[1]:top_left[1]] = 100
        map[bottom_right[0]:top_right[0], top_right[1]] = 100
        map[bottom_right[0],bottom_right[1]:bottom_left[1]] = 100
        map[bottom_left[0]:top_left[0],top_left[1]] = 100


    for cen in aff_centers:
        top_left = position_to_map([cen[0]+CUBE_EDGE/2,cen[1]+CUBE_EDGE/2])
        top_right = position_to_map([cen[0] + CUBE_EDGE / 2, cen[1] - CUBE_EDGE / 2])
        bottom_right = position_to_map([cen[0] - CUBE_EDGE / 2, cen[1] - CUBE_EDGE / 2])
        bottom_left = position_to_map([cen[0] - CUBE_EDGE / 2, cen[1] + CUBE_EDGE / 2])
        map[top_right[0],top_right[1]:top_left[1]] = 50
        map[bottom_right[0]:top_right[0], top_right[1]] = 50
        map[bottom_right[0],bottom_right[1]:bottom_left[1]] = 50
        map[bottom_left[0]:top_left[0],top_left[1]] = 50

    return map
def point_in_square(point, center):
    x, y = point
    cx, cy = center

    return cx - CUBE_EDGE <= x <= cx + CUBE_EDGE and cy - CUBE_EDGE <= y <= cy + CUBE_EDGE

class ActionReqResponse():
    def __init__(self):
        self.success = None
        self.message = ""

class AffordanceServ:

    def __init__(self, aff_cen, ws_actions, req_tasks,map,init_pos):
        self.aff_cen = aff_cen
        self.possible_actions = ws_actions
        self.tasks = req_tasks
        self.orig_tasks = req_tasks.copy()
        # self.mv_DWA_client = dynamic_reconfigure.client.Client("/move_base/DWAPlannerROS/")
        # self.mv_DWA_client.update_configuration({"max_vel_x": 0.22})
        # # rospy.Service('/affordance_service', Trigger, self.handle_request)
        # # rospy.Service('/do_action', ActionReq, self.action_request)
        # # rospy.Service('/initial_costmap', GetCostmap, self.get_costmap)
        # # self.odom_sub = rospy.Subscriber('/odom', Odometry, self.update_status)
        # self.rviz_pub = RvizPublisher()
        # self.cost_pub = rospy.Publisher('current_cost', String, queue_size=10)
        self.init_pos = init_pos
        self.curr_pose = init_pos
        self.curr_reward = 0
        self.last_action = None
        self.current_object = None
        self.busy = False
        self.done_actions = ''
        self.cost = 0
        self.time = 0
        self.max_vel = 0.22
        self.map=map


    def reset(self):
        self.curr_pose = self.init_pos
        self.curr_reward = 0
        self.last_action = None
        self.current_object = None
        self.busy = False
        self.cost = 0
        self.time = 0
        self.max_vel = 0.22
        self.done_actions = ''
        self.tasks = self.orig_tasks.copy()


    def get_costmap(self, req):
        return self.map

    def check_and_update_reward(self):
        for key, val in self.tasks.items():
            curr_task = key.replace('->', '')
            if self.done_actions.endswith(curr_task):
                del self.tasks[key]
                self.curr_reward += val
                self.last_action = None
                self.busy = False
                self.done_actions = ''
                break



    def action_request(self, act,ws):

        res = ActionReqResponse()
        curr_ws = None
        for key, val in self.aff_cen.items():
            if point_in_square(self.curr_pose, val):
                curr_ws = key
        if curr_ws is None:
            res.success = False
            res.message = 'Not At a workstation'
            return res

        if curr_ws != ws:
            res.success = False
            res.message = "The current station is different"
            return res

        if act not in self.possible_actions[ws]:
            res.success = False
            res.message = 'Not a valid action for this station'
            return res

        if self.last_action is None:
            if curr_ws == ws:
                self.last_action = [ws, act]
                if act.startswith('ACT'):
                    self.done_actions += act
                if act.startswith('PU'):
                    self.max_vel=0.15
                    self.current_object = act[-1]
                res.success = True
                self.busy = True
                return res

        if act.startswith('PU') and self.current_object is not None:
            res.success = False
            res.message = "Trying picking another object while already loaded"
            return res

        if act.startswith('PL') and self.current_object is None:
            res.success = False
            res.message = 'Trying to place an object while nothing in your hold'
            return res

        if act.startswith('ACT') and self.current_object is not None:
            res.success = False
            res.message = 'Trying to act while holding an object'
            return res

        if self.busy and act.startswith('ACT') and not self.last_action[1].startswith('PL'):
            res.success = False
            res.message = 'Trying to cheat without moving an object'
            return res

        if self.last_action[0] == ws:
            if act.startswith('ACT') and self.last_action[1].startswith('ACT'):
                res.success = False
                res.message = 'Too many actions at the same workstation'
                return res
            if act.startswith('PL') and self.last_action[1].startswith('PU'):
                res.success = False
                res.message = 'Cannot place an object at the same station without doing some action with it'
                return res

            self.last_action = [ws, act]
            res.success = True
            if act.startswith('PU'):
                self.max_vel = 15
                self.current_object = act[-1]
            if act.startswith('ACT'):
                self.done_actions += act
                self.check_and_update_reward()
            return res

        if act.startswith('PL') and self.last_action[1].startswith('PU'):
            ob = act[-1]
            if self.last_action[1].endswith(ob):
                self.last_action = [ws, act]
                res.success = True
                self.max_vel = 0.22
                self.current_object = None
                return res
            res.success = False
            res.message = 'Trying to place an unmatched object'
            return res

        self.last_action = [ws, act]
        res.success = True
        if act.startswith('PU'):
            self.max_vel = 0.1
            self.current_object = act[-1]
        if act.startswith('ACT'):
            self.done_actions += act
        self.check_and_update_reward()
        return res

    def handle_request(self):
        # Do some processing here and return a response
        res = ActionReqResponse()
        res.success = True
        res.message = ''
        for key, val in self.aff_cen.items():
            res.message += key + ' affordance center is at ' + str(val[0]) + ' ' + str(val[1]) + '\n'
        return res

    # def update_status(self, msg):
    #     pose_x = msg.pose.pose.position.x
    #     pose_y = msg.pose.pose.position.y
    #     self.curr_pose = np.array([pose_x, pose_y])
    #     point = (pose_x, pose_y)
    #     for key, val in self.aff_cen.items():
    #         if point_in_square(point, val):
    #             self.rviz_pub.update_and_publish_markers(0, 1., 0, val[0], val[1], int(key[2]))
    #         else:
    #             self.rviz_pub.update_and_publish_markers(1., 0, 0, val[0], val[1], int(key[2]))
    #














def get_plan_dist(start_coord, goal):
    # use movebase to get plan from start to goal, return length of trajectory. First start should be [0,0], goals should be as received in cmd input

    return l2_norm(map_to_position(np.array(start_coord)),map_to_position(np.array(goal)))



def form_center_travel_costs(centers, start_coord):
    # do all in input position ([0,0] instead of [-1.45,-1.45] etc)

    deps = np.zeros([len(centers) + 1, len(centers) + 1])
    nodes = {i: ([start_coord] + centers)[i] for i in range(len(centers) + 1)}

    for i in range(len(centers) + 1):
        for j in range(len(centers) + 1):
            if i == j:
                continue
            if i > j:
                deps[i][j] = deps[j][i]
                continue
            deps[i][j] = get_plan_dist(nodes[i], nodes[j]) #TODO Add wall cost to plan dist func
    return deps

def get_acts_num(tasks):
    max_act = 1
    for seq in tasks:
        for task in seq.split("->"):
            if int(task.split("ACT")[1]) > max_act:
                max_act = int(task.split("ACT")[1])
    return max_act

def l2_norm(x1,x2):
    return np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)

def encode_seqs(tasks):
    seq_encodes = []
    rewards = []
    for seq,reward in tasks.items():
        seq_encode_str = ""
        for act in seq.split("->"):
            seq_encode_str += act[-1]
        seq_encodes.append(int(seq_encode_str))
        rewards.append(reward)

    return np.array([seq_encodes,rewards]).astype(np.float32).flatten()


class TaskPerformerEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, map, workstations, tasks, aff, init_pos, overall_time):
        super(TaskPerformerEnv, self).__init__()

        self.map = map
        self.acts_num = get_acts_num(tasks)
        self.wss = workstations  # all workstations
        self.sequences = tasks
        self.seq_encode = encode_seqs(tasks)
        self.init_pos = init_pos  # TODO pass init pos to reset to start from various poses
        # self.service =rospy.ServiceProxy('/do_action',ActionReq)
        self.overall_time = overall_time
        self.affordance_centers = {f"ws{i}": position_to_map(aff[f"ws{i}"]) for i in range(len(aff))}
        # self.reward_decay = 0.9
        # self.decay_every = 10 #every how many steps decay reward
        self.service = AffordanceServ(aff, {k: ws[k].tasks for k in ws}, tasks, self.map, self.init_pos)

        self.action_space = spaces.Discrete(
            self.acts_num + len(self.wss)*4)  # acts num for doing action, wss to go to ws
        self.observation_space = spaces.Box(shape=(
        (len(self.wss) + 1) ** 2 + 1 + 1 + 2*len(self.sequences) + 2 * ((len(self.wss)) + 1) + len(self.wss) * (
                    self.acts_num + 1) + 1,), low=-np.inf, high=np.inf, dtype=np.float32)
        # self.observation_space = spaces.Box(shape=(self.map.shape),low=0,high=256)

    def _next_observation(self):
        # return self.map
        travel_costs = form_center_travel_costs([self.wss[ws].location for ws in self.wss], self.init_pos)
        positions = np.append(self.pos, [self.wss[ws].location for ws in self.wss])
        acts_per_ws = np.zeros((len(self.wss), self.acts_num + 1))  # +1 for place action
        for i, ws in enumerate(self.wss):
            possible_act_indices = [int(act.split('ACT')[1]) - 1 for act in self.wss[ws].tasks if
                                    'ACT' in act]  # subtract 1 because actions are 1 based
            acts_per_ws[i][possible_act_indices] = 1
            acts_per_ws[i, -1] = 1  # place always possible

        prev_act = int(self.service.last_action[0].split("ws")[1]) if self.service.last_action is not None else 0



        return np.concatenate([arr for arr in (travel_costs.astype(np.float32).flatten(),self.seq_encode,np.array(prev_act).astype(np.float32).flatten(),np.array(self.holding_obj).astype(np.float32).flatten(),positions.astype(np.float32).flatten(),acts_per_ws.astype(np.float32).flatten(),np.array(self.overall_time-time()).astype(np.float32).flatten())])

    def movebase_client(self,goal_pos):
        dist = l2_norm(map_to_position(np.array(self.pos)), map_to_position(np.array(goal_pos)))
        self.service.curr_pose = map_to_position(goal_pos)
        self.service.cost += dist/self.service.max_vel
        self.service.time += (dist/self.service.max_vel)/60

    def _take_action(self, action):

        self.current_step += 1
        if action >= self.acts_num: #go to wss

            target_ws = (action-self.acts_num)//4
            target_aff = (action-self.acts_num)%4
            if target_aff == 0:
                map_target_aff = np.array(self.wss[f"ws{target_ws}"].location) + np.array([ CUBE_EDGE,0])
            elif target_aff == 1:
                map_target_aff = np.array(self.wss[f"ws{target_ws}"].location) - np.array([ CUBE_EDGE,0])
            elif target_aff == 2:
                map_target_aff = np.array(self.wss[f"ws{target_ws}"].location) + np.array([0,CUBE_EDGE])
            else:
                map_target_aff = np.array(self.wss[f"ws{target_ws}"].location) - np.array([0, CUBE_EDGE])
            #map_target_aff = position_to_map(np.array(self.wss[f"ws{target_ws}"].affordance_center))
            map_target_aff = position_to_map(map_target_aff)
            self.movebase_client(map_target_aff)
            self.curr_ws = target_ws
            # pickup
            if not self.holding_obj:
                pickup_actions = [task for task in self.wss[f"ws{self.curr_ws}"].tasks if "PU" in task]
                assert len(pickup_actions)
                pu = pickup_actions[0]
                res = self.service.action_request(pu,f"ws{self.curr_ws}")
                if res.success:
                    #self.curr_reward += 50
                    self.holding_obj = ord(pu[-1])-ord('A')+1

            self.pos = self.affordance_centers[f"ws{target_ws}"]


            return

        if not self.holding_obj:
            return

        #if got here - perform action on workstation
        if self.holding_obj and f"ws{self.curr_ws}" != self.service.last_action[0]:  # do place action
            place_actions = [task for task in self.wss[f"ws{self.curr_ws}"].tasks if "PL" in task]
            assert len(place_actions)
            if f"PL-{chr(self.holding_obj +ord('A')-1)}" in place_actions:
                pl = f"PL-{chr(self.holding_obj +ord('A')-1)}"
            else:
                return

            res = self.service.action_request(pl, f"ws{self.curr_ws}")
            if res.success:
                #self.curr_reward += 50
                self.holding_obj = 0
                # for seq in self.sequences:
                #     if self.curr_taken_actions == [int(x.split("ACT")[1]) for x in seq.split("->")]:
                #         self.curr_reward += self.sequences[
                #             seq]  # TODO add cost subtraction. Take cost from Affordance Server Publisher

        #perform non-place action
        res = self.service.action_request("ACT"+str(action+1),f"ws{self.curr_ws}") #add 1 to actions because actions are 1 based
        # if res.success:
        #       self.curr_reward += 100
            #self.curr_taken_actions.append(action+1)


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        eps = 0.1
        reward = self.service.curr_reward-self.current_step/100 #- self.service.cost
        done = (self.service.time >= self.overall_time*60 - eps) or not len(self.service.tasks)


        obs = self._next_observation()
        if done or not self.current_step%RENDER_EACH:
            self.render()
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.pos = self.init_pos
        self.holding_obj = 0
        #self.curr_taken_actions = []
        self.curr_ws = -1
        self.start_time = time()
        self.time_left = self.overall_time
        self.current_step = 0
        self.curr_reward = 0
        self.service.reset()
        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen

        print(f'Step: {self.current_step}')
        print(f'Curr Taken: {self.service.done_actions}')
        print(f'Curr Reward: {self.service.curr_reward}')
        print(f'Curr Cost: {self.current_step}')


CUBE_EDGE = 0.5



class TurtleBot:
    def __init__(self):
        self.initial_position = [250,250]
        self.time = None


        print("The initial position is {}".format(self.initial_position))

    def set_initial_position(self, msg):
        initial_pose = msg.pose.pose
        self.initial_position = np.array([initial_pose.position.x, initial_pose.position.y])

    def run(self, ws, tasks,aff, time):
        self.time = time
        #env = TaskPerformerEnv(ws, tasks, aff, self.initial_position, time)
        # ==== You can delete =======

        print(tasks)
        for w, val in ws.items():
            print(w + ' center is at ' + str(val.location) + ' and its affordance center is at ' + str(
                val.affordance_center))
        # ===========================

    def train(self, ws, tasks,aff, time):
        map = create_map(ws,3)  # np.zeros((320,320))
        env = TaskPerformerEnv(map, ws, tasks, aff, self.initial_position, time)

        print("============================================================================================")

        ####### initialize environment hyperparameters ######

        has_continuous_action_space = False  # continuous action space; else discrete

        max_ep_len = 100000  # max timesteps in one episode
        max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps

        print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
        log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
        save_model_freq = int(1e5)  # save model frequency (in num timesteps)

        action_std = 0.6  # starting std for action distribution (Multivariate Normal)
        action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
        action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
        #####################################################

        ## Note : print/log frequencies should be > than max_ep_len

        ################ PPO hyperparameters ################
        update_timestep = max_ep_len * 4  # update policy every n timesteps
        K_epochs = 80  # update policy for K epochs in one PPO update

        eps_clip = 0.2  # clip parameter for PPO
        gamma = 0.99  # discount factor

        lr_actor = 0.0003  # learning rate for actor network
        lr_critic = 0.001  # learning rate for critic network

        random_seed = 0  # set random seed if required (0 = no random seed)
        #####################################################

        # state space dimension
        state_dim = env.observation_space.shape[0]

        # action space dimension
        if has_continuous_action_space:
            action_dim = env.action_space.shape[0]
        else:
            action_dim = env.action_space.n

        ###################### logging ######################

        #### log files for multiple runs are NOT overwritten
        log_dir = "PPO_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + "log" + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #### get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        log_f_name = log_dir + '/PPO_' + "_log_" + str(run_num) + ".csv"

        print("logging at : " + log_f_name)
        #####################################################

        ################### checkpointing ###################
        run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

        directory = "PPO_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + "run" + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        #####################################################

        ############# print all hyperparameters #############
        print("--------------------------------------------------------------------------------------------")
        print("max training timesteps : ", max_training_timesteps)
        print("max timesteps per episode : ", max_ep_len)
        print("model saving frequency : " + str(save_model_freq) + " timesteps")
        print("log frequency : " + str(log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
        print("state space dimension : ", state_dim)
        print("action space dimension : ", action_dim)
        print("--------------------------------------------------------------------------------------------")
        if has_continuous_action_space:
            print("Initializing a continuous action space policy")
            print("--------------------------------------------------------------------------------------------")
            print("starting std of action distribution : ", action_std)
            print("decay rate of std of action distribution : ", action_std_decay_rate)
            print("minimum std of action distribution : ", min_action_std)
            print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
        else:
            print("Initializing a discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("PPO update frequency : " + str(update_timestep) + " timesteps")
        print("PPO K epochs : ", K_epochs)
        print("PPO epsilon clip : ", eps_clip)
        print("discount factor (gamma) : ", gamma)
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", lr_actor)
        print("optimizer learning rate critic : ", lr_critic)
        if random_seed:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", random_seed)
            torch.manual_seed(random_seed)
            env.seed(random_seed)
            np.random.seed(random_seed)
        #####################################################

        print("============================================================================================")

        ################# training procedure ################

        # initialize a PPO agent
        ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space, action_std)

        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # logging file
        log_f = open(log_f_name, "w+")
        log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        # training loop
        while time_step <= max_training_timesteps:

            state = env.reset()
            current_ep_reward = 0

            for t in range(1, max_ep_len + 1):

                # select action with policy
                action = ppo_agent.select_action(state)
                state, reward, done, _ = env.step(action)

                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                # update PPO agent
                if time_step % update_timestep == 0:
                    ppo_agent.update()

                # if continuous action space; then decay action std of ouput action distribution
                if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                # log in logging file
                if time_step % log_freq == 0:
                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % print_freq == 0:
                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                            print_avg_reward))

                    print_running_reward = 0
                    print_running_episodes = 0

                # save model weights
                if time_step % save_model_freq == 0:
                    print(
                        "--------------------------------------------------------------------------------------------")
                    print("saving model at : " + "weigths")
                    ppo_agent.save("weights")
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print(
                        "--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

        log_f.close()
        env.close()


# ======================================================================================================================

def analyse_res(msg):
    result = {}
    for line in msg.split("\n"):
        if line:
            parts = line.split(" ")
            key = parts[0]
            x = float(parts[-2])
            y = float(parts[-1])
            result[key] = [x, y]
    return result


class Workstation:
    def __init__(self, location, tasks):
        self.location = location
        self.tasks = tasks
        self.affordance_center = None

    def update_affordance_center(self, new_center):
        self.affordance_center = new_center


# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':



    ws_file = "ws_cfg3.yaml"#"workstations_config.yaml"
    tasks_file = "tasks_config.yaml"

    with open(tasks_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.CLoader)
        tasks = data['tasks']

    locs = {}
    actions = {}
    with open(ws_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.CLoader)
        num_ws = data['num_ws']
        for i in range(num_ws):
            locs['ws' + str(i)] = data['ws' + str(i)]['location']
            actions['ws' + str(i)] = data['ws' + str(i)]['tasks']

    moves = [[-CUBE_EDGE, 0], [0, -CUBE_EDGE], [CUBE_EDGE, 0], [0, CUBE_EDGE]]
    global aff_cen_list
    aff_cen_list = {}
    for key, val in locs.items():
        move = random.choice(moves)
        aff_cen_list[key] = [val[0] + move[0], val[1] + move[1]]

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--time",
        type=float,
        default=1,
        help="time in mins"
    )
    args = CLI.parse_args()
    time_ovr = args.time




    ws = {}
    with open(ws_file, 'r') as f:
        data = yaml.load(f,Loader=yaml.CLoader)
        num_ws = data['num_ws']
        for i in range(num_ws):
            ws['ws' + str(i)] = Workstation(data['ws' + str(i)]['location'], data['ws' + str(i)]['tasks'])



    aff = aff_cen_list
    for key, val in ws.items():
        val.update_affordance_center(aff[key])

    with open(tasks_file, 'r') as f:
        data = yaml.load(f,Loader=yaml.CLoader)
        tasks = data['tasks']

    tb3 = TurtleBot()
    #tb3.run(ws, tasks, aff, time)
    tb3.train(ws, tasks, aff, time_ovr)