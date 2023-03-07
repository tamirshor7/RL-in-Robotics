#!/usr/bin/env python
import gym
from gym import spaces
import time
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from time import time
import pdb
from datetime import datetime
import yaml
import os
import argparse
import numpy as np

from matplotlib import pyplot as plt


import random
import numpy as np

import pdb
import sys

import yaml
import os


CUBE_EDGE = 0.5


def position_to_map(pos):
    return ((np.array(pos) - np.array([-10,-10])) // 0.05).astype(int)

def map_to_position(indices):
    return indices * 0.05 + np.array([-10,-10])

def create_map(ws):
    ws_centers = [x.location for x in ws.values()]
    aff_centers = [x.affordance_center for x in ws.values()]
    map = np.zeros((320,320))
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

    # def publish_cost(self):
    #     if self.curr_pose is None:
    #         return
    #     self.time += 1
    #     idx_x, idx_y = self.cmu.position_to_map(self.curr_pose)
    #     c = self.cmu.cost_map[int(idx_x)][int(idx_y)]
    #
    #     message = 'cost: ' + str(c + self.time) + ', reward: ' + str(self.curr_reward)
    #     return message

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

    return l2_norm(start_coord,goal)



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



class TaskPerformerEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg):

        super(TaskPerformerEnv, self).__init__()

        map = cfg["map"]
        tasks = cfg["tasks"]
        workstations = cfg["workstations"]
        init_pos = cfg["init_pos"]
        overall_time = cfg["overall_time"]
        aff = cfg["aff"]

        self.map = map
        self.acts_num = get_acts_num(tasks)
        self.wss = workstations  # all workstations
        self.sequences = tasks
        self.init_pos = init_pos  # TODO pass init pos to reset to start from various poses
        # self.service =rospy.ServiceProxy('/do_action',ActionReq)
        self.overall_time = overall_time
        self.affordance_centers = {f"ws{i}": position_to_map(aff[f"ws{i}"]) for i in range(len(aff))}
        # self.reward_decay = 0.9
        # self.decay_every = 10 #every how many steps decay reward
        self.service = AffordanceServ(aff, {k: ws[k].tasks for k in ws}, tasks, self.map, self.init_pos)

        self.action_space = spaces.Discrete(
            self.acts_num + len(self.wss))  # acts num for doing action, wss to go to ws
        self.observation_space = spaces.Box(shape=(
        (len(self.wss) + 1) ** 2 + 1 + 1 + 2 * ((len(self.wss)) + 1) + len(self.wss) * (
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



        return np.concatenate([arr for arr in (travel_costs.astype(np.float32).flatten(),np.array(prev_act).astype(np.float32).flatten(),np.array(self.holding_obj).astype(np.float32).flatten(),positions.astype(np.float32).flatten(),acts_per_ws.astype(np.float32).flatten(),np.array(self.overall_time-time()).astype(np.float32).flatten())])

    def movebase_client(self,goal_pos):
        dist = l2_norm(self.pos, goal_pos)
        self.service.curr_pose = map_to_position(goal_pos)
        self.service.cost += dist/self.service.max_vel
        self.service.time += dist/self.service.max_vel

    def _take_action(self, action):

        self.current_step += 1
        if action >= self.acts_num: #go to wss

            self.movebase_client(self.affordance_centers[f"ws{action-self.acts_num}"])
            self.curr_ws = action-self.acts_num
            # pickup
            if not self.holding_obj:
                pickup_actions = [task for task in self.wss[f"ws{self.curr_ws}"].tasks if "PU" in task]
                assert len(pickup_actions)
                pu = pickup_actions[0]
                res = self.service.action_request(pu,f"ws{self.curr_ws}")
                if res.success:
                    self.curr_reward += 50
                    self.holding_obj = ord(pu[-1])-ord('A')+1

            self.pos = self.affordance_centers[f"ws{action-self.acts_num}"]


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
                self.curr_reward += 50
                self.holding_obj = 0
                for seq in self.sequences:
                    if self.curr_taken_actions == [int(x.split("ACT")[1]) for x in seq.split("->")]:
                        self.curr_reward += self.sequences[
                            seq]  # TODO add cost subtraction. Take cost from Affordance Server Publisher

        #perform non-place action
        res = self.service.action_request("ACT"+str(action+1),f"ws{self.curr_ws}") #add 1 to actions because actions are 1 based
        if res.success:
            self.curr_reward += 100
            self.curr_taken_actions.append(action+1)


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        eps = 0.1
        reward = self.curr_reward +100*(len(self.curr_taken_actions))#- self.service.cost + 100*(len(self.curr_taken_actions))
        done = time()-self.start_time >= self.overall_time*60 - eps

        obs = self._next_observation()
        self.render()
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.pos = self.init_pos
        self.holding_obj = 0
        self.curr_taken_actions = []
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
        print(f'Curr Taken: {self.curr_taken_actions}')
        print(f'Curr Reward: {self.curr_reward}')


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
        map = create_map(ws)  # np.zeros((320,320))

        ray.init()

        config = {
            "env": TaskPerformerEnv,
            "env_config": {"map": map, "workstations": ws, "tasks": tasks, "aff": aff,
                           "init_pos": self.initial_position, "overall_time": time},
            "framework": "torch",
            "num_workers": 1,
            "num_gpus": 0,
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
            },
            "lr": 0.0005,
            "gamma": 0.99,
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 64,
            "train_batch_size": 100,
            "rollout_fragment_length": "auto",
            "num_envs_per_worker": 1,
            "seed": 42,
            "monitor": False,
        }
        # env = TaskPerformerEnv(config["env_config"])
        # env._next_observation()

        trainer = PPOTrainer  # (config=config)
        tune.run(trainer, config=config, stop={"timesteps_total": 100000})


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



    ws_file = "workstations_config.yaml"
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
        default=6.0,
    )
    args = CLI.parse_args()
    time_ovr = args.time



    ws_file = "workstations_config.yaml"
    tasks_file = "tasks_config.yaml"
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