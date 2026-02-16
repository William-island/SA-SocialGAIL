import gym
import pygame
import numpy as np
import os
import imageio
import pickle
import argparse
import random
import math
import os
from statistics import mean
from scipy.spatial.distance import euclidean
from scipy.stats import norm
from core.network import GraphData
import torch
from tqdm import tqdm
from hausdorff import hausdorff_distance
from torch_geometric.data import Batch

from core.buffer import RolloutBuffer_for_Latent_Feature



class CrowdEnv(gym.Env):  # can extend from gym.Env
    def __init__(self, args, TT_type):
        super(CrowdEnv, self).__init__()

        # GC_continue = {'trajectories':trajectories, 'overlap_noise':overlap_noise, 'train_ids':train_list, 'test_ids':test_list}
        with open(args.dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        self.time_interval = args.time_interval
        self.frame_interval = args.frame_interval

        self.trajectories, self.frame_data, self.start_data, self.goal_data = self._load_dataset(dataset['trajectories'])  #(dataset['trajectories']) 
        # self.overlap_noise = dataset['overlap_noise']
        if TT_type == 'Train':
            self.test_flag = False
            self.people_ids = dataset['train_ids']
            print(f'Training dataset size:{len(self.people_ids)}')
        else:
            self.test_flag = True
            self.people_ids = dataset['test_ids']
            print(f'Testing dataset size:{len(self.people_ids)}')

        self.regions = args.regions
        self.radius = args.radius
        self.num_agents = len(self.trajectories.keys())
        self.map_size_bound = args.map_size_bound # [low_x, high_x, low_y, high_y]
        self.with_last_speed = args.with_last_speed
        self.observation_type = args.observation_type # 'radar' / 'graph'
        self.entity_type = args.entity_type
        if self.observation_type == 'graph':
            self.graph_obs_past_len = args.graph_obs_past_len
            self.padd_to_number = args.padd_to_number
            self.graph_feature_len = args.graph_feature_len

        self.agent_radius = args.agent_radius # 0.25
        self.target_circle_width = args.target_circle_width # 1.0 # 1.5 # 1.0 
        self.new_width = self.target_circle_width
        self.relaxed_steps = args.relaxed_steps  # 40 # 40 # 80
        self._max_episode_steps = args.max_trajectory_length # 400
        print(f'Now env target circle width: {self.target_circle_width}')
        if not self.test_flag:
            print(f'relaxed steps: {self.relaxed_steps}')
            print(f'relaxed frames: {self.relaxed_steps*self.frame_interval}')

        self.agent_id = 1
        self.frame_number = 0
        self.current_position = [self.trajectories[self.agent_id][0][0], self.trajectories[self.agent_id][0][1]]
        self.time_steps = 0
        self.end_frame = 120000
        self.ADE_list = []
        self.new_traj = [self.current_position]
        self.old_traj = np.array(self.trajectories[self.agent_id])[:,0:2]

        self.collisions = 0
        self.normal_collisions, self.ccd_collisions = 0, 0
        # self.companys = 0

        # gym action space & observation sapce
        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)
        if self.with_last_speed:
            self.observation_space = gym.spaces.Box(low=-400, high=400, shape=(3*self.regions+2+2,), dtype=np.float64)
        else:
            self.observation_space = gym.spaces.Box(low=-400, high=400, shape=(3*self.regions+2,), dtype=np.float64)  # self.regions+


        # if args.manual_reward_done exists
        if hasattr(args, 'manual_reward_done'):
            self.manual_reward_done = args.manual_reward_done
        else:
            self.manual_reward_done = False

        
        self.latent_code = args.latent_code

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]





    def reset(self, person_id=None):
        # decide the agent of current eposide and start frame
        if person_id==None:
            self.agent_id = random.choice(self.people_ids) # (list(self.trajectories.keys()))   |   (self.people_ids)
        else:
            self.agent_id = person_id
        # self.agent_id = 129
        # while(not self._frame_continues_check()):
        #     self.agent_id = random.randint(1,self.num_agents)
        # print("now agent_id{}".format(self.agent_id))
        self.frame_number = self.trajectories[self.agent_id][0][2]
        self.current_position = np.array([self.trajectories[self.agent_id][0][0], self.trajectories[self.agent_id][0][1]])
        self.time_steps = 0
        self.end_frame = self.trajectories[self.agent_id][-1][2]
        self.ADE_list = []
        self.new_traj = [self.current_position]
        self.old_traj = np.array(self.trajectories[self.agent_id])[:,0:2]
        self.collisions = 0
        self.companys = 0

        goal_v = np.array(self.trajectories[self.agent_id][-1][:2])-np.array(self.trajectories[self.agent_id][0][:2])
        self.r_matrix = self._rotate_matrix_v(goal_v)
        self.cr_matrix = self._counter_rotate_matrix_v(goal_v)

        self.normal_collisions, self.ccd_collisions = 0, 0

        return self._get_observation()




    def step(self, action):
        # update frame
        self._update_frame_number()

        # update the position of target agent
        # valid check: map & collision
        reward = 0
        done = False
        self.time_steps += 1

        # update new position , get position change via speed!!
        # dx = action[0]
        # dy = action[1]
        dx = action[0]*self.time_interval
        dy = action[1]*self.time_interval

        dx, dy  = np.dot([dx,dy],self.cr_matrix)

        new_position = np.array(self.current_position) + np.array([dx, dy])

        # check map validation
        if not self._valid_check_in_map():
            new_position = self.current_position

        # # check collision
        # if self.frame_number in self.frame_data.keys():
        #     for other, opos in self.frame_data[self.frame_number].items():
        #         if self.agent_id != other:
        #             dist = np.linalg.norm(new_position - np.array([opos[0],opos[1]]))
        #             if dist < 2*self.agent_radius:
        #                 self.collisions += 1
                        ## handle collision：back to the old position and punish it
                        # if self.entity_type:
                        #     new_position = self.current_position # no step back
                        # reward -= 1
                        # if self._abs_angle([dx,dy],[opos[2],opos[3]])<=30:
                        #     self.companys += 1
                        # else:
                        #     self.collisions += 1

        self.current_position = new_position

        self.new_traj.append(self.current_position)

        # end condition 1
        dist = np.linalg.norm(np.array(self.current_position) - np.array(self.goal_data[self.agent_id]))
        # self.ADE_list.append(dist)
        if dist < self.target_circle_width:
            # to the target circle
            # reward += 10
            done = True
        # end condition2
        if self.test_flag:
            if self.frame_number == self.end_frame:
                done = True
        else:
            # if dist < self.target_circle_width:
            #     # to the target circle
            #     done = True
            if self.frame_number == self.end_frame + int(self.relaxed_steps*self.frame_interval):
                done = True
        # end condition 3
        if self.time_steps > self._max_episode_steps:
            done = True

        # if using manual reward done
        if self.manual_reward_done:
            reward, man_done = self._manual_reward_done()
            done = done or man_done

        

        return self._get_observation(), reward, done, {}
    








    ## compute and store latent feature each step
    def get_latent_features(self , action):

        # extract features from states and actions
        ext_features = {}
        for k, v in self.latent_code.items():
            # for 'speed'
            if k == 'speed':
                feature = self.get_speed(action)
                ext_features[k] = feature
            elif k == 'nn_dist':
                feature = self.get_nn_dist()
                ext_features[k] = feature
            elif k == 'social_comfort':
                feature = self.get_social_comfort(action)
                ext_features[k] = feature
            elif k == 'social_gradient':
                feature = self.get_social_gradient(action)
                ext_features[k] = feature
            else:
                # not finished
                pass

        return ext_features


    ## define each feature extractor
    def get_speed(self, action):
        # compute norm of speed from action
        feature = np.linalg.norm(action, keepdims=True)
        return feature
    
    def get_nn_dist(self, within_radius = True):
        # compute the nearest neighbour distance
        if within_radius:
            nn_dist = self.radius
        else:
            nn_dist = 100. # similar to np.inf
        
        if self.frame_number in self.frame_data.keys():
            for other, opos in self.frame_data[self.frame_number].items():
                if self.agent_id != other:
                    dist = np.linalg.norm(self.current_position - np.array([opos[0], opos[1]]), keepdims=True)
                    if dist < nn_dist:
                        nn_dist = dist
        
        return np.array(nn_dist)


    def get_social_comfort(self, action, sigma_long=2.0, sigma_lat=0.8):
        # compute the social comfort

        # should warn that this action is not the action input to the step
        # compute the velocity vector
        dx = action[0] * self.time_interval * 2
        dy = action[1] * self.time_interval * 2

        vel_vector  = np.dot([dx,dy],self.cr_matrix)


        # 处理静止或速度极小的情况 (防止除以0)
        speed = np.linalg.norm(vel_vector)
        if speed < 1e-4:
            # 如果行人静止，假设一个默认朝向，或者沿用上一帧的朝向
            if len(self.new_traj) > 1:
                last_move = np.array(self.new_traj[-1]) - np.array(self.new_traj[-2])
                if np.linalg.norm(last_move, keepdims=False) > 1e-4:
                    direction_u = last_move / np.linalg.norm(last_move, keepdims=False)
                else:
                    # 假设默认朝向为 x 轴正方向
                    direction_u = np.array([1, 0])
                direction_n = np.array([-direction_u[1], direction_u[0]])
            else:
                # 假设默认朝向为 x 轴正方向
                direction_u = np.array([1, 0])
                direction_n = np.array([-direction_u[1], direction_u[0]])
        else:
            # 单位方向向量 (Longitudinal / 纵向)
            direction_u = vel_vector / speed
            # 法向量 (Lateral / 横向) - 旋转90度
            direction_n = np.array([-direction_u[1], direction_u[0]])
            
        # 计算相对位置向量 (Agent - Pedestrian)
        p_p = np.array(self.current_position)
        # find the nearest pedestrian as p_p
        nn_dist = 1000.
        if self.frame_number in self.frame_data.keys():
            for other, opos in self.frame_data[self.frame_number].items():
                if self.agent_id != other:
                    dist = np.linalg.norm(self.current_position - np.array([opos[0], opos[1]]), keepdims=True)
                    if dist < nn_dist:
                        nn_dist = dist
                        p_a = np.array([opos[0], opos[1]])
                        delta_p = p_a - p_p

        if delta_p is None:
            assert False, "delta_p is None"
        
        # 投影到局部坐标系
        # d_long: 在行人前进方向上的距离 (正值在前方，负值在后方)
        d_long = np.dot(delta_p, direction_u)
        # d_lat: 在行人侧向的距离
        d_lat = np.dot(delta_p, direction_n)
        
        # 可选优化：对后方不敏感处理 (如果 Agent 在行人身后，势场衰减更快)
        # 如果你只关心前方避障，可以不加这个，或者把身后的 sigma_long 设得很小
        # current_sigma_long = sigma_long if d_long > 0 else sigma_long * 0.5 
        current_sigma_long = sigma_long
        
        # G. 计算高斯值 (0 ~ 1)
        # Feature = exp( -0.5 * ( (dx/sx)^2 + (dy/sy)^2 ) )
        exponent = -0.5 * ( (d_long**2 / current_sigma_long**2) + 
                            (d_lat**2 / sigma_lat**2) )
        social_comfort = np.exp(exponent)*10.
        
        return np.array(social_comfort)


    def get_social_gradient(self, action, sigma_long=4.0, sigma_lat=1.0):
        # compute the social comfort

        # should warn that this action is not the action input to the step
        # compute the velocity vector
        dx = action[0] * self.time_interval * 2
        dy = action[1] * self.time_interval * 2

        vel_vector  = np.dot([dx,dy],self.cr_matrix)


        # 处理静止或速度极小的情况 (防止除以0)
        speed = np.linalg.norm(vel_vector)
        if speed < 1e-4:
            # 如果行人静止，假设一个默认朝向，或者沿用上一帧的朝向
            if len(self.new_traj) > 1:
                last_move = np.array(self.new_traj[-1]) - np.array(self.new_traj[-2])
                if np.linalg.norm(last_move, keepdims=False) > 1e-4:
                    direction_u = last_move / np.linalg.norm(last_move, keepdims=False)
                else:
                    # 假设默认朝向为 x 轴正方向
                    direction_u = np.array([1, 0])
                direction_n = np.array([-direction_u[1], direction_u[0]])
            else:
                # 假设默认朝向为 x 轴正方向
                direction_u = np.array([1, 0])
                direction_n = np.array([-direction_u[1], direction_u[0]])
        else:
            # 单位方向向量 (Longitudinal / 纵向)
            direction_u = vel_vector / speed
            # 法向量 (Lateral / 横向) - 旋转90度
            direction_n = np.array([-direction_u[1], direction_u[0]])
            
        # 计算相对位置向量 (Agent - Pedestrian)
        p_p = np.array(self.current_position)
        # find the nearest pedestrian as p_p
        nn_dist = 1000.
        if self.frame_number in self.frame_data.keys():
            for other, opos in self.frame_data[self.frame_number].items():
                if self.agent_id != other:
                    dist = np.linalg.norm(self.current_position - np.array([opos[0], opos[1]]), keepdims=True)
                    if dist < nn_dist:
                        nn_dist = dist
                        p_a = np.array([opos[0], opos[1]])
                        delta_p = p_a - p_p

        if delta_p is None:
            assert False, "delta_p is None"
        
        # 投影到局部坐标系
        d_long = np.dot(delta_p, direction_u)
        d_lat = np.dot(delta_p, direction_n)
        
        # 高斯指数项内部: E = -0.5 * ( (dx/sx)^2 + (dy/sy)^2 )
        # 这里不对后方做特殊衰减，保持梯度的连续性，便于网络学习
        term_long = (d_long / sigma_long)**2
        term_lat = (d_lat / sigma_lat)**2
        exponent = -0.5 * (term_long + term_lat)
        
        # 势能标量值 (Potential)
        V = np.exp(exponent)
        
        # --- 核心修改：计算梯度 (Gradient) ---
        # 对于高斯函数 f(x) = exp(-x^2 / 2s^2)
        # 导数 f'(x) = f(x) * (-x / s^2)
        # 这里我们需要计算的是力，通常定义为势场的负梯度 F = -grad(V)
        # 但作为特征输入给网络，直接给梯度值即可，让网络自己学符号
        
        # 纵向梯度分量 (关于 d_long 的偏导)
        g_long = V * (-d_long / (sigma_long**2))
        
        # 横向梯度分量 (关于 d_lat 的偏导)
        g_lat = V * (-d_lat / (sigma_lat**2))

        return np.array([g_long*10, g_lat]) # *10 for align the scale




















    # from start frame to end frame of target agent, render all agents by policy
    def step_multi(self, policy, latent_c):
        # init
        start_frame = self.trajectories[self.agent_id][0][2]
        end_frame = self.trajectories[self.agent_id][-1][2]

        # cut init frame
        current_frame = {}
        for id, value in self.frame_data[start_frame].items():
            current_frame[id] = np.array(value[:2])
        self.past_frames = {}

        target_traj = []
        for frame_id in range(start_frame, end_frame+1, self.frame_interval):
            # 获取在场行人
            present_agent = np.array(list(self.frame_data[frame_id].keys()))
            # 添加新行人进当前帧, 删除旧行人
            last_present_agent = np.array(list(current_frame.keys()))
            for id in np.setdiff1d(present_agent, last_present_agent):
                current_frame[id] = np.array(self.frame_data[frame_id][id][:2])
            for id in np.setdiff1d(last_present_agent, present_agent):
                if id!=self.agent_id:
                    del current_frame[id]
            # 压栈
            self.past_frames[frame_id] = current_frame
            target_traj.append(current_frame[self.agent_id])
            # end or continue
            dist = np.linalg.norm(np.array(current_frame[self.agent_id]) - np.array(self.goal_data[self.agent_id]))
            if dist < self.target_circle_width:
                break
            # generate next frame
            next_frame = {}
            states = []
            latent_cs = []
            for id in current_frame.keys():
                current_position = np.array([current_frame[id][0], current_frame[id][1]])

                # get unique r_matrix
                goal_v = np.array(self.trajectories[id][-1][:2])-np.array(self.trajectories[id][0][:2])
                r_matrix = self._rotate_matrix_v(goal_v)

                state = self._get_ar_relative_graph_observation(id, r_matrix, frame_id, current_position, current_frame, self.past_frames)
                states.append(state)

            # compute actions
            states = Batch.from_data_list(states).to(policy.device)
            latent_cs = latent_c.expand(len(current_frame.keys()), -1).to(policy.device)
            with torch.no_grad():
                actions = policy.actor(states, latent_cs).cpu().numpy()*self.action_space.high

            for i,id in enumerate(current_frame.keys()):
                cr_matrix = self._counter_rotate_matrix_v(np.array(self.trajectories[id][-1][:2])-np.array(self.trajectories[id][0][:2]))
                dx, dy = actions[i][0]*self.time_interval, actions[i][1]*self.time_interval
                dx, dy  = np.dot([dx,dy],cr_matrix)
                new_position = np.array([current_frame[id][0], current_frame[id][1]]) + np.array([dx, dy])
                next_frame[id] = new_position

            current_frame = next_frame
        
        self.new_traj = target_traj
    

    # from start frame to end frame of target agent, render all agents by policy
    def step_all(self, policy, flow_ids):
        # init
        start_frame = self.trajectories[self.agent_id][0][2]
        end_frame = self.trajectories[self.agent_id][-1][2]

        # cut init frame
        current_frame = {}
        for id, value in self.frame_data[start_frame].items():
            current_frame[id] = np.array(value[:2])
        past_frames = {}

        target_traj = []
        for frame_id in range(start_frame, end_frame+1, self.frame_interval):
            # 获取在场行人
            present_agent = np.array(list(self.frame_data[frame_id].keys()))
            # 添加新行人进当前帧, 删除旧行人
            last_present_agent = np.array(list(current_frame.keys()))
            for id in np.setdiff1d(present_agent, last_present_agent):
                current_frame[id] = np.array(self.frame_data[frame_id][id][:2])
            for id in np.setdiff1d(last_present_agent, present_agent):
                if id!=self.agent_id:
                    del current_frame[id]
            # 压栈
            past_frames[frame_id] = current_frame
            target_traj.append(current_frame[self.agent_id])
            # end or continue
            dist = np.linalg.norm(np.array(current_frame[self.agent_id]) - np.array(self.goal_data[self.agent_id]))
            if dist < self.target_circle_width:
                break
            # generate next frame
            next_frame = {}
            for id in current_frame.keys():
                current_position = np.array([current_frame[id][0], current_frame[id][1]])

                # get unique r_matrix
                goal_v = np.array(self.trajectories[id][-1][:2])-np.array(self.trajectories[id][0][:2])
                r_matrix = self._rotate_matrix_v(goal_v)
                cr_matrix = self._counter_rotate_matrix_v(goal_v)

                state = self._get_ar_relative_graph_observation(id, r_matrix, frame_id, current_position, current_frame, past_frames)
                action = policy.exploit(state)

                dx = action[0]*self.time_interval
                dy = action[1]*self.time_interval

                dx, dy  = np.dot([dx,dy],cr_matrix)

                new_position = current_position + np.array([dx, dy])

                next_frame[id] = new_position
            current_frame = next_frame
        # get base traj list
        base_traj_list = []
        for id in flow_ids:
            base_traj_list.append(np.array(self.trajectories[id])[:,:2])
        return self._log_KDE(base_traj_list, target_traj, 5)
    

    def render(self):
        # 创建 Pygame Surface 对象
        x_shift, y_shift = 12, 15
        scale = 10
        screen_width, screen_height = 60*scale, 60*scale
        screen = pygame.Surface((screen_width, screen_height))

        screen.fill((255, 255, 255))  # clear Surface

        for waypoint in self.trajectories[self.agent_id]:
            pygame.draw.circle(screen, (185, 199, 141), (int((waypoint[0]+x_shift)*scale), int((waypoint[1]+y_shift)*scale)), 0.5*scale)

        for person,value in self.frame_data[self.frame_number].items():
            # draw the agent
            if person != self.agent_id:
                agent_x, agent_y = value[0]+x_shift, value[1]+y_shift
                pygame.draw.circle(screen, (62, 115, 158), (int(agent_x*scale), int(agent_y*scale)), 0.5*scale)
            else:
                pygame.draw.circle(screen, (101, 136, 115), (int((self.frame_data[self.frame_number][self.agent_id][0]+x_shift)*scale), int((self.frame_data[self.frame_number][self.agent_id][1]+y_shift)*scale)), 0.5*scale)
                pygame.draw.circle(screen, (255, 0, 0), (int((self.current_position[0]+x_shift)*scale), int((self.current_position[1]+y_shift)*scale)), 6*scale, 2)
                pygame.draw.circle(screen, (255, 0, 0), (int((self.current_position[0]+x_shift)*scale), int((self.current_position[1]+y_shift)*scale)), 0.5*scale)
                pygame.draw.circle(screen, (255, 0, 0), (int((self.goal_data[self.agent_id][0]+x_shift)*scale), int((self.goal_data[self.agent_id][1]+y_shift)*scale)), 1*scale, 2)
        

        # 保存为图片文件
        pygame.image.save(screen, f'./gym_render/temp_pic/{int(self.frame_number/self.frame_interval)}.png')
    

    def generate_gif(self):
        # draw gif
        pic_list = sorted(os.listdir('./gym_render/temp_pic/'),key = lambda x:int(x[:-4]))
        with imageio.get_writer(uri=f'./gym_render/id_{self.agent_id}.gif', mode='i', fps=15) as writer:
            for pic in pic_list:
                writer.append_data(imageio.imread('./gym_render/temp_pic/'+pic))
        # delete all pictures
        for pic in pic_list:
            os.remove('./gym_render/temp_pic/'+pic)

    def compute_ADE(self):
        ade = mean(self.ADE_list)
        return ade
    
    def compute_FDE(self):
        fde = euclidean(self.new_traj[-1], self.old_traj[-1])
        return fde
    
    def compute_Frechet_Distance(self):
        P = self.old_traj
        Q = self.new_traj
        p_length = len(P)
        q_length = len(Q)
        distance_matrix = np.ones((p_length, q_length)) * -1

        # fill the first value with the distance between
        # the first two points in P and Q
        distance_matrix[0, 0] = euclidean(P[0], Q[0])

        # load the first column and first row with distances (memorize)
        for i in range(1, p_length):
            distance_matrix[i, 0] = max(distance_matrix[i - 1, 0], euclidean(P[i], Q[0]))
        for j in range(1, q_length):
            distance_matrix[0, j] = max(distance_matrix[0, j - 1], euclidean(P[0], Q[j]))

        for i in range(1, p_length):
            for j in range(1, q_length):
                distance_matrix[i, j] = max(
                    min(distance_matrix[i - 1, j], distance_matrix[i, j - 1], distance_matrix[i - 1, j - 1]),
                    euclidean(P[i], Q[j]))
        # distance_matrix[p_length - 1, q_length - 1]
        return distance_matrix[p_length-1,q_length-1]
    
    def compute_DTW(self):
        s1 = self.old_traj
        s2 = self.new_traj
        m = len(s1)
        n = len(s2)

        # 构建二位dp矩阵,存储对应每个子问题的最小距离
        dp = [[0]*n for _ in range(m)] 

        # 起始条件,计算单个字符与一个序列的距离
        for i in range(m):
            dp[i][0] = euclidean(s1[i],s2[0])
        for j in range(n):
            dp[0][j] = euclidean(s1[0],s2[j])
        
        # 利用递推公式,计算每个子问题的最小距离,矩阵最右下角的元素即位最终两个序列的最小值
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1]) + euclidean(s1[i],s2[j])
        
        return dp[-1][-1]
    
    def get_collisions(self):
        return self.collisions
    
    def get_companys(self):
        return self.companys

    def get_agent_id(self):
        return self.agent_id
    
    def get_target_circle_width(self):
        return self.target_circle_width

    def update_target_circle_width(self, fde):
        w = 0.4
        self.new_width = w*fde + (1-w)*self.new_width
        if self.new_width<self.target_circle_width:
            self.target_circle_width = float(format(self.new_width,'.1f'))
            print(f'Now env target circle width: {self.target_circle_width}')
        return self.target_circle_width

    def compute_log_kde(self, flow_ids):
        base_traj_list = []

        for id in flow_ids:
            base_traj_list.append(np.array(self.trajectories[id])[:,:2])

        target_traj = np.array(self.new_traj)[:,:2]


        return self._log_KDE(base_traj_list, target_traj, 5)
    
    def compute_traj_hausdorff_distance(self):
        # compute the hausdorff distance of states in traj
        hdd = 0
        len_traj = len(self.old_traj) if len(self.new_traj)>len(self.old_traj) else len(self.new_traj)


        # pairwise comparison
        for i in range(len_traj):
            new_pos = self.new_traj[i]
            new_state = self.get_haus_state(new_pos,i)
            gt_pos = self.old_traj[i]
            gt_state = self.get_haus_state(gt_pos,i)
            if isinstance(new_state,np.ndarray) and isinstance(gt_state,np.ndarray):
                hdd += hausdorff_distance(new_state, gt_state, distance='euclidean')
        return hdd

    def reach_goal(self):
        flag = False
        dist = np.linalg.norm(np.array(self.new_traj[-1]) - np.array(self.goal_data[self.agent_id]))
        # self.ADE_list.append(dist)
        if dist < self.target_circle_width:
            # to the target circle
            # reward += 10
            flag = True
        return flag



    def compute_geodesic_rate(self):
        # compute the project of new traj on old traj

        real_distance = np.array(self.goal_data[self.agent_id])- np.array(self.trajectories[self.agent_id][0][:2])
        real_len = np.linalg.norm(real_distance)

        new_distance = np.array(self.new_traj[-1])- np.array(self.trajectories[self.agent_id][0][:2])
        # compute the project length of new distance on real distance
        project_len = np.dot(new_distance, real_distance)/np.linalg.norm(real_distance)

        return project_len/real_len

    def theoretical_geodesic_rate(self):
        # compute the theoretical geodesic rate of the agent

        real_distance = np.array(self.goal_data[self.agent_id])- np.array(self.trajectories[self.agent_id][0][:2])
        real_len = np.linalg.norm(real_distance)

        the_distance = real_len - self.target_circle_width
        
        return the_distance/real_len
    
    def compute_remining_steps(self):
        # compute the remaining steps of the agent
        # return the remaining steps
        dist = np.linalg.norm(np.array(self.new_traj[-1]) - np.array(self.goal_data[self.agent_id]))
        if dist < self.target_circle_width:
            return 0
        else:
            # remaining distance
            AVG_SPEED = 1.2065383937121106  # average speed of the agent
            remaining_distance = dist - self.target_circle_width
            # remaining steps
            remaining_steps = int(remaining_distance / AVG_SPEED / self.time_interval) + 1
            return remaining_steps


    def get_collision_2type(self):
        return self.normal_collisions, self.ccd_collisions
        



    def compute_short_FD(self):
        # True ADE, fake FD
        len_traj = len(self.new_traj)

        ADE = 0
        for i in range(len_traj):
            ADE += np.linalg.norm(np.array(self.new_traj[i])-np.array(self.old_traj[i]))/len_traj

        return ADE

    def get_haus_state(self, position, frame_order, past_len=3):
        # init
        nodes = []
        x1, y1 = position
        frame_number = frame_order*self.frame_interval+self.trajectories[self.agent_id][0][2]

        if frame_number<=20:
            return None

        goal = np.array(self.goal_data[self.agent_id]) - np.array(position)
        r_matrix = self._rotate_matrix_v(goal)

        flag = False
        for other,ovalue in self.frame_data[frame_number].items():
            if other!=self.agent_id:
                x2 = ovalue[0]
                y2 = ovalue[1]
                dis = math.sqrt((x1-x2)**2+(y1-y2)**2)
                if dis<self.radius:
                    # create per person node
                    pernode = []
                    relative_pos=[x2-x1,y2-y1]
                    pernode.extend(np.dot(relative_pos,r_matrix))
                    for i in range(1,past_len):
                        if ((frame_number-i*self.frame_interval) in self.frame_data.keys()) and other in self.frame_data[frame_number-i*self.frame_interval].keys():
                            nx, ny = self.frame_data[frame_number-i*self.frame_interval][other][:2]
                            rela_pos_plus = [nx-x1, ny-y1]
                            pernode.extend(np.dot(rela_pos_plus,r_matrix))
                    # only assert complete node
                    if len(pernode)==(2*past_len):
                        flag = True
                        nodes.append(np.array(pernode))
        if flag:
            nodes = np.array(nodes,dtype=float)
            assert isinstance(nodes,np.ndarray)
            assert nodes.shape[1]==2*past_len
            assert nodes.ndim==2
            return nodes
        else:
            return None


    def get_short_traj(self):
        return self.new_traj


    def _valid_check_in_map(self):
        valid = True

        # get parameters
        current_x = self.current_position[0]
        current_y = self.current_position[1]
        low_x = self.map_size_bound[0]
        high_x = self.map_size_bound[1]
        low_y = self.map_size_bound[2]
        high_y = self.map_size_bound[3]

        # bound validation check
        if (current_x<low_x) or (current_x>high_x) or (current_y<low_y) or (current_y>high_y):
            valid = False

        # grid map obstacle check
        if not self._gridmap_check:
            valid = False

        return valid
    

    def _load_gridmap(self):
        # init grid map
        grid_map = {}
        x_length = math.ceil(self.map_size_bound[1]-self.map_size_bound[0])
        y_length = math.ceil(self.map_size_bound[3]-self.map_size_bound[2])
        for i in range(x_length*y_length):
            grid_map[i] = True
        return grid_map
    
    # 还没实现！
    def _gridmap_check(self):
        return True

    def _load_dataset(self, trajectories):
        # The dataset is organized as per person trajectory dict as form of pickle
        # fist load that then transfer to per frame form
        # trajectories contains the trajectory of every pedestrian, each waypoint is [x,y,frame_number]
        # with open(dataset_path, 'rb') as f:
        #     trajectories = pickle.load(f)

        for id, traj in trajectories.items():
            len_traj = len(traj)
            for i in range(1, len_traj - 1):
                past_flag = 0
                future_flag = 0
                if traj[i][2]-traj[i - 1][2] == self.frame_interval:
                    past_v_x = (traj[i][0] - traj[i - 1][0]) / self.time_interval
                    past_v_y = (traj[i][1] - traj[i - 1][1]) / self.time_interval
                else:
                    past_flag = 1
                if traj[i+1][2]-traj[i][2] == self.frame_interval:
                    future_v_x = (traj[i + 1][0] - traj[i][0]) / self.time_interval
                    future_v_y = (traj[i + 1][1] - traj[i][1]) / self.time_interval
                else:
                    future_flag = 1

                if past_flag == 0 and future_flag ==1:
                    future_v_x = past_v_x
                    future_v_y = past_v_y
                elif past_flag == 1 and future_flag ==0:
                    past_v_x = future_v_x
                    past_v_y = future_v_y

                traj[i].append(past_v_x)
                traj[i].append(past_v_y)
                traj[i].append(future_v_x)
                traj[i].append(future_v_y)
            traj[0].append(traj[1][3])
            traj[0].append(traj[1][4])
            traj[0].append(traj[1][3])
            traj[0].append(traj[1][4])
            traj[len_traj - 1].append(traj[len_traj - 2][5])
            traj[len_traj - 1].append(traj[len_traj - 2][6])
            traj[len_traj - 1].append(traj[len_traj - 2][5])
            traj[len_traj - 1].append(traj[len_traj - 2][6])

        # transfer to per frame dict
        # frame_data: key is frame id, value is all the pedestrains' waypoints
        frame_data = {}
        for id in list(trajectories.keys()):
            for waypoint in trajectories[id]:
                frame_data[waypoint[2]] = {}
        for id in list(trajectories.keys()):
            for waypoint in trajectories[id]:
                frame_data[waypoint[2]][id] = [waypoint[0], waypoint[1], \
                                               waypoint[3], waypoint[4], waypoint[5],waypoint[6]]

        # save each pedestrian's start position
        start_data={}
        for id in list(trajectories.keys()):
            info=trajectories[id][0]
            start_data[id]=[info[0],info[1]]

        # save each pedestrian's goal
        goal_data={}
        for id in list(trajectories.keys()):
            info=trajectories[id][-1]
            goal_data[id]=[info[0],info[1]]

        return trajectories, frame_data, start_data, goal_data
    

    # Choose the type of observation
    def _get_observation(self):
        # if self.observation_type=='radar':
        #     return self._get_relative_radar_observation()
        if self.observation_type=='graph':
            return self._get_relative_graph_observation()
        
    def _get_relative_radar_observation(self):
        # get the observation of current state

        # init state
        state=[0 for i in range(3*self.regions+2)]
        for i in range(self.regions):
            state[i*3] = self.radius


        # get the direction to goal
        dx, dy = self.goal_data[self.agent_id][0]-self.current_position[0], self.goal_data[self.agent_id][1]-self.current_position[1]
        relative_goal_dis = np.dot([dx,dy],self.r_matrix)
        state[-2], state[-1] = relative_goal_dis

        if len(self.new_traj) == 1:
            last_v = self.frame_data[self.frame_number][self.agent_id][2:4]
        else:
            last_v = (self.current_position - self.new_traj[-2])/self.time_interval

        
        # get nearby information
        for other,ovalue in self.frame_data[self.frame_number].items():
            if other!= self.agent_id:
                x2=ovalue[0]
                y2=ovalue[1]
                vec=np.dot([x2-self.current_position[0],y2-self.current_position[1]],self.r_matrix)
                dis=math.sqrt((self.current_position[0]-x2)**2+(self.current_position[1]-y2)**2)
                ang=self._angle_by_x(vec)
                region_order=int((ang*self.regions)/360)
                if dis<self.radius:
                    # state[region_order] = 1
                    if dis<state[3*region_order]:
                            state[3*region_order]=dis
                            orev = [ovalue[2]-last_v[0],ovalue[3]-last_v[1]]
                            orev = np.dot(orev,self.r_matrix)
                            state[3*region_order+1]=orev[0]
                            state[3*region_order+2]=orev[1]
        return np.array(state)

    def _get_radar_observation(self):
        # get the observation of current state

        # init state
        state = [0 for i in range(3*self.regions+2)]

        for i in range(self.regions):
            state[i*3] = self.radius


        # get the direction to goal
        dx, dy = self.goal_data[self.agent_id][0]-self.current_position[0], self.goal_data[self.agent_id][1]-self.current_position[1]
        if dx!=0 or dy!=0: # 归一化
            vx, vy = dx/math.sqrt(dx**2+dy**2),dy/math.sqrt(dx**2+dy**2)
            state[-2], state[-1] = vx, vy
        else:
            state[-2], state[-1] = 0, 0

        # with last_speed
        if self.with_last_speed:
            if len(self.new_traj) == 1:
                last_v = self.frame_data[self.frame_number][self.agent_id][2:4]
                # last_v = [0,0] # old edition
            else:
                last_v = (self.current_position - self.new_traj[-2])/self.time_interval
            state.extend(last_v)

        
        # get nearby information
        for other,ovalue in self.frame_data[self.frame_number].items():
            if other!= self.agent_id:
                x2=ovalue[0]
                y2=ovalue[1]
                vec=[x2-self.current_position[0],y2-self.current_position[1]]
                dis=math.sqrt((self.current_position[0]-x2)**2+(self.current_position[1]-y2)**2)
                ang=self._angle_by_x(vec)
                region_order=int((ang*self.regions)/360)
                if dis<self.radius:
                    if dis<state[3*region_order]:
                        state[3*region_order]=dis
                        state[3*region_order+1]=ovalue[2]
                        state[3*region_order+2]=ovalue[3]
        return np.array(state)

    def _get_relative_graph_observation(self):
        X=[]
        cluster=[]
        edge_index=[[],[]]

        # goal vector
        dx, dy= self.goal_data[self.agent_id][0]-self.current_position[0], self.goal_data[self.agent_id][1]-self.current_position[1]
        relative_goal_dis = np.dot([dx,dy],self.r_matrix)
        goal = relative_goal_dis

        # # 计算自身上一时刻速度
        # if len(self.new_traj) == 1:
        #     last_speed = [0, 0]
        # else:
        #     last_speed = (self.current_position - self.new_traj[-2])/self.time_interval

        # agent当前位置
        x1=self.current_position[0]
        y1=self.current_position[1]

        # Insert target person
        if len(self.new_traj) == 1:
            last_x = 0
            last_y = 0
        else:
            last_x = self.new_traj[-2][0]-x1
            last_y = self.new_traj[-2][1]-y1
        last_x, last_y = np.dot([last_x,last_y],self.r_matrix)
        X.append([last_x, last_y, 0, 0, 0])
        sum_ped = 0
        cluster.append(sum_ped)
        sum_ped += 1
        # Insert other person
        if self.frame_number in self.frame_data.keys():
            for other,ovalue in self.frame_data[self.frame_number].items():
                if other!=self.agent_id:
                    x2=ovalue[0]
                    y2=ovalue[1]
                    dis=math.sqrt((x1-x2)**2+(y1-y2)**2)
                    if dis<self.radius:  # 只考虑当前半径范围内的行人
                        # 每个行人最多取past_len个历史点向量
                        frame_id=self.frame_number
                        len_nodes=0
                        while_flag=0   # 一个不会让聚类编号凭空增加的flag
                        while(frame_id>=self.frame_interval and len_nodes<self.graph_obs_past_len and  ((frame_id-self.frame_interval) in self.frame_data.keys()) and (other in self.frame_data[frame_id-self.frame_interval])):
                            start_x=self.frame_data[frame_id-self.frame_interval][other][0]-x1
                            start_y=self.frame_data[frame_id-self.frame_interval][other][1]-y1
                            rela_start_x, rela_start_y = np.dot([start_x,start_y],self.r_matrix)

                            end_x=self.frame_data[frame_id][other][0]-x1
                            end_y=self.frame_data[frame_id][other][1]-y1
                            rela_end_x, rela_end_y = np.dot([end_x,end_y],self.r_matrix)

                            # 计算当前历史向量终点的相对agent的前后,前为1，后为0
                            front_flag=1
                            if len(self.new_traj) == 1:
                                last_v = goal
                            else:
                                last_v = (self.current_position - self.new_traj[-2])/self.time_interval
                            ang=self._counter_clockwise_angle(last_v, [end_x,end_y])
                            if ang>=90 and ang<=270:
                                front_flag=0

                            X.append([rela_start_x, rela_start_y, rela_end_x, rela_end_y, front_flag])
                            cluster.append(sum_ped)
                            while_flag=1

                            if len_nodes>0:
                                link_start=len(X)-1
                                edge_index[0].append(link_start)
                                edge_index[1].append(link_start-1)

                            len_nodes += 1
                            frame_id -= self.frame_interval
                        if while_flag == 1:
                            sum_ped += 1
        assert len(cluster)!=0
        X = np.array(X)
        cluster = np.array(cluster)
        valid_len = cluster.max()+1
        X = np.vstack([X, np.zeros((self.padd_to_number - cluster.max()-1, self.graph_feature_len), dtype=X.dtype)])
        cluster = np.hstack([cluster, np.arange(cluster.max()+1, self.padd_to_number)])
        g_data = GraphData(
            x=torch.tensor(X,dtype=torch.float32),
            cluster=torch.tensor(cluster,dtype=torch.int64),
            edge_index=torch.tensor(edge_index,dtype=torch.int64),
            valid_len=torch.tensor([valid_len]),
            time_step_len=torch.tensor([self.padd_to_number]),
            goal=torch.tensor(goal,dtype=torch.float32),
            # last_speed=torch.tensor(last_speed,dtype=torch.float32)
        )
        return g_data

    def _get_ar_relative_graph_observation(self, agent_id, r_matrix, frame_id, current_position, current_frame, past_frames):

        X=[]
        cluster=[]
        edge_index=[[],[]]

        # goal vector
        dx, dy= self.goal_data[agent_id][0]-current_position[0], self.goal_data[agent_id][1]-current_position[1]
        relative_goal_dis = np.dot([dx,dy],r_matrix)
        goal = relative_goal_dis

        # agent当前位置
        x1=current_position[0]
        y1=current_position[1]

        # Insert target person
        if len(past_frames.keys()) == 1 or (agent_id not in  past_frames[frame_id-self.frame_interval].keys()):
            last_x = 0
            last_y = 0
        else:
            last_x, last_y = (past_frames[frame_id-self.frame_interval][agent_id]-current_position)
        last_x, last_y = np.dot([last_x,last_y],r_matrix)
        X.append([last_x, last_y, 0, 0, 0])
        sum_ped = 0
        cluster.append(sum_ped)
        sum_ped += 1
        # Insert other person
        for other, ovalue in current_frame.items():
            if other!=agent_id:
                x2=ovalue[0]
                y2=ovalue[1]
                dis=math.sqrt((x1-x2)**2+(y1-y2)**2)
                if dis<self.radius:  # 只考虑当前半径范围内的行人
                    # 每个行人最多取past_len个历史点向量
                    cur_frame_id = frame_id
                    len_nodes=0
                    while_flag=0   # 一个不会让聚类编号凭空增加的flag
                    while(cur_frame_id>=self.frame_interval and len_nodes<self.graph_obs_past_len and  len_nodes<(len(past_frames.keys())-1) and (other in past_frames[cur_frame_id-self.frame_interval])):
                        start_x=past_frames[cur_frame_id-self.frame_interval][other][0]-x1
                        start_y=past_frames[cur_frame_id-self.frame_interval][other][1]-y1
                        rela_start_x, rela_start_y = np.dot([start_x,start_y],r_matrix)

                        end_x=past_frames[cur_frame_id][other][0]-x1
                        end_y=past_frames[cur_frame_id][other][1]-y1
                        rela_end_x, rela_end_y = np.dot([end_x,end_y],r_matrix)

                        # 计算当前历史向量终点的相对agent的前后,前为1，后为0
                        front_flag=1
                        if len(past_frames.keys()) == 1 or (agent_id not in  past_frames[frame_id-self.frame_interval].keys()):
                            last_v = goal
                        else:
                            last_v = (current_position - past_frames[frame_id-self.frame_interval][agent_id])/self.time_interval
                        ang=self._counter_clockwise_angle(last_v, [end_x,end_y])
                        if ang>=90 and ang<=270:
                            front_flag=0

                        X.append([rela_start_x, rela_start_y, rela_end_x, rela_end_y, front_flag])
                        cluster.append(sum_ped)
                        while_flag=1

                        if len_nodes>0:
                            link_start=len(X)-1
                            edge_index[0].append(link_start)
                            edge_index[1].append(link_start-1)

                        len_nodes += 1
                        cur_frame_id -= self.frame_interval
                    if while_flag == 1:
                        sum_ped += 1
        assert len(cluster)!=0
        X = np.array(X)
        cluster = np.array(cluster)
        valid_len = cluster.max()+1
        X = np.vstack([X, np.zeros((self.padd_to_number - cluster.max()-1, self.graph_feature_len), dtype=X.dtype)])
        cluster = np.hstack([cluster, np.arange(cluster.max()+1, self.padd_to_number)])
        g_data = GraphData(
            x=torch.tensor(X,dtype=torch.float32),
            cluster=torch.tensor(cluster,dtype=torch.int64),
            edge_index=torch.tensor(edge_index,dtype=torch.int64),
            valid_len=torch.tensor([valid_len]),
            time_step_len=torch.tensor([self.padd_to_number]),
            goal=torch.tensor(goal,dtype=torch.float32),
            # last_speed=torch.tensor(last_speed,dtype=torch.float32)
        )
        return g_data

    def _get_graph_observation(self):
        X=[]
        cluster=[]
        edge_index=[[],[]]

        # goal vector
        dx, dy= self.goal_data[self.agent_id][0]-self.current_position[0], self.goal_data[self.agent_id][1]-self.current_position[1]
        if dx!=0 or dy!=0: # 归一化
            vx,vy=dx/math.sqrt(dx**2+dy**2),dy/math.sqrt(dx**2+dy**2)
            goal=[vx,vy]
        else:
            goal=[0,0]

        # 添加自身上一时刻速度
        if len(self.new_traj) == 1:
            last_speed = [0, 0]
        else:
            last_speed = (self.current_position - self.new_traj[-2])/self.time_interval

        # agent当前位置
        x1=self.current_position[0]
        y1=self.current_position[1]

        # Insert target person
        if len(self.new_traj) == 1:
            last_x = 0
            last_y = 0
        else:
            last_x = self.new_traj[-2][0]-x1
            last_y = self.new_traj[-2][1]-y1
        X.append([last_x, last_y, 0, 0, 0])
        sum_ped = 0
        cluster.append(sum_ped)
        sum_ped += 1
        # Insert other person
        for other,ovalue in self.frame_data[self.frame_number].items():
            if other!=self.agent_id:
                x2=ovalue[0]
                y2=ovalue[1]
                dis=math.sqrt((x1-x2)**2+(y1-y2)**2)
                if dis<self.radius:  # 只考虑当前半径范围内的行人
                    # 每个行人最多取past_len个历史点向量
                    frame_id=self.frame_number
                    len_nodes=0
                    while_flag=0   # 一个不会让聚类编号凭空增加的flag
                    while(frame_id>=self.frame_interval and len_nodes<self.graph_obs_past_len and  (other in self.frame_data[frame_id-self.frame_interval])):
                        start_x=self.frame_data[frame_id-self.frame_interval][other][0]-x1
                        start_y=self.frame_data[frame_id-self.frame_interval][other][1]-y1
                        end_x=self.frame_data[frame_id][other][0]-x1
                        end_y=self.frame_data[frame_id][other][1]-y1

                        # 计算当前历史向量终点的相对agent的前后,前为1，后为0
                        front_flag=1
                        if len(self.new_traj) == 1:
                            last_v = goal
                        else:
                            last_v = last_speed
                        ang=self._clockwise_angle(last_v, [end_x,end_y])
                        if ang>=90 and ang<=270:
                            front_flag=0

                        X.append([start_x,start_y,end_x,end_y,front_flag])
                        cluster.append(sum_ped)
                        while_flag=1

                        if len_nodes>0:
                            link_start=len(X)-1
                            edge_index[0].append(link_start)
                            edge_index[1].append(link_start-1)

                        len_nodes += 1
                        frame_id -= self.frame_interval
                    if while_flag == 1:
                        sum_ped += 1
        assert len(cluster)!=0
        X = np.array(X)
        cluster = np.array(cluster)
        valid_len = cluster.max()+1
        X = np.vstack([X, np.zeros((self.padd_to_number - cluster.max()-1, self.graph_feature_len), dtype=X.dtype)])
        cluster = np.hstack([cluster, np.arange(cluster.max()+1, self.padd_to_number)])
        g_data = GraphData(
            x=torch.tensor(X,dtype=torch.float32),
            cluster=torch.tensor(cluster,dtype=torch.int64),
            edge_index=torch.tensor(edge_index,dtype=torch.int64),
            valid_len=torch.tensor([valid_len]),
            time_step_len=torch.tensor([self.padd_to_number]),
            goal=torch.tensor(goal,dtype=torch.float32),
            last_speed=torch.tensor(last_speed,dtype=torch.float32)
        )
        return g_data

    
    def _update_frame_number(self):
        self.frame_number += self.frame_interval

    # angle by x axis
    def _angle_by_x(self, v):
        x1,y1 = 1,0
        x2,y2 = v
        dot = x1*x2+y1*y2
        det = x1*y2-y1*x2
        theta = np.arctan2(det, dot)
        theta = theta if theta>0 else 2*np.pi+theta
        theta = theta*180/np.pi
        return theta%360
    
    # clockwise angle from v1 to v2
    def _counter_clockwise_angle(self,v1, v2):
        x1,y1 = v1
        x2,y2 = v2
        dot = x1*x2+y1*y2
        det = x1*y2-y1*x2
        theta = np.arctan2(det, dot)
        theta = theta if theta>=0 else 2*np.pi+theta
        theta = theta*180/np.pi
        return theta
    
    # 顺时针旋转，向量*矩阵
    def _rotate_matrix(self,angle):
        ang=angle*np.pi/180
        return np.array([[math.cos(ang),-math.sin(ang)],[math.sin(ang),math.cos(ang)]])

    # 向量要顺时针旋转到x轴正方向需要的旋转矩阵
    def _rotate_matrix_v(self, v):
        angle=self._angle_by_x(v)
        return self._rotate_matrix(angle)
    
    # 向量要顺时针旋转到x轴正方向需要的旋转矩阵de反
    def _counter_rotate_matrix_v(self, v):
        ang=self._angle_by_x(v)*np.pi/180
        return np.array([[math.cos(ang),math.sin(ang)],[-math.sin(ang),math.cos(ang)]])
    
    # abs_angle between v1 and v2
    def _abs_angle(self, v1, v2):
        x1,y1 = v1
        x2,y2 = v2
        dot = x1*x2+y1*y2
        det = x1*y2-y1*x2
        theta = np.arctan2(det, dot)
        # theta = theta if theta>=0 else 2*np.pi+theta
        theta = theta*180/np.pi
        return abs(theta)

    def _frame_continues_check(self):
        # check if trajectories is continues
        check = True
        if self.agent_id not in self.trajectories.keys():
            return False
        start_frame = self.trajectories[self.agent_id][0][2]
        end_frame = self.trajectories[self.agent_id][-1][2]
        for i in range(int(start_frame/self.frame_interval),int(end_frame/self.frame_interval)+1):
            if self.agent_id not in self.frame_data[i*self.frame_interval].keys():
                check = False
                break
        return check

    def _Frechet_Distance(self,P,Q):
        p_length = len(P)
        q_length = len(Q)
        distance_matrix = np.ones((p_length, q_length)) * -1

        # fill the first value with the distance between
        # the first two points in P and Q
        distance_matrix[0, 0] = euclidean(P[0], Q[0])

        # load the first column and first row with distances (memorize)
        for i in range(1, p_length):
            distance_matrix[i, 0] = max(distance_matrix[i - 1, 0], euclidean(P[i], Q[0]))
        for j in range(1, q_length):
            distance_matrix[0, j] = max(distance_matrix[0, j - 1], euclidean(P[0], Q[j]))

        for i in range(1, p_length):
            for j in range(1, q_length):
                distance_matrix[i, j] = max(
                    min(distance_matrix[i - 1, j], distance_matrix[i, j - 1], distance_matrix[i - 1, j - 1]),
                    euclidean(P[i], Q[j]))
        # distance_matrix[p_length - 1, q_length - 1]
        return distance_matrix[p_length-1,q_length-1]
    
    def _KDE(self, base_traj_list, target_traj, h):
        n = len(base_traj_list)
        pdf_value = 0
        for traj in (base_traj_list):
            fd = self._Frechet_Distance(traj,target_traj)
            fd /= h
            new_pdf = norm.pdf(fd, loc=0, scale=1)
            pdf_value += new_pdf
        return pdf_value/(n*h)

    def _log_KDE(self, base_traj_list, target_traj, h):
        return np.log(self._KDE(base_traj_list, target_traj, h))



    # manual_reward: special for diversity_aware RL
    def _manual_reward_done(self):
        # # init value
        reward = 0.0
        done = False


        # closer to goal reward
        goal = np.array(self.goal_data[self.agent_id])
        cur_dist = np.linalg.norm(np.array(self.current_position) - goal)
        if len(self.new_traj) >= 2:
            prev_dist = np.linalg.norm(np.array(self.new_traj[-2]) - goal)
            delta = prev_dist - cur_dist
            reward += 2.5 * delta # 2.5 # 2.0
            done = False # False # False
        
        ## collision reward

        collided = self._ccd_collision_detection()
        _ = self._normal_collision_detection()

        # _ = self._ccd_collision_detection()
        # collided = self._normal_collision_detection()

        if collided:
            reward += -5 # -15 # -20.0
            done = False # False # True
            return float(reward), done
        
        # reach goal reward
        goal = np.array(self.goal_data[self.agent_id])
        cur_dist = np.linalg.norm(np.array(self.current_position) - goal)
        if cur_dist < self.target_circle_width:
            reward += 15 # 15 # 10.0
            done = True # True # True
            return float(reward), done
            
        return float(reward), done

    # collision detection using CCD
    def _ccd_collision_detection(self):
        collided = False
        if self.frame_number in self.frame_data.keys():
            pa0 = np.array(self.new_traj[-2]) if len(self.new_traj) >= 2 else np.array(self.current_position)
            pa1 = np.array(self.current_position)
            prev_frame = self.frame_number - self.frame_interval

            # collect current positions excluding self
            others = [pid for pid in self.frame_data[self.frame_number].keys() if pid != self.agent_id]
            if len(others) == 0:
                return False
            cur_pos = np.array([[self.frame_data[self.frame_number][pid][0], self.frame_data[self.frame_number][pid][1]] for pid in others])

            # previous positions (fallback to current if missing)
            if prev_frame in self.frame_data.keys():
                prev_pos = np.array([
                    [self.frame_data[prev_frame][pid][0], self.frame_data[prev_frame][pid][1]] if pid in self.frame_data[prev_frame].keys() else [cur_pos[i,0], cur_pos[i,1]]
                    for i, pid in enumerate(others)
                ])
            else:
                prev_pos = cur_pos.copy()

            pa0_mat = np.repeat(pa0.reshape(1,2), len(others), axis=0)
            pa1_mat = np.repeat(pa1.reshape(1,2), len(others), axis=0)
            p0 = pa0_mat - prev_pos
            v = (pa1_mat - pa0_mat) - (cur_pos - prev_pos)

            vv = np.einsum('ij,ij->i', v, v)
            dot_p0v = np.einsum('ij,ij->i', p0, v)

            mask_static = np.isclose(vv, 0.0)
            t_star = np.zeros_like(vv)
            valid = ~mask_static
            t_star[valid] = -dot_p0v[valid] / vv[valid]
            t_star = np.clip(t_star, 0.0, 1.0)

            p_min = p0 + (t_star[:, None] * v)
            min_dist = np.linalg.norm(p_min, axis=1)
            if (min_dist < 2 * self.agent_radius).any():
                collided = True
        if collided:
            self.ccd_collisions += 1
        return collided

    def _normal_collision_detection(self):
        collided = False
        # collision check
        if self.frame_number in self.frame_data.keys():
            for other, opos in self.frame_data[self.frame_number].items():
                if self.agent_id != other:
                    dist = np.linalg.norm(np.array(self.current_position) - np.array([opos[0],opos[1]]))
                    if dist < 2*self.agent_radius:
                        collided = True
                        break
        if collided:
            self.normal_collisions += 1
        return collided
























































































class CrowdRenderEnv():  
    def __init__(self, args):
        super(CrowdRenderEnv, self).__init__()

        self.time_interval = args.time_interval
        self.trajectories, self.frame_data, self.start_data, self.goal_data = self._load_dataset(args.dataset_path)
        self.regions = args.regions
        self.radius = args.radius
        self.num_agents = len(self.trajectories.keys())
        self.frame_interval = args.frame_interval
        self.map_size_bound = args.map_size_bound # [low_x, high_x, low_y, high_y]
        self.with_last_speed = args.with_last_speed
        self.observation_type = args.observation_type # 'radar' / 'graph'
        self.entity_type = args.entity_type
        if self.observation_type == 'graph':
            self.graph_obs_past_len = args.graph_obs_past_len
            self.padd_to_number = args.padd_to_number
            self.graph_feature_len = args.graph_feature_len
        
    def render_segment(self, policy, start_frame, end_frame):
        # validity check
        assert start_frame>= 0 and start_frame%self.frame_interval==0
        assert end_frame<=max(self.frame_data.keys()) and end_frame%self.frame_interval==0

        # cut init frame
        current_frame = self.frame_data[start_frame]
        past_frames = {}
        # save each agent's quit time
        # agent_quit_frame = {}
        # for id in present_agent:
        #     agent_quit_frame[id] = self.trajectories[id][-1][2]

        # start render
        dy_coll = 0

        for frame_id in tqdm(range(start_frame, end_frame+1, self.frame_interval)):
            # 获取在场行人
            present_agent = self.frame_data[frame_id].keys()
            # 添加新行人进当前帧, 删除旧行人
            last_present_agent = list(current_frame.keys())
            for id in present_agent:
                if id not in last_present_agent:
                    current_frame[id] = self.frame_data[frame_id][id]
            for id in last_present_agent:
                if id not in present_agent:
                    del current_frame[id]
            # 压栈
            past_frames[frame_id] = current_frame
            # draw
            dy_coll += self.draw_one_frame(frame_id, current_frame)
            # generate next frame
            next_frame = {}
            for id in present_agent:
                current_position = np.array([current_frame[id][0], current_frame[id][1]])

                state = self._get_graph_observation(id, frame_id, current_position, current_frame, past_frames)
                action = policy.select_action(state)

                dx = action[0]*self.time_interval
                dy = action[1]*self.time_interval
                new_position = current_position + np.array([dx, dy])

                # collision check

                next_frame[id] = new_position
            current_frame = next_frame
        print(f'Dynamic Collisions: {dy_coll}')
        self.generate_gif('dynamic')
        # draw the old to compare
        sta_coll = 0
        for frame_id in tqdm(range(start_frame, end_frame+1, self.frame_interval)):
            coll = self.draw_one_frame(frame_id, self.frame_data[frame_id])
            sta_coll += coll
        print(f'Static Collisions: {sta_coll}')
        self.generate_gif('static')



    def draw_one_frame(self, frame_id, current_frame):
        # 创建 Pygame Surface 对象
        x_shift, y_shift = 12, 15
        scale = 10
        screen_width, screen_height = 60*scale, 60*scale
        screen = pygame.Surface((screen_width, screen_height))

        screen.fill((255, 255, 255))  # clear Surface

        collisions = 0

        for person, value in current_frame.items():
            agent_color = (62, 115, 158)
            # check collision
            for other, opos in current_frame.items():
                if person != other:
                    dist = np.linalg.norm(np.array([value[0], value[1]]) - np.array([opos[0], opos[1]]))
                    if dist < 1.0:
                        agent_color = (255, 0, 0)
                        collisions += 1
            # draw the agent
            agent_x, agent_y = value[0]+x_shift, value[1]+y_shift
            pygame.draw.circle(screen, agent_color, (int(agent_x*scale), int(agent_y*scale)), 0.5*scale)

        # 保存为图片文件
        pygame.image.save(screen, f'./gym_render/temp_pic/{int(frame_id/self.frame_interval)}.png')

        return collisions

    def generate_gif(self, file_name):
        # draw gif
        pic_list = sorted(os.listdir('./gym_render/temp_pic/'),key = lambda x:int(x[:-4]))
        with imageio.get_writer(uri=f'./gym_render/render_all/{file_name}.gif', mode='i', fps=15) as writer:
            for pic in pic_list:
                writer.append_data(imageio.imread('./gym_render/temp_pic/'+pic))
        # delete all pictures
        for pic in pic_list:
            os.remove('./gym_render/temp_pic/'+pic)


    def _get_graph_observation(self, agent_id, frame_id, current_position, current_frame, past_frames):
        X=[]
        cluster=[]
        edge_index=[[],[]]

        # goal vector
        dx, dy= self.goal_data[agent_id][0]-current_position[0], self.goal_data[agent_id][1]-current_position[1]
        if dx!=0 or dy!=0: # 归一化
            vx,vy=dx/math.sqrt(dx**2+dy**2),dy/math.sqrt(dx**2+dy**2)
            goal=[vx,vy]
        else:
            goal=[0,0]

        # 添加自身上一时刻速度
        if len(past_frames.keys()) == 1 or (agent_id not in  past_frames[frame_id-self.frame_interval].keys()):
            last_speed = [0, 0]
        else:
            last_speed = (current_position - past_frames[frame_id-self.frame_interval][agent_id])/self.time_interval

        # agent当前位置
        x1=current_position[0]
        y1=current_position[1]

        # Insert target person
        if len(past_frames.keys()) == 1 or (agent_id not in  past_frames[frame_id-self.frame_interval].keys()):
            last_x = 0
            last_y = 0
        else:
            last_x = past_frames[frame_id-self.frame_interval][agent_id][0]-x1
            last_y = past_frames[frame_id-self.frame_interval][agent_id][1]-y1
        X.append([last_x, last_y, 0, 0, 0])
        sum_ped = 0
        cluster.append(sum_ped)
        sum_ped += 1
        # Insert other person
        for other,ovalue in current_frame.items():
            if other!=agent_id:
                x2=ovalue[0]
                y2=ovalue[1]
                dis=math.sqrt((x1-x2)**2+(y1-y2)**2)
                if dis<self.radius:  # 只考虑当前半径范围内的行人
                    # 每个行人最多取past_len个历史点向量
                    cur_frame_id = frame_id
                    len_nodes=0
                    while_flag=0   # 一个不会让聚类编号凭空增加的flag
                    while(cur_frame_id>=self.frame_interval and len_nodes<self.graph_obs_past_len and  len(past_frames.keys())>1 and (other in past_frames[cur_frame_id-self.frame_interval])):
                        start_x=past_frames[cur_frame_id-self.frame_interval][other][0]-x1
                        start_y=past_frames[cur_frame_id-self.frame_interval][other][1]-y1
                        end_x=past_frames[cur_frame_id][other][0]-x1
                        end_y=past_frames[cur_frame_id][other][1]-y1

                        # 计算当前历史向量终点的相对agent的前后,前为1，后为0
                        front_flag=1
                        if len(past_frames.keys()) == 1:
                            last_v = goal
                        else:
                            last_v = last_speed
                        ang = self._clockwise_angle(last_v, [end_x,end_y])
                        if ang>=90 and ang<=270:
                            front_flag=0

                        X.append([start_x,start_y,end_x,end_y,front_flag])
                        cluster.append(sum_ped)
                        while_flag=1

                        if len_nodes>0:
                            link_start=len(X)-1
                            edge_index[0].append(link_start)
                            edge_index[1].append(link_start-1)

                        len_nodes += 1
                        cur_frame_id -= self.frame_interval
                    if while_flag == 1:
                        sum_ped += 1
        assert len(cluster)!=0
        X = np.array(X)
        cluster = np.array(cluster)
        valid_len = cluster.max()+1
        X = np.vstack([X, np.zeros((self.padd_to_number - cluster.max()-1, self.graph_feature_len), dtype=X.dtype)])
        cluster = np.hstack([cluster, np.arange(cluster.max()+1, self.padd_to_number)])
        g_data = GraphData(
            x=torch.tensor(X,dtype=torch.float32),
            cluster=torch.tensor(cluster,dtype=torch.int64),
            edge_index=torch.tensor(edge_index,dtype=torch.int64),
            valid_len=torch.tensor([valid_len]),
            time_step_len=torch.tensor([self.padd_to_number]),
            goal=torch.tensor(goal,dtype=torch.float32),
            last_speed=torch.tensor(last_speed,dtype=torch.float32)
        )
        return g_data
    
    # clockwise angle from v1 to v2
    def _clockwise_angle(self,v1, v2):
        x1,y1 = v1
        x2,y2 = v2
        dot = x1*x2+y1*y2
        det = x1*y2-y1*x2
        theta = np.arctan2(det, dot)
        theta = theta if theta>0 else 2*np.pi+theta
        theta = theta*180/np.pi
        return theta

    def _load_dataset(self, dataset_path):
        # The dataset is organized as per person trajectory dict as form of pickle
        # fist load that then transfer to per frame form
        # trajectories contains the trajectory of every pedestrian, each waypoint is [x,y,frame_number]
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

        # delet pedestrians with only one or two waypoints
        for id in list(trajectories.keys()):
            if len(trajectories[id]) < 3:
                del trajectories[id]
        print(f'dataset pedestrians:{len(trajectories)}')

        # calculate last speed and next speed
        for traj in trajectories.values():
            len_traj = len(traj)
            for i in range(1, len_traj - 1):
                past_v_x = (traj[i][0] - traj[i - 1][0]) / self.time_interval
                past_v_y = (traj[i][1] - traj[i - 1][1]) / self.time_interval
                future_v_x = (traj[i + 1][0] - traj[i][0]) / self.time_interval
                future_v_y = (traj[i + 1][1] - traj[i][1]) / self.time_interval
                traj[i].append(past_v_x)
                traj[i].append(past_v_y)
                traj[i].append(future_v_x)
                traj[i].append(future_v_y)
            traj[0].append(traj[1][3])
            traj[0].append(traj[1][4])
            traj[0].append(traj[1][3])
            traj[0].append(traj[1][4])
            traj[len_traj - 1].append(traj[len_traj - 2][5])
            traj[len_traj - 1].append(traj[len_traj - 2][6])
            traj[len_traj - 1].append(traj[len_traj - 2][5])
            traj[len_traj - 1].append(traj[len_traj - 2][6])

        # transfer to per frame dict
        # frame_data: key is frame id, value is all the pedestrains' waypoints
        frame_data = {}
        for id in list(trajectories.keys()):
            for waypoint in trajectories[id]:
                frame_data[waypoint[2]] = {}
        for id in list(trajectories.keys()):
            for waypoint in trajectories[id]:
                frame_data[waypoint[2]][id] = np.array([waypoint[0], waypoint[1]])
                                               # waypoint[3], waypoint[4], waypoint[5],waypoint[6]]

        # save each pedestrian's start position
        start_data={}
        for id in list(trajectories.keys()):
            info=trajectories[id][0]
            start_data[id]=[info[0],info[1]]

        # save each pedestrian's goal
        goal_data={}
        for id in list(trajectories.keys()):
            info=trajectories[id][-1]
            goal_data[id]=[info[0],info[1]]

        return trajectories, frame_data, start_data, goal_data

















if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args for SocialGAIL')
    parser.add_argument('--env_name',default="CrowdEnv")
    parser.add_argument('--dataset_path',default="./datasets/gc_interpolated_trajectory.pkl")
    parser.add_argument('--frame_interval',default=10)
    parser.add_argument('--time_interval',default=0.4)
    parser.add_argument('--regions',default=16)
    parser.add_argument('--radius',default=6.0)
    parser.add_argument('--map_size_bound',default=[-10,40,-20,50])         # [low_x, high_x, low_y, high_y] (int)
    parser.add_argument('--with_last_speed',default=True)
    parser.add_argument('--entity_type',default=False)
    parser.add_argument('--observation_type',default='radar')               # 'radar' / 'graph'
    parser.add_argument('--graph_obs_past_len',default=5)
    parser.add_argument('--padd_to_number',default=60)                      # the max number of people in radius to form a mini-batch
    parser.add_argument('--graph_feature_len',default=5)
    parser.add_argument('--output_len',default=2)
    # training hyperparamater
    parser.add_argument('--max_timesteps',default=800)                      # max time steps in one episode
    parser.add_argument('--total_steps',default=8e5)                        # int(1.6e6) 800000
    parser.add_argument('--training_interval',default=2048)                  # 2048
    parser.add_argument('--n_iter',default=10)                               # updates per epoch    # 10
    parser.add_argument('--batch_size',default=64)                          # num of transitions sampled from expert    # 64
    parser.add_argument('--action_std_decay_rate',default=0.05)             # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    parser.add_argument('--min_action_std',default=0.1)                     # minimum action_std (stop decay after action_std <= min_action_std)
    parser.add_argument('--action_std_decay_freq',default=int(2.5e5))       # action_std decay frequency (in num timesteps)
    parser.add_argument('--pretrained_model_path',default='./pretrained_models/GC_GNN.pt')
    args = parser.parse_args()


    # 用于检查自定义的gym环境
    # 如果你安装了pytorch，则使用上面的，如果你安装了tensorflow，则使用from stable_baselines.common.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env 
    env = CrowdEnv(args)
    check_env(env)




    # from stable_baselines3 import PPO
    # env = CrowdEnv(args)
    
    # model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./PPO/")
    # model.learn(total_timesteps=5000000)   # 400000

    # obs = env.reset()
    # # 验证十次
    # for i in range(400):
    #     action, state = model.predict(observation=obs)
    #     obs, reward, done, info = env.step(action)
    #     env.render(i)
    #     if done:
    #         obs = env.reset()
    # env.draw_gif()
