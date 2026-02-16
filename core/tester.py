import os
from time import time, sleep
from datetime import timedelta
import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import random
import numpy as np

class Tester:

    def __init__(self, env_test, algo, model_dir, seed=0):
        super().__init__()


        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2**31-seed)

        self.algo = algo
        self.model_dir = model_dir


    def semi_info_flow_test(self, flow_ids, dim_c):
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        log_kde = 0

        for test_id in tqdm(flow_ids):
            episode_log_kde = -np.inf

            for i in range(dim_c):
                # create current latent_c
                latent_c = torch.zeros(dim_c, device=self.algo.device)
                latent_c[i] = 1.0
                self.algo.latent_code_value = torch.cat([latent_c], dim=0).to(self.algo.device)

                state = self.env_test.reset(person_id=test_id)
                done = False

                while (not done):
                    action = self.algo.exploit(state,self.algo.latent_code_value)
                    # action, _ = self.algo.explore(state,self.algo.latent_code_value) # big bug!

                    state, reward, done, _ = self.env_test.step(action)
                
                episode_log_kde = max(episode_log_kde, self.env_test.compute_log_kde(flow_ids))

            log_kde += episode_log_kde
        
        return log_kde


    def semi_info_test_ffh_once(self, dim_c, test_ids):
        # ## get test_ids
        # with open(args.dataset_path, 'rb') as f:
        #     dataset = pickle.load(f)
        # test_ids = dataset['test_ids']

        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        # begin test
        # SCALE_FACTOR = 2.0
        frechet_distance = 0.0
        fde = 0.0
        hausdorff_distance = 0.0

        

        for id in tqdm(test_ids):
            # init 
            episode_fd = np.inf
            episode_fde = np.inf
            episode_hdd = np.inf
            
            for i in range(dim_c):
                # create current latent_c
                latent_c = torch.zeros(dim_c, device=self.algo.device)
                latent_c[i] = 1.0
                self.algo.latent_code_value = torch.cat([latent_c], dim=0).to(self.algo.device)

                state = self.env_test.reset(person_id=id)
                done = False

                while (not done):
                    action = self.algo.exploit(state,self.algo.latent_code_value)
                    # action, _ = self.algo.explore(state,self.algo.latent_code_value) # big bug!

                    state, reward, done, _ = self.env_test.step(action)
                
                episode_fd = min(episode_fd,self.env_test.compute_Frechet_Distance())
                episode_fde = min(episode_fde,self.env_test.compute_FDE())
                episode_hdd = min(episode_hdd,self.env_test.compute_traj_hausdorff_distance())
            
            frechet_distance += episode_fd
            fde += episode_fde
            hausdorff_distance += episode_hdd
                
        return frechet_distance, fde, hausdorff_distance



    def semi_info_test_ffh(self, num_ids, dim_c, test_subdir, args):
        ## get test_ids
        with open(args.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        test_ids = dataset['test_ids']

        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        # begin test
        # SCALE_FACTOR = 2.0
        frechet_distance = 0.0
        fde = 0.0
        hausdorff_distance = 0.0

        
        # num_ids = len(test_ids) # all test
        for order in tqdm(range(num_ids)):
            # init 
            episode_fd = np.inf
            episode_fde = np.inf
            episode_hdd = np.inf
            
            for i in range(dim_c):
                # create current latent_c
                latent_c = torch.zeros(dim_c, device=self.algo.device)
                latent_c[i] = 1.0
                self.algo.latent_code_value = torch.cat([latent_c], dim=0).to(self.algo.device)

                state = self.env_test.reset(person_id=test_ids[order])
                done = False

                while (not done):
                    action = self.algo.exploit(state,self.algo.latent_code_value)
                    # action, _ = self.algo.explore(state,self.algo.latent_code_value) # big bug！

                    state, reward, done, _ = self.env_test.step(action)
                
                episode_fd = min(episode_fd,self.env_test.compute_Frechet_Distance())
                episode_fde = min(episode_fde,self.env_test.compute_FDE())
                episode_hdd = min(episode_hdd,self.env_test.compute_traj_hausdorff_distance())
            
            frechet_distance += episode_fd/num_ids
            fde += episode_fde/num_ids
            hausdorff_distance += episode_hdd/num_ids
                
                
        
        # save log
        print(f'frechet_distance: {frechet_distance}')
        print(f'fde: {fde}')
        print(f'hausdorff_distance: {hausdorff_distance}')
        with open(test_subdir + f"logs/ffh_log_D{dim_c}_{num_ids}_{args.algo}.pkl",'wb') as f:
            pickle.dump({'frechet_distance':frechet_distance,
                         'fde':fde,
                         'hausdorff_distance':hausdorff_distance}, f)
                        
        

    def semi_info_test_social_gradient(self, num_ids, dim_c, test_subdir, args):
        ## get test_ids
        with open(args.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        test_ids = dataset['test_ids']

        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        # begin test
        SCALE_FACTOR = 2.0

        social_gradient_log  = {}
        trajectory_log = {}

        for order in tqdm(range(num_ids)):
            social_gradient_log[order] = {}
            trajectory_log[order] = {}
            for i in range(dim_c):
                # init 
                social_gradient_list = []

                # create current latent_c
                latent_c = torch.zeros(dim_c, device=self.algo.device)
                latent_c[i] = 1.0
                self.algo.latent_code_value = torch.cat([latent_c], dim=0).to(self.algo.device)

                state = self.env_test.reset(person_id=test_ids[order])
                done = False

                while (not done):
                    # action = self.algo.exploit(state,self.algo.latent_code_value)
                    action, _ = self.algo.explore(state,self.algo.latent_code_value)

                    social_gradient_list.append(self.env_test.get_social_gradient(action))

                    state, reward, done, _ = self.env_test.step(action)
                
                social_gradient_log[order][i] = social_gradient_list

                trajectory_log[order][i] = self.env_test.get_short_traj()
        
        # save social_gradient log
        with open(test_subdir + f"logs/social_gradient_log_D{dim_c}_{num_ids}_{args.algo}.pkl",'wb') as f:
            pickle.dump(social_gradient_log, f)
        
        # save trajectory log
        with open(test_subdir + f"logs/trajectory_log_D{dim_c}_{num_ids}_{args.algo}.pkl",'wb') as f:
            pickle.dump(trajectory_log, f)

    def semi_info_test_social_comfort(self, num_ids, dim_c, test_subdir, args):
        ## get test_ids
        with open(args.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        test_ids = dataset['test_ids']

        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        # begin test
        SCALE_FACTOR = 2.0

        social_comfort_log  = {}
        trajectory_log = {}

        for order in tqdm(range(num_ids)):
            social_comfort_log[order] = {}
            trajectory_log[order] = {}
            for i in range(dim_c):
                # init 
                social_comfort_list = []

                # create current latent_c
                latent_c = torch.zeros(dim_c, device=self.algo.device)
                latent_c[i] = 1.0
                self.algo.latent_code_value = torch.cat([latent_c], dim=0).to(self.algo.device)

                state = self.env_test.reset(person_id=test_ids[order])
                done = False

                while (not done):
                    # action = self.algo.exploit(state,self.algo.latent_code_value)
                    action, _ = self.algo.explore(state,self.algo.latent_code_value)

                    social_comfort_list.append(self.env_test.get_social_comfort(action).item())

                    state, reward, done, _ = self.env_test.step(action)
                
                social_comfort_log[order][i] = social_comfort_list

                trajectory_log[order][i] = self.env_test.get_short_traj()
        
        # save social_comfort log
        with open(test_subdir + f"logs/social_comfort_log_D{dim_c}_{num_ids}_{args.algo}.pkl",'wb') as f:
            pickle.dump(social_comfort_log, f)
        
        # save trajectory log
        with open(test_subdir + f"logs/trajectory_log_D{dim_c}_{num_ids}_{args.algo}.pkl",'wb') as f:
            pickle.dump(trajectory_log, f)

    def semi_info_test_nn_dist(self, num_ids, dim_c, test_subdir, args):
        ## get test_ids
        with open(args.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        test_ids = dataset['test_ids']

        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        # begin test
        SCALE_FACTOR = 2.0

        nn_dist_log  = {}

        for order in tqdm(range(num_ids)):
            nn_dist_log[order] = {}
            for i in range(dim_c):
                # init 
                nn_dist_list = []

                # create current latent_c
                latent_c = torch.zeros(dim_c, device=self.algo.device)
                latent_c[i] = 1.0
                self.algo.latent_code_value = torch.cat([latent_c], dim=0).to(self.algo.device)

                state = self.env_test.reset(person_id=test_ids[order])
                done = False

                while (not done):
                    action = self.algo.exploit(state,self.algo.latent_code_value)
                    state, reward, done, _ = self.env_test.step(action)

                    nn_dist_list.append(self.env_test.get_nn_dist(within_radius=False).item())
                
                nn_dist_log[order][i] = nn_dist_list
        
        # save speed log
        with open(test_subdir + f"logs/nn_dist_log_D{dim_c}_{num_ids}_{args.algo}.pkl",'wb') as f:
            pickle.dump(nn_dist_log, f)



    def semi_info_test_speed_synthetic(self, num_ids, dim_c, test_subdir, args):
        ## get test_ids
        with open(args.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        test_ids = dataset['test_ids']

        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        # begin test
        SCALE_FACTOR = args.scale_factor
        speed_log  = {}
        trajectory_log = {}

        for order in tqdm(range(num_ids)):
            speed_log[order] = {}
            trajectory_log[order] = {}
            for i in range(dim_c):
                # init 
                speed_list = []

                # create current latent_c
                latent_c = torch.zeros(dim_c, device=self.algo.device)
                latent_c[i] = 1.0
                self.algo.latent_code_value = torch.cat([latent_c], dim=0).to(self.algo.device)

                state = self.env_test.reset(person_id=test_ids[order%len(test_ids)])
                done = False

                while (not done):
                    # action = self.algo.exploit(state,self.algo.latent_code_value)
                    action, _ = self.algo.explore(state,self.algo.latent_code_value)

                    state, reward, done, _ = self.env_test.step(action)

                    speed_list.append(SCALE_FACTOR * np.linalg.norm(action))
                
                speed_log[order][i] = speed_list
                trajectory_log[order][i] = self.env_test.get_short_traj()
                
        
        # save speed log
        with open(test_subdir + f"logs/speed_log_D{dim_c}_{num_ids}_{args.algo}.pkl",'wb') as f:
            pickle.dump(speed_log, f)

        # save trajectory log
        with open(test_subdir + f"logs/trajectory_log_D{dim_c}_{num_ids}_{args.algo}.pkl",'wb') as f:
            pickle.dump(trajectory_log, f)




    def semi_info_test_speed(self, num_ids, dim_c, test_subdir, args):
        ## get test_ids
        with open(args.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        test_ids = dataset['test_ids']

        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        # begin test
        SCALE_FACTOR = 2.0
        speed_log  = {}

        for order in tqdm(range(num_ids)):
            speed_log[order] = {}
            for i in range(dim_c):
                # init 
                speed_list = []

                # create current latent_c
                latent_c = torch.zeros(dim_c, device=self.algo.device)
                latent_c[i] = 1.0
                self.algo.latent_code_value = torch.cat([latent_c], dim=0).to(self.algo.device)

                state = self.env_test.reset(person_id=test_ids[order])
                done = False

                while (not done):
                    action = self.algo.exploit(state,self.algo.latent_code_value)
                    state, reward, done, _ = self.env_test.step(action)

                    speed_list.append(SCALE_FACTOR * np.linalg.norm(action))
                
                speed_log[order][i] = speed_list
                
        # print(f"Average Speeds under different latent codes:")
        # for order in speed_log.keys():
        #     for i in speed_log[order].keys():
        #         print(f"Order {order}, Latent Code {i}: {sum(speed_log[order][i])/len(speed_log[order][i]):.3f} m/s")
        
        # save speed log
        with open(test_subdir + f"logs/speed_log_D{dim_c}_{num_ids}_{args.algo}.pkl",'wb') as f:
            pickle.dump(speed_log, f)
        

        
    


        


    def test(self):
        # test id list
        with open('./core/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        mean_return = 0.0
        frechet_distance = 0.0
        fde = 0.0

        for id in tqdm(test_ids):   
            state = self.env_test.reset(person_id=id)
            episode_return = 0.0
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            mean_return += episode_return 
            frechet_distance += self.env_test.compute_Frechet_Distance() 
            fde += self.env_test.compute_FDE()     

        mean_return /= len(test_ids)
        frechet_distance /= len(test_ids)
        fde /= len(test_ids)

        return mean_return, frechet_distance, fde
    
    def ffh_test_multi(self, dim_c, test_ids):
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        frechet_distance = 0.0
        fde = 0.0
        hausdorff_distance = 0.0

        # test_ids = test_ids[:20]

        for id in tqdm(test_ids):
            episode_fd = float("inf")
            episode_fde = float("inf")
            episode_hdd = float("inf")

            for i in range(dim_c):
                if hasattr(self.algo,'latent_c'):
                    self.algo.latent_c = torch.zeros(self.algo.dim_c, device=self.algo.device)
                    self.algo.latent_c[i] = 1.0

                self.env_test.reset(person_id=id)
                self.env_test.step_multi(self.algo, self.algo.latent_c)
                
                episode_fd = min(episode_fd,self.env_test.compute_Frechet_Distance())
                episode_fde = min(episode_fde,self.env_test.compute_FDE())
                episode_hdd = min(episode_hdd,self.env_test.compute_traj_hausdorff_distance())

            frechet_distance += episode_fd
            fde += episode_fde
            hausdorff_distance += episode_hdd

        return frechet_distance, fde, hausdorff_distance

    def ffh_test(self, dim_c, test_ids):
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        frechet_distance = 0.0
        fde = 0.0
        hausdorff_distance = 0.0

        # test_ids = test_ids[:20]

        for id in tqdm(test_ids):
            episode_fd = float("inf")
            episode_fde = float("inf")
            episode_hdd = float("inf")

            for i in range(dim_c):
                if hasattr(self.algo,'latent_c'):
                    self.algo.latent_c = torch.zeros(self.algo.dim_c, device=self.algo.device)
                    self.algo.latent_c[i] = 1.0
                state = self.env_test.reset(person_id=id)
                done = False

                while (not done):
                    if hasattr(self.algo,'latent_c'):
                        action = self.algo.exploit(state,self.algo.latent_c)
                    else:
                        action = self.algo.exploit(state)
                    state, reward, done, _ = self.env_test.step(action)
                
                episode_fd = min(episode_fd,self.env_test.compute_Frechet_Distance())
                episode_fde = min(episode_fde,self.env_test.compute_FDE())
                episode_hdd = min(episode_hdd,self.env_test.compute_traj_hausdorff_distance())

            frechet_distance += episode_fd
            fde += episode_fde
            hausdorff_distance += episode_hdd

        return frechet_distance, fde, hausdorff_distance

    def info_test(self, dim_c, test_ids):
        # test id list
        # with open('./core/test_people.pkl','rb') as f:
        #     test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        frechet_distance = 0.0
        fde = 0.0

        # test_ids = test_ids[:20]

        for id in tqdm(test_ids):
            episode_fd = float("inf")
            episode_fde = float("inf")

            for i in range(dim_c):
                self.algo.latent_c = torch.zeros(self.algo.dim_c, device=self.algo.device)
                self.algo.latent_c[i] = 1.0
                state = self.env_test.reset(person_id=id)
                done = False

                while (not done):
                    action = self.algo.exploit(state,self.algo.latent_c)
                    state, reward, done, _ = self.env_test.step(action)
                
                episode_fd = min(episode_fd,self.env_test.compute_Frechet_Distance())
                episode_fde = min(episode_fde,self.env_test.compute_FDE())

            frechet_distance += episode_fd
            fde += episode_fde

        # mean_return /= len(test_ids)
        # frechet_distance /= len(test_ids)
        # fde /= len(test_ids)

        return frechet_distance, fde
    

    def latent_long_special(self, index, dim_c)->list:
        # test id list
        with open('./core/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')


        test_id = test_ids[index]
        traj_new = {}
        for i in range(dim_c):
            state = self.env_test.reset(person_id=test_id)
            done = False
            # create current latent_c
            self.algo.latent_c = torch.zeros(dim_c, device=self.algo.device)
            self.algo.latent_c[i] = 1.0

            while (not done):
                action = self.algo.exploit(state,self.algo.latent_c)
                state, _, done, _ = self.env_test.step(action)

            traj_new[i] = self.env_test.get_short_traj()
        
        return traj_new

    def lots_latent_long_special(self, num_traj, dim_c)->list:
        # test id list
        with open('./core/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        # random select num_traj indexs
        skip_list = os.listdir(f'./render/trajectories/C{dim_c}/')
        skip_list = [int(i[:-4]) for i in skip_list]
        # indexs = random.sample(range(len(test_ids)), num_traj)
        indexs = range(len(test_ids))
        for index in tqdm(indexs):
            test_id = test_ids[index]
            # skip if already exist
            if test_id in skip_list: 
                continue
            # print(f"add new traj:{test_id}")
            traj_new = {}
            for i in range(dim_c):
                state = self.env_test.reset(person_id=test_id)
                done = False
                # create current latent_c
                self.algo.latent_c = torch.zeros(dim_c, device=self.algo.device)
                self.algo.latent_c[i] = 1.0

                while (not done):
                    action = self.algo.exploit(state,self.algo.latent_c)
                    state, _, done, _ = self.env_test.step(action)

                traj_new[i] = self.env_test.get_short_traj()
            
            with open(f'./render/trajectories/C{dim_c}/{test_id}.pkl','wb') as f:
                pickle.dump(traj_new,f)
    


    def flow_test(self, flow_ids):
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        log_kde = 0

        for test_id in tqdm(flow_ids):
            state = self.env_test.reset(person_id=test_id)
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, _, done, _ = self.env_test.step(action)

            log_kde += self.env_test.compute_log_kde(flow_ids)
        
        return log_kde
    

    def flow_test_ar(self, flow_ids):
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        log_kde = 0

        for test_id in tqdm(flow_ids):
            # init 
            self.env_test.reset(person_id=test_id)
            # render and compute
            log_kde += self.env_test.step_all(self.algo, flow_ids)
        
        return log_kde
    

    def states_test(self):
        # test id list
        with open('./core/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        hdd = 0

        for test_id in tqdm(test_ids):
            state = self.env_test.reset(person_id=test_id)
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, _, done, _ = self.env_test.step(action)

            hdd += self.env_test.compute_traj_hausdorff_distance()
        
        return hdd/len(test_ids)
    

    def short_test(self)->list:
        # test id list
        with open('./core/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        FD_list = []

        for test_id in tqdm(test_ids[:300]):
            if test_id!=12323:
                state = self.env_test.reset(person_id=test_id)
                count = 0

                while (count<5):
                    action = self.algo.exploit(state)
                    state, _, done, _ = self.env_test.step(action)
                    count += 1

                FD_list.append(self.env_test.compute_short_FD())
        
        return FD_list
    
    def short_special(self, index)->list:
        # test id list
        with open('./core/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')


        test_id = test_ids[index]
        state = self.env_test.reset(person_id=test_id)
        count = 0

        while (count<5):
            action = self.algo.exploit(state)
            state, _, done, _ = self.env_test.step(action)
            count += 1
        
        return self.env_test.get_short_traj()

    

    def long_test(self)->list:
        # test id list
        with open('./core/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        FD_list = []

        for test_id in tqdm(test_ids[:]):
            state = self.env_test.reset(person_id=test_id)
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, _, done, _ = self.env_test.step(action)

            FD_list.append(self.env_test.compute_Frechet_Distance())
        
        return FD_list
    

    def long_special(self, index)->list:
        # test id list
        with open('./core/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')


        test_id = test_ids[index]
        state = self.env_test.reset(person_id=test_id)
        done = False

        while (not done):
            action = self.algo.exploit(state)
            state, _, done, _ = self.env_test.step(action)
        
        return self.env_test.get_short_traj()
    
    
    def collision_test(self)->list:
        # test id list
        with open('./core/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        col_list = []

        for test_id in tqdm(test_ids[:]):
            state = self.env_test.reset(person_id=test_id)
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, _, done, _ = self.env_test.step(action)

            col_list.append(self.env_test.get_collisions())
        
        return col_list
