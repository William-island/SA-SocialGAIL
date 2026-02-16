import os
from time import time, sleep
from datetime import timedelta
import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch

class Trainer:

    def __init__(self, env, env_test, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=10):
        super().__init__()

        # Env to collect samples.
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2**31-seed)

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)


        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes


    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()

        for step in range(1, self.num_steps + 1):
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}'))
                

        # self.evaluate_saved_models(0)
        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self, step):
        if hasattr(self.algo, 'latent_code'):
            self.evaluate_with_c(step)
        else:
            self.evaluate_wo_c(step)



    def evaluate_wo_c(self, step):
        mean_return = 0.0
        frechet_distance = 0.0
        fde = 0.0

        reach_goal = 0.0
        geodesic_rate = 0.0
        tgr = 0.0 # theoretical geodesic rate
        remaining_steps = 0

        normal_collisions = 0.0
        ccd_collisions = 0.0

        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            episode_return = 0.0
            done = False


            while (not done):
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes
            frechet_distance += self.env_test.compute_Frechet_Distance() / self.num_eval_episodes
            fde += self.env_test.compute_FDE() / self.num_eval_episodes

            reach_goal += self.env_test.reach_goal()
            geodesic_rate += self.env_test.compute_geodesic_rate()
            tgr += self.env_test.theoretical_geodesic_rate()
            remaining_steps += self.env_test.compute_remining_steps()

            normal_collisions += self.env_test.get_collision_2type()[0]
            ccd_collisions += self.env_test.get_collision_2type()[1]

        reach_goal = reach_goal / self.num_eval_episodes
        geodesic_rate = geodesic_rate / self.num_eval_episodes
        tgr = tgr / self.num_eval_episodes
        remaining_steps = remaining_steps / self.num_eval_episodes

        normal_collisions = normal_collisions / self.num_eval_episodes
        ccd_collisions = ccd_collisions / self.num_eval_episodes

        # self.writer.add_scalar('test/return', mean_return, step)
        self.writer.add_scalar('test/frechet_distance', frechet_distance, step)
        self.writer.add_scalar('test/fde', fde, step)
        self.writer.add_scalar('test/reach_goal', reach_goal, step)
        self.writer.add_scalar('test/geodesic_rate', geodesic_rate, step)
        self.writer.add_scalar('test/theoretical_geodesic_rate', tgr, step)
        self.writer.add_scalar('test/theoretical_geodesic_rate_error', tgr-geodesic_rate, step)
        self.writer.add_scalar('test/remaining_steps', remaining_steps, step)
        print(f'Num steps: {step:<6}   '
              # f'Return: {mean_return:<5.1f}   '
              f'Frechet Distance: {frechet_distance:<5.2f}   '
              f'FDE: {fde:<5.2f}   '
              f'Time: {self.time}'
              f'   ;   '
                f'Reach Goal: {reach_goal:<5.3f}   '
                f'Geodesic Rate: {geodesic_rate:<5.2f} ' #  (TGR: {tgr:<5.2f})
                # f'Remaining Steps: {remaining_steps:<5.2f}   '
                f'Nor_Coll: {normal_collisions:<5.2f}   '
                f'CCD_Coll: {ccd_collisions:<5.2f}   '
              )

    def evaluate_with_c(self, step):
        mean_return = 0.0
        frechet_distance = 0.0
        fde = 0.0

        reach_goal = 0.0
        geodesic_rate = 0.0
        tgr = 0.0 # theoretical geodesic rate
        remaining_steps = 0

        normal_collisions = 0.0
        ccd_collisions = 0.0

        for _ in range(self.num_eval_episodes):
            latent_c = self.algo.sample_latent_code_value()
            state = self.env_test.reset()
            episode_return = 0.0
            done = False


            while (not done):
                action = self.algo.exploit(state, latent_c)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward


            # log
            mean_return += episode_return / self.num_eval_episodes
            frechet_distance += self.env_test.compute_Frechet_Distance() / self.num_eval_episodes
            fde += self.env_test.compute_FDE() / self.num_eval_episodes

            reach_goal += self.env_test.reach_goal()
            geodesic_rate += self.env_test.compute_geodesic_rate()
            tgr += self.env_test.theoretical_geodesic_rate()
            remaining_steps += self.env_test.compute_remining_steps()

            normal_collisions += self.env_test.get_collision_2type()[0]
            ccd_collisions += self.env_test.get_collision_2type()[1]


        reach_goal = reach_goal / self.num_eval_episodes
        geodesic_rate = geodesic_rate / self.num_eval_episodes
        tgr = tgr / self.num_eval_episodes
        remaining_steps = remaining_steps / self.num_eval_episodes

        normal_collisions = normal_collisions / self.num_eval_episodes
        ccd_collisions = ccd_collisions / self.num_eval_episodes

        # self.writer.add_scalar('test/return', mean_return, step)
        self.writer.add_scalar('test/frechet_distance', frechet_distance, step)
        self.writer.add_scalar('test/fde', fde, step)
        self.writer.add_scalar('test/reach_goal', reach_goal, step)
        self.writer.add_scalar('test/geodesic_rate', geodesic_rate, step)
        self.writer.add_scalar('test/theoretical_geodesic_rate', tgr, step)
        self.writer.add_scalar('test/theoretical_geodesic_rate_error', tgr-geodesic_rate, step)
        self.writer.add_scalar('test/remaining_steps', remaining_steps, step)

        self.writer.add_scalar('test/normal_collisions', normal_collisions, step)
        self.writer.add_scalar('test/ccd_collisions', ccd_collisions, step)

        print(f'Num steps: {step:<6}   '
              # f'Return: {mean_return:<5.1f}   '
              f'Frechet Distance: {frechet_distance:<5.2f}   '
              f'FDE: {fde:<5.2f}   '
              f'Time: {self.time}'
              f'   ;   '
                f'Reach Goal: {reach_goal:<5.3f}   '
                f'Geodesic Rate: {geodesic_rate:<5.2f}   (TGR: {tgr:<5.2f})  '
                # f'Remaining Steps: {remaining_steps:<5.2f}   '
                # f'Nor_Coll: {normal_collisions:<5.2f}   '
                f'CCD_Coll: {ccd_collisions:<5.2f}   '
              )
        
        # judge if to update Q function
        if hasattr(self.algo, 'q_update_flag'):
            if not self.algo.q_update_flag:
                # if reach_goal > 0.9: # 0.7:
                if frechet_distance < 3 and reach_goal > 0.9:
                    self.algo.q_update_flag = True
                    print('Reach goal rate > 0.9, start to update Q function!')
                    self.writer.add_scalar('test/start_update_Q', 1, step)
            else:
                # close Q function update from begin
                self.writer.add_scalar('test/start_update_Q', 0, step)

        # judge if use reward_us_coef
        if hasattr(self.algo, 'reward_us_coef_flag'):
            if not self.algo.reward_us_coef_flag:
                if frechet_distance < 3 and reach_goal > 0.9:
                    self.algo.reward_us_coef_flag = True
                    print('Frechet distance < 0.3 and reach goal rate > 0.9, start to use reward_us_coef!')
                    self.writer.add_scalar('test/start_use_reward_us_coef', 1, step)
            else:
               self.writer.add_scalar('test/start_use_reward_us_coef', 0, step)



    
    def evaluate_with_c_old(self, step):
        print(f'Num steps: {step:<6}   ')
        for i in range(self.algo.dim_c):
            self.algo.latent_c = torch.zeros(self.algo.dim_c, device=self.algo.device)
            self.algo.latent_c[i] = 1.0
            # mean_return = 0.0
            frechet_distance = 0.0
            fde = 0.0

            for _ in range(self.num_eval_episodes):
                state = self.env_test.reset()
                episode_return = 0.0
                done = False
                # self.algo.sample_latent_c()

                while (not done):
                    action = self.algo.exploit(state,self.algo.latent_c)
                    state, reward, done, _ = self.env_test.step(action)
                    episode_return += reward

                # mean_return += episode_return / self.num_eval_episodes
                frechet_distance += self.env_test.compute_Frechet_Distance() / self.num_eval_episodes
                fde += self.env_test.compute_FDE() / self.num_eval_episodes

            # self.writer.add_scalar('test/return', mean_return, step)
            self.writer.add_scalar(f'test/dim_{i}/frechet_distance', frechet_distance, step)
            self.writer.add_scalar(f'test/dim_{i}/fde', fde, step)
            print(f'           dim_{i}   '
                # f'Return: {mean_return:<5.1f}   '
                f'Frechet Distance: {frechet_distance:<5.2f}   '
                f'FDE: {fde:<5.2f}   '
                f'Time: {self.time}')
        
        # if self.env.get_target_circle_width()>=0.5:
        #     self.env.update_target_circle_width(fde)
        #     self.env_test.update_target_circle_width(fde)
    



    
    def evaluate_saved_models(self, num_models):
        # test id list
        with open('./core/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # test models
        models_list = sorted(os.listdir(self.model_dir), key = lambda x:int(x[4:-3]))
        for i in range(num_models):
            model_subdir = models_list[-num_models+i]
            self.evaluate_one_model(test_ids, model_subdir)

    def evaluate_one_model(self, test_ids, model_subdir):
        model_state_dict = torch.load(self.model_dir+'/'+model_subdir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {model_subdir} success!')

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


        print(f'Test Model: {model_subdir}'
              f'Return: {mean_return:<5.1f}   '
              f'Frechet Distance: {frechet_distance:<5.2f}   '
              f'FDE: {fde:<5.2f}   '
              f'Time: {self.time}')

    def evaluate_all_test(self):
        with open('./core/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # with open("./core/crowd_env/datasets/GC_continue.pkl",'rb') as f:
        #     gc = pickle.load(f)
        # test_ids = gc['test_ids']

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


        print('Test Res:\t'
              f'Return: {mean_return:<5.1f}   '
              f'Frechet Distance: {frechet_distance:<5.2f}   '
              f'FDE: {fde:<5.2f}   '
              f'Time: {self.time}')


    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
