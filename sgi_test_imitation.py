import os
import argparse
from datetime import datetime
import torch
import pickle
import copy
import yaml

from core.env import make_env
from core.buffer import *
from core.algos import ALGOS, SI_PPO, DA_PPO
from core.tester import Tester

import concurrent 



def test_speed(args, test_subdir):
    env_test = make_env(args,'Test')
    
    # load algo
    if args.algo == 'sgi_socialgail' or args.algo == 'sgi_socialgail_v2':
        algo = SI_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),  
            seed = args.seed
        )
    elif args.algo == 'diversity_aware_rl' or args.algo == 'sgi_socialgail_v4' or args.algo == 'sgi_socialgail_v6':
        algo = DA_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),  
            seed = args.seed
        )

    # hyperparameters
    TEST_DIM = 2
    NUM_IDS = 3000

    tester = Tester(
        env_test = env_test,
        algo = algo,
        model_dir = test_subdir + f"models/model_D{TEST_DIM}_{args.algo}.pt"
    )

    
    tester.semi_info_test_speed(num_ids=NUM_IDS, dim_c=TEST_DIM, test_subdir=test_subdir, args = args)

def test_speed_synthetic(args, test_subdir):
    env_test = make_env(args,'Test')
    
    # load algo
    if args.algo == 'sgi_socialgail' or args.algo == 'sgi_socialgail_v2':
        algo = SI_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),  
            seed = args.seed
        )
    elif args.algo == 'diversity_aware_rl' or args.algo == 'sgi_socialgail_v4' or args.algo == 'sgi_socialgail_v6':
        algo = DA_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),  
            seed = args.seed
        )

    # hyperparameters
    TEST_DIM = 2
    NUM_IDS = 500

    tester = Tester(
        env_test = env_test,
        algo = algo,
        model_dir = test_subdir + f"models/model_D{TEST_DIM}_{args.algo}.pt"
    )

    
    tester.semi_info_test_speed_synthetic(num_ids=NUM_IDS, dim_c=TEST_DIM, test_subdir=test_subdir, args = args)








def test_nn_dist(args, test_subdir):
    env_test = make_env(args,'Test')
    
    # load algo
    if args.algo == 'sgi_socialgail' or args.algo == 'sgi_socialgail_v2':
        algo = SI_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),  
            seed = args.seed
        )
    elif args.algo == 'diversity_aware_rl' or args.algo == 'sgi_socialgail_v4':
        algo = DA_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),  
            seed = args.seed
        )

    # hyperparameters
    TEST_DIM = 3
    NUM_IDS = 200

    tester = Tester(
        env_test = env_test,
        algo = algo,
        model_dir = test_subdir + f"models/model_D{TEST_DIM}_{args.algo}.pt"
    )

    
    tester.semi_info_test_nn_dist(num_ids=NUM_IDS, dim_c=TEST_DIM, test_subdir=test_subdir, args = args)








def test_social_comfort(args, test_subdir):
    env_test = make_env(args,'Test')
    
    # load algo
    if args.algo == 'sgi_socialgail' or args.algo == 'sgi_socialgail_v2':
        algo = SI_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),  
            seed = args.seed
        )
    elif args.algo == 'diversity_aware_rl' or args.algo == 'sgi_socialgail_v4' or args.algo == 'sgi_socialgail_v6':
        algo = DA_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),  
            seed = args.seed
        )

    # hyperparameters
    TEST_DIM = 2
    NUM_IDS = 100

    tester = Tester(
        env_test = env_test,
        algo = algo,
        model_dir = test_subdir + f"models/model_D{TEST_DIM}_{args.algo}.pt"
    )

    
    tester.semi_info_test_social_comfort(num_ids=NUM_IDS, dim_c=TEST_DIM, test_subdir=test_subdir, args = args)



def test_social_gradient(args, test_subdir):
    env_test = make_env(args,'Test')
    
    # load algo
    if args.algo == 'sgi_socialgail' or args.algo == 'sgi_socialgail_v2':
        algo = SI_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),  
            seed = args.seed
        )
    elif args.algo == 'diversity_aware_rl' or args.algo == 'sgi_socialgail_v4' or args.algo == 'sgi_socialgail_v6':
        algo = DA_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),  
            seed = args.seed
        )

    # hyperparameters
    TEST_DIM = 2
    NUM_IDS = 200

    tester = Tester(
        env_test = env_test,
        algo = algo,
        model_dir = test_subdir + f"models/model_D{TEST_DIM}_{args.algo}.pt"
    )

    
    tester.semi_info_test_social_gradient(num_ids=NUM_IDS, dim_c=TEST_DIM, test_subdir=test_subdir, args = args)
    




































def test_once_ffh(args, test_subdir):
    env_test = make_env(args,'Test')
    
    # load algo
    if args.algo == 'sgi_socialgail' or args.algo == 'sgi_socialgail_v2':
        algo = SI_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),  
            seed = args.seed
        )
    elif args.algo == 'diversity_aware_rl' or args.algo == 'sgi_socialgail_v4' or args.algo == 'sgi_socialgail_v6':
        algo = DA_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),  
            seed = args.seed
        )

    # hyperparameters
    TEST_DIM = 3
    NUM_IDS = 200

    tester = Tester(
        env_test = env_test,
        algo = algo,
        model_dir = test_subdir + f"models/model_D{TEST_DIM}_{args.algo}.pt"
    )


    tester.semi_info_test_ffh(num_ids=NUM_IDS, dim_c=TEST_DIM, test_subdir=test_subdir, args = args)



# run once, test FDE FD Hausdorff
def run_once_ffh_paral(args, model_subdir, cuda_device, test_ids):
    env_test = make_env(args,'Test')

    # load algo
    if args.algo == 'sgi_socialgail' or args.algo == 'sgi_socialgail_v2':
        algo = SI_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(cuda_device),  
            seed = args.seed
        )
    elif args.algo == 'diversity_aware_rl' or args.algo == 'sgi_socialgail_v4' or args.algo == 'sgi_socialgail_v6':
        algo = DA_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(cuda_device),  
            seed = args.seed
        )

    tester = Tester(
        env_test=env_test,
        algo=algo,
        model_dir = args.model_dir+'/'+model_subdir
    )

    frechet_distance, fde, hausdorff_distance = tester.semi_info_test_ffh_once(args.dim_c, test_ids)

    return model_subdir, frechet_distance, fde, hausdorff_distance


# test FDE FD Hausdorff
def test_fde_fd_hausdorff_paral(args):
    # test models
    models_list = os.listdir(args.model_dir)
    # test id list
    # with open(args.test_ids_path,'rb') as f:
    #     whole_test_ids = pickle.load(f)
    with open(args.dataset_path,'rb') as f:
        whole_test_ids = pickle.load(f)['test_ids']

    # devide the test_ids, may not be complete devided
    # whole_test_ids = whole_test_ids[:20]
    NUM_SUBTEST = 10
    # RANGE = 200
    GPU_LIST = [0, 2, 3]

    selected_test_ids = whole_test_ids[:]
    # test_ids_list = [selected_test_ids[i:i + len(selected_test_ids)//NUM_DEVIDE] for i in range(0, len(selected_test_ids), len(selected_test_ids)//NUM_DEVIDE)]
    chunk_size = len(selected_test_ids) // NUM_SUBTEST
    test_ids_list = [
        selected_test_ids[i * chunk_size : (i + 1) * chunk_size] 
        if i < NUM_SUBTEST - 1 
        else selected_test_ids[i * chunk_size :] 
        for i in range(NUM_SUBTEST) 
    ]

    num_workers = int(len(test_ids_list)*len(models_list))

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            to_dos = []
            res_list = []
            # submit
            for i in range(len(models_list)):
                for j in range(len(test_ids_list)):
                    model_subdir = models_list[i]

                    cuda_device = "cuda:" + str(GPU_LIST[(i*len(test_ids_list)+j)%len(GPU_LIST)])

                    future = executor.submit(run_once_ffh_paral, args, model_subdir, cuda_device, test_ids_list[j])
                    to_dos.append(future)
            # wait all to finish
            for future in concurrent.futures.as_completed(to_dos):
                res_list.append(future.result())
            # show res
            # res_list = sorted(res_list, key=lambda tup:int(tup[0][4:-3])) # no need to sort
            res_dict = {}
            for tup in res_list:
                res_dict[tup[0]] = {}
                res_dict[tup[0]]['frechet_distance'] = 0
                res_dict[tup[0]]['fde'] = 0
                res_dict[tup[0]]['hausdorff_distance'] = 0
            for tup in res_list:
                model_dir, frechet_distance, fde, hausdorff_distance = tup
                res_dict[model_dir]['frechet_distance'] += frechet_distance
                res_dict[model_dir]['fde'] += fde
                res_dict[model_dir]['hausdorff_distance'] += hausdorff_distance
            for model_dir in res_dict.keys():
                frechet_distance = res_dict[model_dir]['frechet_distance']/len(selected_test_ids)
                fde = res_dict[model_dir]['fde']/len(selected_test_ids)
                hausdorff_distance = res_dict[model_dir]['hausdorff_distance']/len(selected_test_ids)
                print(f'Test Model: {model_dir}  '
                f'Frechet Distance: {frechet_distance:<5.3f}   '
                f'FDE: {fde:<5.3f}   '
                f'Hausdorff Distance: {hausdorff_distance:<5.3f}')
    except KeyboardInterrupt:
        print("收到 Ctrl+C，正在关闭程序...")
        executor.shutdown(wait=False)
        print("程序已关闭。")
    finally:
        # 确保所有进程都被终止
        executor.shutdown(wait=True)
        print("清理完成。")

























# test global similarity
def test_global_similarity(args, cuda_device, flow_path, test_subdir):
    # test flow
    with open(flow_path,'rb') as f:
        flow_ids = pickle.load(f)

    env_test = make_env(args,'Test')
    
    # load algo
    if args.algo == 'sgi_socialgail' or args.algo == 'sgi_socialgail_v2':
        algo = SI_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(cuda_device),  
            seed = args.seed
        )
    elif args.algo == 'diversity_aware_rl' or args.algo == 'sgi_socialgail_v4' or args.algo == 'sgi_socialgail_v6':
        algo = DA_PPO(
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = env_test.observation_space.shape,
            action_shape = env_test.action_space.shape,
            latent_code = args.latent_code,
            device = torch.device(cuda_device),  
            seed = args.seed
        )

    TEST_DIM = 3

    tester = Tester(
        env_test=env_test,
        algo=algo,
        model_dir = test_subdir + f"models/model_D{TEST_DIM}_{args.algo}.pt"
    )

    log_kde = tester.semi_info_flow_test(flow_ids, TEST_DIM)
    
    # print(f"{flow_path[-9:-4]}  log KDE: {log_kde}")
    return flow_path, log_kde


def test_global_similarity_parallel(args, test_subdir):
    # test flows
    flow_list = os.listdir(args.flow_dir)

    GPU_LIST = [1, 3]

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(flow_list)) as executor:
            # submit
            to_dos = []
            for i in range(len(flow_list)):
                flow_path = os.path.join(args.flow_dir, flow_list[i])
                cuda_device = "cuda:" + str(GPU_LIST[i%len(GPU_LIST)])

                # new_args = copy.copy(args)
                # new_args.device = cuda_device
                # new_args.flow_path = flow_path

                future = executor.submit(test_global_similarity, args, cuda_device, flow_path, test_subdir)
                to_dos.append(future)

            # wait all to finish
            res_list = []
            for future in concurrent.futures.as_completed(to_dos):
                res_list.append(future.result())
            
            # show res
            res_list.sort(key=lambda x: x[0])
            for flow_path, log_kde in res_list:
                print(f"{flow_path[-9:-4]}  log KDE: {log_kde}")

            with open(test_subdir + "logs/"+ f"global_similarity_{args.algo}.txt", "w") as file:
                for flow_path, log_kde in res_list:
                    file.write(f"{flow_path[-9:-4]}  log KDE: {log_kde}\n")

    except KeyboardInterrupt:
        print("收到 Ctrl+C，正在关闭程序...")
        executor.shutdown(wait=False)
        print("程序已关闭。")
    finally:
        # 确保所有进程都被终止
        executor.shutdown(wait=True)
        print("清理完成。")






















if __name__ == '__main__':
    ## define parser
    p = argparse.ArgumentParser()

    p.add_argument('--algo', type=str, default='sa_socialgail') 
    p.add_argument('--dataset',default='synthetic_w_speed') 

    p.add_argument('--model_dir', type=str, default=None)
    p.add_argument('--flow_dir', type=str, default='./info_test/flow_test/global_flows/')
    p.add_argument('--flow_path', type=str, default=None)


    ## load config files
    known, _ = p.parse_known_args()
    # Load algorithm-specific configuration
    with open(f"./configs/algo_{known.algo}.yaml", "r") as file:
        config = yaml.safe_load(file)
        p.set_defaults(**config)
    # Load dataset-specific configuration
    with open(f"./core/crowd_env/datasets/datasets_configs.yaml", "r") as file:
        dataset_config = yaml.safe_load(file)
        p.set_defaults(**dataset_config[known.dataset])

    

    args = p.parse_args()

    
    ## test 
    test_speed_synthetic(args, f"./info_test/speed/")
    
    