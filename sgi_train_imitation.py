import os
import argparse
from datetime import datetime
import torch
import yaml

from core.env import make_env
from core.buffer import *
from core.algos import ALGOS
from core.trainer import Trainer
from core.utils import save_args_to_yaml


def run(args):
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.dataset, args.algo, f'seed{args.seed}-{time}-V4-UC1.0-T1.0-Speed-D3') 
    print('Log directory:', log_dir)

    env = make_env(args,'Train')
    env_test = make_env(args,'Test')

    buffer_exp = SerializedBuffer_SA(
        path=args.buffer,
        device=torch.device(args.cuda_device if args.cuda else "cpu")
    )
    

    algo = ALGOS(
        buffer_exp = buffer_exp,
        state_shape = env.observation_space.shape,
        action_shape = env.action_space.shape,
        args = args
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )

    # save args for this time
    save_args_to_yaml(args, log_dir)
    
    trainer.train()


if __name__ == '__main__':
    ## define parser
    p = argparse.ArgumentParser()

    p.add_argument('--algo', type=str, default='sa_socialgail') 
    p.add_argument('--dataset',default='GC') 


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

    

    run(args)



    