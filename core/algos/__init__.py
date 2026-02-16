from .sa_socialgail import SA_SOCIAL_GAIL



import torch

def ALGOS(buffer_exp, state_shape, action_shape, args):
    if args.algo == 'sa_socialgail':
        return SA_SOCIAL_GAIL(
            args = args,
            buffer_exp = buffer_exp,
            graph_feature_channels = args.graph_obs_past_len,
            state_shape = state_shape,
            action_shape = action_shape,
            latent_code = args.latent_code,
            device = torch.device(args.cuda_device if args.cuda else "cpu"),
            seed = int(args.seed),
            rollout_length = args.rollout_length
        )
    else:
        raise NotImplementedError(f"Algorithm {args.algo} is not implemented.")
