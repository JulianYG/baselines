'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os.path as osp
import os
import logging
from mpi4py import MPI
from tqdm import tqdm

import numpy as np
import gym

from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', help='Name of environment', default='SawyerLiftEnv')
    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument('--mix_reward', action='store_true', help='using true env rewards mixed with discriminator reward')
    parser.add_argument('--rew_lambda', type=float, help='lambda coefficient to mix true rew and discriminator rew', default=0.5)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)

    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=1)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)

    parser.add_argument('--frame_stack', help='number of frames to stack', type=int, default=1)

    
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0.)
    # parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=0.)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()


def get_task_name(env_name, user_name):
    # task_name = args.algo + "_gail."
    # if args.pretrained:
    #     task_name += "with_pretrained."
    # if args.traj_limitation != np.inf:
    #     task_name += "transition_limitation_%d." % args.traj_limitation
    # task_name += user_name
    task_name = '%s.%s' % (env_name, user_name)
    return task_name

def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)

    import MujocoManip as MM
    if args.task == 'train':
      env_name, user_name = osp.basename(args.expert_path).split('.')[0].split('_')
    else:
      env_name, user_name = osp.basename(args.load_model_path).split('.')[:2]
    wrapper = '%sWrapper' % env_name
    render = True if args.task=='evaluate' else False

    if env_name == 'SawyerLiftEnv':
      env = MM.make(wrapper, 
                  ignore_done=False, 
                  use_eef_ctrl=False, 
                  gripper_visualization=True, 
                  use_camera_obs=False, 
                  has_renderer=render,
                  reward_shaping=True,
                  has_offscreen_renderer=render
                  )
    elif env_name == 'SawyerBinsEnv':
      env = MM.make(wrapper, 
                  ignore_done=False, 
                  use_eef_ctrl=False, 
                  gripper_visualization=True, 
                  use_camera_obs=False, 
                  has_renderer=render,
                  reward_shaping=True,
                  single_object_mode=False if 'hard' in user_name.lower() else True,
                  has_offscreen_renderer=render
                  )
    elif env_name == 'SawyerPegsEnv':
      env = MM.make(wrapper, 
                  ignore_done=False, 
                  use_eef_ctrl=False, 
                  gripper_visualization=True, 
                  use_camera_obs=False, 
                  has_renderer=render,
                  reward_shaping=True,
                  single_object_mode=False if 'hard' in user_name.lower() else True,
                  has_offscreen_renderer=render
                  )
    else:
      raise NotImplementedError

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(env_name, user_name) + '_%s_%s' % (args.algo, 1 if not args.mix_reward else args.rew_lambda)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.task == 'train':
        dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
        train(env,
              args.seed,
              policy_fn,
              reward_giver,
              dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              args.rew_lambda,
              args.mix_reward,
              task_name,
              args.frame_stack
              )
    elif args.task == 'evaluate':
        visualizer(env,
               policy_fn,
               args.load_model_path,
               timesteps_per_batch=env.env.horizon,
               number_trajs=10,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample
               )
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, rew_lambda,
          mix_reward=False, task_name=None, frame_stack=1):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.gail import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
                                                 max_iters=BC_max_iter)

    # Set up for MPI seed
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env.seed(workerseed)

    if algo == 'trpo':
        from baselines.gail import trpo_mpi
        trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_timesteps=num_timesteps,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=env.env.horizon * 5,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1,
                       gamma=0.995, lam=0.97,
                       vf_iters=5, vf_stepsize=1e-4,
                       mix_reward=mix_reward, r_lambda=rew_lambda,
                       task_name=task_name, frame_stack=frame_stack)

    elif algo == 'ppo':
        from baselines.gail import ppo_mpi
        ppo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_timesteps=num_timesteps,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=env.env.horizon, #env.env.horizon * 5,
                       gamma=0.995, lam=0.97, clip_param=0.2, 
                       optim_epochs=50, optim_stepsize=1e-4, optim_batchsize=100, # optimization hypers
                       mix_reward=mix_reward, r_lambda=rew_lambda,
                       task_name=task_name, frame_stack=frame_stack)
    else:
        raise NotImplementedError

def visualizer(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):
  
    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    U.load_state(load_model_path)
    # obs_list = []
    # acs_list = []
    # len_list = []
    # ret_list = []
    for _ in tqdm(range(number_trajs)):
      ob = env.reset()
      total_rew = 0
      env.env.viewer.set_camera(2)
      env.env.viewer.viewer._hide_overlay = True
      for _ in range(timesteps_per_batch):

        ac, vpred = pi.act(False, ob)
        ob, rew, new, _ = env.step(ac)
        env.render()
        total_rew += rew
        if new: 
          break
      print('total reward: ', total_rew)

        # traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        # obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        # obs_list.append(obs)
        # acs_list.append(acs)
        # len_list.append(ep_len)
        # ret_list.append(ep_ret)

def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    U.load_state(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)
