'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''

import argparse
import tempfile
import os.path as osp
import gym
import logging
from tqdm import tqdm

import tensorflow as tf

from baselines.gail import mlp_policy
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.common.mpi_adam import MpiAdam
from baselines.gail.run_mujoco import runner, visualizer
from baselines.gail.dataset.mujoco_dset import Mujoco_Dset


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Behavior Cloning")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/SawyerLiftEnv_easy.npz') #this has to be absolute
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='BC_checkpoint')
    parser.add_argument('--load_model_path', help='path to the saved model', default='BC_checkpoint/.')
    parser.add_argument('--log_dir', help='the directory to save log file', default='BC_log')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')

    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=128)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e5)
    return parser.parse_args()


def learn(env, policy_func, dataset, optim_batch_size=128, max_iters=1e4,
          adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False):

    val_per_iter = int(max_iters/10)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")
    loss = tf.reduce_mean(tf.square(ac-pi.ac))
    
    tf.summary.scalar('BC_loss', loss)
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter(log_dir)
   
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()
    logger.log("Pretraining with Behavior Cloning...")
    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, task_name)
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        train_loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)
        
        summary = tf.get_default_session().run(merged, feed_dict={ob: ob_expert, ac: ac_expert, stochastic: True})
        writer.add_summary(summary, iter_so_far)
        writer.flush()
        
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))
        
        if iter_so_far % 20 == 0:
            from tensorflow.core.protobuf import saver_pb2
            saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
            saver.save(tf.get_default_session(), savedir_fname)
    return savedir_fname


def get_task_name(env_name, user_name):
    return 'BC_%s.%s' % (env_name, user_name)

def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)

    import MujocoManip as MM
    if args.task == 'train':
      env_name, user_name = osp.basename(args.expert_path).split('.')[0].split('_')
    else:
      uenv, user_name = osp.basename(args.load_model_path).split('.')[:2]
      env_name = uenv.split('_')[1]

    wrapper = '%sWrapper' % env_name
    render = True if args.task=='evaluate' else False
    print('%s initialized.' % wrapper)

    bin_dict = dict(milk=0, bread=1, cereal=2, can=3)
    peg_dict = dict(square=0, round=1)

    if env_name == 'SawyerLiftEnv':
      env = MM.make(wrapper, 
                  ignore_done=False, 
                  use_eef_ctrl=False, 
                  gripper_visualization=True, 
                  use_camera_obs=False, 
                  has_renderer=render,
                  reward_shaping=True,
                  has_offscreen_renderer=False
                  )
    elif env_name == 'SawyerBinsEnv':
      env = MM.make(wrapper, 
                  ignore_done=False, 
                  use_eef_ctrl=False, 
                  gripper_visualization=True, 
                  use_camera_obs=False, 
                  has_renderer=render,
                  reward_shaping=True,
                  single_object_mode=False if user_name.lower() == 'hard' else True,
                  has_offscreen_renderer=False, 
                  selected_bin=None if user_name.lower() == 'hard' else bin_dict[user_name.lower()]
                  )
    elif env_name == 'SawyerPegsEnv':
      env = MM.make(wrapper, 
                  ignore_done=False, 
                  use_eef_ctrl=False, 
                  gripper_visualization=True, 
                  use_camera_obs=False, 
                  has_renderer=render,
                  reward_shaping=True,
                  single_object_mode=False if user_name.lower() == 'hard' else True,
                  has_offscreen_renderer=False,
                  selected_bin=None if user_name.lower() == 'hard' else peg_dict[user_name.lower()]
                  )
    else:
      raise NotImplementedError
        
    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=3)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(env_name, user_name)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    
    if args.task == 'train':
      dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
      savedir_fname = learn(env,
                            policy_fn,
                            dataset,
                            max_iters=args.BC_max_iter,
                            ckpt_dir=args.checkpoint_dir,
                            log_dir=args.log_dir,
                            task_name=task_name,
                            verbose=True)

    elif args.task == 'evaluate':
      visualizer(env, policy_fn, args.load_model_path, env.env.horizon, 10,
           args.stochastic_policy, save=args.save_sample)
      # avg_len, avg_ret = runner(env,
      #                           policy_fn,
      #                           savedir_fname,
      #                           timesteps_per_batch=env.env.horizon,
      #                           number_trajs=10,
      #                           stochastic_policy=args.stochastic_policy,
      #                           save=args.save_sample,
      #                           reuse=True)
      # print('avg ret: {}, avg len: {}'.format(avg_ret, avg_len))


if __name__ == '__main__':
    args = argsparser()
    main(args)
