'''
Disclaimer: The ppo part highly rely on ppo_mpi at @openai/baselines
'''

import time
import os
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common import Dataset, explained_variance, zipsame, dataset, fmt_row
from baselines import logger
from baselines.common import colorize
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from baselines.common.cg import cg
from baselines.gail.statistics import stats


def traj_segment_generator(pi, env, reward_giver, horizon, mix_rew, lam, stochastic, frame_stack):

    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    true_rew = 0.0
    ob = env.reset()
    # env.env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []
    ep_lens = []

    ob_tmp = np.concatenate([ob for _ in range(frame_stack)])
    # Initialize history arrays
    obs = np.array([ob_tmp for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    #print('obs', obs.shape)
    frame_buf = obs[0]
    #print('ob', ob.shape)
    #print('frame buf', frame_buf.shape)
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, frame_buf) #ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets}
            _, vpred = pi.act(stochastic, frame_buf) #ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
        i = t % horizon
        #print('obs', obs.shape)
        #print('ob', ob.shape)
        #print(frame_buf.shape)
        frame_buf = np.concatenate([frame_buf[ob.shape[0]:], ob ])
        #print(frame_buf.shape)
        #print(obs.shape)
        obs[i] = frame_buf
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        rew = reward_giver.get_reward(ob, ac)
        ob, true_rew, new, info = env.step(ac)

        # Use hybrid reward
        if mix_rew:
            rew = lam * rew + (1. - lam) * true_rew
        
        rews[i] = rew
        true_rews[i] = true_rew
        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, policy_func, reward_giver, expert_dataset, rank, 
          pretrained, pretrained_weight, *, clip_param,
          g_step, d_step, entcoeff, save_per_iter,
          optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
          ckpt_dir, log_dir, timesteps_per_batch, task_name,
          gamma, lam, d_stepsize=3e-4, adam_epsilon=1e-5,
          max_timesteps=0, max_episodes=0, max_iters=0,
          mix_reward=False, r_lambda=0.44,
          callback=None,
          schedule='constant', # annealing for stepsize parameters (epsilon and adam),
          frame_stack=1
          ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ob_space.shape = (ob_space.shape[0] * frame_stack,)
    print(ob_space)
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=(pretrained_weight != None))
    oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    # kloldnew = oldpi.pd.kl(pi.pd)
    # ent = pi.pd.entropy()
    # meankl = tf.reduce_mean(kloldnew)
    # meanent = tf.reduce_mean(ent)
    # entbonus = entcoeff * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    # vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    # ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    # surrgain = tf.reduce_mean(ratio * atarg)

    # optimgain = surrgain + entbonus
    # losses = [optimgain, meankl, entbonus, surrgain, meanent]
    # loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    d_adam = MpiAdam(reward_giver.get_trainable_variables())

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    # dist = meankl

    # all_var_list = pi.get_trainable_variables()
    # var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    # vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]
    # assert len(var_list) == len(vf_var_list) + 1
    # d_adam = MpiAdam(reward_giver.get_trainable_variables())
    # vfadam = MpiAdam(vf_var_list)

    # get_flat = U.GetFlat(var_list)
    # set_from_flat = U.SetFromFlat(var_list)
    # klgrads = tf.gradients(dist, var_list)
    # flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    # shapes = [var.get_shape().as_list() for var in var_list]
    # start = 0
    # tangents = []
    # for shape in shapes:
    #     sz = U.intprod(shape)
    #     tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
    #     start += sz
    # gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    # fvp = U.flatgrad(gvp, var_list)

    # assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
    #                                                 for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    # compute_losses = U.function([ob, ac, atarg], losses)
    # compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    # compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    # compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    if rank == 0:
        generator_loss = tf.placeholder(tf.float32, [], name='generator_loss')
        expert_loss = tf.placeholder(tf.float32, [], name='expert_loss')
        entropy = tf.placeholder(tf.float32, [], name='entropy')
        entropy_loss = tf.placeholder(tf.float32, [], name='entropy_loss')
        generator_acc = tf.placeholder(tf.float32, [], name='genrator_acc')
        expert_acc = tf.placeholder(tf.float32, [], name='expert_acc')
        eplenmean = tf.placeholder(tf.int32, [], name='eplenmean')
        eprewmean = tf.placeholder(tf.float32, [], name='eprewmean')
        eptruerewmean = tf.placeholder(tf.float32, [], name='eptruerewmean')
        # _meankl = tf.placeholder(tf.float32, [], name='meankl')
        # _optimgain = tf.placeholder(tf.float32, [], name='optimgain')
        # _surrgain = tf.placeholder(tf.float32, [], name='surrgain')
        _ops_to_merge = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc, eplenmean, eprewmean, eptruerewmean]
        ops_to_merge = [ tf.summary.scalar(op.name, op) for op in _ops_to_merge]

        merged = tf.summary.merge(ops_to_merge)

    ### TODO: report these stats ### 
    #     generator_loss = tf.placeholder(tf.float32, [], name='generator_loss')
    #     expert_loss = tf.placeholder(tf.float32, [], name='expert_loss')
    #     generator_acc = tf.placeholder(tf.float32, [], name='genrator_acc')
    #     expert_acc = tf.placeholder(tf.float32, [], name='expert_acc')
    #     eplenmean = tf.placeholder(tf.int32, [], name='eplenmean')
    #     eprewmean = tf.placeholder(tf.float32, [], name='eprewmean')
    #     eptruerewmean = tf.placeholder(tf.float32, [], name='eptruerewmean')

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    adam.sync()
    d_adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, reward_giver, timesteps_per_batch, 
        mix_reward, r_lambda,
        stochastic=True, frame_stack=frame_stack)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=100)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    g_loss_stats = stats(loss_names)
    d_loss_stats = stats(reward_giver.loss_name)
    ep_stats = stats(["True_rewards", "Rewards", "Episode_length"])

    # if provide pretrained weight
    if pretrained_weight is not None:
        U.load_state(pretrained_weight, var_list=pi.get_variables())

    if rank == 0:
        filenames = [f for f in os.listdir(log_dir) if 'logs' in f]
        writer = tf.summary.FileWriter('{}/logs-{}'.format(log_dir, len(filenames)))

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # Save model
        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            fname = os.path.join(ckpt_dir, task_name)
            os.makedirs(os.path.dirname(fname), exist_ok=True)

            from tensorflow.core.protobuf import saver_pb2
            saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
            saver.save(tf.get_default_session(), fname)
            # U.save_state(fname)

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)

        # ------------------ Update G ------------------
        logger.log("Optimizing Policy...")
        for _ in range(g_step):
            with timed("sampling"):
                seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]


            # # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            # ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            # vpredbefore = seg["vpred"]  # predicted value function before udpate
            # atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

            # args = seg["ob"], seg["ac"], atarg
            # fvpargs = [arr[::5] for arr in args]

            assign_old_eq_new()  # set old parameter values to new parameter values

            with timed("policy optimization"):
                logger.log("Optimizing...")
                logger.log(fmt_row(13, loss_names))
                # Here we do a bunch of optimization epochs over the data
                for _ in range(optim_epochs):
                    losses = [] # list of tuples, each of which gives the loss for a minibatch
                    for batch in d.iterate_once(optim_batchsize):
                        *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                        adam.update(g, optim_stepsize * cur_lrmult)
                        losses.append(newlosses)
                    logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))


        # g_losses = meanlosses
        # for (lossname, lossval) in zip(loss_names, meanlosses):
        #     logger.record_tabular(lossname, lossval)
        # logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))


        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, reward_giver.loss_name))
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob))
        batch_size = len(ob) // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch

        for _ in range(optim_epochs // 10):
            for ob_batch, ac_batch in dataset.iterbatches((ob, ac),
                                                          include_final_partial_batch=False,
                                                          batch_size=batch_size):
                ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
                # update running mean/std for reward_giver
                ob_batch = ob_batch[:, -ob_expert.shape[1]:][:-30]
                if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0)[:, :-30])
                # *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
                *newlosses, g = reward_giver.lossandgrad(ob_batch[:, :-30], ob_expert[:, :-30])
                d_adam.update(allmean(g), d_stepsize)
                d_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))


        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rewbuffer.extend(true_rets)
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0 and iters_so_far % 10 == 0:
            disc_losses = np.mean(d_losses, axis=0)
            res = tf.get_default_session().run(merged, feed_dict={
                generator_loss: disc_losses[0],
                expert_loss: disc_losses[1],
                entropy: disc_losses[2],
                entropy_loss: disc_losses[3],
                generator_acc: disc_losses[4],
                expert_acc: disc_losses[5],
                eplenmean: np.mean(lenbuffer),
                eprewmean: np.mean(rewbuffer),
                eptruerewmean: np.mean(true_rewbuffer),
            })
            writer.add_summary(res, iters_so_far)
            writer.flush()

        if rank == 0:
            logger.dump_tabular()


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
