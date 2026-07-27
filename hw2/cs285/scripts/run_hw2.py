import hashlib
import json
import os
import re
import time

import gym
import numpy as np
import torch

from cs285.agents.pg_agent import PGAgent
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.infrastructure.action_noise_wrapper import ActionNoiseWrapper
from cs285.infrastructure.logger import Logger

MAX_NVIDEO = 2


def _checkpoint_config(args):
    """Return the command-line arguments that identify one training run."""
    excluded_args = {
        "checkpoint_dir",
        "checkpoint_freq",
        "checkpoint_path",
        "logdir",
        "n_iter",
        "no_gpu",
        "resume",
        "which_gpu",
    }
    return {
        key: value
        for key, value in vars(args).items()
        if key not in excluded_args
    }


def _checkpoint_path(args):
    config = _checkpoint_config(args)
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:12]

    def sanitize(value):
        return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-_") or "run"

    exp_name = sanitize(args.exp_name)[:40]
    env_name = sanitize(args.env_name)[:40]
    learning_rate = sanitize(f"{args.learning_rate:g}")
    filename = (
        f"{exp_name}_{env_name}_seed{args.seed}_lr{learning_rate}"
        f"_bs{args.batch_size}_{config_hash}.ckpt"
    )
    return os.path.join(args.checkpoint_dir, filename)


def _optimizer_state(agent):
    state = {"actor": agent.actor.optimizer.state_dict()}
    if agent.critic is not None:
        state["critic"] = agent.critic.optimizer.state_dict()
    return state


def _capture_generator_state(generator):
    if hasattr(generator, "bit_generator"):
        return {"type": "generator", "state": generator.bit_generator.state}
    if hasattr(generator, "get_state"):
        return {"type": "random_state", "state": generator.get_state()}
    return None


def _restore_generator_state(generator, saved_state):
    if saved_state is None:
        return
    if saved_state["type"] == "generator" and hasattr(generator, "bit_generator"):
        generator.bit_generator.state = saved_state["state"]
    elif saved_state["type"] == "random_state" and hasattr(generator, "set_state"):
        generator.set_state(saved_state["state"])


def _capture_rng_state(env):
    state = {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    env_rng = getattr(env.unwrapped, "np_random", None)
    environment_state = _capture_generator_state(env_rng)
    if environment_state is not None:
        state["environment"] = environment_state

    action_noise_rng = getattr(env, "rng", None)
    action_noise_state = _capture_generator_state(action_noise_rng)
    if action_noise_state is not None:
        state["action_noise"] = action_noise_state

    return state


def _restore_rng_state(env, state):
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [cuda_state.cpu() for cuda_state in state["torch_cuda"]]
        )

    env_rng = getattr(env.unwrapped, "np_random", None)
    _restore_generator_state(env_rng, state.get("environment"))

    action_noise_rng = getattr(env, "rng", None)
    _restore_generator_state(action_noise_rng, state.get("action_noise"))


def _save_checkpoint(
    checkpoint_path,
    args,
    agent,
    env,
    iteration,
    total_envsteps,
    elapsed_time,
):
    checkpoint = {
        "iteration": iteration,
        "total_envsteps": total_envsteps,
        "elapsed_time": elapsed_time,
        "config": _checkpoint_config(args),
        "agent": agent.state_dict(),
        "optimizers": _optimizer_state(agent),
        "rng": _capture_rng_state(env),
    }
    temporary_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def _load_checkpoint(checkpoint_path, args, agent, env):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found for these command-line arguments: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=ptu.device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=ptu.device)

    expected_config = _checkpoint_config(args)
    if checkpoint["config"] != expected_config:
        raise ValueError(
            "Checkpoint arguments do not match the current command-line arguments."
        )

    agent.load_state_dict(checkpoint["agent"])
    agent.actor.optimizer.load_state_dict(checkpoint["optimizers"]["actor"])
    if agent.critic is not None:
        agent.critic.optimizer.load_state_dict(checkpoint["optimizers"]["critic"])
    _restore_rng_state(env, checkpoint["rng"])

    print(
        f"Resumed from {checkpoint_path} "
        f"(completed iteration {checkpoint['iteration']})"
    )
    return checkpoint


def run_training_loop(args):
    logger = Logger(args.logdir)

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = gym.make(args.env_name, render_mode=None)
    #from gym.vector import AsyncVectorEnv          
    #num_envs = 16  
    #env = AsyncVectorEnv([ lambda: gym.make(args.env_name, render_mode=None) for _ in range(num_envs)])
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # add action noise, if needed
    if args.action_noise_std > 0:
        assert not discrete, f"Cannot use --action_noise_std for discrete environment {args.env_name}"
        env = ActionNoiseWrapper(env, args.seed, args.action_noise_std)

    max_ep_len = args.ep_len or env.spec.max_episode_steps

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
    )

    checkpoint_path = _checkpoint_path(args)
    total_envsteps = 0
    start_iteration = 0
    elapsed_time = 0.0

    if args.resume:
        checkpoint = _load_checkpoint(checkpoint_path, args, agent, env)
        total_envsteps = checkpoint["total_envsteps"]
        start_iteration = checkpoint["iteration"] + 1
        elapsed_time = checkpoint["elapsed_time"]

    start_time = time.time() - elapsed_time

    for itr in range(start_iteration, args.n_iter):
        print(f"\n********** Iteration {itr} ************")
        # TODO: sample `args.batch_size` transitions using utils.sample_trajectories
        # make sure to use `max_ep_len`
        trajs, envsteps_this_batch = utils.sample_trajectories(env,agent.actor,
                                                               args.batch_size,max_ep_len)
        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        # TODO: train the agent using the sampled trajectories and the agent's update function
        train_info: dict = agent.update(obs=trajs_dict["observation"],actions=trajs_dict["action"],
                                        rewards=trajs_dict["reward"],terminals=trajs_dict["terminal"])

        if itr % args.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, args.eval_batch_size, max_ep_len
            )

            logs = utils.compute_metrics(trajs, eval_trajs)
            # compute additional metrics
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            logger.flush()

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

        if (itr + 1) % args.checkpoint_freq == 0 or itr + 1 == args.n_iter:
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                args=args,
                agent=agent,
                env=env,
                iteration=itr,
                total_envsteps=total_envsteps,
                elapsed_time=time.time() - start_time,
            )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    parser.add_argument("--action_noise_std", type=float, default=0)
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=1,
        help="Save a checkpoint every N training iterations.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory used for parameter-named checkpoints (default: hw2/data/checkpoints).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint matching the current experiment arguments.",
    )

    args = parser.parse_args()
    if args.checkpoint_freq <= 0:
        parser.error("--checkpoint_freq must be a positive integer")

    # create directory for logging
    logdir_prefix = "q2_pg_"  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(data_path, "checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logdir = (
        logdir_prefix
        + args.exp_name
        + "_"
        + args.env_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    args.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    run_training_loop(args)


if __name__ == "__main__":
    main()
