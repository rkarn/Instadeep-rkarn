import os
import ray
from ray import tune

if __name__ == "__main__":

    if "REDIS_PASSWORD" in os.environ:
        ray.init(
            address=os.environ.get("RAY_SERVER", "auto"),
            _redis_password=os.environ.get("REDIS_PASSWORD", ""),
        )
    else:
        # according to the docs local_mode, if true, forces serial execution which is meant for debugging
        # unfortunately, it also allows requests for resources such as GPUs to subsequently ignore them without
        # any error or warning
        ray.init()

    analysis = tune.run(
        "PPO",
        name="pong",
        metric="episode_reward_mean",
        mode="max",
        # stop={"training_iteration": 100},
        config = {
            "env": "PongNoFrameskip-v4",
            "framework": "tf",
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "clip_rewards": True,
            "clip_param": 0.1,
            "vf_clip_param": 10.0,
            "entropy_coeff": 0.01,
            "train_batch_size": 5000,
            "rollout_fragment_length": 20,
            "sgd_minibatch_size": 500,
            "num_sgd_iter": 10,
            "num_workers": 8,
            "num_envs_per_worker": 5,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "num_gpus": 0,
            "model": {
                "dim": 42,
                "vf_share_layers": True,
            }
        }
    )
