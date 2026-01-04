import torch.multiprocessing as mp
from functools import partial
import numpy as np
import torch
import pickle
import gc
import time
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
from maml_rl.utils.reinforcement_learning import reinforce_loss, get_returns
from sklearn.feature_extraction.text import CountVectorizer
import sampler_lang_policy as S
from sampler_lang_policy import (
    BabyAIMissionTaskWrapper,
    MultiTaskSampler,
    preprocess_obs,
)
from environment import (LOCAL_MISSIONS,
                        DOOR_MISSIONS,
                        OPEN_DOOR_MISSIONS,
                        DOOR_LOC_MISSIONS,
                        PICKUP_MISSIONS,
                        OPEN_DOORS_ORDER_MISSIONS,
                        ACTION_OBJ_DOOR_MISSIONS,
                        PUTNEXT_MISSIONS)                        
from environment import (GoToLocalMissionEnv, 
                            GoToOpenMissionEnv, 
                            GoToObjDoorMissionEnv, 
                            PickupDistMissionEnv,
                            OpenDoorMissionEnv,
                            OpenDoorLocMissionEnv,
                            OpenDoorsOrderMissionEnv,
                            ActionObjDoorMissionEnv,
                            PutNextLocalMissionEnv)
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S.device = device 

# argparser
p = argparse.ArgumentParser()
p.add_argument("--env", dest="env_name",
               choices=["GoToLocal","PickupDist","GoToObjDoor","GoToOpen","OpenDoor",
                        "OpenDoorLoc","OpenDoorsOrder","ActionObjDoor","PutNextLocal"],
               default="GoToLocal")
p.add_argument("--room-size", type=int, default=7)
p.add_argument("--num-dists", type=int, default=3)
p.add_argument("--max-steps", type=int, default=300)
p.add_argument("--delta-theta", type=int, default=0.7)
p.add_argument("--meta-iters", type=int, default=50, help="number of meta-batches")
p.add_argument("--batch-size", type=int, default=50, help="episodes per meta-batch (per task)")
p.add_argument("--num-workers", type=int, default=4)

args = p.parse_args()


# Build the environment
def build_env(env, room_size, num_dists, max_steps, missions):

    if env == "GoToLocal":
        base = GoToLocalMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env == "PickupDist":
        base = PickupDistMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env == "GoToObjDoor":
        base = GoToObjDoorMissionEnv(max_steps=max_steps, num_distractors=num_dists)
    elif env == "GoToOpen":
        base = GoToOpenMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env == "OpenDoor":
        base = OpenDoorMissionEnv(room_size=room_size, max_steps=max_steps)
    elif env == "OpenDoorLoc":
        base = OpenDoorLocMissionEnv(room_size=room_size, max_steps=max_steps)
    elif env == "OpenDoorsOrder":
        base = OpenDoorsOrderMissionEnv(room_size=room_size)
    elif env == "ActionObjDoor":
        base = ActionObjDoorMissionEnv(objects=None, door_colors=None, obj_colors=None, num_dists=num_dists)
    elif env == "PutNextLocal":
        base = PutNextLocalMissionEnv(room_size=room_size, max_steps=max_steps, num_dists=None)
    else:
        raise ValueError(f"Unknown env_name: {env}")

    return BabyAIMissionTaskWrapper(base, missions=missions)


# Select for missions based on environment
def select_missions(env):

    if env == "GoToLocal":
        return LOCAL_MISSIONS
    if env == "PickupDist":
        return PICKUP_MISSIONS
    if env == "GoToObjDoor":
        return (LOCAL_MISSIONS + DOOR_MISSIONS)
    if env == "GoToOpen":
        return LOCAL_MISSIONS
    if env == "OpenDoor":
        return OPEN_DOOR_MISSIONS
    if env == "OpenDoorLoc":
        return (OPEN_DOOR_MISSIONS + DOOR_LOC_MISSIONS)
    if env == "OpenDoorsOrder":
        return OPEN_DOORS_ORDER_MISSIONS
    if env == "ActionObjDoor":
        return ACTION_OBJ_DOOR_MISSIONS
    if env == "PutNextLocal":
        return PUTNEXT_MISSIONS 
    raise ValueError(f"Unknown env for missions/vocab: {env}")


def main():

    env_name  = args.env_name
    room_size = args.room_size
    num_dists = args.num_dists
    max_steps = args.max_steps
    delta_theta = args.delta_theta
    num_workers = args.num_workers
    num_batches = args.meta_iters
    batch_size = args.batch_size


    missions = select_missions(env_name)

    make_env = partial(
        build_env,
        env_name,
        room_size,
        num_dists,
        max_steps,
        missions
    )

    env = make_env()
    print(f"Using environment: {env_name}\n"
          f"room_size: {room_size}  num_dists: {num_dists}  max_steps: {max_steps}  ")


    # Finding Policy Parameters
    env.reset_task(missions[0])
    obs, _ = env.reset()
    vec = preprocess_obs(obs, mission_str=missions[0])
    input_size = vec.shape[0]
    hidden_sizes = (64, 64)
    nonlinearity = torch.nn.functional.tanh
    output_size = env.action_space.n


    policy = CategoricalMLPPolicy(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        nonlinearity=nonlinearity,
    ).to(device)
    policy.share_memory()
    baseline = LinearFeatureBaseline(input_size).to(device)

    
    # Sampler setup
    sampler = MultiTaskSampler(
        env=env,
        env_fn=make_env,
        batch_size=batch_size,
        policy=policy,
        baseline=baseline,
        seed=1,
        num_workers=num_workers
    )


    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # Training loop
    avg_steps_per_batch = []
    meta_batch_size = globals().get("meta_batch_size") or min(5, len(env.missions))

    start_time = time.time()

    for batch in range(num_batches):
        print(f"\nBatch {batch + 1}/{num_batches}")
        valid_episodes, step_counts = sampler.sample(meta_batch_size=meta_batch_size ,gamma=0.99, gae_lambda=1.0, device=device)
        
        # Compute policy-gradient loss over all tasks (NO meta-learner)
        optimizer.zero_grad()
        total_loss = 0.0

        for episodes in valid_episodes:
            loss = reinforce_loss(policy, episodes)
            total_loss += loss

        if len(valid_episodes) > 0:
            total_loss = total_loss / len(valid_episodes)

        total_loss.backward()
        optimizer.step()


        avg_steps = np.mean(step_counts) if len(step_counts) > 0 else float('nan')
        avg_steps_per_episode = avg_steps / sampler.batch_size 
        avg_steps_per_batch.append(avg_steps_per_episode)
        print(f"Average steps in Meta-batch {batch+1}: {avg_steps_per_episode}\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    # Save the trained meta-policy parameters
    torch.save({
        "policy": policy.state_dict()
    }, f"lang_policy_model/lang_policy_{env_name}.pth")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()