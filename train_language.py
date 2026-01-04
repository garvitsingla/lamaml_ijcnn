import os
import warnings
import logging
# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore") 
logging.getLogger("gymnasium").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.multiprocessing as mp
from functools import partial
import numpy as np
import torch
import pickle
import gc
import time
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
from maml_rl.metalearners.lang_trpo import MAMLTRPO
import sampler_lang as S
from sampler_lang import (BabyAIMissionTaskWrapper, 
                        MissionEncoder,
                        SentenceMissionEncoder,
                        MissionParamAdapter, 
                        MultiTaskSampler, 
                        preprocess_obs)
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
    raise ValueError(f"Unknown env for missions: {env}")


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
          f"room_size: {room_size}  num_dists: {num_dists}  max_steps: {max_steps}  "
          f"delta_theta: {delta_theta}")

    # Policy setup 
    hidden_sizes = (64, 64)
    nonlinearity = torch.nn.functional.tanh

    # Instantiate the encoder
    # S.vectorizer = vectorizer 
    # mission_encoder_input_dim = len(S.vectorizer.get_feature_names_out())
    mission_encoder = SentenceMissionEncoder(
        model_name="all-MiniLM-L6-v2",
        frozen=True,          # keep backbone fixed (recommended first)
        normalize=True,       # stable scale
        cache=True,           # missions repeat, speeds up
        device=device
    )
    mission_encoder_output_dim = mission_encoder.output_dim
    mission_adapter_input_dimension = mission_encoder_output_dim

    # mission_encoder = MissionEncoder(mission_encoder_input_dim,  hidden_dim1=32, hidden_dim2=64, output_dim=mission_encoder_output_dim).to(device)
    # S.mission_encoder = mission_encoder  

    # Finding Policy Parameters shape
    obs, _ = env.reset()
    vec = preprocess_obs(obs)
    input_size = vec.shape[0]
    output_size = env.action_space.n


    policy = CategoricalMLPPolicy(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        nonlinearity=nonlinearity,
    ).to(device)
    policy.share_memory()
    baseline = LinearFeatureBaseline(input_size).to(device)

    policy_param_shapes = [p.shape for p in policy.parameters()]

    mission_adapter_input_dimension = mission_encoder_output_dim
    mission_adapter = MissionParamAdapter(mission_adapter_input_dimension, policy_param_shapes).to(device)

    
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

    # Meta-learner setup
    meta_learner = MAMLTRPO(
        policy=policy,
        mission_encoder=mission_encoder,
        mission_adapter=mission_adapter,
        # vectorizer=vectorizer,
        delta_theta=delta_theta,
        fast_lr=1e-4,
        first_order=True,
        device=device
    )

    # Training loop
    avg_steps_per_batch = []
    meta_batch_size = globals().get("meta_batch_size") or min(5, len(env.missions))

    start_time = time.time()

    for batch in range(num_batches):
        print(f"\nBatch {batch + 1}/{num_batches}")
        valid_episodes, step_counts = sampler.sample(
            meta_batch_size,
            meta_learner,
            gamma=0.99,
            gae_lambda=1.0,
            device=device
        )
        
        avg_steps = np.mean(step_counts) if len(step_counts) > 0 else float('nan')
        avg_steps_per_episode = avg_steps / sampler.batch_size 
        avg_steps_per_batch.append(avg_steps_per_episode)
        print(f"Average steps in Meta-batch {batch+1}: {avg_steps_per_episode}\n")

        meta_learner.step(valid_episodes,valid_episodes)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")


    os.makedirs("lang_model", exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "mission_encoder": mission_encoder.state_dict(),
        "mission_adapter": mission_adapter.state_dict(),
    }, f"lang_model/lang_{env_name}.pth")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()