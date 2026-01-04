import torch
import numpy as np
import random
from functools import partial
from environment import (GoToLocalMissionEnv, 
                         GoToOpenMissionEnv, 
                         GoToObjDoorMissionEnv,  
                         PickupDistMissionEnv,
                         OpenDoorMissionEnv, 
                         OpenDoorLocMissionEnv,
                         OpenDoorsOrderMissionEnv,
                         ActionObjDoorMissionEnv)
from sampler_lang import BabyAIMissionTaskWrapper, MissionEncoder, MissionParamAdapter
import sampler_lang 
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
import pickle
from collections import OrderedDict
import argparse

import builtins, io
from contextlib import contextmanager, redirect_stdout, redirect_stderr

@contextmanager
def silence_sampling_rejected():
    real_print = builtins.print
    buf = io.StringIO()
    def filtered_print(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("Sampling rejected: unreachable object"):
            return
        return real_print(*args, **kwargs)
    builtins.print = filtered_print
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            yield
    finally:
        builtins.print = real_print
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

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
p.add_argument("--skip-random", action="store_true",
               help="Skip the random-policy baseline to speed up evaluation")

args = p.parse_args()



OBJECTS = ['box']
COLORS = ['red', 'green', 'blue', 'purple','yellow', 'grey']
PREP_LOCS = ['on', 'at', 'to']

# Location names
LOC_NAMES = ['right', 'front']

DOOR_COLORS = ['yellow', 'grey']

# For Pickup
PICKUP_MISSIONS = [f"pick up the {color} {obj}" for color in COLORS for obj in OBJECTS]

# For GoToLocal
LOCAL_MISSIONS = [f"go to the {color} {obj}" for color in COLORS for obj in OBJECTS]

# For environments that include doors (GoToObjDoor, GoToOpen, Open)
DOOR_MISSIONS = [f"go to the {color} door" for color in DOOR_COLORS]
OPEN_DOOR_MISSIONS = [f"open the {color} door" for color in DOOR_COLORS]
DOOR_LOC_MISSIONS = [f"open the door {prep} the {loc}" for prep in PREP_LOCS for loc in LOC_NAMES]
OPEN_TWO_DOORS_MISSIONS = [f"open the {c1} door, then open the {c2} door" for c1 in DOOR_COLORS for c2 in DOOR_COLORS]
OPEN_DOORS_ORDER_MISSIONS = (
    [f"open the {c1} door" for c1 in DOOR_COLORS] +
    [f"open the {c1} door, then open the {c2} door" for c1 in DOOR_COLORS for c2 in DOOR_COLORS] +
    [f"open the {c1} door after you open the {c2} door" for c1 in DOOR_COLORS for c2 in DOOR_COLORS]
)

ACTION_OBJ_DOOR_MISSIONS = (
    [f"pick up the {c} box" for c in COLORS] +
    [f"go to the {c} box"   for c in COLORS] +
    [f"go to the {c} door"   for c in DOOR_COLORS] +
    [f"open a {c} door"      for c in DOOR_COLORS]
)


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
    else:
        raise ValueError(f"Unknown env_name: {env}")

    return BabyAIMissionTaskWrapper(base, missions=missions)


def select_missions(env_name):
    mission_map = {
        "GoToLocal": LOCAL_MISSIONS,
        "PickupDist": PICKUP_MISSIONS,
        "GoToObjDoor": LOCAL_MISSIONS + DOOR_MISSIONS,
        "GoToOpen": LOCAL_MISSIONS,
        "OpenDoor": OPEN_DOOR_MISSIONS,
        "OpenDoorLoc": OPEN_DOOR_MISSIONS + DOOR_LOC_MISSIONS,
        "OpenDoorsOrder": OPEN_DOORS_ORDER_MISSIONS,
        "ActionObjDoor": ACTION_OBJ_DOOR_MISSIONS
    }
    return mission_map[env_name]


env_name  = args.env_name
room_size = args.room_size
num_dists = args.num_dists
max_steps = args.max_steps
delta_theta = args.delta_theta


missions = select_missions(env_name)
make_env = partial(build_env, env_name, room_size, num_dists, max_steps, missions)
env = make_env()
  
print(f"env name {env} \n")

# restore saved lang-adapted policy 

lang_model = torch.load(f"lang_model/lang_policy_{env_name}.pth", map_location=device)
with open(f"lang_model/vectorizer_lang_{env_name}.pkl", "rb") as f:
    vectorizer = pickle.load(f)

sampler_lang.vectorizer = vectorizer  
mission_encoder_output_dim = 32
sampler_lang.mission_encoder = MissionEncoder(len(sampler_lang.vectorizer.get_feature_names_out()), 32, 64, mission_encoder_output_dim).to(device)
sampler_lang.mission_encoder.load_state_dict(lang_model["mission_encoder"])
sampler_lang.mission_encoder.eval()
mission_encoder = sampler_lang.mission_encoder
preprocess_obs = sampler_lang.preprocess_obs

dummy_obs, _ = env.reset()
input_size = preprocess_obs(dummy_obs).shape[0]
output_size = env.action_space.n
hidden_sizes = (64, 64)
nonlinearity = torch.nn.functional.tanh

# Policy language
policy_lang = CategoricalMLPPolicy(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=hidden_sizes,
    nonlinearity=nonlinearity,
).to(device)  
policy_lang.load_state_dict(lang_model["policy"])
policy_lang.eval()
policy_param_shapes = [p.shape for p in policy_lang.parameters()]

# Adapter
mission_adapter = MissionParamAdapter(mission_encoder_output_dim, policy_param_shapes).to(device)
mission_adapter.load_state_dict(lang_model["mission_adapter"])    
mission_adapter.eval()


# restore saved unadapted language policy 

unadpated_lang_policy = torch.load(f"ablation_language_parameters_untrained_model/lang_policy_{env_name}.pth", map_location=device)
with open(f"ablation_language_parameters_untrained_model/vectorizer_lang_{env_name}.pkl", "rb") as g:
    vectorizer_2 = pickle.load(g)


sampler_lang.vectorizer = vectorizer_2  
mission_encoder_output_dim = 32
sampler_lang.mission_encoder = MissionEncoder(len(sampler_lang.vectorizer.get_feature_names_out()), 32, 64, mission_encoder_output_dim).to(device)
sampler_lang.mission_encoder.load_state_dict(unadpated_lang_policy["mission_encoder"])
sampler_lang.mission_encoder.eval()
mission_encoder_2 = sampler_lang.mission_encoder
preprocess_obs = sampler_lang.preprocess_obs

dummy_obs, _ = env.reset()
input_size = preprocess_obs(dummy_obs).shape[0]
output_size = env.action_space.n
hidden_sizes = (64, 64)
nonlinearity = torch.nn.functional.tanh

# Policy unadapted language
policy_unadapted_lang = CategoricalMLPPolicy(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=hidden_sizes,
    nonlinearity=nonlinearity,
).to(device)  

# Adapter
policy_unadapted_lang.load_state_dict(unadpated_lang_policy["policy"])
policy_unadapted_lang.eval()

policy_param_shapes = [p.shape for p in policy_unadapted_lang.parameters()]

mission_adapter_2 = MissionParamAdapter(mission_encoder_output_dim, policy_param_shapes).to(device)
mission_adapter_2.load_state_dict(unadpated_lang_policy["mission_adapter"])    
mission_adapter_2.eval()


def get_language_adapted_params(policy, mission_str, mission_encoder, mission_adapter, vectorizer, device):
    mission_vec = vectorizer.transform([mission_str]).toarray()[0]
    mission_tensor = torch.from_numpy(mission_vec.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        mission_emb = mission_encoder(mission_tensor)
        mission_emb = mission_emb.to(device)
        delta_thetas = mission_adapter(mission_emb)
        delta_thetas = [delta * delta_theta for delta in delta_thetas]
    policy_params = list(policy.parameters())
    param_names = list(dict(policy.named_parameters()).keys())
    theta_prime = OrderedDict(
        (name, param + delta.squeeze(0))
        for name, param, delta in zip(param_names, policy_params, delta_thetas)
    )
    return theta_prime


def evaluate_policy(env, policy, params=None, max_steps=max_steps, render=False):
    with silence_sampling_rejected():
        obs, info = env.reset()
    steps = 0
    done = False
    while not done and steps < max_steps:
        if render:
            env.render("human")
        obs_vec = preprocess_obs(obs)
        obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)
        with torch.no_grad():
            if params is not None:
                dist = policy(obs_tensor, params=params)
            else:
                dist = policy(obs_tensor)
            action = dist.sample().item()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    return steps


# Evaluation
N_MISSIONS = 20
N_EPISODES = 40

results_lang = []
results_lang_unadapted = []

print("Comparing language-adapted policy and random policy on random missions:")
for i in range(N_MISSIONS):
    mission = random.choice(missions)

    # 1. Lang-adapted policy
    sampler_lang.vectorizer = vectorizer
    sampler_lang.mission_encoder = mission_encoder
    theta_prime = get_language_adapted_params(policy_lang, mission, mission_encoder, mission_adapter, vectorizer, device)
    lang_steps = []
    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(env, policy_lang, params=theta_prime)
        lang_steps.append(steps)
    mean_lang = np.mean(lang_steps)
    std_lang = np.std(lang_steps)
    results_lang.append(mean_lang)

    # 2. Unadapted lang policy
    sampler_lang.vectorizer = vectorizer_2
    sampler_lang.mission_encoder = mission_encoder_2
    theta_prime_unadapted = get_language_adapted_params(policy_unadapted_lang, mission, mission_encoder_2, mission_adapter_2, vectorizer_2, device)
    lang_unadapted_steps = []
    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(env, policy_unadapted_lang, params=theta_prime_unadapted)
        lang_unadapted_steps.append(steps)
    mean_lang_unadapted = np.mean(lang_unadapted_steps)
    std_lang_unadapted = np.std(lang_unadapted_steps)
    results_lang_unadapted.append(mean_lang_unadapted)

# Results
print("\n===== FINAL AGGREGATE RESULTS =====")
print(f"LA-MAML policy: {np.mean(results_lang):.2f} ± {np.std(results_lang):.2f}")
print(f"unadapted Lang policy: {np.mean(results_lang_unadapted):.2f} ± {np.std(results_lang_unadapted):.2f}")
