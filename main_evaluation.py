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
                         ActionObjDoorMissionEnv,
                         PutNextLocalMissionEnv, _aug_phrase)
from sampler_lang import BabyAIMissionTaskWrapper, MissionEncoder, MissionParamAdapter
import sampler_lang_policy
import sampler_lang
import sampler_maml
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
import pickle
from maml_rl.utils.reinforcement_learning import reinforce_loss
from maml_rl.episode import BatchEpisodes
from maml_rl.baseline import LinearFeatureBaseline
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


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = argparse.ArgumentParser()
p.add_argument("--env", dest="env_name",
               choices=["GoToLocal","PickupDist","GoToObjDoor","GoToOpen","OpenDoor",
                        "OpenDoorLoc","OpenDoorsOrder","ActionObjDoor","PutNextLocal"],
               default="GoToLocal")
p.add_argument("--room-size", type=int, default=6)
p.add_argument("--num-dists", type=int, default=6)
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


PUTNEXT_MISSIONS = [
    _aug_phrase(c1, t1, c2, t2)
    for c1 in COLORS for t1 in OBJECTS
    for c2 in COLORS for t2 in OBJECTS
    if not (c1 == c2 and t1 == t2)]


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


def select_missions(env_name):
    mission_map = {
        "GoToLocal": LOCAL_MISSIONS,
        "PickupDist": PICKUP_MISSIONS,
        "GoToObjDoor": LOCAL_MISSIONS + DOOR_MISSIONS,
        "GoToOpen": LOCAL_MISSIONS,
        "OpenDoor": OPEN_DOOR_MISSIONS,
        "OpenDoorLoc": OPEN_DOOR_MISSIONS + DOOR_LOC_MISSIONS,
        "OpenDoorsOrder": OPEN_DOORS_ORDER_MISSIONS,
        "ActionObjDoor": ACTION_OBJ_DOOR_MISSIONS,
        "PutNextLocal": PUTNEXT_MISSIONS
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

print(f"env name {env}\n")
print(f"room_size: {room_size}\nnum_dists: {num_dists}\nmax_steps: {max_steps}\n")


# restore saved lang-adapted policy 
ckpt = torch.load(f"lang_model/lang_policy_{env_name}.pth", map_location=device)
with open(f"lang_model/vectorizer_lang_{env_name}.pkl", "rb") as f:
    vectorizer = pickle.load(f)


sampler_lang.vectorizer = vectorizer  
mission_encoder_output_dim = 32
sampler_lang.mission_encoder = MissionEncoder(len(sampler_lang.vectorizer.get_feature_names_out()), 32, 64, mission_encoder_output_dim).to(device)
sampler_lang.mission_encoder.load_state_dict(ckpt["mission_encoder"])
sampler_lang.mission_encoder.eval()
mission_encoder = sampler_lang.mission_encoder
preprocess_obs = sampler_lang.preprocess_obs

dummy_obs, _ = env.reset()
input_size_lang = sampler_lang.preprocess_obs(dummy_obs).shape[0]
output_size = env.action_space.n
hidden_sizes = (64, 64)
nonlinearity = torch.nn.functional.tanh

# Policy language
policy_lang = CategoricalMLPPolicy(
    input_size=input_size_lang,
    output_size=output_size,
    hidden_sizes=hidden_sizes,
    nonlinearity=nonlinearity,
).to(device)  
policy_lang.load_state_dict(ckpt["policy"])
policy_lang.eval()
policy_param_shapes = [p.shape for p in policy_lang.parameters()]

# Adapter
mission_adapter = MissionParamAdapter(mission_encoder_output_dim, policy_param_shapes).to(device)
mission_adapter.load_state_dict(ckpt["mission_adapter"])    
mission_adapter.eval()



# restore saved lang-adapted policy 
ckpt_ablation = torch.load(f"lang_policy_model/lang_policy_{env_name}.pth", map_location=device)
with open(f"lang_policy_model/vectorizer_lang_{env_name}.pkl", "rb") as f:
    vectorizer_ablation = pickle.load(f)

# wire globals for sampler_lang_policy
sampler_lang_policy.device = device
sampler_lang_policy.vectorizer = vectorizer_ablation

# mission encoder for ablation baseline
sampler_lang_policy.mission_encoder = sampler_lang_policy.MissionEncoder(
    len(sampler_lang_policy.vectorizer.get_feature_names_out()),
    hidden_dim1=32, hidden_dim2=64, output_dim=32
).to(device)
sampler_lang_policy.mission_encoder.load_state_dict(ckpt_ablation["mission_encoder"])
sampler_lang_policy.mission_encoder.eval()

# figure out input size for this baseline
dummy_obs, _ = env.reset()
dummy_vec = sampler_lang_policy.preprocess_obs(dummy_obs, mission_str=missions[0])
input_size_ablation = dummy_vec.shape[0]

policy_ablation = CategoricalMLPPolicy(
    input_size=input_size_ablation,
    output_size=output_size,
    hidden_sizes=hidden_sizes,
    nonlinearity=nonlinearity,
).to(device)
policy_ablation.load_state_dict(ckpt_ablation["policy"])
policy_ablation.eval()


# restore saved maml policy
ckpt_base = f"maml_model/maml_{env_name}"
with open(ckpt_base + "_vectorizer.pkl", "rb") as f:
    sampler_maml.vectorizer = pickle.load(f)

# Policy maml
sampler_maml.mission_encoder = sampler_maml.MissionEncoder(
    len(sampler_maml.vectorizer.get_feature_names_out()),
    hidden_dim1=32, hidden_dim2=64, output_dim=32
).to(device)
sampler_maml.mission_encoder.load_state_dict(torch.load(ckpt_base + "_encoder.pth", map_location=device))
sampler_maml.mission_encoder.eval()

dummy_obs, _ = env.reset()
input_size_maml = sampler_maml.preprocess_obs(dummy_obs).shape[0]

policy_maml = CategoricalMLPPolicy(
        input_size=input_size_maml,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        nonlinearity=nonlinearity,      
    ).to(device)

policy_maml.load_state_dict(torch.load(ckpt_base + ".pth", map_location=device))
policy_maml.eval()


baseline = LinearFeatureBaseline(input_size_maml).to(device)


def get_language_adapted_params(policy, mission_str, mission_encoder, mission_adapter, vectorizer, device):
    mission_vec = vectorizer.transform([mission_str]).toarray()[0]
    mission_tensor = torch.from_numpy(mission_vec.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        mission_emb = mission_encoder(mission_tensor)
        mission_emb = mission_emb.to(device)
        delta_thetas = mission_adapter(mission_emb)
        delta_thetas = [delta * delta_theta  for delta in delta_thetas]
    policy_params = list(policy.parameters())
    param_names = list(dict(policy.named_parameters()).keys())
    from collections import OrderedDict
    theta_prime = OrderedDict(
        (name, param + delta.squeeze(0))
        for name, param, delta in zip(param_names, policy_params, delta_thetas)
    )

    return theta_prime

def adapt_policy_for_task(task, policy, num_grad_steps=1, fast_lr=0.5,
                          batch_size=10, baseline=None):
    """
    Perform num_grad_steps inner-loop gradient updates using REINFORCE
    on trajectories from the current task.

    If num_grad_steps == 0, returns None (i.e., use θ without adaptation).
    """
    env.reset_task(task)

    # No adaptation: use initialization θ
    if num_grad_steps == 0:
        return None

    params = None  # start from θ
    for step in range(num_grad_steps):
        batch = BatchEpisodes(batch_size=batch_size, gamma=0.99, device=device)

        # Collect trajectories with current params
        for ep in range(batch_size):
            with silence_sampling_rejected():
                obs, info = env.reset()
            done = False
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            while not done:
                obs_vec = sampler_maml.preprocess_obs(obs)
                obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    if params is not None:
                        dist = policy(obs_tensor, params=params)
                    else:
                        dist = policy(obs_tensor)
                    action = dist.sample().item()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_obs.append(obs_vec)
                episode_actions.append(np.array(action))
                episode_rewards.append(np.array(reward, dtype=np.float32))

            batch.append(episode_obs, episode_actions, episode_rewards,
                         [ep] * len(episode_obs))

        # Advantage + one inner-loop gradient update
        batch.compute_advantages(baseline, gae_lambda=1.0, normalize=True)
        loss = reinforce_loss(policy, batch, params=params)
        params = policy.update_params(loss, params=params,
                                      step_size=fast_lr, first_order=True)

    return params


def evaluate_policy(env, policy,preprocess_obs=None, params=None, max_steps=max_steps, render=False):
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
results_lang_conditioned = []
results_maml_zero = []   # θ only, no adaptation
results_maml_adapt = []  # θ adapted with k gradient steps

print("Comparing language-adapted policy and random policy on random missions:")
for i in range(N_MISSIONS):
    mission = random.choice(missions)

    # 1. Lang-adapted policy
    theta_prime = get_language_adapted_params(policy_lang, mission, mission_encoder, mission_adapter, vectorizer, device)
    lang_steps = []

    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(env, policy_lang, preprocess_obs= sampler_lang.preprocess_obs, params=theta_prime)
        lang_steps.append(steps)
    mean_lang = np.mean(lang_steps)
    std_lang = np.std(lang_steps)
    results_lang.append(mean_lang)


    # 2. Language-conditioned policy WITHOUT meta-learning
    lang_conditioned_policy_steps = []
    preprocess_ablation = lambda obs, m=mission: sampler_lang_policy.preprocess_obs(obs, mission_str=m)

    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(env,
                                policy_ablation,
                                preprocess_obs=preprocess_ablation)
        lang_conditioned_policy_steps.append(steps)

    mean_ablation = np.mean(lang_conditioned_policy_steps)
    results_lang_conditioned.append(mean_ablation)

    # 3. MAML (no adaptation): use θ directly (no inner-loop gradient)
    maml_zero_steps = []
    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(
            env, policy_maml,
            preprocess_obs=sampler_maml.preprocess_obs,
            params=None            # <-- no adaptation
        )
        maml_zero_steps.append(steps)
    results_maml_zero.append(np.mean(maml_zero_steps))

    # 4. MAML (k-shot adaptation): inner-loop gradients with few examples
    #    Choose k = 1 or 2 (reviewer wants "a few examples")
    maml_params = adapt_policy_for_task(
        mission, policy_maml,
        num_grad_steps=2,         # e.g., 2 gradient steps
        fast_lr=0.25,
        batch_size=10,
        baseline=baseline
    )

    maml_adapt_steps = []
    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(
            env, policy_maml,
            preprocess_obs=sampler_maml.preprocess_obs,
            params=maml_params     # <-- adapted parameters
        )
        maml_adapt_steps.append(steps)
    results_maml_adapt.append(np.mean(maml_adapt_steps))

# Results
print("\n===== FINAL AGGREGATE RESULTS =====")
print(f"LA-MAML policy (lang only):      {np.mean(results_lang):.2f} ± {np.std(results_lang):.2f}")
print(f"MAML (θ only, no adaptation):    {np.mean(results_maml_zero):.2f} ± {np.std(results_maml_zero):.2f}")
print(f"MAML (k-shot, 2 grad steps):     {np.mean(results_maml_adapt):.2f} ± {np.std(results_maml_adapt):.2f}")
print(f"language-conditioned policy:   {np.mean(results_lang_conditioned):.2f} ± {np.std(results_lang_conditioned):.2f}")
