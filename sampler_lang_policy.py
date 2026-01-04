import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from maml_rl.episode import BatchEpisodes

import builtins, io
from contextlib import contextmanager, redirect_stdout
from sentence_transformers import SentenceTransformer

_SENT_MODEL = None
_SENT_CACHE = {}

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
        with redirect_stdout(buf):
            yield
    finally:
        builtins.print = real_print

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Mission Encoder
# class MissionEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim1=32, hidden_dim2=64, output_dim=32):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim1),
#             nn.ReLU(),
#             nn.Linear(hidden_dim1, hidden_dim2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim2, output_dim)
#         )
#     def forward(self, x):
#         return self.net(x)

# Mission Wrapper
class BabyAIMissionTaskWrapper(gym.Wrapper):
    def __init__(self, env, missions=None):
        assert missions is not None, "tasks not there"
        super().__init__(env)
        self.missions = missions
        self.current_mission = None

    def sample_tasks(self, n_tasks):
        return list(np.random.choice(self.missions, n_tasks, replace=False))

    def reset_task(self, mission):
        self.current_mission = mission
        if hasattr(self.env, 'set_forced_mission'):
            self.env.set_forced_mission(mission)

    def reset(self, **kwargs):        
        result = super().reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        if self.current_mission is not None:
            obs['mission'] = self.current_mission
        if isinstance(result, tuple):
            return obs, info
        else:
            return obs
        

def _get_sent_model(device):
    global _SENT_MODEL
    if _SENT_MODEL is None:
        # loads once per process (important for num_workers > 0)
        _SENT_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
        _SENT_MODEL.eval()
    return _SENT_MODEL


def preprocess_obs(obs, mission_str):

    image = obs["image"].flatten() / 255.0
    direction = np.eye(4)[obs["direction"]]
    
    base = np.concatenate([image, direction]).astype(np.float32)

    if mission_str is None:
        return base

    # ---- sentence embedding for mission_str ----
    emb = _SENT_CACHE.get(mission_str)
    if emb is None:
        model = _get_sent_model(device)
        with torch.no_grad():
            e = model.encode(
                [mission_str],
                convert_to_tensor=True,
                normalize_embeddings=True
            )[0]  # torch tensor [d]
        emb = e.detach().cpu().numpy().astype(np.float32)
        _SENT_CACHE[mission_str] = emb
    return np.concatenate([base, emb]).astype(np.float32)    

# Sampler
class MultiTaskSampler(object):
    def __init__(self,
                 env=None,
                 env_fn=None,
                 batch_size=None,
                 policy=None,
                 baseline=None,
                 seed=None,
                 num_workers=0):
        assert env is not None, "Must pass BabyAI env"
        assert env_fn is not None, "Must pass env_fn constructor"
        assert policy is not None
        assert baseline is not None

        self.env = env
        self.env_fn = env_fn
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.seed = seed
        self.num_workers = num_workers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_tasks(self, num_tasks):
        return self.env.sample_tasks(num_tasks)

    def sample(self, meta_batch_size, gamma=0.95, gae_lambda=1.0, device='cpu'):

        tasks = self.sample_tasks(meta_batch_size)  

        valid_episodes_all = []
        step_counts = []

        for mission in tasks:
            env = self.env_fn()
            env.reset_task(mission)

            batch = BatchEpisodes(batch_size=self.batch_size,
                                  gamma=gamma,
                                  device=self.device)
            batch.mission = mission

            total_steps = 0

            for ep in range(self.batch_size):
                with silence_sampling_rejected():
                    obs, info = env.reset()
                done = False
                while not done:
                    obs_vec = preprocess_obs(obs, mission_str=mission)
                    obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        pi = self.policy(obs_tensor)
                        action = pi.sample().item()

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    batch.append(
                        [obs_vec],
                        [np.array(action)],
                        [np.array(reward, dtype=np.float32)],
                        [np.array(ep)]
                    )

                    obs = next_obs
                    total_steps += 1

            self.baseline.fit(batch)
            batch.compute_advantages(self.baseline,
                                     gae_lambda=gae_lambda,
                                     normalize=True)

            valid_episodes_all.append(batch)
            step_counts.append(total_steps)

        return valid_episodes_all, step_counts