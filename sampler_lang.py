import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
import gymnasium as gym
from maml_rl.episode import BatchEpisodes
from sentence_transformers import SentenceTransformer
# import logging

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

vectorizer = None
mission_encoder = None

def rollout_one_task(args):
    (make_env_fn, mission, policy_cls, policy_kwargs, 
     policy_state_dict,adapted_params_cpu, batch_size, gamma) = args
    
    import warnings
    import logging
    warnings.filterwarnings("ignore") # Silence everything in workers
    logging.getLogger("gym").setLevel(logging.ERROR) # Silence gym specific prints

    env = make_env_fn()
    env.reset_task(mission)

    policy = policy_cls(**policy_kwargs)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    obs_list, action_list, reward_list, episode_list = [], [], [], []
    total_steps = 0

    for episode in range(batch_size):
        with silence_sampling_rejected():
            obs, info = env.reset()
        done = False; steps = 0
        while not done:
            obs_vec = preprocess_obs(obs)
            with torch.no_grad():
                pi = policy(torch.from_numpy(obs_vec[None, :]).float(), 
                            params=adapted_params_cpu)
                action = pi.sample().item()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            obs_list.append(obs_vec)
            action_list.append(action)
            reward_list.append(reward)
            episode_list.append(episode)
        total_steps += steps

    return (mission, total_steps, obs_list, action_list, reward_list, episode_list)


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
        

# Mission Encoder
class MissionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1=32, hidden_dim2=64, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )
    def forward(self, x):
        return self.net(x)


class SentenceMissionEncoder(nn.Module):
    """
    Mission text -> sentence embedding (SBERT / SentenceTransformer)

    This is "frozen" by default: the pretrained backbone does NOT get updated.
    Your MissionParamAdapter and policy still learn normally.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", frozen=True, normalize=True, cache=True, device=None):
        super().__init__()
        self.normalize = normalize
        self.cache = cache
        self._cache = {}  # mission_str -> embedding tensor on CPU (to save GPU mem)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # SentenceTransformer is itself a torch module
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.output_dim = self.model.get_sentence_embedding_dimension()

        if frozen:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, missions):
        # missions: str or list[str]
        if isinstance(missions, str):
            missions = [missions]

        # simple caching helps because missions repeat a lot
        out = []
        to_encode = []
        idxs = []
        for i, m in enumerate(missions):
            if self.cache and m in self._cache:
                out.append(self._cache[m].to(self.device))
            else:
                out.append(None)
                to_encode.append(m)
                idxs.append(i)

        if len(to_encode) > 0:
            with torch.no_grad():
                emb = self.model.encode(
                    to_encode,
                    convert_to_tensor=True,
                    normalize_embeddings=self.normalize
                )  # [k, d] on self.device (because SentenceTransformer got device)

            # fill + cache (store cache on CPU)
            for j, i in enumerate(idxs):
                e = emb[j]
                out[i] = e
                if self.cache:
                    self._cache[to_encode[j]] = e.detach().cpu()

        return torch.stack(out, dim=0)  # [B, d]
        

# MissionParamAdapter 
class MissionParamAdapter(nn.Module):
    def __init__(self, mission_adapter_input_dim, policy_param_shapes):
        super().__init__()
        self.policy_param_shapes = policy_param_shapes
        total_params = sum([torch.Size(shape).numel() for shape in policy_param_shapes])
        self.net = nn.Sequential(
            nn.Linear(mission_adapter_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, total_params),
            nn.Tanh()  
        )
    def forward(self, mission_emb):

        out = self.net(mission_emb)  
        split_points = []
        total = 0
        for shape in self.policy_param_shapes:
            num = torch.Size(shape).numel()
            split_points.append(total + num)
            total += num
        chunks = torch.split(out, [torch.Size(shape).numel() for shape in self.policy_param_shapes], dim=1)
        reshaped = [chunk.view(-1, *shape) for chunk, shape in zip(chunks, self.policy_param_shapes)]
        return reshaped 
     

def preprocess_obs(obs):

    image = obs["image"].flatten() / 255.0
    direction = np.eye(4)[obs["direction"]]
    
    return np.concatenate([image, direction])
    

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
        self.env = env
        self.env_fn = env_fn
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.seed = seed
        self.num_workers = num_workers

    def sample_tasks(self, num_tasks):
        return self.env.sample_tasks(num_tasks)

    def sample(self, meta_batch_size, meta_learner, gamma=0.95, gae_lambda=1.0, device='cpu'):

        tasks = self.sample_tasks(meta_batch_size)  
        all_step_counts = []
        valid_episodes_all = []
        if (self.num_workers or 0) > 0:
            assert self.env_fn is not None, "env_fn no there "
 
            policy_state_dict_cpu = {k: v.cpu() for k, v in self.policy.state_dict().items()}
            policy_cls = self.policy.__class__
            policy_kwargs = dict(
                input_size=self.policy.input_size,
                output_size=self.policy.output_size,
                hidden_sizes=self.policy.hidden_sizes,
                nonlinearity=self.policy.nonlinearity
            )

            tasks = self.sample_tasks(meta_batch_size)

            # compute theta' per task
            adapted_params_cpu = []
            for t in tasks:
                theta_prime = meta_learner.adapt_one(t)
                adapted_params_cpu.append({k: v.detach().cpu() for k, v in theta_prime.items()})

            worker_args = []
            for t, p in zip(tasks, adapted_params_cpu):
                worker_args.append((self.env_fn, t, policy_cls, policy_kwargs,
                                    policy_state_dict_cpu, p, self.batch_size, gamma))

            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                results = list(ex.map(rollout_one_task, worker_args))

            valid_episodes_all, all_step_counts = [], []
            for (mission, step_count, obs_list, action_list, reward_list, episode_list) in results:
                batch_episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                batch_episodes.mission = mission
                for obs, action, reward, episode in zip(obs_list, action_list, reward_list, episode_list):
                    batch_episodes.append([obs], [np.array(action)], [np.array(reward)], [np.array(episode)])
                self.baseline.fit(batch_episodes)
                batch_episodes.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                valid_episodes_all.append(batch_episodes)
                all_step_counts.append(step_count)

            return (valid_episodes_all, all_step_counts)