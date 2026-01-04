import torch
import torch.multiprocessing as mp
import asyncio
import threading
import time
import multiprocessing as mp
mp.set_start_method("spawn", force=True)


from datetime import datetime, timezone
from copy import deepcopy

from maml_rl.samplers.sampler import Sampler, make_env
from maml_rl.envs.utils.sync_vector_env import SyncVectorEnv
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning import reinforce_loss


def _create_consumer(queue, futures, loop=None):
    if loop is None:
        loop = asyncio.get_event_loop()
    while True:
        data = queue.get()
        if data is None:
            break
        index, step, episodes = data
        future = futures if (step is None) else futures[step]
        if not future[index].cancelled():
            loop.call_soon_threadsafe(future[index].set_result, episodes)


class MultiTaskSampler(Sampler):

    def __init__(self,
                 env_name,
                 env_kwargs,
                 batch_size,
                 policy,
                 baseline,
                 env=None,
                 seed=None,
                 num_workers=0):
        super(MultiTaskSampler, self).__init__(env_name,
                                               env_kwargs,
                                               batch_size,
                                               policy,
                                               seed=seed,
                                               env=env)

        self.num_workers = num_workers
        self.baseline = baseline


        self.task_queue = mp.JoinableQueue()
        self.train_episodes_queue = mp.Queue()
        self.valid_episodes_queue = mp.Queue()
        policy_lock = mp.Lock()

        self.workers = [SamplerWorker(index,
                                      env_name,
                                      env_kwargs,
                                      batch_size,
                                      self.env.observation_space,
                                      self.env.action_space,
                                      self.policy,
                                      deepcopy(baseline),
                                      self.seed,
                                      self.task_queue,
                                      self.train_episodes_queue,
                                      self.valid_episodes_queue,
                                      policy_lock)
            for index in range(num_workers)]
        
        # Workers will sample tasks in parllel not episodes
        for worker in self.workers:
            worker.daemon = True
            worker.start()

        self._waiting_sample = False
        self._event_loop = asyncio.get_event_loop()
        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def sample_tasks(self, num_tasks):
        return self.env.unwrapped.sample_tasks(num_tasks)

    def sample_async(self, tasks, **kwargs):
        if self._waiting_sample:
            raise RuntimeError('Calling `sample_async` while waiting '
                               'for a pending call to `sample_async` '
                               'to complete. Please call `sample_wait` '
                               'before calling `sample_async` again.')

        for index, task in enumerate(tasks):
            self.task_queue.put((index, task, kwargs))  # each task in this form and then sample episodes accordingly
            # print(f"Task {index} added to queue for sampling.")  
        num_steps = kwargs.get('num_steps', 1)
        futures = self._start_consumer_threads(tasks,
                                               num_steps=num_steps)
        self._waiting_sample = True
        return futures

    def sample_wait(self, episodes_futures):
        if not self._waiting_sample:
            raise RuntimeError('Calling `sample_wait` without any '
                               'prior call to `sample_async`.')

        async def _wait(train_futures, valid_futures):
            # Gather the train and valid episodes
            train_episodes = await asyncio.gather(*[asyncio.gather(*futures)
                                                  for futures in train_futures])
            valid_episodes = await asyncio.gather(*valid_futures)
            return (train_episodes, valid_episodes)

        samples = self._event_loop.run_until_complete(_wait(*episodes_futures))
        self._join_consumer_threads()
        self._waiting_sample = False
        return samples

    def sample(self, tasks, **kwargs):
        if self.num_workers == 0:
            train_episodes_all = []
            valid_episodes_all = []
            for index, task in enumerate(tasks):
                self.env.reset_task(task)
                episodes = self.work_single_task(index, **kwargs)
                train_episodes_all.append(episodes[0])
                valid_episodes_all.append(episodes[1])
                print(f"Task {index} sampled with {len(episodes[0])} training episodes and {len(episodes[1])} validation episodes.")
            return ([train_episodes_all], valid_episodes_all)
        else:
            futures = self.sample_async(tasks, **kwargs)
            return self.sample_wait(futures)
        
    def work_single_task(self, index, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):
        from maml_rl.episode import BatchEpisodes
        from maml_rl.utils.reinforcement_learning import reinforce_loss

        from datetime import datetime
        import torch

        envs = SyncVectorEnv(
            [lambda: self.env for _ in range(self.batch_size)],
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )

        try:
            envs.reset(seed=self.seed + index * self.batch_size if self.seed is not None else None)
        except TypeError:
            if hasattr(envs, 'seed'):
                envs.seed(self.seed + index * self.batch_size if self.seed is not None else None)

        def sample_episodes(envs, policy, params=None):
            episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
            observations = envs.reset()
            with torch.no_grad():
                while not envs.dones.all():
                    observations_tensor = torch.from_numpy(observations)
                    pi = policy(observations_tensor, params=params)
                    actions_tensor = pi.sample()
                    actions = actions_tensor.cpu().numpy()
                    new_observations, rewards, _, infos = envs.step(actions)
                    batch_ids = infos['batch_ids']
                    episodes.append(observations, actions, rewards, batch_ids)
                    observations = new_observations

            self.baseline.fit(episodes)
            episodes.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
            return episodes

        params = None
        train_episodes_all = []
        for _ in range(num_steps):
            eps = sample_episodes(envs, self.policy, params=params)
            loss = reinforce_loss(self.policy, eps, params=params)
            params = self.policy.update_params(loss, params=params, step_size=fast_lr, first_order=True)
            train_episodes_all.append(eps)

        valid_eps = sample_episodes(envs, self.policy, params=params)

        return (train_episodes_all, valid_eps)


    @property
    def train_consumer_thread(self):
        if self._train_consumer_thread is None:
            raise ValueError()
        return self._train_consumer_thread

    @property
    def valid_consumer_thread(self):
        if self._valid_consumer_thread is None:
            raise ValueError()
        return self._valid_consumer_thread

    def _start_consumer_threads(self, tasks, num_steps=1):
        # Start train episodes consumer thread
        train_episodes_futures = [[self._event_loop.create_future() for _ in tasks]
                                  for _ in range(num_steps)]
        self._train_consumer_thread = threading.Thread(target=_create_consumer,
            args=(self.train_episodes_queue, train_episodes_futures),
            kwargs={'loop': self._event_loop},
            name='train-consumer')
        self._train_consumer_thread.daemon = True
        self._train_consumer_thread.start()

        # Start valid episodes consumer thread
        valid_episodes_futures = [self._event_loop.create_future() for _ in tasks]
        self._valid_consumer_thread = threading.Thread(target=_create_consumer,
            args=(self.valid_episodes_queue, valid_episodes_futures),
            kwargs={'loop': self._event_loop},
            name='valid-consumer')
        self._valid_consumer_thread.daemon = True
        self._valid_consumer_thread.start()

        return (train_episodes_futures, valid_episodes_futures)

    def _join_consumer_threads(self):
        if self._train_consumer_thread is not None:
            self.train_episodes_queue.put(None)
            self.train_consumer_thread.join()

        if self._valid_consumer_thread is not None:
            self.valid_episodes_queue.put(None)
            self.valid_consumer_thread.join()

        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def close(self):
        if self.closed:
            return

        for _ in range(self.num_workers):
            self.task_queue.put(None)
        self.task_queue.join()
        self._join_consumer_threads()

        self.closed = True


class SamplerWorker(mp.Process):
    def __init__(self,
                 index,
                 env_name,
                 env_kwargs,
                 batch_size,    # number of trajectories per task
                 observation_space,
                 action_space,
                 policy,
                 baseline,
                 seed,
                 task_queue,
                 train_queue,
                 valid_queue,
                 policy_lock):
        super(SamplerWorker, self).__init__()

        env_fns = [make_env(env_name, env_kwargs=env_kwargs)
                   for _ in range(batch_size)]
        self.envs = SyncVectorEnv(env_fns,
                                  observation_space=observation_space,
                                  action_space=action_space)
        try:
            self.envs.reset(seed=seed + index * batch_size if seed is not None else None)
        except TypeError:
            # fallback for older gym versions
            if hasattr(self.envs, 'seed'):
                self.envs.seed(seed + index * batch_size if seed is not None else None)

        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline

        self.task_queue = task_queue
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.policy_lock = policy_lock

    def sample(self,
               index,
               num_steps=1,
               fast_lr=0.5,
               gamma=0.95,
               gae_lambda=1.0,
               device='cpu'):
        # Sample the training trajectories with the initial policy and adapt the
        # policy to the task, based on the REINFORCE loss computed on the
        # training trajectories. The gradient update in the fast adaptation uses
        # `first_order=True` no matter if the second order version of MAML is
        # applied since this is only used for sampling trajectories, and not
        # for optimization.
        params = None
        for step in range(num_steps):
            train_episodes = self.create_episodes(params=params,
                                                  gamma=gamma,
                                                  gae_lambda=gae_lambda,
                                                  device=device)
            train_episodes.log('_enqueueAt', datetime.now(timezone.utc))
            # QKFIX: Deep copy the episodes before sending them to their
            # respective queues, to avoid a race condition. This issue would 
            # cause the policy pi = policy(observations) to be miscomputed for
            # some timesteps, which in turns makes the loss explode.
            self.train_queue.put((index, step, deepcopy(train_episodes)))

            with self.policy_lock:
                loss = reinforce_loss(self.policy, train_episodes, params=params)
                params = self.policy.update_params(loss,
                                                   params=params,
                                                   step_size=fast_lr,
                                                   first_order=True)

        # Sample the validation trajectories with the adapted policy
        valid_episodes = self.create_episodes(params=params,
                                              gamma=gamma,
                                              gae_lambda=gae_lambda,
                                              device=device)
        valid_episodes.log('_enqueueAt', datetime.now(timezone.utc))
        self.valid_queue.put((index, None, deepcopy(valid_episodes)))

    def create_episodes(self,
                        params=None,
                        gamma=0.95,
                        gae_lambda=1.0,
                        device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size,
                                 gamma=gamma,
                                 device=device)
        episodes.log('_createdAt', datetime.now(timezone.utc))
        episodes.log('process_name', self.name)

        t0 = time.time()
        for item in self.sample_trajectories(params=params):
            # print("********************************************************",item)
            episodes.append(*item)
        episodes.log('duration', time.time() - t0)

        self.baseline.fit(episodes)
        episodes.compute_advantages(self.baseline,
                                    gae_lambda=gae_lambda,
                                    normalize=True)
        return episodes

    def sample_trajectories(self, params=None):
        step_count = 0
        t0 = time.time()    #added line
        observations = self.envs.reset()
        # print(observations)
        # print(f"[{self.name}] Env reset in {time.time() - t0:.2f}s")    #added line
        with torch.no_grad():
            # for _ in range(10):                
            while not self.envs.dones.all():
                step_count += 1
                # print(f"[{self.name}] Step #{step_count}, dones: {self.envs.dones}")
                observations_tensor = torch.from_numpy(observations)
                pi = self.policy(observations_tensor, params=params)
                actions_tensor = pi.sample()
                actions = actions_tensor.cpu().numpy()

                t1 = time.time()    #added line
                new_observations, rewards, _, infos = self.envs.step(actions)
                # print("---------------------------------------------")
                # print(new_observations, rewards, _, infos)
                # print(f"[{self.name}] Step took {time.time() - t1:.2f}s")    #added line
                
                batch_ids = infos['batch_ids']
                yield (observations, actions, rewards, batch_ids)
                observations = new_observations

    def run(self):
        while True:
            data = self.task_queue.get()

            if data is None:
                self.envs.close()
                self.task_queue.task_done()
                break

            index, task, kwargs = data
            self.envs.reset_task(task)
            self.sample(index, **kwargs)
            self.task_queue.task_done()
            # print(f"[{self.name}] Task {index} completed sampling.")
