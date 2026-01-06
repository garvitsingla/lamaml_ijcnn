import torch
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence
from maml_rl.metalearners.base import GradientBasedMetaLearner
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters)
from maml_rl.utils.optimization import conjugate_gradient
from collections import OrderedDict


class MAMLTRPO(GradientBasedMetaLearner):

    def __init__(self,  
                 policy,
                 mission_encoder,
                 mission_adapter,
                 fast_lr=0.5,
                 delta_theta=None,
                 first_order=False,
                 device='cpu'):
        super(MAMLTRPO, self).__init__(policy, device=device)
        self.fast_lr = fast_lr
        self.delta_theta = delta_theta
        self.first_order = first_order
        self.mission_encoder = mission_encoder
        self.mission_adapter = mission_adapter
 
    def adapt_one(self, mission_str):
        
        if mission_str is None:
            raise RuntimeError("Mission string is None! Make sure each BatchEpisodes has a valid mission.")

        # SentenceTransformer encoding (handles tokenization internally)
        mission_emb = self.mission_encoder(mission_str)
        mission_emb = mission_emb.to(next(self.mission_adapter.parameters()).device)

        # Mission Adapter
        delta_thetas = self.mission_adapter(mission_emb)

        # scale the delta thetas
        delta_thetas = [delta * self.delta_theta for delta in delta_thetas]  
       
        # task adapted parameters dict (theta_prime)
        policy_params = list(self.policy.parameters())
        param_names = list(dict(self.policy.named_parameters()).keys())
        theta_prime = OrderedDict(
            (name, param + delta.squeeze(0))
            for name, param, delta in zip(param_names, policy_params, delta_thetas)
        )

        return theta_prime


    def hessian_vector_product(self, kl, meta_params, damping=1e-2):
        grads = torch.autograd.grad(kl,
                                    meta_params,
                                    create_graph=True)      
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                         meta_params,
                                         retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product


    def surrogate_loss(self, train_futures, valid_futures, old_pi=None):
        
        if not isinstance(train_futures, list):
            train_futures = [train_futures]
        if not isinstance(valid_futures, list):
            valid_futures = [valid_futures]

        task_params_list = [self.adapt_one(getattr(valid_batch, "mission", None)) for valid_batch in valid_futures]

        with torch.set_grad_enabled(old_pi is None):
            losses = []
            kls = []
            old_pis = []
            for task_params, valid_episodes in zip(task_params_list, valid_futures):
                pi = self.policy(valid_episodes.observations, params=task_params)

                if old_pi is None:
                    old_pi_task = detach_distribution(pi)
                else:
                    old_pi_task = old_pi

                if isinstance(old_pi_task, list):
                    old_log_prob = old_pi_task[0].log_prob(valid_episodes.actions)
                else:
                    old_log_prob = old_pi_task.log_prob(valid_episodes.actions)
                log_ratio = pi.log_prob(valid_episodes.actions) - old_log_prob

                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * valid_episodes.advantages,
                                    lengths=valid_episodes.lengths)
                
                if isinstance(old_pi_task, list):
                    kl_vals = [kl_divergence(pi, q) for q in old_pi_task]
                    kl = weighted_mean(torch.stack(kl_vals).mean(dim=0), lengths=valid_episodes.lengths)
                else:
                    kl = weighted_mean(kl_divergence(pi, old_pi_task), lengths=valid_episodes.lengths)

                losses.append(loss)
                kls.append(kl)
                old_pis.append(old_pi_task)

        return torch.stack(losses).mean(), torch.stack(kls).mean(), old_pis
    

    def step(self,
             train_futures,  
             valid_futures,
             max_kl=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5):
        num_tasks = len(train_futures)
        logs = {}

        # Compute the surrogate loss
        old_losses, old_kls, old_pis = self._async_gather([
            self.surrogate_loss(train, valid, old_pi=None)
            for (train, valid) in zip(train_futures, valid_futures)])

        logs['loss_before'] = to_numpy(old_losses)
        logs['kl_before'] = to_numpy(old_kls)

        old_loss = sum(old_losses) / num_tasks

        meta_params = list(self.policy.parameters()) 
                        
        grads = torch.autograd.grad(old_loss,
                                    meta_params,
                                    retain_graph=True)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl,meta_params=meta_params,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir,
                              hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(meta_params)

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 meta_params)

            losses, kls, _ = self._async_gather([
                self.surrogate_loss(train, valid, old_pi=old_pi)
                for (train, valid, old_pi)
                in zip(train_futures, valid_futures, old_pis)])

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks   
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                logs['loss_after'] = to_numpy(losses)
                logs['kl_after'] = to_numpy(kls)
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params,meta_params)

        return logs
