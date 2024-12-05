import functools
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union, cast

import ale_py
import gymnasium as gym
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tensordict import TensorDict
from torch.distributed.fsdp import BackwardPrefetch, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer,
    ProcessorMixin,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    get_scheduler,
    set_seed,
)
from transformers.trainer_pt_utils import get_module_class_from_name

from game_agent import Qwen2VLInputProcessor, get_action_meanings

gym.register_envs(ale_py)


class RunningStats:
    pass


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Adapted from https://github.com/huggingface/trl/blob/c10cc8995b6fd45f3a876ec98cade97251abe733/trl/core.py#L115-L124
    logp = F.log_softmax(logits, dim=-1)
    logpy = logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logpy


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # Adapted from https://github.com/huggingface/trl/blob/c10cc8995b6fd45f3a876ec98cade97251abe733/trl/core.py#L181-L185
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy


@dataclass
class PPOConfig:
    # sampling
    do_sample: bool = field(default=True)
    temperature: float = field(default=1.0)
    top_p: float = field(default=1.0)
    # rollout
    num_rollout_timesteps: int = field(default=16384)
    rollout_batch_size: int = field(default=64)
    # training
    actor_lr: float = field(default=1e-6)
    critic_lr: float = field(default=1e-5)
    lr_scheduler_type: str = field(default="cosine")
    train_batch_size: int = field(default=2048)
    train_micro_batch_size: int = field(default=256)
    eval_micro_batch_size: int = field(default=16384)
    max_steps: int = field(default=100)
    warmup_steps: int = field(default=8)
    num_ppo_epochs: int = field(default=1)
    max_grad_norm: float = field(default=1.0)
    logging_steps: int = field(default=1)
    output_dir: str = field(default="game-agent")
    save_steps: int = field(default=10)
    force_sync_grad: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    use_liger_kernel: bool = field(default=False)
    # ppo
    gamma: float = field(default=0.9)
    gae_lambda: float = field(default=0.95)
    clip_range: float = field(default=0.2)
    clip_range_vf: Optional[float] = field(default=None)
    clip_range_reward: Optional[float] = field(default=None)
    entropy_coef: float = field(default=0.05)
    normalize_advantage: bool = field(default=True)
    normalize_return: bool = field(default=False)
    critic_warmup_steps: int = field(default=10)
    # profiler
    profiler_enable: bool = field(default=False)
    profiler_output: str = field(default="profiler_logs")
    # memory profiler
    record_memory: bool = field(default=False)
    memory_output: str = field(default="memory_logs")

    @staticmethod
    def _check_div(x, y):
        q, r = divmod(x, y)
        if r != 0:
            raise ValueError(f"{x} is not divisible by {y}")
        return q

    def __post_init__(self):
        self.num_timesteps_per_env = self._check_div(self.num_rollout_timesteps, self.rollout_batch_size)
        self.accumulation_steps = self._check_div(self.train_batch_size, self.train_micro_batch_size)

    def normalize(self, data_parallel_size: int):
        self.num_rollout_timesteps = self._check_div(self.num_rollout_timesteps, data_parallel_size)
        self.rollout_batch_size = self._check_div(self.rollout_batch_size, data_parallel_size)
        self.train_batch_size = self._check_div(self.train_batch_size, data_parallel_size)
        self.train_micro_batch_size = self._check_div(self.train_micro_batch_size, data_parallel_size)
        self.eval_micro_batch_size = self._check_div(self.eval_micro_batch_size, data_parallel_size)


class PPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        env: gym.vector.VectorEnv,
        processor: ProcessorMixin,
        actor_tokenizer: PreTrainedTokenizer,
        actor: PreTrainedModel,
        critic_tokenizer: PreTrainedTokenizer,
        critic: PreTrainedModel,
    ) -> None:
        if env.num_envs != config.rollout_batch_size:
            raise ValueError(f"num_envs ({env.num_envs}) must equal rollout_batch_size ({config.rollout_batch_size})")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        env.reset(seed=0)

        if config.use_liger_kernel:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=actor)
            _apply_liger_kernel_to_instance(model=critic)

        if config.gradient_checkpointing:
            actor.gradient_checkpointing_enable()
        actor = parallelize_model(actor)

        if critic.lm_head.out_features != 1:
            critic.lm_head = nn.Linear(critic.config.hidden_size, 1, bias=False)
        if config.gradient_checkpointing:
            critic.gradient_checkpointing_enable()
        critic = parallelize_model(critic)

        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

        # optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=config.lr)
        actor_lr_scheduler: torch.optim.lr_scheduler.LRScheduler = get_scheduler(
            name=config.lr_scheduler_type,
            optimizer=actor_optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps - config.critic_warmup_steps,
        )
        critic_lr_scheduler: torch.optim.lr_scheduler.LRScheduler = get_scheduler(
            name=config.lr_scheduler_type,
            optimizer=critic_optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )

        self.config = config
        self.env = env
        self.processor = processor
        self.actor_tokenizer = actor_tokenizer
        self.actor = actor
        self.critic_tokenizer = critic_tokenizer
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.actor_lr_scheduler = actor_lr_scheduler
        self.critic_lr_scheduler = critic_lr_scheduler

        self.global_step = 0
        self.input_processor = Qwen2VLInputProcessor(env=env, processor=self.processor)

        self.legal_action_texts = get_action_meanings(env)
        action_ids = self.actor_tokenizer(self.legal_action_texts).input_ids
        max_new_tokens = max(len(x) for x in action_ids) + 1  # plus 1 for the eos token
        self.generation_config = GenerationConfig(
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=max_new_tokens,
            # eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
            # pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        self.profiler = None
        if self.config.profiler_enable:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                # schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"{self.config.profiler_output}/rank_{self.rank}", use_gzip=True
                ),
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                with_flops=False,
                with_modules=False,
            )

    def rollout_step(self):
        observation = self.env.render()

        processor_outputs = self.input_processor(observation=observation)
        batch = processor_outputs["batch"]
        batch_size, input_len = batch["input_ids"].shape

        outputs = self.actor.generate(
            **processor_outputs["inputs"], generation_config=self.generation_config, synced_gpus=False
        )

        sequence_ids = outputs.sequences
        sequence_mask = (sequence_ids != self.actor_tokenizer.pad_token_id).long()

        output_ids = sequence_ids[:, input_len:]
        output_mask = sequence_mask[:, input_len:]

        old_logits = torch.stack(outputs.scores, dim=1)
        old_log_probs = logprobs_from_logits(old_logits, output_ids)
        # Let's say action a consists of n tokens: x1, x2, ..., xn. Then:
        # p(a) = p(x1) * p(x2|x1) * ... * p(xn|x1...x_{n-1})
        # log p(a) = log p(x1) + log p(x2|x1) + ... + log p(xn|x1...x_{n-1})
        old_log_probs = old_log_probs.masked_fill_(output_mask == 0, 0).sum(dim=-1)

        # map response to action id
        action_texts = self.actor_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        is_illegal = torch.zeros(batch_size, dtype=torch.bool)
        actions = []
        for i, action_text in enumerate(action_texts):
            if action_text not in self.legal_action_texts:
                action_text = "NOOP"
                is_illegal[i] = True
            action = self.legal_action_texts.index(action_text)
            actions.append(action)

        _, rewards, terminations, truncations, _ = self.env.step(actions)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        if self.config.clip_range_reward is not None:
            rewards = rewards.clamp(min=-self.config.clip_range_reward, max=self.config.clip_range_reward)
        rewards[is_illegal] -= 1  # punish illegal actions

        dones = terminations | truncations

        # pad response to max_new_tokens to be safe to concatenate with other batches
        pad_size = input_len + self.generation_config.max_new_tokens - outputs.sequences.shape[1]
        if pad_size > 0:
            sequence_ids = F.pad(sequence_ids, (0, pad_size), mode="constant", value=self.actor_tokenizer.pad_token_id)
            sequence_mask = F.pad(sequence_mask, (0, pad_size), mode="constant", value=False)

        batch.update(
            dict(
                sequence_ids=sequence_ids,
                sequence_mask=sequence_mask,
                old_log_probs=old_log_probs,
                actions=torch.tensor(actions, dtype=torch.long),
                rewards=rewards,
                dones=torch.tensor(dones, dtype=torch.bool),
                is_illegal=is_illegal,
            )
        )
        return batch

    # TODO: is numpy + numba faster?
    @staticmethod
    def compute_advantages_and_returns(
        values: torch.Tensor,
        rewards: torch.Tensor,
        last_values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> torch.Tensor:
        # Adapted from https://github.com/DLR-RM/stable-baselines3/blob/8a3e3ccb4e08e42f60240dfa510b97c47a34b787/stable_baselines3/common/buffers.py#L403-L438

        # TODO: handle dones truncated
        num_steps, _ = values.shape
        next_values = torch.cat((values[1:], last_values.unsqueeze(0)), dim=0)
        next_non_terminals = (~dones).float()
        deltas = rewards + gamma * next_values * next_non_terminals - values
        advantages = torch.empty_like(values)
        last_gae_lam = torch.zeros_like(last_values)
        for step in reversed(range(num_steps)):
            last_gae_lam = deltas[step] + gamma * gae_lambda * next_non_terminals[step] * last_gae_lam
            advantages[step] = last_gae_lam

        returns = advantages + values
        return advantages, returns

    @torch.no_grad()
    def rollout(self) -> TensorDict:
        self.actor.eval()
        self.critic.eval()

        rollout_buffer = []

        with FSDP.summon_full_params(self.actor, writeback=False), unwrap_fsdp_model_with_full_params(
            self.actor
        ), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for _ in range(self.config.num_timesteps_per_env):
                step_sample = self.rollout_step()
                rollout_buffer.append(step_sample)

        rollout_buffer: TensorDict = TensorDict.cat(rollout_buffer, dim=0)

        # compute old values
        old_values = []
        for micro_batch in rollout_buffer.split(self.config.eval_micro_batch_size):
            critic_inputs = self.input_processor(batch=micro_batch)["inputs"]
            micro_old_values = self.critic(**critic_inputs).logits[:, -1].squeeze(-1)
            old_values.append(micro_old_values)
        old_values = torch.cat(old_values, dim=0)

        # compute last values for truncated trajectory
        last_observation = self.env.render()
        last_inputs = self.input_processor(observation=last_observation)["inputs"]
        last_values = self.critic(**last_inputs).logits[:, -1].squeeze(-1)

        # compute advantages & returns
        batch_size = self.env.num_envs
        values = old_values.view(self.config.num_timesteps_per_env, batch_size)
        rewards = rollout_buffer.get("rewards").view(self.config.num_timesteps_per_env, batch_size)
        dones = rollout_buffer.get("dones").view(self.config.num_timesteps_per_env, batch_size)
        advantages, returns = self.compute_advantages_and_returns(
            values=values,
            rewards=rewards,
            last_values=last_values,
            dones=dones,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rollout_buffer.update(dict(old_values=old_values, advantages=advantages.flatten(), returns=returns.flatten()))
        return rollout_buffer

    def train_actor_micro_batch(self, batch: TensorDict) -> dict[str, float]:
        _, input_len = batch.get("input_ids").shape
        sequence_ids = batch.get("sequence_ids")
        sequence_mask = batch.get("sequence_mask")
        old_log_probs = batch.get("old_log_probs")
        advantages = batch.get("advantages")

        output_ids = sequence_ids[:, input_len:]
        output_mask = sequence_mask[:, input_len:]

        actor_inputs = self.input_processor(batch=batch)["inputs"]
        actor_inputs.update(input_ids=sequence_ids, attention_mask=sequence_mask)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits: torch.Tensor = self.actor(**actor_inputs).logits

        # warp logits to the real distribution that responses were sampled from
        logits_processor = LogitsProcessorList()
        if self.config.temperature > 0 and self.config.temperature != 1:
            logits_processor.append(TemperatureLogitsWarper(temperature=self.config.temperature))
        if self.config.top_p > 0 and self.config.top_p < 1:
            logits_processor.append(TopPLogitsWarper(top_p=self.config.top_p))
        logits = logits_processor(sequence_ids, logits)

        log_probs = logprobs_from_logits(logits[:, input_len - 1 : -1], output_ids)
        log_probs = log_probs.masked_fill(output_mask == 0, 0).sum(dim=-1)

        log_ratio = log_probs - old_log_probs
        ratio = log_ratio.exp()

        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * ratio.clamp(min=1 - self.config.clip_range, max=1 + self.config.clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        if self.config.entropy_coef > 0:
            entropy = entropy_from_logits(logits[:, input_len - 1 : -1])
            # TODO: is token-level sum a correct estimator?
            entropy = (entropy * output_mask).sum(dim=-1).mean()
        else:
            entropy = torch.zeros_like(policy_loss)

        loss = policy_loss - self.config.entropy_coef * entropy
        loss = loss / self.config.accumulation_steps
        loss.backward()

        with torch.no_grad():
            policy_clip_frac = ((ratio - 1).abs() > self.config.clip_range).mean(dtype=torch.float32)
            approx_kl = (ratio - 1 - log_ratio).mean()

        return {
            "policy/loss": policy_loss.item(),
            "policy/ratio_mean": ratio.mean().item(),
            "policy/entropy": entropy.item(),
            "policy/approx_kl": approx_kl.item(),
            "policy/clip_frac": policy_clip_frac.item(),
        }

    def train_critic_micro_batch(self, batch: TensorDict) -> dict[str, float]:
        old_values = batch.get("old_values")
        returns = batch.get("returns")

        critic_inputs = self.input_processor(batch=batch)["inputs"]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values: torch.Tensor = self.critic(**critic_inputs).logits[:, -1].squeeze(-1)

        if self.config.clip_range_vf is not None:
            values_clip = values.clamp(
                min=old_values - self.config.clip_range_vf, max=old_values + self.config.clip_range_vf
            )
            value_loss_1 = F.mse_loss(values, returns, reduction="none")
            value_loss_2 = F.mse_loss(values_clip, returns, reduction="none")
            value_loss = torch.max(value_loss_1, value_loss_2).mean()

            with torch.no_grad():
                value_clip_frac = (value_loss_2 > value_loss_1).mean(dtype=torch.float32)
        else:
            value_loss = F.mse_loss(values, returns)
            value_clip_frac = torch.zeros((), dtype=torch.float32)

        # loss =  value_loss * self.config.value_coef
        loss = value_loss
        loss = loss / self.config.accumulation_steps
        loss.backward()

        with torch.no_grad():
            explained_var = 1 - (values - returns).var() / returns.var()

        return {
            "value/loss": value_loss.item(),
            "value/values_mean": values.mean().item(),
            "value/values_var": values.var().item(),
            "value/explained_var": explained_var.item(),
            "value/clip_frac": value_clip_frac.item(),
        }

    def train_batch(self, batch: TensorDict) -> dict[str, float]:

        def auto_sync(model: FSDP, micro_batch_idx: int):
            if not self.config.force_sync_grad and (micro_batch_idx + 1) % self.config.accumulation_steps != 0:
                return model.no_sync()
            return nullcontext()

        self.actor.train()
        self.critic.train()

        # TODO: micro stats
        stats = {}
        if self.global_step >= self.config.critic_warmup_steps:
            # TODO: mean stats
            for micro_batch_idx, micro_batch in enumerate(batch.split(self.config.train_micro_batch_size)):
                with auto_sync(self.actor, micro_batch_idx):
                    stats.update(self.train_actor_micro_batch(micro_batch))
            policy_grad_norm = self.actor.clip_grad_norm_(max_norm=self.config.max_grad_norm)
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()
        else:
            policy_grad_norm = torch.zeros(())

        for micro_batch_idx, micro_batch in enumerate(batch.split(self.config.train_micro_batch_size)):
            with auto_sync(self.critic, micro_batch_idx):
                stats.update(self.train_critic_micro_batch(micro_batch))
        # TODO: FSDP clip grad norm
        value_grad_norm = self.critic.clip_grad_norm_(max_norm=self.config.max_grad_norm)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        stats.update(
            {
                "policy/grad_norm": policy_grad_norm.item(),
                "value/grad_norm": value_grad_norm.item(),
            }
        )
        return stats

    def training_step(self):
        start = time.time()
        batch = self.rollout()
        torch.cuda.empty_cache()
        elapsed_rollout = time.time() - start

        start = time.time()
        exp_loader = DataLoader(batch, batch_size=self.config.train_batch_size, shuffle=True, collate_fn=lambda x: x)

        # TODO: average batch stats
        for _ in range(self.config.num_ppo_epochs):
            for exp_batch in exp_loader:
                train_stats = self.train_batch(exp_batch)

        torch.cuda.empty_cache()
        elapsed_train = time.time() - start

        if self.global_step >= self.config.critic_warmup_steps:
            self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        # stats
        advantages = batch.get("advantages")
        returns = batch.get("returns")
        rewards = batch.get("rewards")
        old_values = batch.get("old_values")
        actions = batch.get("actions")
        is_illegal_actions = batch.get("is_illegal")

        action_stats = {f"policy/action_prob_ILLEGAL": is_illegal_actions.mean(dtype=torch.float32).item()}
        for action_idx, action_text in enumerate(self.legal_action_texts):
            action_stats[f"policy/action_prob_{action_text}"] = (actions == action_idx).mean(dtype=torch.float32).item()

        stats = {
            "step": self.global_step,
            **{f"policy/lr_group_{i}": pg["lr"] for i, pg in enumerate(self.actor_optimizer.param_groups)},
            **{f"value/lr_group_{i}": pg["lr"] for i, pg in enumerate(self.critic_optimizer.param_groups)},
            **train_stats,
            **action_stats,
            "policy/advantages_mean": advantages.mean().item(),
            "policy/advantages_var": advantages.var().item(),
            "policy/rewards_mean": rewards.mean().item(),
            "policy/rewards_var": rewards.var().item(),
            "policy/returns_mean": returns.mean().item(),
            "policy/returns_var": returns.var().item(),
            "value/old_values_mean": old_values.mean().item(),
            "value/old_values_var": old_values.var().item(),
            "timing/rollout": elapsed_rollout,
            "timing/train": elapsed_train,
        }

        return stats

    def train(self):
        for self.global_step in tqdm(range(self.config.max_steps)):
            # memory profiler start
            if self.global_step == 1 and self.config.record_memory:
                torch.cuda.memory._record_memory_history()

            # torch profiler
            if self.global_step == 1 and self.profiler is not None:
                profiler_ctx = self.profiler
            else:
                profiler_ctx = nullcontext()

            # train!
            with profiler_ctx:
                stats = self.training_step()

            # memory profiler end
            if self.global_step == 1 and self.config.record_memory:
                torch.cuda.memory._dump_snapshot(self.config.memory_output)
                torch.cuda.memory._record_memory_history(enabled=None)

            # checkpoint
            if (self.global_step + 1) % self.config.save_steps == 0:
                checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
                self.save_model(checkpoint_dir)

            # logging
            if self.rank == 0:
                wandb.log(stats, step=self.global_step)
                if self.global_step % self.config.logging_steps == 0:
                    tqdm.write(f"{stats}")

    def save_model(self, output_dir: Optional[Union[str, os.PathLike]] = None) -> None:
        if output_dir is None:
            output_dir = self.config.output_dir

        full_state_dict_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
        with FSDP.state_dict_type(self.actor, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            state_dict = self.actor.state_dict()

        if self.rank == 0:
            self.processor.save_pretrained(output_dir)
            self.actor.save_pretrained(output_dir, state_dict=state_dict)


@contextmanager
def unwrap_fsdp_model_with_full_params(model: FSDP):
    def unwrapped_forward(self: FSDP, *args, **kwargs):
        return self.module(*args, **kwargs)

    for module in FSDP.fsdp_modules(model):
        module.old_forward = module.forward
        module.forward = functools.partial(unwrapped_forward, module)

    try:
        yield
    finally:
        for module in FSDP.fsdp_modules(model):
            module.forward = module.old_forward
            del module.old_forward


def parallelize_model(model: PreTrainedModel) -> FSDP:
    transformer_layer_cls = {get_module_class_from_name(model, name) for name in model._no_split_modules}
    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls)
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16
    )
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
    )
    return model


@dataclass
class Arguments:
    task: str = field(default="ALE/Breakout-v5")
    vec_env_mode: str = field(default="sync")
    actor: str = field(default="Qwen/Qwen2-VL-2B-Instruct")
    critic: str = field(default="Qwen/Qwen2-VL-2B-Instruct")
    seed: int = field(default=12345)


def main():
    import sys

    sys.argv += """
--use_liger_kernel --clip_range_reward 1 --vec_env_mode async
""".split()

#     # debug
#     sys.argv += """
# --num_rollout_timesteps 128 --rollout_batch_size 32
# --train_batch_size 64 --train_micro_batch_size 64
# """.split()

    #     sys.argv += '''
    # --profiler_enable --num_rollout_timesteps 64 --rollout_batch_size 64 --train_batch_size 64 --num_ppo_epochs 1
    # '''.split()

    parser = HfArgumentParser((Arguments, PPOConfig))
    args, ppo_config = cast(tuple[Arguments, PPOConfig], parser.parse_args_into_dataclasses())

    if os.getenv("RANK") is None and os.getenv("WORLD_SIZE") is None:
        os.environ.update(
            {"RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500", "LOCAL_RANK": "0"}
        )

    local_rank = int(os.getenv("LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # normalize batch size across world size
    ppo_config.normalize(data_parallel_size=world_size)

    if rank == 0:
        wandb.init(project="ppo", config={**asdict(args), **asdict(ppo_config)})

    set_seed(args.seed)

    env = gym.make_vec(
        args.task, num_envs=ppo_config.rollout_batch_size, vectorization_mode=args.vec_env_mode, render_mode="rgb_array"
    )

    actor_processor = AutoProcessor.from_pretrained(args.actor)
    actor_tokenizer = AutoTokenizer.from_pretrained(args.actor, padding_side="left", trust_remote_code=True)
    critic_tokenizer = AutoTokenizer.from_pretrained(args.critic, padding_side="left", trust_remote_code=True)

    actor_config = AutoConfig.from_pretrained(args.actor)
    critic_config = AutoConfig.from_pretrained(args.critic)

    def get_auto_model_class(config):
        if type(config) in AutoModelForVision2Seq._model_mapping:
            return AutoModelForVision2Seq
        return AutoModelForCausalLM

    actor: PreTrainedModel = (
        get_auto_model_class(actor_config)
        .from_pretrained(
            args.actor,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float32,
            use_cache=False,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
    )

    critic: PreTrainedModel = (
        get_auto_model_class(critic_config)
        .from_pretrained(
            args.critic,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float32,
            use_cache=False,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
    )

    trainer = PPOTrainer(
        config=ppo_config,
        env=env,
        processor=actor_processor,
        actor_tokenizer=actor_tokenizer,
        actor=actor,
        critic_tokenizer=critic_tokenizer,
        critic=critic,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
