from dataclasses import dataclass, field
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
    PreTrainedModel,
    ProcessorMixin,
)

from game_agent import Qwen2VLInputProcessor, get_action_meanings

gym.register_envs(ale_py)


@dataclass
class Arguments:
    model_id: str
    task: str = "ALE/Breakout-v5"
    vec_env_mode: str = "sync"
    max_step: int = 256
    batch_size: int = 8
    # sampling
    do_sample: bool = field(default=True)
    temperature: float = field(default=1.0)
    top_p: float = field(default=1.0)


def get_auto_model_class(config):
    if type(config) in AutoModelForVision2Seq._model_mapping:
        return AutoModelForVision2Seq
    return AutoModelForCausalLM


def save_gif():
    pass


@torch.no_grad()
def main():
    import sys

    sys.argv += "--model_id game-agent/checkpoint-39".split()

    parser = HfArgumentParser(Arguments)
    args: Arguments = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    processor: ProcessorMixin = AutoProcessor.from_pretrained(args.model_id)
    model: PreTrainedModel = (
        get_auto_model_class(config)
        .from_pretrained(
            args.model_id,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_cache=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda",
        )
        .eval()
    )

    env = gym.make_vec(
        args.task, num_envs=args.batch_size, vectorization_mode=args.vec_env_mode, render_mode="rgb_array"
    )
    env.reset(seed=0)

    input_processor = Qwen2VLInputProcessor(env=env, processor=processor)

    legal_action_texts = get_action_meanings(env)
    action_ids = tokenizer(legal_action_texts).input_ids
    max_new_tokens = max(len(x) for x in action_ids) + 1  # plus 1 for the eos token

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=max_new_tokens,
        # eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
        # pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    batch_observation = []

    for _ in tqdm(range(args.max_step)):
        observation = env.render()
        step_observation = np.stack(observation, axis=0)
        batch_observation.append(step_observation)

        processor_outputs = input_processor(observation=observation)
        batch = processor_outputs["batch"]
        batch_size, input_len = batch["input_ids"].shape

        sequence_ids = model.generate(**processor_outputs["inputs"], generation_config=generation_config)

        # sequence_mask = (sequence_ids != tokenizer.pad_token_id).long()

        output_ids = sequence_ids[:, input_len:].cpu()

        # map response to action id
        action_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        is_illegal = torch.zeros(batch_size, dtype=torch.bool)
        actions = []
        for i, action_text in enumerate(action_texts):
            if action_text not in legal_action_texts:
                action_text = "NOOP"
                is_illegal[i] = True
            action = legal_action_texts.index(action_text)
            actions.append(action)

        _, rewards, terminations, truncations, _ = env.step(actions)

        # rewards = torch.tensor(rewards, dtype=torch.float32)
        # if args.clip_range_reward is not None:
        #     rewards = rewards.clamp(min=-args.clip_range_reward, max=args.clip_range_reward)
        # rewards[is_illegal] -= 1  # punish illegal actions

        # dones = terminations | truncations

    batch_observation = np.stack(batch_observation, axis=1)

    output_dir = Path("trace")
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, images in enumerate(batch_observation):
        images = [Image.fromarray(img) for img in images]
        images[0].save(output_dir / f"{i}.gif", format="GIF", append_images=images, save_all=True, duration=10, loop=0)


if __name__ == "__main__":
    main()
