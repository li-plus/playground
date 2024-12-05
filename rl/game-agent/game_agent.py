import json
import random
import re
from pathlib import Path
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from tensordict import TensorDict
from transformers import ProcessorMixin


def get_action_meanings(env: gym.vector.VectorEnv) -> list[str]:
    with gym.make(env.spec.id) as dummy_env:
        action_meanings = dummy_env.unwrapped.get_action_meanings()
    return action_meanings


Conversation = list[dict[str, Union[str, list]]]
Conversations = list[Conversation]


class Qwen2VLInputProcessor:
    def __init__(self, env: gym.vector.VectorEnv, processor: ProcessorMixin):
        env_docs = json.loads(Path("environment-docs.json").read_text())
        env_doc = env_docs[self.camel_case_to_snake_case(env.spec.name)]
        env_prompt = env_doc["env_description"]
        if env_doc["reward_description"]:
            env_prompt += " " + env_doc["reward_description"]
        self.legal_action_texts = get_action_meanings(env)
        self.instruction_format = f"You are an excellent video game player able to achieve high scores by making proper decisions. The image shows the current game status of Atari game {env.spec.name}. {env_prompt} The current legal actions are: {{legal_action_prompt}}. You must choose a legal action to maximize the final scores. Answer only the action name in upper case without explanation."
        self.processor = processor

    @staticmethod
    def camel_case_to_snake_case(name: str) -> str:
        # https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def get_instruction(self):
        action_texts = self.legal_action_texts.copy()
        random.shuffle(action_texts)
        return self.instruction_format.format(legal_action_prompt=", ".join(action_texts))

    def build_conversations(self, observation: list[np.ndarray]) -> Conversations:
        conversations = []
        for obs in observation:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": Image.fromarray(obs),
                        },
                        {"type": "text", "text": self.get_instruction()},
                    ],
                }
            ]
            conversations.append(conversation)

        return conversations

    def build_batch(self, conversations: Conversations) -> TensorDict:
        from qwen_vl_utils import process_vision_info

        batch_size = len(conversations)
        texts = [
            self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            for conversation in conversations
        ]
        image_inputs, video_inputs = process_vision_info(conversations)
        inputs: dict[str, torch.Tensor] = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        for k in ("pixel_values", "image_grid_thw"):
            inputs[k] = inputs[k].unflatten(dim=0, sizes=(batch_size, -1))
        batch = TensorDict(**inputs, batch_size=batch_size, device=torch.cuda.current_device())
        return batch

    def build_inputs(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        inputs: dict[str, torch.Tensor] = batch.select(
            "input_ids", "attention_mask", "pixel_values", "image_grid_thw"
        ).to_dict()
        for k in ("pixel_values", "image_grid_thw"):
            inputs[k] = inputs[k].flatten(start_dim=0, end_dim=1)
        return inputs

    def __call__(
        self, *, observation: Optional[list[np.ndarray]] = None, batch: Optional[TensorDict] = None
    ) -> dict[str, Union[TensorDict, dict[str, torch.Tensor]]]:
        if observation is None == batch is None:
            raise RuntimeError("inputs must be either observation or batch")
        if observation is not None:
            conversations = self.build_conversations(observation)
            batch = self.build_batch(conversations)
        inputs = self.build_inputs(batch)
        return dict(batch=batch, inputs=inputs)
