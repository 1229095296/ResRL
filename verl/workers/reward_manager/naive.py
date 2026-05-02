# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", enable_length_penalty=False) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            enable_length_penalty: Whether to enable length penalty for positive samples. Can be overridden by
                VERL_ENABLE_LENGTH_PENALTY environment variable. Defaults to False.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        # Read from environment variable if set, otherwise use the parameter value
        env_enable_length_penalty = os.getenv("VERL_ENABLE_LENGTH_PENALTY", "").lower()
        if env_enable_length_penalty in ("true", "1", "yes"):
            self.enable_length_penalty = True
        elif env_enable_length_penalty in ("false", "0", "no"):
            self.enable_length_penalty = False
        else:
            self.enable_length_penalty = enable_length_penalty  # Enable length-based reward penalty

    def _apply_length_penalty(self, reward, response_length):
        """
        Apply length penalty to reward for positive samples.

        Rules:
        - If response_length <= penalty_start: reward * 1.0 (100%)
        - If penalty_start < response_length <= max_length:
            linear discount from 100% to 70%
        - If response_length > max_length: reward * 0.7 (70%)

        Defaults follow the paper: no penalty up to 3500 tokens, then linearly decay to 0.7
        at 4096 tokens. MAX_RESPONSE_LENGTH, LENGTH_PENALTY_START, and
        LENGTH_PENALTY_END_SCALE can override these values.

        Args:
            reward: The original reward value
            response_length: The length of the response in tokens

        Returns:
            The reward after applying length penalty
        """
        if reward <= 0:
            # Only apply penalty to positive samples
            return reward

        max_length = int(os.getenv("MAX_RESPONSE_LENGTH", "4096"))
        penalty_start = int(os.getenv("LENGTH_PENALTY_START", "3500"))
        end_scale = float(os.getenv("LENGTH_PENALTY_END_SCALE", "0.7"))

        if response_length <= penalty_start:
            return reward
        elif response_length <= max_length:
            if max_length <= penalty_start:
                return reward * end_scale
            discount_ratio = (response_length - penalty_start) / (max_length - penalty_start)
            penalty_factor = 1.0 - (1.0 - end_scale) * discount_ratio
            return reward * penalty_factor
        else:
            return reward * end_scale

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        # Track which keys are present in all samples
        common_keys = None

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Track common keys across all samples
                if common_keys is None:
                    common_keys = set(score.keys())
                else:
                    common_keys = common_keys.intersection(set(score.keys()))
                # Store all values for now, we'll filter later
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # Apply length penalty if enabled
            if self.enable_length_penalty:
                original_reward = reward
                reward = self._apply_length_penalty(reward, valid_response_length.item())

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # Filter reward_extra_info to only include keys that are present in all samples
        if common_keys is not None:
            filtered_reward_extra_info = {k: v for k, v in reward_extra_info.items() if k in common_keys}
        else:
            filtered_reward_extra_info = reward_extra_info

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": filtered_reward_extra_info,
            }
        else:
            return reward_tensor

@register("naive_code")
class NaiveCodeRewardManager:
    """The reward manager for code-based tasks with parallel processing."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            # data_source = data_item.non_tensor_batch['data_source']
            # compute_score_fn = _select_rm_score_fn(data_source)
            # score = compute_score_fn(data_source=data_source, llm_solution=sequences_str, ground_truth=ground_truth)
# if isinstance(score, dict):
            #     reward = score["score"]
            #     # Store the information including original reward
            #     for key, value in score.items():
            #         reward_extra_info[key].append(value)
            # else:
            #     reward = score
            
            # with print_lock:
            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print(sequences_str)      
            return i, score, valid_response_length

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=24) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
