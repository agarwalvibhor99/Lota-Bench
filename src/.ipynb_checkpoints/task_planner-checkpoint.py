import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    PretrainedConfig,   # <- added for Qwen3 workaround
)
from torch.nn import CrossEntropyLoss
import guidance
import logging


class TaskPlanner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.planner.device
        self.max_steps = cfg.planner.max_steps
        self.model_name = cfg.planner.model_name
        self.scoring_batch_size = cfg.planner.scoring_batch_size
        self.score_function = cfg.planner.score_function
        self.scoring_mode = cfg.planner.scoring_mode
        self.use_predefined_prompt = cfg.planner.use_predefined_prompt

        # Load pre-trained model
        print(f"Loading LLM and tokenizer: {self.model_name}")

        model_args = {
            "pretrained_model_name_or_path": self.model_name,
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        if cfg.planner.use_accelerate_device_map:
            model_args["device_map"] = "auto"
            if getattr(cfg.planner, "load_in_8bit", False):
                model_args["load_in_8bit"] = True
        model_args["use_auth_token"] = cfg.planner.hf_auth_token

        # ========== SCORING MODE: GUIDANCE ==========
        if cfg.planner.scoring_mode == "guidance":
            # guidance manages the model itself
            model_args.pop("pretrained_model_name_or_path")
            tokenizer = None
            if "OpenAI" in self.model_name:
                openai_model_name = self.model_name.split("/")[1]
                guidance.llm = guidance.llms.OpenAI(
                    openai_model_name,
                    api_key=cfg.planner.openai_api_key,
                )
            else:
                if "decapoda-research/llama" in self.model_name or "chainyo/alpaca" in self.model_name:
                    tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
                if "bigscience/bloom" == self.model_name:  # bloom 175B
                    model_args["max_memory"] = {
                        0: "60GB",
                        1: "80GB",
                        2: "48GB",
                        3: "48GB",
                        4: "48GB",
                    }

                # your updated guidance call with temperature + sampling
                guidance.llm = guidance.llms.Transformers(
                    self.model_name,
                    tokenizer=tokenizer,
                    temperature=0.7,   # any > 0, tune later
                    do_sample=True,    # sampling + logprobs
                    **model_args,
                )

            self.guidance_program = guidance(
                """{{prompt}} {{select 'step' options=candidates logprobs='score'}}"""
            )

            self.model = None
            self.tokenizer = None

            logging.getLogger("guidance").setLevel(logging.WARNING)

        # ========== SCORING MODE: REUSE_PROMPT / NAIVE ==========
        else:
            # Special handling for some model families
            if "decapoda-research/llama" in self.model_name or "chainyo/alpaca" in self.model_name:
                # Original LLaMA path
                self.model = LlamaForCausalLM.from_pretrained(**model_args)
                self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)

            # Qwen3 workaround: avoid AutoConfig CONFIG_MAPPING['qwen3'] KeyError
            elif "qwen3" in self.model_name.lower():
                model_name = model_args["pretrained_model_name_or_path"]

                # Drop keys that we will pass explicitly (or don't need for local load)
                safe_args = {
                    k: v
                    for k, v in model_args.items()
                    if k not in ["pretrained_model_name_or_path", "trust_remote_code", "use_auth_token"]
                }

                # For a local model path, we don't need any token
                config = PretrainedConfig.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )

                # Important: trust_remote_code ONLY here, not also inside **safe_args
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=config,
                    trust_remote_code=True,
                    **safe_args,
                )

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_fast=False,
                )

            # Default path for all other models
            else:
                self.model = AutoModelForCausalLM.from_pretrained(**model_args)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Move model if not using accelerate's device_map
            if not cfg.planner.use_accelerate_device_map:
                self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer.pad_token_id = 0
            print("Loading done\n")

        # Load prompt
        self.prompt = self.init_prompt(cfg)

    # NOTE: original repo has two reset() definitions; keeping same behavior
    def reset(self, nl_act_list, nl_obj_list):
        self.nl_obj_list = nl_obj_list
        self.skill_set = self.init_skill_set(nl_act_list, nl_obj_list)

    def reset(self):
        self.skill_set = self.init_skill_set()

    def init_prompt(self, cfg):
        raise NotImplementedError()

    def init_skill_set(self, nl_act_list, nl_obj_list):
        raise NotImplementedError()

    def update_skill_set(self, previous_step, nl_obj_list):
        raise NotImplementedError()

    def score(self, prompt, skill_set):
        scores = {}
        batch_skill_set_list = [
            skill_set[chunk : chunk + self.scoring_batch_size]
            for chunk in range(0, len(skill_set), self.scoring_batch_size)
        ]

        if self.scoring_mode == "guidance":
            out = self.guidance_program(prompt=prompt, candidates=skill_set)
            scores = out["score"]

        elif self.scoring_mode in ("reuse_prompt", "naive"):
            prompt_tokens = self.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            )
            if not self.cfg.planner.use_accelerate_device_map:
                prompt_tokens = prompt_tokens.to(self.device)
            prompt_len = prompt_tokens.attention_mask[0].sum().item()

            for batch_skill_set in batch_skill_set_list:
                batch_sentence = [f"{prompt} {skill}" for skill in batch_skill_set]
                size_B = len(batch_skill_set)

                if (
                    "decapoda-research/llama" in self.model_name
                    or "chainyo/alpaca" in self.model_name
                ):
                    batch_skill_set_for_model = batch_skill_set
                else:
                    batch_skill_set_for_model = [f" {skill}" for skill in batch_skill_set]

                with torch.no_grad():
                    if self.scoring_mode == "reuse_prompt":
                        prompt_output = self.model(**prompt_tokens, use_cache=True)
                        skill_tokens = self.tokenizer(
                            batch_skill_set_for_model,
                            add_special_tokens=False,
                            return_tensors="pt",
                            padding=True,
                        )
                        if not self.cfg.planner.use_accelerate_device_map:
                            skill_tokens = skill_tokens.to(self.device)

                        concat_attention_mask = torch.cat(
                            (
                                prompt_tokens.attention_mask.repeat(size_B, 1),
                                skill_tokens.attention_mask,
                            ),
                            dim=1,
                        )
                        batch_past_key_values = self.duplicate_past_key_values(
                            prompt_output.past_key_values, size_B
                        )

                        output = self.model(
                            input_ids=skill_tokens.input_ids,
                            attention_mask=concat_attention_mask,
                            past_key_values=batch_past_key_values,
                            return_dict=True,
                        )
                        prompt_last_logits = prompt_output.logits[:, -1:, :].repeat(
                            size_B, 1, 1
                        )  # [B, 1, C]
                        logits = torch.cat(
                            (prompt_last_logits, output.logits[:, :-1, :]),
                            dim=1,
                        )
                        labels = skill_tokens.input_ids
                        attention_mask = skill_tokens.attention_mask

                    elif self.scoring_mode == "naive":
                        sentence_tokens = self.tokenizer(
                            batch_sentence,
                            add_special_tokens=False,
                            return_tensors="pt",
                            padding=True,
                        )

                        # your Qwen device fix: always push tokens to model.device
                        if not self.cfg.planner.use_accelerate_device_map:
                            sentence_tokens = sentence_tokens.to(self.device)
                        device = (
                            self.model.device
                            if hasattr(self.model, "device")
                            else "cuda"
                        )
                        sentence_tokens = sentence_tokens.to(device)

                        output = self.model(
                            sentence_tokens.input_ids,
                            attention_mask=sentence_tokens.attention_mask,
                            return_dict=True,
                        )
                        logits = output.logits[:, prompt_len - 1 : -1]
                        labels = sentence_tokens.input_ids[:, prompt_len:]
                        attention_mask = sentence_tokens.attention_mask[:, prompt_len:]

                    size_B, size_L, size_C = logits.shape
                    logits = logits.reshape([size_B * size_L, size_C])
                    labels = labels.reshape([size_B * size_L])
                    loss_fn = CrossEntropyLoss(reduction="none")
                    loss = loss_fn(logits.float(), labels.long())
                    loss = loss.reshape([size_B, size_L])
                    skill_len = attention_mask.count_nonzero(axis=1)

                    if self.score_function == "sum":
                        score = -(loss * attention_mask).sum(axis=1)
                    elif self.score_function == "avg":
                        score = -(loss * attention_mask).sum(axis=1) / skill_len

                    for skill_id, skill in enumerate(batch_skill_set):
                        scores[skill] = score[skill_id].item()
        else:
            assert False, "unknown scoring mode"
        return scores

    def plan_whole(self, query):
        step_seq = []
        skill_set_size_seq = []
        print(f"Input query: {query}")

        prompt_lines = self.prompt.split("\n")
        prompt_examples = prompt_lines[2:]
        example_text = "\n".join(prompt_examples)
        skills_text = ", ".join([x.strip() for x in self.skill_set])

        self.guidance_program = guidance(
            """
        {{#system~}}
        You are a robot operating in a home. A human user can ask you to do various tasks and you are supposed to tell the sequence of actions you would do to accomplish your task.
        {{~/system}}
        
        {{#user~}}
        Examples of human instructions and possible your (robot) answers:
        {{example_text}}
        
        Now please answer the sequence of actions for the input instruction.
        You should use one of actions of this list: {{skills_text}}.
        List the actions with comma seperator.
        
        Input user instruction:   
        {{query}}
        {{~/user}}
        
        {{#assistant~}}
        {{gen 'answer' temperature=0.7 max_tokens=500}}
        {{~/assistant}}
        """
        )

        out = self.guidance_program(
            example_text=example_text,
            skills_text=skills_text,
            query=query,
        )
        answer = out["answer"]
        print(answer)

        answer = answer.replace("Robot: ", "")
        actions = [action.strip(" 1234567890.") for action in answer.split(",")]
        step_seq = actions

        return step_seq, skill_set_size_seq

    def plan_step_by_step(self, query, prev_steps=(), prev_msgs=()):
        if len(prev_steps) >= self.max_steps:
            return None, None

        prompt = self.prompt + f"Human: {query.strip()}\nRobot: 1. "

        for i, (step, msg) in enumerate(zip(prev_steps, prev_msgs)):
            if self.use_predefined_prompt and len(msg) > 0:
                prompt += step + f" (this action failed: {msg.lower()}), {i + 2}. "
            else:
                prompt += step + f", {i + 2}. "

        scores = self.score(prompt, self.skill_set)

        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_step = results[0][0].strip()

        return best_step, prompt

    def duplicate_past_key_values(self, past_key_values, batch_size):
        batch_past_key_values = []
        for layer in range(len(past_key_values)):
            batch_past_key_values_layer = []
            for kv in range(len(past_key_values[layer])):
                batch_past_key_values_layer.append(
                    past_key_values[layer][kv].repeat(batch_size, 1, 1, 1)
                )
            batch_past_key_values_layer = tuple(batch_past_key_values_layer)
            batch_past_key_values.append(batch_past_key_values_layer)
        batch_past_key_values = tuple(batch_past_key_values)
        return batch_past_key_values
