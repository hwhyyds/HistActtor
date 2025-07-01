import ruamel.yaml as yaml
from pathlib import Path
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed, PreTrainedModelWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, PeftConfig
from accelerate.utils import DeepSpeedPlugin
from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
from finetune import DataConfig, DataManager
from typing import Union, Annotated
import requests
from prompt import prompts
import torch
import typer
import tqdm
import functools
import os
import multiprocessing
import dataclasses as dc

app = typer.Typer(pretty_exceptions_show_locals=False)


@dc.dataclass
class Prompts(object):
    system_prompt: str
    evaluatist_prompt: str


@dc.dataclass
class RLAIFConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int

    ppo_config: PPOConfig = None

    generate_kwargs: dict = None

    prompts: Prompts = None

    @classmethod
    def from_dict(cls, **kwargs) -> 'RLAIFConfig':
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=HfTrainerDeepSpeedConfig(kwargs["deepspeed"]["deepspeed"]))

        accelerator_kwargs = {
            "deepspeed_plugin": deepspeed_plugin,
            **kwargs["accelerator_config"]
        }
        ppo_config = PPOConfig(
            accelerator_kwargs=accelerator_kwargs,
            **kwargs["ppo_config"]
        )
        kwargs['ppo_config'] = ppo_config
        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)
        kwargs = prompts(**kwargs["actor"])

        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'RLAIFConfig':
        path = "configs/ppo.yaml"
        path = Path(path)
        parser = yaml.YAML(typ='safe', pure=True)
        parser.indent(mapping=2, offset=2, sequence=4)
        parser.default_flow_style = False
        kwargs = parser.load(path)
        return cls.from_dict(**kwargs)


def preprocess_function(batch, tokenizer, system_prompt, max_input_length, max_output_length):
    batched_input_ids = []
    batched_messages = []
    for conv in batch['messages']:
        input_ids = [151331, 151333]
        conv[0]["content"] = system_prompt
        conv[1]["content"] = conv[1]["content"].split("response:")[0].replace("query:", "").strip(";")
        input_ids = input_ids + tokenizer.apply_chat_template(conv[:-1], tokenize=True, return_dict=False)[2:]
        batched_messages.append(conv[1]["content"])
        batched_input_ids.append(input_ids[:max_input_length + max_output_length] + [151336])
    del conv, input_ids
    torch.cuda.empty_cache()
    return {'input_ids': batched_input_ids, 'query': batched_messages}


def get_reward_value(evaluatist_prompt, texts):
    scores = []
    messages = [[{"role": "system", "content": evaluatist_prompt}, {"role": "user", "content": text}] for text in texts]
    for message in messages:
        try:
            answer = requests.post("http://localhost:5000", json=message)
            scores.append(float(answer.text))
        except Exception as e:
            print(f"Error fetching reward for message {message}: {e}")
            scores.append(0.0)  # 或其他默认值
    return scores


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


@app.command()
def main(
        data_dir: Annotated[str, typer.Argument(help='')],
        model_dir: str,
        lora_model_dir: str,
        ref_model_dir: str,
        config_file: Annotated[str, typer.Argument(help='')],
):
    #
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    # 加载基础模型并应用LoRA
    base_model_for_PPO = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    base_model_for_PPO_with_sft_lora = PeftModel.from_pretrained(
        base_model_for_PPO,
        lora_model_dir
    )
    # 包装模型
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_for_PPO_with_sft_lora
    )

    # 使LoRA模块可训练
    for name, param in base_model_for_PPO_with_sft_lora.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    ref_model_for_PPO = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    ref_model_for_PPO_with_sft_lora = PeftModel.from_pretrained(
        ref_model_for_PPO,
        ref_model_dir
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_model_for_PPO_with_sft_lora)

    config = RLAIFConfig.from_file(config_file)
    ppo_trainer = PPOTrainer(
        config.ppo_config,
        model=ppo_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
        # optimizer=DummyOptim,
    )

    data_manager = DataManager(data_dir, config.data_config)
    train_dataset = data_manager.get_dataset(
        "train",
        functools.partial(
            preprocess_function,
            tokenizer=tokenizer,
            system_prompt=config.prompts.system_prompt,
            max_input_length=config.max_input_length,
            max_output_length=config.max_output_length,
        ),
        batched=True,
    )
    train_dataset.set_format(type="torch")

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if epoch >= config.ppo_config.steps:
            break
        question_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=True,
            batch_size=1,
            generate_ref_response=False,
            generation_kwargs=config.generate_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # 计算奖励值
        if config.actor == "sushi":
            texts = [f"问题：{query}；回复：{response[len(query) + len(config.prompts.system_prompt):]}" for
                     query, response in
                     zip(batch["query"], batch["response"])]
        else:
            texts = [f"Question: {query};Response: {response[len(query) + len(config.prompts.system_prompt):]}" for
                     query, response in
                     zip(batch["query"], batch["response"])]

        scores = get_reward_value(config.prompts.evaluatist_prompt, texts)

        rewards = [torch.tensor(score - 0.0) for score in scores]

        for q, r, s in zip(batch["query"], batch["response"], scores):
            print(epoch, 'query:', q)
            print(f'response:{len(r)}', r[len(q) + 245:])
            print('score:', s)

        # 运行PPO步骤
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        def get_unique_dir(epoch):
            return f"{config.training_args.output_dir}/epoch_{epoch}_{multiprocessing.current_process().name}_{os.getpid()}"

        def ensure_dir_exists(dir_path):
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        def save_model(ppo_trainer, epoch):
            save_dir = get_unique_dir(epoch)
            ensure_dir_exists(save_dir)
            ppo_trainer._save_pretrained(save_dir)
            print(f"Model is saved in {save_dir}")

        if epoch % 5 == 0:
            save_model(ppo_trainer, epoch)