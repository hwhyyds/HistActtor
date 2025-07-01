# -*- coding: utf-8 -*-
import os
import jieba
import dataclasses as dc
import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Union
import numpy as np
import ruamel.yaml as yaml
import torch
import typer
from datasets import Dataset, Split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import PeftConfig, get_peft_config, get_peft_model
from rouge_chinese import Rouge
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer
from datasets import load_dataset, DatasetDict, NamedSplit
from typing import Optional
from transformers.trainer import _is_peft_model
from transformers.trainer_pt_utils import LabelSmoother

app = typer.Typer(pretty_exceptions_show_locals=False)

import json

with open("tokenizer_vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
    vocab_tensor = torch.tensor(list(vocab.values())).to("cuda:0")
    num_cuda_device = torch.cuda.device_count()
    vocab_tensor[list(vocab.values())] = True
    vocab_tensor_cuda0 = vocab_tensor.to("cuda:0") if num_cuda_device > 0 else vocab_tensor
    vocab_tensor_cuda1 = vocab_tensor.to("cuda:1") if num_cuda_device > 1 else vocab_tensor
    vocab_tensor_cuda2 = vocab_tensor.to("cuda:2") if num_cuda_device > 2 else vocab_tensor
    vocab_tensor_cuda3 = vocab_tensor.to("cuda:3") if num_cuda_device > 2 else vocab_tensor


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output_ids = ([feature['output_ids'] for feature in features] if 'output_ids' in features[0].keys() else None)
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        return super().__call__(features, return_tensors)


def mask_penalty(log_probs, padding_mask):
    '''
    惩罚项计算流程为：
    1.先计算每个seq的实际概率分布
    2.获取每个seq预测的最终值
    3.判断该值是否属于在词表里面并
    4.将3中得到的判断为是否在词表里面的结果再次判断是否属于padding_mask的部分（即计算loss的部分）
    5.计算4中得到的需要计算惩罚项的值对应的mask的变化
    '''
    really_prob = torch.exp(-log_probs)  # 实际预测的每个值的概率
    # 计算预测项，判断有哪些预测结果是不为vocab里面的值
    predicted_token_ids = torch.argmax(really_prob, dim=-1)  # predicted_token_ids代表最后一维中概率最大的词，也就是预计要输出的值
    if predicted_token_ids.device == torch.device("cuda:0"):
        temp_vocab_tensor = vocab_tensor_cuda0
    elif predicted_token_ids.device == torch.device("cuda:1"):
        temp_vocab_tensor = vocab_tensor_cuda1
    elif predicted_token_ids.device == torch.device("cuda:2"):
        temp_vocab_tensor = vocab_tensor_cuda2
    else:
        temp_vocab_tensor = vocab_tensor_cuda3

    temp_vocab_tensor[predicted_token_ids[padding_mask]] = True # 将输入中的token添加入此表内

    penalty_term = ~temp_vocab_tensor[predicted_token_ids].to(
        torch.bool)  # 惩罚项，即将predicted_token_ids最后一维替换为True/False， True代表该预测的值不在词表中，需要惩罚


    paddinged_mask = ~padding_mask.squeeze(
        -1)  # mask中为False的值代表需要Loss计算中考虑的值，也就是惩罚项要参考的值，替换True来判断，构建新的mask项来选择实际概率中需要考虑的部分

    # 计算惩罚mask
    mask = penalty_term & paddinged_mask  # 计算两个都为True的时候的mask，即这时候mask为True的really_prob需要被惩罚
    mask = mask.float()

    # 计算最后需要惩罚的项
    masked_of_prob = really_prob[..., ~vocab_tensor]  # 从really_prob中选择所有需要被惩罚的预测输出概率

    sum_of_prob = masked_of_prob.sum(dim=-1)
    penalty_prob = sum_of_prob * mask

    penalty_coefficient = torch.exp(penalty_prob.unsqueeze(-1))

    final_padding_mask = torch.where(~padding_mask, penalty_coefficient, padding_mask)
    final_padding_mask = torch.where(~padding_mask, final_padding_mask, torch.tensor(0.0).to(final_padding_mask.device))
    return final_padding_mask


class customLabelSmoother(LabelSmoother):
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """
    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=True):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        padding_mask = labels.eq(self.ignore_index)
        padding_mask = mask_penalty(log_probs, padding_mask)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        nll_loss = nll_loss * padding_mask
        smoothed_loss = smoothed_loss * padding_mask
        # nll_loss.masked_fill_(padding_mask, 0.0)
        # smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        # return nll_loss
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


class Seq2SeqTrainer(_Seq2SeqTrainer):
    # Not Support for apex

    def alter_label_smoother(self):
        self.label_smoother = customLabelSmoother(epsilon=self.args.label_smoothing_factor)
        # self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)

    def compute_loss(self, model, inputs, return_outputs=False):
        self.alter_label_smoother()
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: dict[str, Any], *args, **kwargs) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        detached_loss = loss.detach() / self.args.gradient_accumulation_steps
        del inputs
        torch.cuda.empty_cache()
        return detached_loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys=None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        with torch.no_grad():  # Ensure no gradient computation
            if self.args.predict_with_generate:
                output_ids = inputs.pop('output_ids')
            input_ids = inputs['input_ids']

            loss, generated_tokens, labels = super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
            )

            generated_tokens = generated_tokens[:, input_ids.size()[1]:]
            labels = output_ids

            del inputs, input_ids, output_ids
            torch.cuda.empty_cache()

        return loss, generated_tokens, labels


@dc.dataclass
class DataConfig(object):
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }


@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int

    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output')
    )
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        if not self.training_args.do_eval or self.data_config.val_file is None:
            self.training_args.do_eval = False
            self.training_args.eval_strategy = 'no'
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size
                    or self.training_args.per_device_train_batch_size
            )

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            gen_config = training_args.get('generation_config')
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)

        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(config_dict=peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        path = Path(path)
        parser = yaml.YAML(typ='safe', pure=True)
        parser.indent(mapping=2, offset=2, sequence=4)
        parser.default_flow_style = False
        kwargs = parser.load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
        data_dir: str,
        data_format: str,
        data_files: dict[NamedSplit, str],
        num_proc: Optional[int],
) -> DatasetDict:
    if data_format == '.json':
        dataset_dct = load_dataset(
            data_dir,
            data_files=data_files,
            split=None,
            num_proc=num_proc,
        )
    else:
        raise NotImplementedError(f"Cannot load dataset in the '{data_format}' format.")
    return dataset_dct


class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = _load_datasets(
            data_dir,
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
            self,
            split: NamedSplit,
            process_fn: Callable[[dict[str, Any]], dict[str, Any]],
            batched: bool = True,
            remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return

        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )


def process_message(message):
    if 'tools' in message and message['role'] == 'system':
        for tool in message['tools']:
            parameters = tool['function']['parameters']['properties']
            tool['function']['parameters']['properties'] = \
                {k: v for k, v in parameters.items() if
                 v is not None}
    elif 'tools' in message:
        del message['tools']
    return message


def process_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_conv = batch['messages']
    batched_input_ids = []
    batched_labels = []

    for conv in batched_conv:
        input_ids = [151331, 151333]
        loss_masks = [False, False]
        for message in conv:
            message = process_message(message)
            loss_mask_val = False if message['role'] in ('system', 'user', 'observation') else True
            new_input_ids = tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)[2:]
            new_loss_masks = [loss_mask_val] * len(new_input_ids)
            input_ids += new_input_ids
            loss_masks += new_loss_masks
        input_ids.append(151336)  # EOS for chat
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])

    del batched_conv, conv, input_ids, loss_masks, message, new_input_ids, new_loss_masks, labels, input_id, mask
    torch.cuda.empty_cache()

    return {'input_ids': batched_input_ids, 'labels': batched_labels}


def process_batch_eval(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_conv = batch['messages']
    batched_input_ids = []
    batched_output_ids = []

    for conv in batched_conv:
        input_ids = [151331, 151333]
        for message in conv:
            if len(input_ids) >= max_input_length:
                break
            else:
                message = process_message(message)
                new_input_ids = tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)[2:]
                if message['role'] == 'assistant':
                    output_prompt, output_ids = (
                        new_input_ids[:1],
                        new_input_ids[1:],
                    )
                    output_ids.append(151336)
                    batched_input_ids.append(
                        input_ids[:max_input_length] + output_prompt[:1]
                    )
                    batched_output_ids.append(output_ids[:max_output_length])
                input_ids += new_input_ids

    del batched_conv, conv, input_ids, message, new_input_ids, output_prompt, output_ids
    torch.cuda.empty_cache()

    return {'input_ids': batched_input_ids, 'output_ids': batched_output_ids}


def load_tokenizer_and_model(
        model_dir: str,
        peft_config: Optional[PeftConfig] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if peft_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False,
            torch_dtype=torch.bfloat16  # Must use BFloat 16
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False,
            torch_dtype=torch.bfloat16
        )
    return tokenizer, model


def compute_metrics(eval_preds: EvalPrediction, tokenizer):
    batched_pred_ids, batched_label_ids = eval_preds
    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        metrics_dct['bleu-4'].append(
            sentence_bleu([label_tokens], pred_tokens, smoothing_function=SmoothingFunction().method3))
    return {k: np.mean(v) for k, v in metrics_dct.items()}


@app.command()
def main(
        data_dir: Annotated[str, typer.Argument(help='')],
        model_dir: Annotated[
            str,
            typer.Argument(
                help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
            ),
        ],
        config_file: Annotated[str, typer.Argument(help='')],
        auto_resume_from_checkpoint: str = typer.Argument(
            default='',
            help='If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training'
        ),
):
    ft_config = FinetuningConfig.from_file(config_file)
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    data_manager = DataManager(data_dir, ft_config.data_config)

    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if val_dataset is not None:
        print('val_dataset:', val_dataset)
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:
        print('test_dataset:', test_dataset)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )

    if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint is None:
        with open("model.txt", "a", encoding="utf-8") as f:
            for name, param in model.named_parameters():
                f.write(f"Layer: {name} | Requires Grad: {param.requires_grad}, {model.device}\n")
        trainer.train()
    else:
        output_dir = ft_config.training_args.output_dir
        dirlist = os.listdir(output_dir)
        checkpoint_sn = 0
        for checkpoint_str in dirlist:
            if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                if checkpoint > checkpoint_sn:
                    checkpoint_sn = checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            if checkpoint_sn > 0:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                print("resume checkpoint from checkpoint-" + str(checkpoint_sn))
                print("checkpoint_directory", checkpoint_directory)
                trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                trainer.train()
        else:
            if auto_resume_from_checkpoint.isdigit():
                if int(auto_resume_from_checkpoint) > 0:
                    checkpoint_sn = int(auto_resume_from_checkpoint)
                    model.gradient_checkpointing_enable()
                    model.enable_input_require_grads()
                    checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                    print("resume checkpoint from checkpoint-" + str(checkpoint_sn))
                    trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                print(auto_resume_from_checkpoint,
                      "The specified checkpoint sn(" + auto_resume_from_checkpoint + ") has not been saved. Please search for the correct checkpoint in the model output directory")

    if test_dataset is not None:
        trainer.predict(test_dataset)


if __name__ == '__main__':
    app()
