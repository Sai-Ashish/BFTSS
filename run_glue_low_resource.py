# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import wandb
import torch
from torch import optim
import numpy as np
from datasets import load_dataset, load_metric


import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    default_data_collator,
    set_seed,
)

import eda
from custom_training_arguments import customTrainingArguments as TrainingArguments
from transformers.optimization import get_scheduler
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from ChildTuningD import ChildTuningDtrainer
from ChildTuningF import ChildTuningFtrainer
# from GradientTuning import GradientTuningTrainer
from sim_matrix import *

from MLO_topk import MLOTuningTrainer_topk
from MLO_uv import MLOTuningTrainer_uv
from mixout import change_model_to_mixout
from topk_freeze import create_optimizer_and_scheduler
from recadam import create_recadam_and_scheduler
from r3f import R3FTrainer

import math
import matplotlib.pyplot as plt
from transformers import EarlyStoppingCallback

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # baseline hyperparams
    # mixout
    mixout_prob: float = field(
        default=0.1
    )

    # freeze k
    freeze_k: int = field(
        default=0
    )

    # recadam
    anneal_k: float = field(
        default=0.
    )
    anneal_t0: float = field(
        default=0.
    )

    #R3F
    r3f_eps: float = field(
        default=1.0
    )
    noise_type : str = field(
        default='normal'
    )

    # EDA
    eda_num_augs : int = field(
        default=9
    )

    # myparams
    reserve_p: float = field(
        default=0.0
    )
    mode: str = field(
        default="Vanilla"
    )

    S_epochs: int = field(
        default=8
    )

    S_lr: float = field(
        default=4e-5
    )

    W_lr: float = field(
        default=5e-6
    )

    S_wd: float = field(
        default=0
    )

    report_freq: int = field(
        default=50
    )

    unroll_steps: int = field(
        default=1
    )

    MLO: bool = field(
        default=False
    )

    S_init: str = field(
        default='Gaussian_distance'
    )

    sample_p: float = field(
        default=0.0
    )

    S_lambda: float = field(
        default=1.0
    )

    inv_temp: float = field(
        default=2.0
    )

    train_split: int = field(
        default=300
    )
    device: str = field(
        default='cuda:0'
    )

    cluster_dim: int = field(
        default=50
    )

    uv_split: int = field(
        default=100
    )

    noisy_S: bool = field(
        default=False
    )


def main(args=''):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    gpu_number = None

    if len(args) == 2:
        arg_strings_list, gpu_queue = args
    else:
        arg_strings_list = args

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif arg_strings_list:
        if gpu_queue is not None:
            gpu_number = gpu_queue.get()
            for index, string in enumerate(arg_strings_list):
                if 'GPU_NUMBER' in string:
                    new_string = string.replace("GPU_NUMBER", str(gpu_number))
                    arg_strings_list[index] = new_string

        model_args, data_args, training_args = parser.parse_args_into_dataclasses(arg_strings_list)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.WARN if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {model_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_warning()
        transformers.utils.logging.disable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    training_args.device = model_args.device

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue.py", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    ).to(training_args.device)

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=data_args.max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    train_dataset = datasets["train"]

    # train set
    if len(train_dataset) > model_args.train_split:
        training_subsample_indices = random.sample(range(len(train_dataset)), model_args.train_split)
        datasets["train"] = train_dataset.select(training_subsample_indices)
        # dev set
        remaining_indices = list(set(np.arange(len(train_dataset))) - set(training_subsample_indices))
        dev_subsample_indices = random.sample(remaining_indices, int(model_args.train_split/6))
        datasets["dev"] = train_dataset.select(dev_subsample_indices)
    else:
        model_args.train_split = len(train_dataset)
        training_subsample_indices = random.sample(range(len(train_dataset)), model_args.train_split)
        datasets["train"] = train_dataset.select(training_subsample_indices)
        datasets["dev"] = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]

    ## EDA here
    if model_args.mode == 'EDA':
        datasets["train"] = eda.eda_dataset(datasets["train"], model_args.eda_num_augs)

    # MLO train set
    MLO_train_indices = random.sample(training_subsample_indices, int(model_args.train_split/2))
    datasets["MLO-train"] = train_dataset.select(MLO_train_indices)

    # MLO valid set
    MLO_valid_indices = list(set(training_subsample_indices) - set(MLO_train_indices))
    datasets["MLO-valid"]   = train_dataset.select(MLO_valid_indices)

    print((set(training_subsample_indices) - set(MLO_train_indices)) - set(MLO_valid_indices))

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]
    
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    
    if model_args.mode != 'EDA':
        dev_dataset = datasets["dev"]
        MLO_train_dataset = datasets["MLO-train"]
        MLO_valid_dataset = datasets["MLO-valid"]
        print(len(train_dataset), len(dev_dataset), len(MLO_train_dataset), len(MLO_valid_dataset), len(eval_dataset))
    else:
        print(len(train_dataset), len(eval_dataset))
    
    if data_args.task_name is not None or data_args.test_file is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("metric.py", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            preds = np.nan_to_num(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
        
    # Initialize our Trainer
    assert model_args.mode in ['ChildTuning-F', 'ChildTuning-D', 'SVDEmbedding', 'SimilarityGradientTopK',
                               'SimilarityGradientLinearMixture', 'SimilarityGradientTopK-F', 'SimilarityGradientTopK-D', 'SimilarityGradientLinearMixture-F', 'SimilarityGradientLinearMixture-D',
                               "TopK_freezing", "Mixout", "RecAdam", "EDA", "R3F",
                               'MLO_approach_uv', 'MLO_approach_topk', "Vanilla"]

    num_update_steps_per_epoch = math.ceil(len(train_dataset) // training_args.per_device_train_batch_size)
    max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    optimizer = None
    pretrained_model = None

    if model_args.mode in ["TopK_freezing", "Mixout", "RecAdam", "R3F", "EDA", "Vanilla"]:
        if model_args.mode is None:
            mode = "Vanilla"
        else:
            mode = model_args.mode

        print(mode)

        if mode == 'Mixout':
            mixout_prob = model_args.mixout_prob
            change_model_to_mixout(model, mixout_prob)
        elif mode == 'TopK_freezing':
            optimizer, scheduler = create_optimizer_and_scheduler(model, model_args.freeze_k, training_args, max_steps)
        elif mode == 'RecAdam':
            import copy
            pretrained_model = copy.deepcopy(model)
            optimizer, scheduler = create_recadam_and_scheduler(model, pretrained_model, model_args, training_args, max_steps)


        if mode == 'TopK_freezing' or mode == 'RecAdam':
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                optimizers=(optimizer, scheduler)
            )
        elif mode == 'R3F':
            trainer = R3FTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                # R3F parameters
                r3f_lambda=1.0,
                eps=model_args.r3f_eps,
                noise_type=model_args.noise_type
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

    else:
        if 'MLO_approach_topk' in model_args.mode:
            trainer_cls = MLOTuningTrainer_topk

            # Vocab size and the embedding matrix for the initialization
            vocab_size = tokenizer.vocab_size
            
            embedding_matrix = model.base_model.embeddings.word_embeddings.weight

            embedding_matrix = embedding_matrix.to('cpu')

            print(embedding_matrix.shape)

            cluster_dim = model_args.cluster_dim # pretrained embedding layer dimension

            if not model_args.noisy_S:
                similarity_mt = sim_matrix(embedding_matrix, embedding_matrix, model_args.S_lambda, model_args.inv_temp)
            else:
                if 'base' in model_args.model_name_or_path:
                    with open('S_base_noise.npy', 'rb') as fh:
                        similarity_mt = torch.from_numpy(np.load(fh))
                else:
                    with open('S_large_noise.npy', 'rb') as fh:
                        similarity_mt = torch.from_numpy(np.load(fh))

            S, S_idx = similarity_mt.topk(cluster_dim, dim =1)
            
            S = S.to(training_args.device)

            copy_S = S.clone().detach()
            
            S_idx = S_idx.to(training_args.device)

            print(similarity_mt.sum(1))
            
            ##################

            trainer = trainer_cls(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=dev_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                reserve_p=model_args.reserve_p,
                mode=model_args.mode,
                S = S,
                sample_p=model_args.sample_p,
                S_idx = S_idx,
                cluster_dim = cluster_dim
            )

        elif 'MLO_approach_uv' in model_args.mode:
            trainer_cls = MLOTuningTrainer_uv

            # Vocab size and the embedding matrix for the initialization
            vocab_size = tokenizer.vocab_size
            
            embedding_matrix = model.base_model.embeddings.word_embeddings.weight

            embedding_matrix = embedding_matrix.detach().to('cpu')

            print(embedding_matrix.shape)

            cluster_dim = model_args.cluster_dim # pretrained embedding layer dimension

            print('### Cluster dim: '+str(cluster_dim)+'###')
            
            similarity_mt = sim_matrix(embedding_matrix, embedding_matrix, model_args.S_lambda, model_args.inv_temp)

            if model_args.model_name_or_path == 'bert-base-cased':

                uv_split = 'base_' + str(model_args.uv_split)
            else:
                uv_split = str(model_args.uv_split)

            print('#### UV file loading: ' + str(uv_split))

            if not model_args.noisy_S:
                U = torch.from_numpy(np.load('U_' + str(uv_split) + '.npy')).float()

                V = torch.from_numpy(np.load('V_' + str(uv_split) + '.npy')).float()

            else:
                if 'base' in model_args.model_name_or_path:
                    U = torch.from_numpy(np.load('U_base_' + str(model_args.uv_split) + '_noise.npy')).float()

                    V = torch.from_numpy(np.load('V_base_' + str(model_args.uv_split) + '_noise.npy')).float()
                else:
                    U = torch.from_numpy(np.load('U_large_' + str(model_args.uv_split) + '_noise.npy')).float()

                    V = torch.from_numpy(np.load('V_large_' + str(model_args.uv_split) + '_noise.npy')).float()

            print(U.shape, V.shape)

            copy_U = U.clone().detach()

            copy_V = V.clone().detach()
            U = U.to(training_args.device)
            V = V.to(training_args.device)
            ##################

            trainer = trainer_cls(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                reserve_p=model_args.reserve_p,
                mode=model_args.mode,
                sample_p=model_args.sample_p,
                U = U,
                V = V,
                cluster_dim = cluster_dim
            )
        else: 
            if model_args.mode == 'ChildTuning-F':
                trainer_cls = ChildTuningFtrainer
            elif model_args.mode == 'ChildTuning-D':
                trainer_cls = ChildTuningDtrainer
            elif model_args.mode == 'SVDEmbedding' or 'SimilarityGradient' in model_args.mode:
                trainer_cls = GradientTuningTrainer
                
            trainer = trainer_cls(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                reserve_p=model_args.reserve_p,
                mode=model_args.mode
            )
        

    ##################################

    if 'MLO_approach_uv' in model_args.mode and model_args.MLO:

        ## TODO: Define in training args the necessary arguments for MLO
        # 1) S_lr -
        # 2) S_wd -
        # 3) report_freq -
        # 4) unroll_steps -
        # 5) S_epochs - 
        # Perform the MLO step to get optimal S and W

        from betty.engine import Engine
        from betty.configs import Config, EngineConfig
        from betty.problems import ImplicitProblem

        print(model_args.S_lr, model_args.S_wd)

        print(model_args.S_epochs, training_args.num_train_epochs)

        # num_training_steps  
        num_update_steps_per_epoch = int(( len(MLO_train_dataset) // training_args.per_device_train_batch_size + 1) * model_args.unroll_steps)
        train_iters = math.ceil(model_args.S_epochs * num_update_steps_per_epoch)
        if training_args.max_steps > 0:
            max_steps = training_args.max_steps
            num_train_epochs = training_args.max_steps // num_update_steps_per_epoch + int(
                training_args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            max_steps = train_iters
            num_train_epochs = math.ceil(model_args.S_epochs)

        num_training_steps = train_iters

        # define the similarity matrix parameters

        class Similarity_matrix(torch.nn.Module):
            def __init__(self):
                super(Similarity_matrix, self).__init__()

                self.soft_emb_layer = torch.nn.Embedding.from_pretrained(V).requires_grad_()

                self.U = torch.nn.Parameter(U)

            def forward(self, x):
                # first extract the input ids
                # get the similar words probability
                # please note that the S matrix is row stochastic not column stochastic as in the idea proposal document
                # initialize S appropriately as a row stochastic matrix
                
                x_soft = self.soft_emb_layer(x)

                x_soft = x_soft@((self.U).T)

                x_soft = x_soft.float().to(training_args.device)
                
                return x_soft
                
        # define the model parameters

        class Network(torch.nn.Module):

            def __init__(self, config, tokenizer, model):
                super(Network, self).__init__()

                # initialize the config
                self.config = config

                # initialize the tokenizer
                self.tokenizer = tokenizer
                self.vocab_size = self.tokenizer.vocab_size

                # initialize the model
                self.model = model

            def forward(self, inputs, x, x_soft):

                # dimension of x_soft is batch_size, seq_len, dimension

                # Before diving into the do an experiment of taking the embeddings and passing through the model.
                # check the gradients w.r.t embedding layer
                ### checked ###
                # embedding layer
                model_embeddings = self.model.base_model.embeddings.word_embeddings
                embedding_layer = model_embeddings.weight
                embedding_layer = embedding_layer.to(self.model.device)

                # =================== HACK BEGIN =======================
                # pass the soft data vector through the embedding layer
                vals, idx = x_soft.topk(cluster_dim, dim = -1)
                e_soft = torch.matmul(vals.view(vals.shape[0], vals.shape[1], 1, cluster_dim), embedding_layer[idx])
                e_soft = e_soft.view(x.shape[0], x.shape[1], -1)
                # add the original embeddings back
                original_embs = model_embeddings(x)
                # addition operation
                e_soft = e_soft + original_embs
                # append to the input embeddings
                inputs["inputs_embeds"] = e_soft.to(self.model.device)
                # =================== HACK END =========================

                try:
                    inputs.pop('idx')
                except:
                    pass

                inputs = trainer._prepare_inputs(inputs)
                outputs = self.model(**inputs)

                return outputs


            def loss(self, inputs, x, x_soft):

                outputs = self(inputs, x, x_soft)

                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                return loss
            
        # define the Similarity matrix parameters

        val_avg_loss = torch.zeros(model_args.report_freq,1)

        U_list = []

        V_list = []

        class Outer(ImplicitProblem):
            def forward(self, x):
                return self.module(x)

            def training_step(self, __init__inputs):
                
                # pass through the S and model
                inputs = __init__inputs.copy()
                x = inputs.pop("input_ids")
                x = x.to(training_args.device)
                x_soft = self.forward(x)
                loss = self.inner.module.loss(inputs, x, x_soft)

                # plotting for inference
                U_list.append(((torch.abs(self.module.U.to('cpu')-copy_U)).sum()).detach().numpy())
                V_list.append(((torch.abs(self.module.soft_emb_layer.weight.data.to('cpu')-copy_V)).sum()).detach().numpy())
                
                # val loss list
                val_avg_loss[(self.count-1)%model_args.report_freq] = loss.item()

                if self.count % model_args.report_freq == 0:
                    print(f"step {self.count} || Outer loss: {val_avg_loss.mean()}")

                return loss

            def configure_train_data_loader(self):
                valid_queue = torch.utils.data.DataLoader(
                    MLO_valid_dataset,
                    sampler=torch.utils.data.sampler.RandomSampler(MLO_valid_dataset),
                    batch_size=training_args.per_device_train_batch_size,
                    collate_fn=data_collator,
                    drop_last=training_args.dataloader_drop_last,
                    num_workers=training_args.dataloader_num_workers,
                    pin_memory=training_args.dataloader_pin_memory
                )
                return valid_queue

            def configure_module(self):
                return Similarity_matrix().to(training_args.device)

            def configure_optimizer(self):
                optimizer = optim.Adam(
                    self.module.parameters(),
                    lr=model_args.S_lr,
                    betas=(0.9, 0.999),
                    weight_decay=model_args.S_wd,
                )
                return optimizer

        train_avg_loss = torch.zeros(model_args.report_freq,1)

        class Inner(ImplicitProblem):
            def forward(self, inputs, x, x_soft):
                return self.module(inputs, x, x_soft)

            def training_step(self, __init__inputs):

                inputs = __init__inputs.copy()

                x = inputs.pop("input_ids")

                x = x.to(training_args.device)

                x_soft = self.outer(x)

                loss = self.module.loss(inputs, x, x_soft)

                train_avg_loss[(self.count-1)%model_args.report_freq] = loss.item()

                if self.count % model_args.report_freq == 0:
                    print(f"step {self.count} || Inner loss: {train_avg_loss.mean()}")

                return loss

            def configure_train_data_loader(self):
                train_queue = torch.utils.data.DataLoader(
                    MLO_train_dataset,
                    sampler=torch.utils.data.sampler.RandomSampler(MLO_train_dataset),
                    batch_size=training_args.per_device_train_batch_size,
                    collate_fn=data_collator,
                    drop_last=training_args.dataloader_drop_last,
                    num_workers=training_args.dataloader_num_workers,
                    pin_memory=training_args.dataloader_pin_memory
                )
                return train_queue

            def configure_module(self):
                return Network(config, tokenizer, model).to(training_args.device)

            def configure_optimizer(self):

                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.module.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": training_args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.module.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]

                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

                optimizer_kwargs['lr'] = model_args.W_lr

                print(optimizer_kwargs)

                optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

                return optimizer

            def configure_scheduler(self):

                scheduler = get_scheduler(
                    training_args.lr_scheduler_type,
                    self.optimizer,
                    num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
                return scheduler

        class NASEngine(Engine):
            @torch.no_grad()
            def validation(self):

                tasks = [data_args.task_name]
                eval_datasets = [dev_dataset]
                #if data_args.task_name == "mnli":
                #    tasks.append("mnli-mm")
                #    eval_datasets.append(datasets["validation_mismatched"])

                for _dataset, task in zip(eval_datasets, tasks):
                    eval_result = trainer.evaluate(eval_dataset=_dataset)

                    if trainer.is_world_process_zero():
                        for key, value in sorted(eval_result.items()):
                            print(f"{key} = {value}\n")
        
        # Putting together the pieces and running using Betty
        outer_config = Config(retain_graph=True, first_order=True)
        inner_config = Config(type="darts", unroll_steps=model_args.unroll_steps)
        engine_config = EngineConfig(
            valid_step=model_args.report_freq * model_args.unroll_steps,
            train_iters=train_iters,
            roll_back=True,
            #logger_type='wandb'
        )
        outer = Outer(name="outer", config=outer_config, device=training_args.device)
        inner = Inner(name="inner", config=inner_config, device=training_args.device)

        problems = [outer, inner]
        l2u = {inner: [outer]}
        u2l = {outer: [inner]}
        dependencies = {"l2u": l2u, "u2l": u2l}

        engine = NASEngine(config=engine_config, problems=problems, dependencies=dependencies)
        engine.run()

        #############

        # Code to copy the learned S matrix to the trainer.S

        U = outer.module.U
        V = outer.module.soft_emb_layer.weight.data
        torch.save(U, 'U.pt')
        torch.save(V, 'V.pt')
        print((torch.abs(U.to('cpu')-copy_U)).sum(), (torch.abs(V.to('cpu')-copy_V)).sum())
        
        trainer.U = U
        trainer.V = V

        # Plot the graph
        plt.plot(U_list)
        plt.plot(V_list)
        plt.savefig('U_V.png')
    #################################

    if 'MLO_approach_topk' in model_args.mode and model_args.MLO:

        from betty.engine import Engine
        from betty.configs import Config, EngineConfig
        from betty.problems import ImplicitProblem

        ## TODO: Define in training args the necessary arguments for MLO
        # 1) S_lr -
        # 2) S_wd -
        # 3) report_freq -
        # 4) unroll_steps -
        # 5) S_epochs - 
        # Perform the MLO step to get optimal S and W

        print(model_args.S_lr, model_args.S_wd)

        print(model_args.S_epochs, training_args.num_train_epochs)

        # num_training_steps  
        num_update_steps_per_epoch = int(( len(MLO_train_dataset) // training_args.per_device_train_batch_size + 1) * model_args.unroll_steps)
        train_iters = math.ceil(model_args.S_epochs * num_update_steps_per_epoch)
        if training_args.max_steps > 0:
            max_steps = training_args.max_steps
            num_train_epochs = training_args.max_steps // num_update_steps_per_epoch + int(
                training_args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            max_steps = train_iters
            num_train_epochs = math.ceil(model_args.S_epochs)

        num_training_steps = train_iters

        # define the similarity matrix parameters

        class Similarity_matrix(torch.nn.Module):
            def __init__(self, S):
                super(Similarity_matrix, self).__init__()

                self.soft_emb_layer = torch.nn.Embedding.from_pretrained(S).requires_grad_()

                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # first extract the input ids
                # get the similar words probability
                # please note that the S matrix is row stochastic not column stochastic as in the idea proposal document
                # initialize S appropriately as a row stochastic matrix

                # self.soft_emb_layer.weight.data = self.relu(self.soft_emb_layer.weight.data)

                # self.soft_emb_layer.weight.data = self.soft_emb_layer.weight.data/(self.soft_emb_layer.weight.data.max(1)[0][:,None])
                
                x_soft = self.soft_emb_layer(x)

                x_soft = x_soft.float().to(training_args.device)
                
                return x_soft
                
        # define the model parameters

        class Network(torch.nn.Module):

            def __init__(self, config, tokenizer, model):
                super(Network, self).__init__()

                # initialize the config
                self.config = config

                # initialize the tokenizer
                self.tokenizer = tokenizer
                self.vocab_size = self.tokenizer.vocab_size

                # initialize the model
                self.model = model

            def forward(self, inputs, x, x_soft):

                # dimension of x_soft is batch_size, seq_len, dimension

                # Before diving into the do an experiment of taking the embeddings and passing through the model.
                # check the gradients w.r.t embedding layer
                ### checked ###
                # embedding layer
                embedding_layer = self.model.base_model.embeddings.word_embeddings.weight
                embedding_layer = embedding_layer.to(self.model.device)

                # =================== HACK BEGIN =======================
                # # create a selection matrix
                # # select only 15 percent of words in a sentence
                # seq_lens_wo_pad = inputs["attention_mask"].sum(dim=1)
                # N = int(seq_lens_wo_pad.min())
                # # select indices
                # indices = random.sample(range(N), min(int(np.ceil(model_args.sample_p*N)), N)) # select only p% of words to use for soft aug
                # rem_indices = list(set(np.arange(x.shape[1])) - set(indices))
                # # create mask
                # mask = torch.zeros(x.shape[0], x.shape[1], cluster_dim)
                # mask = mask.to(self.model.device)
                # mask[:,:,0] = 1.0
                # mask[:,indices,:] = torch.ones_like(mask[:,indices,:])
                # # perform the masking operation and make each row a unit norm vector
                # x_soft = x_soft * mask
                # norm_factor = (x_soft[:,rem_indices,:]).sum(dim=2)[:,:, None]
                # (x_soft[:,rem_indices,:]) = (x_soft[:,rem_indices,:]) / norm_factor
                # pass the soft data vector through the embedding layer
                e_soft = torch.matmul(x_soft.view(x.shape[0], x.shape[1], 1, cluster_dim), embedding_layer[S_idx[x,:]])
                e_soft = e_soft.view(x.shape[0], x.shape[1], -1)
                # append to the input embeddings
                inputs["inputs_embeds"] = e_soft.to(self.model.device)
                # =================== HACK END =========================

                try:
                    inputs.pop('idx')
                except:
                    pass

                inputs = trainer._prepare_inputs(inputs)
                outputs = self.model(**inputs)

                return outputs


            def loss(self, inputs, x, x_soft):

                outputs = self(inputs, x, x_soft)

                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                return loss
            
        # define the Similarity matrix parameters

        val_avg_loss = torch.zeros(model_args.report_freq,1)

        S_list = []

        class Outer(ImplicitProblem):
            def forward(self, x):
                return self.module(x)

            def training_step(self, __init__inputs):

                inputs = __init__inputs.copy()

                x = inputs.pop("input_ids")
                x = x.to(training_args.device)

                x_soft = self.forward(x)
                loss = self.inner.module.loss(inputs, x, x_soft)

                # plotting for inference
                S_list.append(((torch.abs(self.module.soft_emb_layer.weight.data.cpu()-copy_S.cpu())).sum()).detach().numpy())

                val_avg_loss[(self.count-1)%model_args.report_freq] = loss.item()

                if self.count % model_args.report_freq == 0:
                    print(f"step {self.count} || Outer loss: {val_avg_loss.mean()}")

                return loss

            def configure_train_data_loader(self):
                valid_queue = torch.utils.data.DataLoader(
                    MLO_valid_dataset,
                    sampler=torch.utils.data.sampler.RandomSampler(MLO_valid_dataset),
                    batch_size=training_args.per_device_train_batch_size,
                    collate_fn=data_collator,
                    drop_last=training_args.dataloader_drop_last,
                    num_workers=training_args.dataloader_num_workers,
                    pin_memory=training_args.dataloader_pin_memory
                )
                return valid_queue

            def configure_module(self):
                return Similarity_matrix(S).to(training_args.device)

            def configure_optimizer(self):
                optimizer = optim.Adam(
                    self.module.parameters(),
                    lr=model_args.S_lr,
                    betas=(0.9, 0.999),
                    weight_decay=model_args.S_wd,
                )
                return optimizer

        train_avg_loss = torch.zeros(model_args.report_freq,1)

        class Inner(ImplicitProblem):
            def forward(self, inputs, x, x_soft):
                return self.module(inputs, x, x_soft)

            def training_step(self, __init__inputs):

                inputs = __init__inputs.copy()

                x = inputs.pop("input_ids")

                x = x.to(training_args.device)

                x_soft = self.outer(x)

                loss = self.module.loss(inputs, x, x_soft)

                train_avg_loss[(self.count-1) % model_args.report_freq] = loss.item()

                if self.count % model_args.report_freq == 0:
                    print(f"step {self.count} || Inner loss: {train_avg_loss.mean()}")

                return loss

            def configure_train_data_loader(self):
                train_queue = torch.utils.data.DataLoader(
                    MLO_train_dataset,
                    sampler=torch.utils.data.sampler.RandomSampler(MLO_train_dataset),
                    batch_size=training_args.per_device_train_batch_size,
                    collate_fn=data_collator,
                    drop_last=training_args.dataloader_drop_last,
                    num_workers=training_args.dataloader_num_workers,
                    pin_memory=training_args.dataloader_pin_memory
                )
                return train_queue

            def configure_module(self):
                return Network(config, tokenizer, model).to(training_args.device)

            def configure_optimizer(self):

                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.module.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": training_args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.module.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]

                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

                optimizer_kwargs['lr'] = model_args.W_lr

                print(optimizer_kwargs)

                optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

                return optimizer

            def configure_scheduler(self):

                scheduler = get_scheduler(
                    training_args.lr_scheduler_type,
                    self.optimizer,
                    num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
                return scheduler

        class NASEngine(Engine):
            @torch.no_grad()
            def validation(self):

                tasks = [data_args.task_name]
                eval_datasets = [dev_dataset]
                # if data_args.task_name == "mnli":
                #     tasks.append("mnli-mm")
                #     eval_datasets.append(datasets["validation_mismatched"])

                for _dataset, task in zip(eval_datasets, tasks):
                    eval_result = trainer.evaluate(eval_dataset=_dataset)

                    if trainer.is_world_process_zero():
                        for key, value in sorted(eval_result.items()):
                            print(f"{key} = {value}\n")
        
        # Putting together the pieces and running using Betty
        outer_config = Config(retain_graph=True, first_order=True)
        inner_config = Config(type="darts", unroll_steps=model_args.unroll_steps)
        engine_config = EngineConfig(
            valid_step=model_args.report_freq * model_args.unroll_steps,
            train_iters=train_iters,
            roll_back=True,
            # logger_type='wandb'
        )
        outer = Outer(name="outer", config=outer_config, device=training_args.device)
        inner = Inner(name="inner", config=inner_config, device=training_args.device)

        problems = [outer, inner]
        l2u = {inner: [outer]}
        u2l = {outer: [inner]}
        dependencies = {"l2u": l2u, "u2l": u2l}

        engine = NASEngine(config=engine_config, problems=problems, dependencies=dependencies)
        engine.run()

        #############

        # Code to copy the learned S matrix to the trainer.S

        S = outer.module.soft_emb_layer.weight.data

        torch.save(S, 'S.pt')

        print((torch.abs(S-copy_S)).sum())
        
        trainer.S = S # S used only for the training process not for evaluation

        print((torch.abs(trainer.S-copy_S)).sum())

        # Plot the graph
        plt.plot(S_list)
        plt.savefig('S.png')
    
    #################################

    # Training
    last_checkpoint = None
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        metrics = train_result.metrics

        # trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        # if trainer.is_world_process_zero():
        #     with open(output_train_file, "w") as writer:
        #         logger.info("***** Train results *****")
        #         for key, value in sorted(metrics.items()):
        #             logger.info(f"  {key} = {value}")
        #             writer.write(f"{key} = {value}\n")
        #
        #     # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        #     trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)
            task_name_result = {key + "/" + task : eval_result[key] for key in eval_result}

            # output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
            # if trainer.is_world_process_zero():
            #     with open(output_eval_file, "w") as writer:
            #         logger.info(f"***** Eval results {task} *****")
            #         for key, value in sorted(eval_result.items()):
            #             logger.info(f"  {key} = {value}")
            #             writer.write(f"{key} = {value}\n")

            eval_results.update(task_name_result)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


    if gpu_number != None:
        gpu_queue.put(gpu_number)

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()


