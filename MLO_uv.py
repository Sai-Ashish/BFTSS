import torch
from transformers import Trainer
from sim_matrix import sim_matrix
from transformers.optimization import get_scheduler
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)
import random
import numpy as np

class MLOTuningTrainer_uv(Trainer):
    def __init__(self, **kwargs):
        self.reserve_p = kwargs.pop('reserve_p')
        self.mode = kwargs.pop('mode')
        self.U = kwargs.pop('U')
        self.V = kwargs.pop('V')
        self.cluster_dim = kwargs.pop('cluster_dim')
        self.sample_p = kwargs.pop('sample_p')
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        if not return_outputs:

            # Before diving into the do an experiment of taking the embeddings and passing through the model.
            # check the gradients w.r.t embedding layer
            ### checked ###
            # embedding layer
            model_embeddings = model.base_model.embeddings.word_embeddings
            embedding_layer = model_embeddings.weight
            embedding_layer = embedding_layer.to(model.device)
            soft_emb_layer = torch.nn.Embedding.from_pretrained(self.V.float().to(model.device))

            # =================== HACK BEGIN =======================
            # first extract the input ids
            x = inputs.pop("input_ids")
            x = x.to(model.device)
            # get the similar words probability
            # please note that the S matrix is row stochastic not column stochastic as in the idea proposal document
            # initialize S appropriately as a row stochastic matrix
            x_soft = soft_emb_layer(x)
            x_soft = x_soft@((self.U).T)
            x_soft = x_soft.float().to(model.device)
            vals, idx = x_soft.topk(self.cluster_dim, dim = -1)
            e_soft = torch.matmul(vals.view(vals.shape[0], vals.shape[1], 1, self.cluster_dim), embedding_layer[idx])
            e_soft = e_soft.view(x.shape[0], x.shape[1], -1)
            # add the original embeddings back
            original_embs = model_embeddings(x).to(model.device)
            # addition operation
            e_soft = e_soft + original_embs
            # append to the input embeddings
            inputs["inputs_embeds"] = e_soft.to(model.device)
            # =================== HACK END =========================
        
        try:
            inputs.pop('idx')
        except:
            pass

        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward(retain_graph=True) # change done!

        return loss.detach()
