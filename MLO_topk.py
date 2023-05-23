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

class MLOTuningTrainer_topk(Trainer):
    def __init__(self, **kwargs):
        self.reserve_p = kwargs.pop('reserve_p')
        self.mode = kwargs.pop('mode')
        self.S = kwargs.pop('S')
        self.S_idx = kwargs.pop('S_idx')
        self.cluster_dim = kwargs.pop('cluster_dim')
        self.sample_p = kwargs.pop('sample_p')
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        if not return_outputs:

            # Before diving into the do an experiment of taking the embeddings and passing through the model.
            # check the gradients w.r.t embedding layer
            ### checked ###
            # embedding layer
            embedding_layer = model.base_model.embeddings.word_embeddings.weight
            embedding_layer = embedding_layer.to(model.device)
            soft_emb_layer = torch.nn.Embedding.from_pretrained(self.S.float().to(model.device))

            if self.S.shape[0] == self.S.shape[1]: # full vocabulary
                # =================== HACK BEGIN =======================
                # first extract the input ids
                x = inputs.pop("input_ids")
                x = x.to(model.device)
                # get the similar words probability
                # please note that the S matrix is row stochastic not column stochastic as in the idea proposal document
                # initialize S appropriately as a row stochastic matrix
                x_soft = soft_emb_layer(x)
                x_soft = x_soft.float().to(model.device) # implement the top k
                # create a selection matrix
                # select only 15 percent of words in a sentence
                seq_lens_wo_pad = inputs["attention_mask"].sum(dim=1)
                N = int(seq_lens_wo_pad.min())
                # select indices
                indices = random.sample(range(N), int(np.ceil(self.sample_p*N))) # select only p% of words to use for soft aug
                rem_indices = list(set(np.arange(x.shape[1])) - set(indices))
                # create mask
                mask = torch.zeros(x.shape[0], x.shape[1], self.tokenizer.vocab_size)
                mask = mask.to(model.device)
                mask.scatter_(-1, x.unsqueeze(-1), 1.).float()
                mask[:,indices,:] = torch.ones_like(mask[:,indices,:])
                # perform the masking operation and make each row a unit norm vector
                x_soft = x_soft * mask
                norm_factor = (x_soft[:,rem_indices,:]).sum(dim=2)[:,:, None]
                (x_soft[:,rem_indices,:]) = (x_soft[:,rem_indices,:]) / norm_factor
                # # topk
                # values, idx = torch.topk(x_soft, 15, dim = 2) # select top 15 words
                # top_k = torch.zeros_like(x_soft)
                # top_k.scatter_(2, idx, values)
                # pass the soft data vector through the embedding layer
                e_soft = torch.matmul(x_soft, embedding_layer)
                # append to the input embeddings
                inputs["inputs_embeds"] = e_soft.to(model.device)
                # =================== HACK END =========================
            else: # topk
                # =================== HACK BEGIN =======================
                # first extract the input ids
                x = inputs.pop("input_ids")
                x = x.to(model.device)
                # get the similar words probability
                # please note that the S matrix is row stochastic not column stochastic as in the idea proposal document
                # initialize S appropriately as a row stochastic matrix
                x_soft = soft_emb_layer(x)
                x_soft = x_soft.float().to(model.device) # implement the top k
                # # # create a selection matrix
                # # # select only 15 percent of words in a sentence
                # seq_lens_wo_pad = inputs["attention_mask"].sum(dim=1)
                # N = int(seq_lens_wo_pad.min())
                # # # select indices
                # indices = random.sample(range(N), min(int(np.ceil(self.sample_p*N)), N)) # select only p% of words to use for soft aug
                # rem_indices = list(set(np.arange(x.shape[1])) - set(indices))
                # # # create mask
                # mask = torch.zeros(x.shape[0], x.shape[1], self.cluster_dim)
                # mask = mask.to(self.model.device)
                # mask[:,:,0] = 1.0
                # mask[:,indices,:] = torch.ones_like(mask[:,indices,:])
                # # # perform the masking operation and make each row a unit norm vector
                # x_soft = x_soft * mask
                # norm_factor = (x_soft[:,rem_indices,:]).sum(dim=2)[:,:, None]
                # (x_soft[:,rem_indices,:]) = (x_soft[:,rem_indices,:]) / norm_factor
                # # pass the soft data vector through the embedding layer
                e_soft = torch.matmul(x_soft.view(x.shape[0], x.shape[1], 1, self.cluster_dim), embedding_layer[self.S_idx[x,:]])
                e_soft = e_soft.view(x.shape[0], x.shape[1], -1)
                # append to the input embeddings
                inputs["inputs_embeds"] = e_soft.to(self.model.device)
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
