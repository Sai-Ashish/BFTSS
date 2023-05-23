from transformers import Trainer
import torch
import transformers
import torch.nn.functional as F

class R3FTrainer(Trainer):
    def __init__(self, *args,  r3f_lambda=1, eps=0.1, noise_type='uniform', **kwargs):
        super(R3FTrainer, self).__init__(*args, **kwargs)
        self.eps = eps
        self.r3f_lambda = r3f_lambda
        self.noise_type = noise_type
        if self.noise_type == "normal":
            self.noise_sampler = torch.distributions.normal.Normal(
                loc=0.0, scale=self.eps
            )
        elif self.noise_type == "uniform":
            self.noise_sampler = torch.distributions.uniform.Uniform(
                low=-self.eps, high=self.eps
            )
        else:
            raise Exception(f"unrecognized noise type {self.noise_type}")

    def compute_loss(self, model: transformers.BertForSequenceClassification, inputs, return_outputs=False):
        labels = inputs.get("labels")
        input_ids = inputs.pop("input_ids")

        # forward pass. first get embeddings
        embedding_module = model.bert.embeddings
        raw_word_embeddings_output = embedding_module.word_embeddings(input_ids)

        sample_size = labels.size(0)

        normal_outputs = model(**inputs, inputs_embeds=raw_word_embeddings_output)
        normal_loss = normal_outputs.loss
        normal_logits = normal_outputs.logits

        total_loss = normal_loss

        if model.training:

            # add noise to the embeddings
            noise = self.noise_sampler.sample(sample_shape=raw_word_embeddings_output.shape).to(
                raw_word_embeddings_output
            )
            noised_embeddings = raw_word_embeddings_output.clone() + noise

            noised_outputs = model(
                **inputs, inputs_embeds=noised_embeddings
            )
            noised_logits = noised_outputs.logits

            symm_kl = self._get_symm_kl(noised_logits, normal_logits)

            symm_kl = symm_kl * sample_size
            total_loss = total_loss + self.r3f_lambda * symm_kl

        return (total_loss, normal_outputs) if return_outputs else total_loss

    def _get_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
        ) / noised_logits.size(0)