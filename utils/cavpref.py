import torch
from torch.nn import functional as F
import torch.nn as nn
import transformers

class CAVPref(nn.Module):
    def __init__(self, lambdas = {"T":1.0, "V": 1.0, "A": 0.8}):
        self.lambdas = lambdas
        self.eta = {"MCIT": 0, "ICIT": 0, "MVIT": 1, "MAIT": 0, "COT-Stitch": 0, "COT-Swap": 1, "CAT": 0, "MVT": 1, "MAT": 0}
        self.gamma = {"MCIT": 0, "ICIT": 0, "MVIT": 0, "MAIT": 1, "COT-Stitch": 0, "COT-Swap": 1, "CAT": 0, "MVT": 0, "MAT": 1}

    def return_log_probs(self, avllm, inputs):
        output_logits = avllm(inputs).logits.to(torch.float32)
        labels = inputs["labels"][:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)
        labels[labels == -100] = 0
        return logits.log_softmax(-1)

    def return_sigmoids(self, winning_logprobs, losing_logprobs, beta=0.1):
        delta = winning_logprobs - losing_logprobs
        return F.logsigmoid(beta * delta)

    def return_preference_loss(self, avllm, T, V, A, betas = {"T": 0.1, "V": 0.1, "A": 0.1}, task_name = "MCIT"):

        # Obtain the tuples of (T["winning"], T["losing"]), (V["winning"], V["losing"]), (A["winning"], A["losing"]) from the batch tokenizer
        # T["winning"] (or T["losing"]) consists of input_ids, attention_masks, labels
        # V["winning"] (or V["losing"]) consists of output tensors from video processor
        # A["winning"] (or A["losing"]) consists of output tensors from audio processor

        winning_text_logprobs = self.return_log_probs(avllm, inputs={'text': T["winning"], 'visual': V["winning"], 'audio': A["winning"]})
        losing_text_logprobs = self.return_log_probs(avllm, inputs={'text': T["losing"], 'visual': V["winning"], 'audio': A["winning"]})
        loss_text = - self.lambdas["T"] * torch.log(torch.mean(torch.exp(self.return_sigmoids(winning_text_logprobs, losing_text_logprobs, beta=betas["T"]) / self.lambdas["T"])))
        loss = loss_text

        if self.eta[task_name] != 0:
            winning_visual_logprobs = self.return_log_probs(avllm, inputs={'text': T["winning"], 'visual': V["winning"], 'audio': A["winning"]})
            losing_visual_logprobs = self.return_log_probs(avllm, inputs={'text': T["winning"], 'visual': V["losing"], 'audio': A["winning"]})
            loss_visual = - self.lambdas["V"] * torch.log(torch.mean(torch.exp(self.return_sigmoids(winning_visual_logprobs, losing_visual_logprobs, beta=betas["V"]) / self.lambdas["V"])))
            loss = loss + self.eta[task_name] * loss_visual

        if self.gamma[task_name] != 0:
            winning_audio_logprobs = self.return_log_probs(avllm, inputs={'text': T["winning"], 'visual': V["winning"], 'audio': A["winning"]})
            losing_audio_logprobs = self.return_log_probs(avllm, inputs={'text': T["winning"], 'visual': V["winning"], 'audio': A["losing"]})
            loss_audio = - self.lambdas["A"] * torch.log(torch.mean(torch.exp(self.return_sigmoids(winning_audio_logprobs, losing_audio_logprobs, beta=betas["A"]) / self.lambdas["A"])))
            loss = loss + self.gamma[task_name] * loss_audio

        return loss