from typing import Sequence, Mapping
from numpy import mod
import torch
from torch import nn
from torch import optim
import  torch.nn.functional as F

from .model_utils import get_fully_connected_layers
from scETM.logging_utils import log_arguments

class BatchClassifier(nn.Module):
    """Docstring (TODO)
    """

    @log_arguments
    def __init__(self,
        n_input: int,
        n_output: int,
        hidden_sizes: Sequence[int],
        bn: bool = False,
        bn_track_running_stats: bool = False,
        dropout_prob = 0.2,
        adversarial_loss = 'confuse',
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ) -> None:
        """Docstring (TODO)
        """

        super().__init__()

        self.batch_clf = get_fully_connected_layers(
            n_trainable_input=n_input,
            n_trainable_output=n_output,
            hidden_sizes=hidden_sizes,
            bn=bn,
            bn_track_running_stats=bn_track_running_stats,
            dropout_prob=dropout_prob,
        ).to(device)
        self.n_output = n_output
        assert adversarial_loss in ('confuse', 'reverse')
        self.adversarial_loss = adversarial_loss

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Docstring (TODO)
        """

        logit = self.batch_clf(X)
        if not self.training:
            return dict(logit=logit)

        clf_loss = F.cross_entropy(logit, y)
        if self.adversarial_loss == 'confuse':
            model_loss = (-F.log_softmax(logit, dim=-1) * torch.zeros_like(logit).fill_(1/self.n_output)).sum(-1).mean()
        else:
            model_loss = -clf_loss
        return clf_loss, dict(logit=logit, model_loss=model_loss), dict(clf_loss=clf_loss.detach().item())

    def train_step(self,
        optimizer: optim.Optimizer,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Mapping[str, torch.Tensor]:
        """Docstring (TODO)
        """

        self.train()
        optimizer.zero_grad()
        loss, fwd_dict, new_records = self(X, y)
        loss.backward()
        optimizer.step()
        new_records['clf_acc'] = (fwd_dict['logit'].argmax(1) == y).to(torch.float).mean().detach().item()
        return new_records