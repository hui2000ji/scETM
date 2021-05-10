from typing import Sequence, Union
from torch import nn
import torch
import math

class InputPartlyTrainableLinear(nn.Module):
    """A linear layer with partially trainable input weights.

    The weights are divided into two parts, one of shape [I_trainable, O] is
    trainable, the other of shape [I_fixed, O] is fixed.
    If bias = True, the trainable part would have a trainbale bias, and if
    n_trainable_input is 0, a trainable bias is added to the layer.

    In the forward pass, the input x of shape [B, I] is split into x_fixed of
    shape [B, I_fixed] and x_trainable of shape [B, I_trainable]. The two parts
    are separately affinely transformed and results are summed.

    B: batch size; I: input dim; O: output dim.

    Attributes:
        fixed: the fixed part of the layer.
        trainable: the trainable part of the layer.
        trainable_bias: a trainable bias. Only present if n_trainable_input is
            0, i.e. all weights are fixed, and bias is True.
        n_fixed_input: number of inputs whose weights should be fixed.
        n_trainable_input: number of inputs whose weights should be trainable.
    """

    def __init__(self, n_fixed_input: int, n_output: int, n_trainable_input: int = 0, bias: bool = True) -> None:
        """Initialize the InputPartlyTrainableLinear layer.

        Args:
            n_fixed_input: number of inputs whose weights should be fixed.
            n_output: number of outputs.
            n_trainable_input: number of inputs whose weights should be
                trainable.
            bias: add a trainable bias if all weights are fixed. This gives
                more flexibility to the model.                
        """

        super().__init__()
        self.fixed: nn.Linear = nn.Linear(n_fixed_input, n_output, bias=False)
        self.fixed.requires_grad_(False)
        self.trainable_bias: Union[None, nn.Parameter] = None
        if n_trainable_input > 0:
            self.trainable: nn.Linear = nn.Linear(n_trainable_input, n_output, bias=bias)
        elif bias:
            self.trainable_bias = nn.Parameter(torch.Tensor(n_output))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fixed.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.trainable_bias, -bound, bound)
        self.n_fixed_input: int = n_fixed_input
        self.n_trainable_input: int = n_trainable_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the InputPartlyTrainableLinear layer.

        Args:
            x: the input tensor of shape [B, I].

        Returns:
            A linear-transformed x.
        """
        
        if self.n_trainable_input > 0:
            x_fixed, x_trainable = x[:, :self.n_fixed_input], x[:, self.n_fixed_input:]
            with torch.no_grad():
                out = self.fixed(x_fixed)
            return out + self.trainable(x_trainable)
        elif self.trainable_bias is not None:
            return self.fixed(x) + self.trainable_bias
        else:
            return self.fixed(x)

    @property
    def weight(self) -> torch.Tensor:
        """A read-only property to access the weights of this layer.

        If both trainable and fixed weights are present, concatenate them and
        return. Else return the fixed weights.
        """

        if self.n_trainable_input > 0:
            return torch.cat([self.fixed.weight, self.trainable.weight], dim=1)
        else:
            return self.fixed.weight
    
    @property
    def bias(self) -> Union[torch.Tensor, None]:
        """A read-only property to access the bias of this layer.

        If the trainable module exists, return the bias of the trainable
        module. Else, return self.trainable_bias (could be None).
        """

        if self.n_trainable_input > 0:
            return self.trainable.bias
        else:
            return self.trainable_bias

class OutputPartlyTrainableLinear(nn.Module):
    """A linear layer with partially trainable output weights.

    The weights are divided into two parts, one of shape [I, O_trainable] is
    trainable, the other of shape [I, O_fixed] is fixed.
    If bias = True, the trainable part would have a bias, and the fixed part
    would also have a trainble bias.

    In the forward pass, the input x of shape [B, I] is separately affinely
    transformed by the fixed and the trainable linear layers and the results
    are concatenated.

    B: batch size; I: input dim; O: output dim.

    Attributes:
        fixed: the fixed part of the layer.
        trainable: the trainable part of the layer.
        trainable_bias: a trainable bias. Only present if bias is True.
        n_fixed_output: number of outputs whose weights should be fixed.
        n_trainable_output: number of outputs whose weights should be
            trainable.
        enable_bias: whether the model has bias.
    """

    def __init__(self, n_input: int, n_fixed_output: int, n_trainable_output: int = 0, bias: bool = True) -> None:
        """Initialize the InputPartlyTrainableLinear layer.

        Args:
            n_fixed_input: number of inputs whose weights should be fixed.
            n_output: number of outputs.
            n_trainable_input: number of inputs whose weights should be
                trainable.
            bias: add a trainable bias if all weights are fixed. This gives
                more flexibility to the model.                
        """

        super().__init__(self)
        self.fixed: nn.Linear = nn.Linear(n_input, n_fixed_output, bias=False)
        self.fixed.requires_grad_(False)
        self.trainable_bias: Union[None, nn.Parameter] = None
        if bias:
            self.trainable_bias = nn.Parameter(torch.Tensor(n_fixed_output))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fixed.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.trainable_bias, -bound, bound)
        if n_trainable_output > 0:
            self.trainable: nn.Linear = nn.Linear(n_input, n_trainable_output, bias=bias)
        self.n_fixed_output: int = n_fixed_output
        self.n_trainable_output: int = n_trainable_output
        self.enable_bias: bool = bias

    def forward(self, x: torch.Tensor):
        """Forward pass of the OutputPartlyTrainableLinear layer.

        Args:
            x: the input tensor of shape [B, I].

        Returns:
            A linear-transformed x.
        """

        # calculate fixed output
        with torch.no_grad():
            fixed_output = self.fixed(x)
        if self.trainable_bias is not None:
            fixed_output = fixed_output + self.trainable_bias
        # return full output
        if self.n_trainable_output > 0:
            return torch.cat([fixed_output, self.trainable(x)], dim=-1) 
        else:
            return fixed_output

    @property
    def weight(self):
        """A read-only property to access the weights of this layer.

        If both trainable and fixed weights are present, concatenate them and
        return. Else return the fixed weights.
        """

        if self.n_trainable_output > 0:
            return torch.cat([self.fixed.weight, self.trainable.weight], dim=0)
        else:
            return self.fixed.weight

    @property
    def bias(self):
        """A read-only property to access the bias of this layer.

        If both trainable and fixed biases are present, concatenate them and
        return. Else, return fixed bias.
        """
        if not self.enable_bias:
            return None
        if self.n_trainable_output > 0:
            return torch.cat([self.trainable_bias, self.trainable.bias], dim=0)
        else:
            return self.trainable_bias

class PartlyTrainableParameter2D(nn.Module):
    """A partly trainable 2D parameter.

    The [H, W] parameter is split to two parts, the fixed [H, W_fixed] and the
    trainbale [H, W_trainable].

    H: height, W: width.

    Args:
        height: the height of the parameter.
        n_fixed_width: the width of the fixed part of the parameter.
        n_trainable_width: the width of the trainable part of the parameter.
        fixed: the fixed part of the parameter.
        trainable: the trainable part of the parameter.
    """

    def __init__(self, height: int, n_fixed_width: int, n_trainable_width: int) -> None:
        super().__init__()
        self.height: int = height
        self.n_fixed_width: int = n_fixed_width
        self.n_trainable_width: int = n_trainable_width
        self.fixed: Union[None, torch.Tensor] = None
        self.trainable: Union[None, nn.Parameter] = None
        if n_fixed_width > 0:
            self.fixed = torch.randn(height, n_fixed_width)
        if n_trainable_width > 0:
            self.trainable = nn.Parameter(torch.randn(height, n_trainable_width))
    
    def get_param(self) -> Union[None, torch.Tensor]:
        """Get the [H, W] parameter as a whole.

        Returns:
            The parameter. If both n_fixed_width and n_trainable_width is 0,
            return None.
        """

        params = [param for param in (self.fixed, self.trainable) if param is not None]
        if len(params) == 2:
            return torch.cat(params, dim=1)
        elif len(params) == 1:
            return params[0]
        else:
            return None

    def __repr__(self):
        return f'{self.__class__.__name__}(height={self.height}, fixed={self.n_fixed_width}, trainable={self.n_trainable_width})'


def get_fully_connected_layers(
    n_trainable_input: int,
    hidden_sizes: Union[int, Sequence[int]],
    n_trainable_output: Union[None, int] = None,
    bn: bool = True,
    bn_track_running_stats: bool = True,
    dropout_prob: float = 0.,
    n_fixed_input: int = 0,
    n_fixed_output: int = 0
) -> nn.Sequential:
    """Construct fully connected layers given the specifications.

    Args:
        n_trainable_input: number of trainable input.
        hidden_sizes: If int, constuct one hidden layer with the given
            size. If sequence of ints, construct a hidden layer for each
            element with the given size.
        n_trainable_output: number of trainable output. If None, do not
            construct the output layer.
        bn: whether to add a BatchNorm1d layer after ReLU activation.
        bn_track_running_stats: the track_running_stats argument of the
            nn.BatchNorm1d constructor.
        dropout_prob: dropout probability. If 0, disable dropout.
        n_fixed_input: number of fixed input. Parameters in the input layer
            related to these genes should be fixed. Useful for the fine-
            tuning stage in transfer learning.
        n_fixed_output: number of fixed output. Parameters in the output
            layer related to these genes should be fixed. Useful for the
            fine-tuning stage in transfer learning.
    
    Returns:
        The constructed fully connected layers.
    """

    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    layers = []
    for i, size in enumerate(hidden_sizes):
        if i == 0 and n_fixed_input > 0:
            layers.append(InputPartlyTrainableLinear(n_fixed_input, size, n_trainable_input))
        else:
            layers.append(nn.Linear(n_trainable_input, size))
        layers.append(nn.ReLU())
        if bn:
            layers.append(nn.BatchNorm1d(size, track_running_stats=bn_track_running_stats))
        if dropout_prob:
            layers.append(nn.Dropout(dropout_prob))
        n_trainable_input = size
    if n_trainable_output is not None:
        if n_fixed_output > 0:
            layers.append(OutputPartlyTrainableLinear(n_trainable_input, n_fixed_output, n_trainable_output))
        else:
            layers.append(nn.Linear(n_trainable_input, n_trainable_output))
    return nn.Sequential(*layers)


def get_kl(mu: torch.Tensor, logsigma: torch.Tensor):
    """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).

    Args:
        mu: the mean of the q distribution.
        logsigma: the log of the standard deviation of the q distribution.

    Returns:
        KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
    """

    logsigma = 2 * logsigma
    return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)