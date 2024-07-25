import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlf.rl.loggers import sanity_checker


def weight_init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def no_bias_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    return m


def def_mlp_weight_init(m):
    return weight_init(
        m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
    )


def reg_mlp_weight_init(m):
    """Does not weight init, defaults to whatever pytorch does."""
    return m


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConcatLayer(nn.Module):
    def __init__(self, concat_dim):
        super().__init__()
        self.concat_dim = concat_dim

    def forward(self, ab):
        a, b = ab
        return torch.cat([a, b], dim=self.concat_dim)


class BaseNet(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super().__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def output_shape(self):
        return (self._hidden_size,)

    def _forward_gru(self, x, hidden_state, masks):
        rnn_hxs = hidden_state["rnn_hxs"]
        if x.size(0) == rnn_hxs.size(0):
            x, rnn_hxs = self.gru(x.unsqueeze(0), (rnn_hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            rnn_hxs = rnn_hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = rnn_hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            rnn_hxs = rnn_hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, rnn_hxs = self.gru(
                    x[start_idx:end_idx], rnn_hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            rnn_hxs = rnn_hxs.squeeze(0)

        hidden_state["rnn_hxs"] = rnn_hxs
        return x, hidden_state


class IdentityBase(BaseNet):
    def __init__(self, input_shape):
        super().__init__(False, None, None)
        self.input_shape = input_shape

    def net(self, x):
        return x

    @property
    def output_shape(self):
        return self.input_shape

    def forward(self, inputs, hxs, masks):
        return inputs, None


class PassThroughBase(BaseNet):
    """
    If recurrent=True, will apply RNN layer, otherwise will just pass through
    """

    def __init__(self, input_shape, recurrent, hidden_size):
        if len(input_shape) != 1:
            raise ValueError("Possible RNN can only work on flat")
        super().__init__(recurrent, input_shape[0], hidden_size)
        self.input_shape = input_shape

    def net(self, x):
        return x

    @property
    def output_shape(self):
        if self.is_recurrent:
            return (self._hidden_size,)
        else:
            return self.input_shape

    def forward(self, inputs, hxs, masks):
        x = inputs
        if self.is_recurrent:
            x, hxs = self._forward_gru(x, hxs, masks)
        return x, hxs


class CNNBase(BaseNet):
    def __init__(self, num_inputs, recurrent, hidden_size):
        super().__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: weight_init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        self.net = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU(),
        )

        self.train()

    def forward(self, inputs, hxs, masks):
        x = self.net(inputs / 255.0) # nav inputs: [32, 4, 19, 19]

        if self.is_recurrent:
            x, hxs = self._forward_gru(x, hxs, masks)

        return x, hxs

class MyMLP(BaseNet):
    def __init__(
        self,
        recurrent,
        num_inputs,
        hidden_sizes,
        weight_init=reg_mlp_weight_init,
    ):
        """
        - no_last_act: if True the activation will not be applied on the final
          output.
        """
        super().__init__(recurrent, num_inputs, hidden_sizes[-1])

        assert len(hidden_sizes) > 0

        layers = [weight_init(nn.Linear(num_inputs, hidden_sizes[0])), nn.ReLU(inplace=True)]
        # Minus one for the input layer
        for i in range(len(hidden_sizes) - 1):
            layers.append(weight_init(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self.train()

    def forward(self, inputs, hxs, masks):

        x = inputs

        hidden_actor = self.net(x)

        return hidden_actor, hxs

class MLPBase(BaseNet):
    def __init__(
        self,
        num_inputs,
        recurrent,
        hidden_sizes,
        weight_init=def_mlp_weight_init,
        get_activation=lambda: nn.Tanh(),
        no_last_act=False,
    ):
        """
        - no_last_act: if True the activation will not be applied on the final
          output.
        """
        super().__init__(recurrent, num_inputs, hidden_sizes[-1])

        assert len(hidden_sizes) > 0

        layers = [weight_init(nn.Linear(num_inputs, hidden_sizes[0])), get_activation()]
        # Minus one for the input layer
        for i in range(len(hidden_sizes) - 1):
            layers.append(weight_init(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])))
            if not (no_last_act and i == len(hidden_sizes) - 2):
                layers.append(get_activation())
        self.net = nn.Sequential(*layers)
        self.train()

    def forward(self, inputs, hxs, masks):

        x = inputs

        if self.is_recurrent:
            x, hxs = self._forward_gru(x, hxs, masks)

        hidden_actor = self.net(x)

        return hidden_actor, hxs


class MLPBasic(MLPBase):
    def __init__(
        self,
        num_inputs,
        hidden_size,
        num_layers,
        weight_init=def_mlp_weight_init,
        get_activation=lambda: nn.Tanh(),
    ):
        super().__init__(
            num_inputs, False, [hidden_size] * num_layers, weight_init, get_activation
        )


class TwoLayerMlpWithAction(BaseNet):
    def __init__(
        self,
        num_inputs,
        hidden_sizes,
        action_dim,
        weight_init=def_mlp_weight_init,
        get_activation=lambda: nn.Tanh(),
    ):
        assert len(hidden_sizes) == 2, "Only two hidden sizes"
        super().__init__(False, num_inputs, hidden_sizes[-1])

        self.net = nn.Sequential(
            weight_init(nn.Linear(num_inputs + action_dim, hidden_sizes[0])),
            get_activation(),
            weight_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            get_activation(),
        )

        self.train()

    def forward(self, inputs, actions, hxs, masks):
        return self.net(torch.cat([inputs, actions], dim=-1)), hxs


class InjectNet(nn.Module):
    def __init__(
        self, base_net, head_net, in_dim, hidden_dim, inject_dim, should_inject
    ):
        super().__init__()
        self.base_net = base_net
        if not should_inject:
            inject_dim = 0
        self.head_net = head_net
        self.inject_layer = nn.Sequential(
            nn.Linear(in_dim + inject_dim, hidden_dim), nn.Tanh()
        )
        self.should_inject = should_inject

    def forward(self, x, inject_x):
        x = self.base_net(x)
        if self.should_inject:
            x = torch.cat([x, inject_x], dim=-1)
        x = self.inject_layer(x)
        x = self.head_net(x)
        return x


class DoubleQCritic(BaseNet):
    """
    Code from https://github.com/denisyarats/pytorch_sac.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__(False, None, hidden_dim)

        dims = [hidden_dim] * hidden_depth
        dims.append(1)

        self.Q1 = MLPBase(
            obs_dim + action_dim,
            False,
            dims,
            weight_init=reg_mlp_weight_init,
            get_activation=lambda: nn.ReLU(inplace=True),
            no_last_act=True,
        )
        self.Q2 = MLPBase(
            obs_dim + action_dim,
            False,
            dims,
            weight_init=reg_mlp_weight_init,
            get_activation=lambda: nn.ReLU(inplace=True),
            no_last_act=True,
        )

        # Apply the weight init exactly the same way as @denisyarats
        self.apply(no_bias_weight_init)

    @property
    def output_shape(self):
        return (2,)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1, _ = self.Q1(obs_action, None, None)
        q2, _ = self.Q2(obs_action, None, None)

        return q1, q2
