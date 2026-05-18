# Copyright 2026 Junbo Ding
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

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMBackbone(nn.Module):

    def __init__(self, input_dim: int=76, hidden_dim: int=256, num_layers: int=2, dropout: float=0.3, bidirectional: bool=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    @staticmethod
    def _lengths_from_mask(seq_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        if seq_mask is None:
            return torch.full((1,), seq_len, dtype=torch.long)
        return seq_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1)

    def forward(self, x: torch.Tensor, seq_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_mask is not None:
            lengths = self._lengths_from_mask(seq_mask, seq_len).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            (_, (h_n, _)) = self.lstm(packed)
        else:
            (_, (h_n, _)) = self.lstm(x)
        if self.bidirectional:
            z = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            z = h_n[-1]
        return self.dropout(z)

    def forward_sequence(self, x: torch.Tensor, seq_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_mask is not None:
            lengths = self._lengths_from_mask(seq_mask, seq_len).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            (packed_out, _) = self.lstm(packed)
            (out, _) = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        else:
            (out, _) = self.lstm(x)
        return self.dropout(out)

class ChannelWiseLSTM(nn.Module):

    def __init__(self, input_dim: int=76, hidden_dim_per_channel: int=16, num_layers: int=1, dropout: float=0.3, output_dim: int=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim_per_channel = hidden_dim_per_channel
        self.channel_lstms = nn.ModuleList([nn.LSTM(input_size=1, hidden_size=hidden_dim_per_channel, num_layers=num_layers, batch_first=True) for _ in range(input_dim)])
        concat_dim = input_dim * hidden_dim_per_channel
        self.fc = nn.Sequential(nn.Linear(concat_dim, output_dim), nn.ReLU(), nn.Dropout(dropout))
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, seq_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        channel_outputs = []
        for (channel_idx, lstm) in enumerate(self.channel_lstms):
            channel_input = x[:, :, channel_idx:channel_idx + 1]
            (_, (h_n, _)) = lstm(channel_input)
            channel_outputs.append(h_n[-1])
        z = torch.cat(channel_outputs, dim=1)
        return self.fc(z)

class ResNetBackbone(nn.Module):

    def __init__(self, model_name: str='resnet50', pretrained: bool=True, allow_random_init: bool=False, **_: object):
        super().__init__()
        from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
        model_name = str(model_name).strip().lower()
        if model_name not in {'resnet18', 'resnet50'}:
            raise ValueError(f"Unsupported NYUv2 backbone '{model_name}'. Expected 'resnet18' or 'resnet50'.")
        if model_name == 'resnet50':
            weight_enum = ResNet50_Weights.DEFAULT if pretrained else None
            builder = resnet50
            output_dim = 2048
        else:
            weight_enum = ResNet18_Weights.DEFAULT if pretrained else None
            builder = resnet18
            output_dim = 512
        try:
            resnet = builder(weights=weight_enum)
        except Exception as exc:
            if pretrained:
                if not allow_random_init:
                    raise RuntimeError(f'Failed to load pretrained weights for {model_name}: {exc}') from exc
                resnet = builder(weights=None)
            else:
                raise
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def build_backbone(dataset_name: str, **kwargs) -> nn.Module:
    if dataset_name in ('mimic3', 'eicu'):
        lstm_kwargs = {'input_dim': kwargs.get('input_dim', 76), 'hidden_dim': kwargs.get('hidden_dim', 256), 'num_layers': kwargs.get('num_layers', 2), 'dropout': kwargs.get('dropout', 0.3), 'bidirectional': kwargs.get('bidirectional', False)}
        return LSTMBackbone(**lstm_kwargs)
    if dataset_name == 'nyuv2':
        resnet_kwargs = {'model_name': kwargs.get('model_name', 'resnet50'), 'pretrained': kwargs.get('pretrained', True)}
        return ResNetBackbone(**resnet_kwargs)
    raise ValueError(f'Unknown dataset: {dataset_name}')
