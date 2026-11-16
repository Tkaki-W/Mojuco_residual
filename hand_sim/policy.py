"""Asymmetric Actor-Critic：Actor は観測の前の N_ACTOR 個だけ、Critic は全部（本当の形なども）を見る。

stable-baselines3 の ActorCriticPolicy で、Actor 側の入力だけ後ろの「Critic だけの情報」を 0 にする。
"""
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

from .env import N_ACTOR


class ActorView(BaseFeaturesExtractor):
    """後ろの Critic だけの情報を 0 にする（次元は同じまま）"""

    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space.shape[0])
        mask = th.zeros(observation_space.shape[0])
        mask[:N_ACTOR] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, obs):
        return obs * self.mask


class AsymmetricPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["share_features_extractor"] = False
        kwargs["features_extractor_class"] = FlattenExtractor
        super().__init__(*args, **kwargs)
        self.pi_features_extractor = ActorView(self.observation_space)   # Actor：前の N_ACTOR 個だけ
