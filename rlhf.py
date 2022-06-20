from typing import List, Tuple

import gym
import torch
import torch.nn as nn
from torchtyping import TensorType


def preference_prob(
    returns: TensorType["batch", 2],
):
    mixture_vector = (
        torch.distributions.OneHotCategorical(probs=torch.tensor([0.9, 0.1]))
        .sample()
        .view(2, 1, 1)
    )
    bradly_terry_prob = nn.functional.softmax(returns, dim=1)
    unif_prob = torch.distributions.Uniform(0, 1).sample(
        torch.Size((returns.shape[0], 1))
    )
    dual_unif_prob = torch.cat([unif_prob, 1 - unif_prob], dim=1)
    probs = torch.stack([bradly_terry_prob, dual_unif_prob])
    return torch.einsum("ijk,imn->mn", mixture_vector, probs)


class Agent(nn.Module):
    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        super().__init__()

    def act(
        self, state: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "action_dim"]:
        return torch.zeros(state.shape[0], self.action_dim)


class RewardModel(nn.Module):
    def __init__(self, layer_size: int, state_dim: int) -> None:
        super().__init__()
        self.first = nn.Linear(state_dim, layer_size)
        self.second = nn.Linear(layer_size, layer_size)
        self.last = nn.Linear(layer_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.first(inputs)
        inputs = nn.functional.relu(inputs)
        inputs = self.second(inputs)
        inputs = nn.functional.relu(inputs)
        return self.last(inputs)


def build_preference(
    trajectories: List[Tuple[TensorType["rewards"], TensorType["rewards"]]],
) -> TensorType["preferences"]:
    out = []
    for left_rewards, right_rewards in trajectories:
        left_return, right_return = torch.sum(left_rewards), torch.sum(right_rewards)
        out.append(left_return > right_return)
    return torch.tensor(out).to(dtype=torch.long)


def rollout(
    agent: Agent, env: gym.Env, n_trajs: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    states = []
    rewards = []
    for _ in range(n_trajs):
        obs = env.reset()
        done = False
        traj_rewards = []
        traj_states = []
        while not done:
            action = agent.act(torch.from_numpy(obs).view(1, -1))
            obs, reward, done, _ = env.step(int(action.item()))
            traj_rewards.append(reward)
            traj_states.append(obs)
        rewards.append(torch.tensor(traj_rewards))
        states.append(torch.tensor(traj_states))
    return states, rewards


def main():
    n_trajs = 10
    n_epochs = 1

    env = gym.make("CartPole-v1")
    # You want to choose an agent that provides interesting comparisons to learn from. On-policy is
    # usually what people use.
    agent = Agent(action_dim=1)

    states, rewards = rollout(agent, env, n_trajs)

    reward_pairs = list(zip(rewards[::2], rewards[1::2]))
    state_pairs = list(zip(states[::2], states[1::2]))

    # This should actually be real human preferences, but here we use the "ground truth" preferences.
    preferences = build_preference(reward_pairs)

    reward_model = RewardModel(256, env.observation_space.sample().shape[0])
    optim = torch.optim.Adam(reward_model.parameters())

    for i in range(n_epochs):
        print(f"Epoch {i}")
        pred_rewards = (
            (reward_model(left), reward_model(right)) for left, right in state_pairs
        )
        pred_returns = torch.stack(
            [
                torch.stack((torch.sum(left), torch.sum(right)))
                for left, right in pred_rewards
            ],
        )
        pred_prefs = preference_prob(pred_returns)

        loss = nn.functional.cross_entropy(pred_prefs, preferences)
        print(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()

    # Use your favorite RL algorithm to train an agent using the learned reward model.


main()
