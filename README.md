# RL From Scratch

A series of clean, self-contained notebooks implementing reinforcement learning algorithms from scratch in PyTorch, built on top of [Gymnasium](https://gymnasium.farama.org/).

Each notebook is self-contained, runs end-to-end on Google Colab (CPU or GPU), and includes a training curve and a GIF of the trained agent.

## Notebooks

| # | Algorithm | Environment | Notebook |
|---|-----------|-------------|----------|
| 1 | REINFORCE | CartPole-v1 | [unit1_reinforce_cartpole.ipynb](notebooks/unit1_reinforce_cartpole.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliBuildsAI/rl-from-scratch/blob/main/notebooks/unit1_reinforce_cartpole.ipynb) |

## Quick start

```bash
# Clone and run locally
git clone https://github.com/AliBuildsAI/rl-from-scratch.git
cd rl-from-scratch
pip install -r requirements.txt
jupyter notebook notebooks/unit1_reinforce_cartpole.ipynb
```

Or click the **Open in Colab** badge next to any notebook above.

## Series roadmap

- [x] Unit 1 — REINFORCE (Monte Carlo policy gradient)
- [ ] Unit 2 — Actor-Critic (A2C)
- [ ] Unit 3 — Proximal Policy Optimization (PPO)
- [ ] Unit 4 — Deep Q-Network (DQN)
- [ ] Unit 5 — Soft Actor-Critic (SAC, continuous actions)

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- HuggingFace Deep RL Course — <https://huggingface.co/learn/deep-rl-course>
