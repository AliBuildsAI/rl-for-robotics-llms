# RL for Robotics & LLMs

A series of clean, annotated notebooks implementing reinforcement learning
algorithms from scratch in PyTorch — built in the spirit of Karpathy's
"zero to hero" lectures.

Each notebook:
- Derives the algorithm from first principles
- Explains the math inline with code and tensor shapes
- Runs end-to-end on Google Colab (CPU or GPU)
- Includes a training curve and a GIF of the trained agent

---

## Series

| # | Algorithm | Key Concept | Environment | Status |
|---|-----------|-------------|-------------|--------|
| 1 | **REINFORCE** | Monte Carlo policy gradient, log-derivative trick | CartPole-v1 | ✅ [notebook](notebooks/unit1_reinforce_cartpole.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliBuildsAI/rl-for-robotics-llms/blob/main/notebooks/unit1_reinforce_cartpole.ipynb) |
| 2 | **A3C + GAE** | Value baseline, Generalised Advantage Estimation, variance reduction | LunarLander-v2 | 🔜 Coming soon |
| 3 | **PPO** | Clipped surrogate objective, multiple epochs per rollout | BipedalWalker-v3 | 🔜 Coming soon |
| 4 | **RLHF** | Reward model from human preferences, PPO fine-tuning | TinyLlama fine-tune | 🔜 Coming soon |
| 5 | **GRPO** | Group relative policy optimisation, chain-of-thought reasoning | GSM8K (math reasoning) | 🔜 Coming soon |

---

## Why this series?

Modern AI is converging on RL as the key post-training ingredient:
- **Robotics** — PPO and SAC train locomotion and manipulation policies
- **LLM alignment** — RLHF (GPT-4, Claude) and GRPO (DeepSeek-R1) use policy
  gradients to steer language models with reward signals

Understanding REINFORCE → PPO → RLHF → GRPO as one connected story makes all
of this legible. That's the goal of this series.

---

## Quick start

```bash
git clone https://github.com/AliBuildsAI/rl-for-robotics-llms.git
cd rl-for-robotics-llms
pip install -r requirements.txt
jupyter notebook notebooks/unit1_reinforce_cartpole.ipynb
```

Or click the **Open in Colab** badge next to any notebook above.

---

## References

- Williams (1992) — *Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning* (the original REINFORCE paper)
- Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed.)
- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
- Ziegler et al. (2019) — *Fine-Tuning Language Models from Human Preferences*
- DeepSeek-AI (2025) — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*
- HuggingFace Deep RL Course — <https://huggingface.co/learn/deep-rl-course>
