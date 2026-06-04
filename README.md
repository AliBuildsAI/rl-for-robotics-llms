# RL for Robotics & LLMs

A series of clean, annotated notebooks implementing reinforcement learning
algorithms from scratch in PyTorch — built in the spirit of Karpathy's
"zero to hero" lectures.

Each notebook:
- Derives the algorithm from first principles with inline math
- Explains every tensor shape and code decision
- Runs end-to-end on Google Colab (CPU or GPU)
- Includes training curves, side-by-side comparisons, and a GIF of the trained agent

---

## Series

| # | Notebook | Algorithms covered | Key concepts | Environment | Status |
|---|----------|--------------------|--------------|-------------|--------|
| 1 | [unit1_reinforce_cartpole.ipynb](notebooks/unit1_reinforce_cartpole.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliBuildsAI/rl-for-robotics-llms/blob/main/notebooks/unit1_reinforce_cartpole.ipynb) | **REINFORCE** → **VPG** | Policy gradient theorem, reward-to-go, value-function baseline, advantage estimation | CartPole-v1 | ✅ Done |
| 2 | [unit2_a2c_gae_lunarlander.ipynb](notebooks/unit2_a2c_gae_lunarlander.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliBuildsAI/rl-for-robotics-llms/blob/main/notebooks/unit2_a2c_gae_lunarlander.ipynb) | **A2C + GAE** | Generalised Advantage Estimation, bias-variance trade-off, N-step rollouts, parallel envs | LunarLander-v3 | ✅ Done |
| 3 | — | **PPO** | Clipped surrogate objective, multiple epochs per rollout, why VPG collapses | BipedalWalker-v3 | 🔜 Coming soon |
| 4 | — | **RLHF** | Reward model from human preferences, PPO fine-tuning of a language model | TinyLlama | 🔜 Coming soon |
| 5 | — | **GRPO** | Group relative policy optimisation, chain-of-thought reasoning rewards | GSM8K (math) | 🔜 Coming soon |

### What's in Unit 1

Unit 1 builds up from the simplest possible policy gradient to Algorithm 1 (Vanilla Policy Gradient) in two parts:

**Part A — REINFORCE** (`b = 0`)
- The policy gradient theorem and the log-derivative trick
- Reward-to-go `R_t` as an O(n) backward pass (and why `insert(0,x)` in a loop is O(n²))
- Why raw returns are high-variance and what that looks like in a training curve

**Part B — VPG with value baseline**
- A separate `ValueNetwork` that learns `V(s)` by minimising MSE against `R_t`
- Advantage `Â_t = R_t − V(s_t)` as a state-specific variance reduction
- Training curves with qualitative interpretation (exact numbers vary by seed/device — see notebook)
- Why both algorithms share the same root problem and how this motivates PPO's clipped objective

### What's in Unit 2

Unit 2 moves to a harder environment (LunarLander-v3) and adds two key ideas:

**Part A — VPG baseline**
- Re-runs Algorithm 1 from Unit 1 on LunarLander to show where it breaks

**Part B — A2C + GAE**
- GAE in its own standalone cell: the backward recurrence, λ interpolation, episode-boundary masking
- A2C in a separate cell: N-step rollouts with 4 parallel environments, orthogonal weight init, logits-based Categorical
- Hyperparameters from [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) tuned config (γ=0.995)
- Honest framing: A2C+GAE clearly improves over VPG but doesn't fully solve LunarLander — consistent with published SB3 benchmarks — which motivates PPO in Unit 3

---

## Why this series?

Modern AI is converging on RL as the key post-training ingredient:
- **Robotics** — PPO and SAC train locomotion and manipulation policies
- **LLM alignment** — RLHF (GPT-4, Claude) and GRPO (DeepSeek-R1) use policy
  gradients to steer language models with reward signals

The goal is to understand REINFORCE → VPG → PPO → RLHF → GRPO as **one connected story**,
where each step fixes a specific failure mode of the previous algorithm.

---

## Quick start

```bash
git clone https://github.com/AliBuildsAI/rl-for-robotics-llms.git
cd rl-for-robotics-llms
pip install -r requirements.txt
jupyter notebook notebooks/unit1_reinforce_cartpole.ipynb
```

Or click an **Open in Colab** badge above.

---

## References

- Williams (1992) — *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*
- Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed.)
- Schulman et al. (2015) — *High-Dimensional Continuous Control Using Generalized Advantage Estimation*
- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
- Ziegler et al. (2019) — *Fine-Tuning Language Models from Human Preferences*
- DeepSeek-AI (2025) — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*
- HuggingFace Deep RL Course — <https://huggingface.co/learn/deep-rl-course>
