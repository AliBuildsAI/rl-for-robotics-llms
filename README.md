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

| # | Notebook | Algorithm | Key concepts | Environment | Result |
|---|----------|-----------|--------------|-------------|--------|
| 1 | [unit1_reinforce_cartpole.ipynb](notebooks/unit1_reinforce_cartpole.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliBuildsAI/rl-for-robotics-llms/blob/main/notebooks/unit1_reinforce_cartpole.ipynb) | **REINFORCE → VPG** | Policy gradient theorem, reward-to-go, value-function baseline, advantage estimation | CartPole-v1 | ✅ VPG solves it |
| 2 | [unit2_a2c_gae_acrobot.ipynb](notebooks/unit2_a2c_gae_acrobot.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliBuildsAI/rl-for-robotics-llms/blob/main/notebooks/unit2_a2c_gae_acrobot.ipynb) | **A2C + GAE** | Generalised Advantage Estimation, bias-variance trade-off, N-step rollouts, parallel envs, episode-boundary masking | Acrobot-v1 | ✅ A2C solves it, VPG doesn't |
| 3 | — | **PPO** | Clipped surrogate objective, multiple epochs per rollout, importance sampling | LunarLander-v3 | 🔜 Coming soon |
| 4 | — | **RLHF** | Reward model from human preferences, PPO fine-tuning of a language model | TinyLlama | 🔜 Coming soon |
| 5 | — | **GRPO** | Group relative policy optimisation, chain-of-thought reasoning rewards | GSM8K (math) | 🔜 Coming soon |

The series tells **one connected story**: each unit fixes a specific failure
mode of the previous algorithm.

| Unit | Problem fixed | How |
|------|--------------|-----|
| 1 → 2 | MC return is high-variance | GAE interpolates between 1-step TD and MC |
| 2 → 3 | No limit on gradient step size → collapse | PPO clips the probability ratio |
| 3 → 4 | Reward must be hand-engineered | Learn reward from human preferences |
| 4 → 5 | RLHF needs human labels at scale | GRPO uses verifiable reasoning rewards |

---

### What's in Unit 1 — REINFORCE → VPG (CartPole-v1)

Unit 1 builds up from the simplest possible policy gradient to full VPG in two parts:

**Part A — REINFORCE** (`b = 0`)
- The policy gradient theorem and the log-derivative trick
- Reward-to-go `R_t` as an O(n) backward pass (and why `list.insert(0, x)` in a loop is O(n²))
- Why raw returns produce high-variance gradients

**Part B — VPG with learned value baseline**
- A separate `ValueNetwork` that learns `V(s)` by minimising MSE against `R_t`
- Advantage `Â_t = R_t − V(s_t)`: state-specific signal instead of episode-level signal
- Side-by-side training curves; the value-function bootstrap trap explained
- Why both algorithms share the same root problem (no step-size limit) → motivates PPO

> Results vary by hardware and library version — seeds guarantee reproducibility
> *within* the same Colab runtime, not across different machines.

---

### What's in Unit 2 — A2C + GAE (Acrobot-v1)

Acrobot is a double-pendulum swing-up task. VPG's Monte Carlo return is useless
here: every episode gives −1/step, so all `R_t ≈ −500` — no signal about which
actions built momentum. A2C+GAE fixes this with two changes.

**Part A — VPG baseline on Acrobot**
- Runs Algorithm 1 from Unit 1 unchanged; plateaus around −116 (not solved)

**Part B — A2C + GAE**
- **GAE** in its own cell: TD residuals `δ_t`, the backward recurrence `Â_t = δ_t + γλÂ_{t+1}`, episode-boundary masking `(1 − done_t)`, and the λ=0/1 special cases
- **A2C** in its own cell: N-step rollouts (N=5) from 8 parallel environments, orthogonal weight init, logits-based `Categorical` distribution, combined actor+critic update
- Hyperparameters from [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) tuned config
- **Result: VPG −116 (not solved) vs A2C+GAE −84 (✓ solved, threshold is −100)**

---

## Why this series?

Modern AI is converging on RL as the key post-training ingredient:
- **Robotics** — PPO and SAC train locomotion and manipulation policies
- **LLM alignment** — RLHF (GPT-4, Claude) and GRPO (DeepSeek-R1) use policy
  gradients to steer language models with human or automated reward signals

The goal is to understand **REINFORCE → VPG → A2C+GAE → PPO → RLHF → GRPO**
as one connected story, where each step fixes a specific failure mode of the
algorithm before it.

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
- Mnih et al. (2016) — *Asynchronous Methods for Deep Reinforcement Learning* (A3C)
- Schulman et al. (2016) — *High-Dimensional Continuous Control Using Generalized Advantage Estimation*
- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
- Ziegler et al. (2019) — *Fine-Tuning Language Models from Human Preferences*
- DeepSeek-AI (2025) — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*
- HuggingFace Deep RL Course — <https://huggingface.co/learn/deep-rl-course>
- RL Baselines3 Zoo — <https://github.com/DLR-RM/rl-baselines3-zoo>
