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
| 3 | [unit3_ppo_lunarlander.ipynb](notebooks/unit3_ppo_lunarlander.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliBuildsAI/rl-for-robotics-llms/blob/main/notebooks/unit3_ppo_lunarlander.ipynb) | **PPO-Clip** | Importance sampling, surrogate loss, trust region, clipped objective, multi-epoch minibatch updates | LunarLander-v3 | ✅ PPO solves it, A2C doesn't |
| 4 | [unit4_sft.ipynb](notebooks/unit4_sft.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliBuildsAI/rl-for-robotics-llms/blob/main/notebooks/unit4_sft.ipynb) | **SFT** | Instruction tuning, chat templates, completion-only loss masking, before/after generation | Qwen2.5-0.5B + Capybara | 🆕 New |
| 5 | [unit5_reward_model.ipynb](notebooks/unit5_reward_model.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliBuildsAI/rl-for-robotics-llms/blob/main/notebooks/unit5_reward_model.ipynb) | **Reward Models** | LLM-as-MDP, Bradley-Terry model, preference data, scalar reward head | Qwen2.5-0.5B + UltraFeedback | 🆕 New |
| 6 | — | **RLHF (PPO on LMs)** | 4-model setup, KL-anchored objective, per-token reward, reward hacking | Qwen2.5-0.5B + TRL | 🔜 Coming soon |
| 7 | — | **DPO** | Closed-form optimal policy, direct preference loss, no reward model, no RL loop | Qwen2.5-0.5B + UltraFeedback | 🔜 Coming soon |
| 8 | — | **RLVR + GRPO** | Verifiable rewards, group-relative baseline, no value network, DeepSeek-R1 recipe | Qwen2.5-1.5B + GSM8K | 🔜 Coming soon |

The series tells **one connected story**: each unit fixes a specific failure
mode of the previous algorithm.

| Unit | Problem fixed | How |
|------|--------------|-----|
| 1 → 2 | MC return is high-variance | GAE interpolates between 1-step TD and MC |
| 2 → 3 | One gradient step per rollout, no step-size limit | PPO reuses each rollout for K epochs with clipped probability ratio |
| 3 → 4 | A pretrained LM doesn't follow instructions | SFT on (instruction, response) demos |
| 4 → 5 | SFT only imitates; no notion of better vs worse | Learn a reward model from human preferences |
| 5 → 6 | A reward model alone doesn't change the policy | PPO-fine-tune the LM against the reward model (RLHF) |
| 6 → 7 | RLHF needs a reward model + a 4-model RL loop | DPO optimises preferences directly — no RM, no RL |
| 7 → 8 | Preferences still need human labels at scale | RLVR uses verifiable rewards; GRPO drops the value network |

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

### What's in Unit 4 — SFT (Qwen2.5-0.5B + Capybara)

Unit 4 is the first hands-on LLM stage: turn a *base* model that only predicts
internet text into one that follows instructions. The model trained here is the
starting point for Units 5–8.

**Theory**
- Where SFT sits: pretraining → **SFT** → RLHF, and why imitation has a ceiling
- SFT = the *same* next-token loss as pretraining, on curated (instruction,
  response) demos with the chat template
- **Completion-only loss masking**: set prompt-token labels to `-100` so loss is
  computed on the response only — the one SFT-specific idea (and a common
  interview question)

**Code**
- Loads the **base** Qwen2.5-0.5B with `AutoModelForCausalLM`
- Builds the masked label tensor **by hand** so you see the `-100`s, then trains
  with TRL's `SFTTrainer` (`completion_only_loss=True`)
- **Before/after generation**: the base model rambles; the SFT model answers and
  stops cleanly
- Saves the checkpoint that **Unit 5** consumes

---

### What's in Unit 5 — Reward Models (Qwen2.5-0.5B + UltraFeedback)

There is no environment reward function for language — so we learn one from human
preference comparisons.

**Theory**
- The LLM as a (peculiar) MDP: state = prompt + tokens so far, action = next
  token, transitions are deterministic, reward is terminal and learned
- Five structural differences from robotics RL (action space, transitions,
  sparse terminal reward, non-random init + KL anchor, learned reward)
- The **Bradley-Terry model**: `P(y_w ≻ y_l) = σ(r(x,y_w) − r(x,y_l))`, and why
  only the *difference* of rewards is identifiable
- Reward-model architecture: pretrained LM backbone + a scalar value head

**Code**
- Computes the Bradley-Terry loss **by hand** on one pair first, then trains
  with TRL's `RewardTrainer`
- **Preference accuracy** on a held-out split (random = 50%), broken down by
  preference clarity (reproduces Llama 2's Table 8 on our own model)
- A probe: hand-written good/bad answers, printing the two scalar rewards so you
  *see* the model prefer the better one
- The reward model trained here is the artifact **Unit 6** consumes

---

### What's in Unit 3 — PPO-Clip (LunarLander-v3)

Unit 3 applies the same GAE + rollout infrastructure from Unit 2, changing only the loss function:

**Part A — A2C baseline on LunarLander**
- Shows A2C plateaus at ~−147 (not solved) — same fundamental step-size problem as Unit 1

**Part B — PPO-Clip**
- Why naive multi-step reuse fails (distribution shift)
- Importance sampling identity with proof (dynamics cancel from trajectory ratio)
- Surrogate loss and why a trust region is needed
- PPO-Clip formula: all 4 cases of `min(unclipped, clipped)` analysed with sign of advantage
- Implementation: `n_steps=1024`, `batch_size=64`, `n_epochs=4` → 512 gradient steps per rollout (vs A2C's 1)
- **Result: A2C −147 (not solved) vs PPO 265 (✓ solved, threshold=200)** — PPO solves at step ~606k

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

The goal is to understand
**REINFORCE → VPG → A2C+GAE → PPO → RLHF → DPO → GRPO**
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
- Rafailov et al. (2023) — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
- Shao et al. (2024) — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning* (GRPO)
- DeepSeek-AI (2025) — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL*
- Lambert — *RLHF Book* — <https://rlhfbook.com>
- HuggingFace Deep RL Course — <https://huggingface.co/learn/deep-rl-course>
- HuggingFace TRL — <https://github.com/huggingface/trl>
- RL Baselines3 Zoo — <https://github.com/DLR-RM/rl-baselines3-zoo>
