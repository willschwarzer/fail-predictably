# Training Agents to Fail Predictably

This repository contains the code for our NeurIPS 2026 paper, [*Training Agents to Fail Predictably: Fine-Tuning for Forecastable Tail Risk*](https://arxiv.org/abs/2605.15134), by Will Schwarzer and Scott Niekum.

An AI system may be evaluated on thousands of inputs before deployment and encounter billions after it, so its worst failures may not appear during evaluation. [Jones et al. (2025)](https://arxiv.org/abs/2502.16797) proposed forecasting how risky the riskiest deployment inputs will be by scoring how close the system comes to failure on each evaluation input and extrapolating from the highest scores. In the paper, we fine-tune the model itself so that this forecast is accurate, using the forecast error as the training loss.

This repository contains the training and evaluation code for the paper's two main settings, a password game for language models and a multi-task gridworld. It also contains a launcher for each experiment in the paper, and the scripts that produce the paper's figures and tables.

## Contents

- `src/forecast_experiment/` contains the training and evaluation code.
  The language-model trainer is `train_lquant.py` (its command-line entry point is `run_train_lquant.py`), the SFT baseline is `train_lquant_sft.py`, and the gridworld trainer is `train_rlquant.py`.
  The same package builds the task banks: password-game prompt banks with GCG suffixes, gridworld layout banks, and WildChat prompt banks.
- `src/importance_sampling/policy_lm.py` is the LoRA wrapper that the language-model trainers use.
- `scripts/` contains the Slurm launchers, with the hyperparameters of our runs, and the readouts and figure scripts that turn run outputs into the paper's tables and figures.
- `tests/` contains CPU unit tests of the forecast, the score caches and the gridworld estimators.
- `data/` contains the word lists and password lists that the task banks are built from.
- `REPRODUCING.md` lists the scripts behind each result in the paper, in the order to run them.

We do not release trained LoRA adapters or gridworld policies, GCG suffix banks, prompt or layout banks, or run outputs. The launchers rebuild all of them with the seeds we used, though not bit for bit (see "How closely a rerun matches ours" below).

## Setup

We ran everything on Linux with Python 3.12, PyTorch 2.10 (CUDA 12.8 build) and NVIDIA GPUs.

```bash
git clone https://github.com/willschwarzer/fail-predictably.git
cd fail-predictably
pip install -r requirements.txt   # the versions we used
pip install -e .                  # optional: the launchers put src/ on PYTHONPATH themselves
```

To check the installation, run the CPU tests from the repository root, either one file at a time (`python -m tests.test_ols_bootstrap`) or all at once with `pytest` if you have it.

The language models download from the Hugging Face Hub on first use. The Qwen3 models are public. The study of pretrained models across scale also uses Gemma 3, which requires accepting its license on the Hub and logging in with `hf auth login`. If your compute nodes have no internet access, download the models beforehand (for example `hf download Qwen/Qwen3-0.6B`); the minimal-change launchers run with `HF_HUB_OFFLINE=1` for this reason.

### WildChat

Many of the language-model experiments use WildChat prompts ([Zhao et al., 2024](https://arxiv.org/abs/2405.01470)).
The password game measures KL divergence from the pretrained model on them, and the natural-prompt, pretrained-scale and decomposition experiments, as well as the WildChat arms of the minimal-change experiments, draw their tasks from them.
We used a July 2024 copy of `allenai/WildChat-1M` (dataset revision `8016827cee8c947143c498df467af52753e5181b`) with 1,039,785 conversations, which included the toxic subset. The default WildChat-1M release has since dropped that subset, which is now in the gated `allenai/WildChat-1M-Full`, so reproducing our experiments requires the Full variant.

The code reads the dataset's Arrow files from a local directory, which it searches recursively for `*.arrow` files. Download the dataset with `datasets.load_dataset`, then point `WILDCHAT_DATASET` at the directory that contains its Arrow files (by default the code looks in `data/wildchat/`). Point it at that dataset's own directory, not at a whole Hugging Face cache, since the search would also find other datasets' files. The KL prompts are sampled from the first Arrow file in sorted order, and the bank builders read the files in sorted order, so a copy with a different file layout will give different prompts from ours.

## Running the launchers

The launchers are Slurm batch scripts, or shell scripts that submit Slurm jobs. Submit them from the repository root, so that they find the code and write their logs to `logs/`:

```bash
export SBATCH_ACCOUNT=<your account> SBATCH_PARTITION=<a GPU partition>
SEED=11 sbatch scripts/password_game/run_lquant_modbulkgcg_iog_full_bootstrap_cached_multiseed.sbatch
```

We removed our cluster's account, partition and node constraints from the `#SBATCH` lines. sbatch reads `SBATCH_ACCOUNT`, `SBATCH_PARTITION` and `SBATCH_QOS` from the environment, or you can pass `-A` and `-p` on the command line. Where one of our runs needed particular hardware, such as an 80 GB GPU or several GPUs for Qwen3-8B, a comment under the `#SBATCH` block says so. Most language-model launchers choose the micro-batch size from the GPU's memory. The micro-batch size sets only how many prompts go through the model at once, so it changes the scores and the update only through rounding in bfloat16.

The scripts read these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `REPO_ROOT` | the submission directory | repository root |
| `PYTHON` | `python` | interpreter |
| `RESULTS_ROOT` | `artifacts/results` | training outputs, which the readouts and figure scripts read |
| `TASK_BANK_ROOT` | `artifacts/task_banks` | prompt and layout banks |
| `FIGURES_ROOT` | `artifacts/figures` | figures |
| `WILDCHAT_DATASET` | `data/wildchat` | directory of WildChat Arrow files |
| `LOGIT_CACHE_ROOT` | under `RESULTS_ROOT` | full-pool WildChat logits for the pretrained-scale study (about 355 GB) |
| `SEED` | none | training seed, for the launchers that run one seed |
| `TAG_DATE` | today | date at the end of run and bank names (see below) |

Bank and run names end in the date on which they were made, and the scripts refer to each other's outputs by the names of our runs. For example, the password-game trainers read the bank `lquant_prompt_bank_40slots_modbulk_gcg6_20260502`, and the password-game figure reads runs named `..._20260505`. To rebuild a bank or rerun an experiment so that the downstream scripts find it, set `TAG_DATE` to the date in the name that they expect, which `REPRODUCING.md` gives for each step, or edit the name in the downstream script.

## Reproducing the paper

`REPRODUCING.md` gives the steps for each result: the bank, the training runs and their seeds, the baselines, the readouts and the figure. In brief:

| Result | Scripts |
|---|---|
| Figure 3 and the finite-k appendix | `scripts/theory/` |
| Figure 4, the password game | `scripts/banks/`, `scripts/password_game/`, `scripts/figures/plot_lquant_three_metric_bars.py` |
| Figure 5, the gridworld | `scripts/banks/`, `scripts/gridworld/`, `scripts/figures/plot_rlquant_three_metric_bars.py` |
| Section 6.3, minimal-change training, and Figure 2 | `scripts/minimal_change/`, `scripts/figures/make_tail_shape.py` |
| Empirical decomposition on WildChat | `scripts/wildchat/decomposition/` |
| Natural user prompts | `scripts/wildchat/natural_prompts/submit_natural_prompt_runs.sh` |
| Pretrained models across scale | `scripts/wildchat/pretrained_scale/` |
| Password-game scaling, fit/deployment shift | `scripts/banks/`, `scripts/password_game/`, `scripts/figures/prepare_revised_figures.py` |
| Sampled returns in the gridworld | `scripts/gridworld/` |
| Checks beyond the training forecast and score | `scripts/audits/` |
| KL divergence with thinking off, and generations with thinking on | `scripts/audits/kl_thinking_off*.sbatch`, `scripts/audits/submit_think_then_answer.sh` |
| The other appendix figures | `scripts/figures/make_appendix_figures.py` |

## How closely a rerun matches ours

The code in `src/` is the last version we ran, from November 2026. The experiments of Sections 6.1 and 6.2 ran on an earlier version from May 2026, and the later experiments on versions in between. We checked that each launcher, run with this version, gives the training settings that our runs saved, apart from the micro-batch size.

The gridworld experiments run on CPU, and there a rerun can match ours exactly. When we reran one seed of Figure 5's forecastability training and one seed of its RL baseline with this version, every value that the runs saved (their training histories, results and forecast traces) was identical to ours. The RL baseline's rerun ran on the type of CPU that our run used (Intel Xeon Gold 5220). On an AMD CPU, it differed from ours slightly, from the first update on.

A rerun of a language-model experiment will not match ours bit for bit, even with our seeds. For the recipe of Section 6.1, the first gradient that this version computes differs from the May version's by 0.5% of its norm, about a tenth of the change that a different micro-batch size makes, so we read the difference as floating-point rounding. Such differences grow during training, and the risk scores, which are computed in bfloat16, also differ between GPU types. We therefore expect a rerun's per-seed numbers to differ from ours somewhat. The post-hoc calibrations can change a lot. When we rescored the training passwords of two of Figure 4's SFT runs on an A100 at micro-batch 96, rather than on the A16 at micro-batch 16 that our calibrations had used, their calibrated held-out errors were 0.15 and 0.42 instead of 1.03 and 0.38.

## Names in the paper and in the code

The code predates some of the paper's terminology. In particular, it calls forecastability training "meta-training", since the model is trained across many task sets, and its file names use `lquant` for the language-model experiments and `rlquant` for the gridworld.

| Paper | Code |
|---|---|
| task set (password game) | one password's prompts, called a "target" or "slot"; each seed trains on 33 passwords and holds out 7 |
| task set (gridworld) | a (fit, deploy) pair or "manifest" of 2,016 layouts; each seed trains on 20 and holds out 5 |
| fit set of M tasks, deployment set of N | the fit and deploy sides of a task set: M = 44 and N = 891 in the password game (`--fit-fraction 0.0476`), M = 96 and N = 1,920 in the gridworld |
| top k = 10 fit scores, Weibull plotting position | `--ols-top-k 10 --plotting-position weibull` |
| forecastability loss over the extrapolated ranks, with log-uniform weights | `--loss-mode ols_primal_extrapolated --extrapolated-weighting deploy_log_uniform` |
| backpropagating through the fit and deployment scores | `--grad-mode both` |
| a new random fit/deployment split at every step | `--bootstrap-permutations 1` (0 keeps each task set's split fixed) |
| cache of the C highest-scoring tasks | `--union-cache-top-c 296 --union-cache-refresh-interval 5` |
| improving-only mask (our primary method) | `--improving-only-grad --iog-scope both`; "IOG" in run names |
| KL penalty | `--kl-coef`, with `--kl-train-datasets` and `--kl-eval-datasets` |
| minimal-change training | the launchers in `scripts/minimal_change/` |
| anchor constraint | `--anchor-gate-margin` and `--anchor-penalty-coef`, together with the mask flags |
| spread constraint | `--sigma-floor-lambda`, `--sigma-floor-margin`, `--sigma-floor-symmetric` and `--sigma-ref-m` |
| post-hoc calibration (Cal.) | `scripts/password_game/posthoc_calibration_baseline.py` and its gridworld counterpart |
| direct risk minimization (SFT, RL) | `train_lquant_sft.py` for SFT; the `baseline_sft_union` gridworld launchers for RL |
| hardened system prompt | `--password-policy non_disclosure_strict_v2` |
| the forecasting method of Jones et al. | "FRLMB" in comments and in the WildChat scripts' names |

## Citation

```bibtex
@inproceedings{schwarzer2026fail,
  title     = {Training Agents to Fail Predictably: Fine-Tuning for Forecastable Tail Risk},
  author    = {Schwarzer, Will and Niekum, Scott},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

## License

The code is released under the MIT license (see `LICENSE`). `data/wordlists/ldnoobw_en.txt` is the English list from the [List of Dirty, Naughty, Obscene, and Otherwise Bad Words](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words), licensed CC-BY-4.0.
