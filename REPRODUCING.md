# Reproducing the paper

This file lists the scripts behind each result in the paper and the order in which to run them.
The sections follow the paper's, and we name appendices by their titles.

A few conventions hold throughout:

- Run every command from the repository root, and set `SBATCH_ACCOUNT` and `SBATCH_PARTITION` first (see the README).
  Files ending in `.sbatch` are Slurm batch scripts that you submit with `sbatch`.
  Files ending in `.sh` submit their own Slurm jobs, so you run them with `bash`.
- The readouts and figure scripts need only a CPU. We ran each of them as a one-core Slurm job, for example
  `sbatch -c 1 --mem=8G -t 1:00:00 --wrap "python scripts/figures/make_appendix_figures.py"`.
  With the default roots they read from `artifacts/results/` and write figures to `artifacts/figures/`.
- The commands below refer to `$RESULTS_ROOT` and the other roots of the README.
  If you keep the defaults, export them first, for example `export RESULTS_ROOT=$PWD/artifacts/results`.
- Run and bank names end in a date, and the later steps look for the names of our runs.
  Each block below gives the `TAG_DATE` that makes a run's name match.
- Most training launchers run one seed, given as `SEED`.
  We list the seeds of each experiment, and loop over them like this:

  ```bash
  for s in 11 17 23 29 31 37 41 43 47 53; do
    SEED=$s TAG_DATE=20260505 sbatch scripts/password_game/run_lquant_modbulkgcg_iog_full_bootstrap_cached_multiseed.sbatch
  done
  ```

A rerun of a language-model experiment will not match our numbers exactly, even with our seeds.
The README's section "How closely a rerun matches ours" says why, and what we checked.

## Theory: decomposition of forecast error (Figure 3)

```bash
python scripts/theory/theory_decomposition_figure.py
python scripts/theory/theory_rank_bias_2d_smooth.py --n-trials 200000 --chunk 20000 --n-k 35 --n-r 35
python scripts/theory/theory_combined_figure.py
```

The first two scripts simulate the inverse-OLS forecast and write `theory_decomposition.json` and `theory_rank_bias_2d_smooth.json` to `$RESULTS_ROOT`.
The third draws both panels into `$FIGURES_ROOT/theory_combined.pdf`.

## Language model: the password game (Section 6.1, Figure 4)

1. Build the prompt bank.
   This runs GCG against Qwen3-0.6B, as one GPU job per password:

   ```bash
   TAG_DATE=20260502 bash scripts/banks/launch_gcg_bank_for_mutate_passwords.sh
   ```

   It writes `$TASK_BANK_ROOT/lquant_prompt_bank_40slots_modbulk_gcg6_20260502`, with the passwords in `data/passwords/`.
   Every password-game launcher reads bank seed 11 (`--prompt-bank-seed 11`), so the seeds of an experiment share one bank and differ in which passwords are held out, how each task set is split, and the other random draws of training.

2. Train with forecastability training (Ours) at ten seeds, 11 17 23 29 31 37 41 43 47 53, with `TAG_DATE=20260505`:
   `scripts/password_game/run_lquant_modbulkgcg_iog_full_bootstrap_cached_multiseed.sbatch`.

3. Train the SFT baseline at five seeds, 11 17 23 29 31, with `TAG_DATE=20260504`:
   `scripts/password_game/run_lquant_sft_modbulkgcg_union_multiseed.sbatch`.
   Our five runs evaluated their forecasts with Gringorten's plotting positions, the trainer's default, by mistake.
   The paper's figure uses the Weibull positions for them, as for every other method, and the launcher now passes `--plotting-position weibull`.

4. Train without the improving-only mask at five seeds, seed 11 with `TAG_DATE=20260502` and seeds 17 23 29 31 with `TAG_DATE=20260504`:
   `scripts/password_game/run_lquant_40slots_modbulkgcg_metatrain_multiseed.sbatch`.
   The figure takes its Cal. bars from these runs, and the table of the subsection "The improving-only mask and hidden rare modes" uses them as the runs without the mask.
   These runs follow the earlier recipe of that table, which kept each password's fit/deployment split fixed and trained for 50 steps rather than 200.
   It also cached the 200 highest-scoring deployment prompts of each password, refreshed every 10 steps, in place of the cache of the C highest-scoring tasks.
   So they are not the Section 6.1 recipe with the mask removed.

5. Fit the post-hoc calibrations.
   Cal. calibrates the pretrained model's forecast, and Ours + cal. and SFT + cal. calibrate the trained models' forecasts.
   Each call is a short GPU job on one run directory:

   ```bash
   # Cal.: in each run directory of step 4
   RUN_DIR=$RESULTS_ROOT/<run>/seed_<s> OUT_JSON=$RESULTS_ROOT/<run>/seed_<s>/posthoc_calibration_baseline.json \
     sbatch scripts/password_game/run_posthoc_calibration_baseline.sbatch
   # Ours + cal. and SFT + cal.: in each run directory of steps 2 and 3
   RUN_DIR=$RESULTS_ROOT/<run>/seed_<s> LORA_CHECKPOINT=$RESULTS_ROOT/<run>/seed_<s>/final \
     HELDOUT_RESULTS_JSON=test_results.json \
     OUT_JSON=$RESULTS_ROOT/<run>/seed_<s>/posthoc_calibration_metatrained.json \
     sbatch scripts/password_game/run_posthoc_calibration_baseline.sbatch
   ```

6. Draw the figure, and write the per-seed numbers that the appendix figures also read:

   ```bash
   python scripts/figures/plot_lquant_three_metric_bars.py --csv-output $FIGURES_ROOT/lm_three_metric_bars.csv
   ```

## Multi-task RL: gridworld navigation with traps (Section 6.2, Figure 5)

1. Build the layout bank for all 30 seeds:

   ```bash
   TAG_DATE=20260503 BANK_SEEDS="11 17 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 101 103 107 109 113 127 131 137 139 149 151" \
     bash scripts/banks/launch_rl_bipolar_bank.sh
   ```

   Each bank seed's layouts depend only on that seed, so one submission with all 30 gives the same banks as ours.

2. Train with forecastability training (Ours) at 30 seeds.
   Each launcher submits a Slurm array, one task per seed, that pretrains the policy and then trains it:

   ```bash
   TAG_DATE=20260505 bash scripts/gridworld/run_rlquant_bipolar_metatrain_iog_full_bootstrap_cached_10seeds.sh
   TAG_DATE=20260506 bash scripts/gridworld/run_rlquant_bipolar_metatrain_iog_full_bootstrap_cached_20moreseeds.sh
   ```

3. Train the RL baseline, which maximizes return directly, at ten seeds:
   `TAG_DATE=20260504 bash scripts/gridworld/run_rlquant_bipolar_baseline_sft_union.sh`.

4. Train the ten-seed fleet whose pretrained policies give the figure's Cal. bars, with an earlier recipe of forecastability training:
   `TAG_DATE=20260503 bash scripts/gridworld/run_rlquant_bipolar_metatrain_10seeds_canonical.sh`.
   Cal. calibrates the pretrained policies' forecasts, so only the pretraining part of these runs matters for the figure.

5. Fit the post-hoc calibrations, once the four fleets have finished.
   This is one CPU job that writes the calibration files into every seed directory:
   `sbatch scripts/gridworld/run_rlquant_posthoc_calibration_sweep.sbatch`.

6. Draw the figure: `python scripts/figures/plot_rlquant_three_metric_bars.py`.
   Add `--csv-output $FIGURES_ROOT/rl_three_metric_bars.csv` to also write the per-seed numbers.

## Changing the model as little as possible (Section 6.3)

This block also covers the appendix "Forecastability training with minimal change to the model" and the two figures that read its runs, Figure 2 and the figure of the appendix "Tail shape before and after training".
The runs go to `$RESULTS_ROOT/lquant_dynamics/runs/`, one directory per arm and seed.
The launchers run with `HF_HUB_OFFLINE=1`, so download Qwen3-0.6B first.

The four arms of the paper, and the run names that the readouts look for (each name ends in `_seed<s>`; the seeds are 11 17 23 29 43):

| Arm | WildChat, 100 and 200 steps | Password game, 100 and 200 steps |
|---|---|---|
| mask alone | `lqdyn_v2e2iogap50`, `lqdyn_v2e2iogap50x200` | `lqpw_pw2iog`, `lqpw_pw2iog200` |
| anchor constraint | `lqdyn_v2e2nbap50`, `lqdyn_v2e2nbap50x200` | `lqpw_pw2nbap`, `lqpw_pw2nbap200` |
| spread constraint | `lqdyn_v2e2iogsbap50sp`, `lqdyn_v2e2iogsbap50spx200` | `lqpw_pw2iogsb200` (200 steps only) |
| both (minimal-change training) | `lqdyn_v2e2ap50sp`, `lqdyn_v2e2ap50spx200` | `lqpw_pw2apsp` (seeds 11 17 23), `lqpw_pw2apsp200` |

1. Build the WildChat bank, whose task sets hold 2,300 prompts each.
   The two Python steps are short CPU jobs:

   ```bash
   TARGETS=16 PER_TARGET=4600 sbatch scripts/banks/run_build_wildchat_ldnoobw_bank.sbatch
   python scripts/banks/rechunk_token_union_bank.py \
     $TASK_BANK_ROOT/wildchat_ldnoobw_qwen_qwen3-0_6b_T16_p4600/bank_seed_11 \
     $TASK_BANK_ROOT/wildchat_ldnoobw_qwen_qwen3-0_6b_T32_p2300/bank_seed_11 --parts 2
   python scripts/banks/derive_wildchat_v2_bank.py \
     $TASK_BANK_ROOT/wildchat_ldnoobw_qwen_qwen3-0_6b_T32_p2300/bank_seed_11 \
     $TASK_BANK_ROOT/wildchat_ldnoobw_v2e2_qwen_qwen3-0_6b_T32_p2300/bank_seed_11 \
     --rule none --pool-size 2300 \
     --drop ASS,Anal,Ass,Cum,Dick,Escort,SEX,Sex,Tit,XX,XXX,anal,ass,butt,cock,coon,cum,escort,mong,poon,sex,sexo,sexual,tit,xx,xxx
   ```

   The readouts also need the base model's score for every prompt in the pool:
   `MODELS_QWEN=0.6B MODELS_GEMMA= bash scripts/wildchat/pretrained_scale/submit_wildchat_fullpool_logits.sh`.

2. Build the hardened password bank `lquant_prompt_bank_40slots_modbulk_gcg6warm_strictv2_qwen3_0_6b_20260724` (steps 1-3 of "Password-game scaling and the role of the evaluation bank", at 0.6B only).
   Keep the bank's default `DATE=20260724`, the date `run_lqpw_arm.sbatch` expects.

3. Train the 100-step runs:

   ```bash
   export BANK_V2=$TASK_BANK_ROOT/wildchat_ldnoobw_v2e2_qwen_qwen3-0_6b_T32_p2300 TRAIN_V2=24
   for s in 11 17 23 29 43; do
     V2TAG=v2e2iog   RUNGS=v2ap50   REC_OVERRIDE=BOOT=16 \
       bash scripts/minimal_change/submit_lqdyn_fleet.sh $s poolscale
     V2TAG=v2e2nb    RUNGS=v2ap50   REC_OVERRIDE=BOOT=16,ANCHOR_MARGIN=0.5,ANCHOR_PENALTY=1.0 \
       bash scripts/minimal_change/submit_lqdyn_fleet.sh $s poolscale
     V2TAG=v2e2iogsb RUNGS=v2ap50sp REC_OVERRIDE=BOOT=16,SIGMA_FLOOR_LAMBDA=50,SIGMA_FLOOR_MARGIN=0.13,SIGMA_FLOOR_SYMMETRIC=1 \
       bash scripts/minimal_change/submit_lqdyn_fleet.sh $s poolscale
     V2TAG=v2e2      RUNGS=v2ap50sp \
       bash scripts/minimal_change/submit_lqdyn_fleet.sh $s poolscale
   done
   for s in 11 17 23; do bash scripts/minimal_change/submit_lqpw_fleet.sh $s; done
   for s in 29 43; do bash scripts/minimal_change/submit_lqpw_fleet.sh $s pw2nbap pw2iog; done
   ```

   `submit_lqdyn_fleet.sh` prints its jobs without submitting them if you set `DRY=1`.
   The password submitter also queues each run's held-out reads; for the WildChat runs, submit them with `bash scripts/minimal_change/submit_wc_heldout_reads.sh` once the trainers have finished.

4. Train the 200-step runs and the runs at other margins.
   Most of the 200-step runs continue the 100-step runs:

   ```bash
   bash scripts/minimal_change/submit_pw_extra_runs.sh       # lqpw_pw2iogsb200, and lqpw_pw2apsp200
   bash scripts/minimal_change/submit_wc_ext200.sh           # the four WildChat arms at 200 steps
   bash scripts/minimal_change/submit_pw_ablation_ext200.sh  # lqpw_pw2iog200 and lqpw_pw2nbap200
   bash scripts/minimal_change/submit_margin_ablation.sh     # seed 43 at margins 0.06 and 0.25, both banks
   ```

   The first two can follow step 3 at once, since they wait for the runs they continue.
   Run `submit_pw_ablation_ext200.sh` only after the 100-step `lqpw_pw2iog` and `lqpw_pw2nbap` runs have finished.
   `submit_lqpw_fleet.sh`, `submit_pw_ablation_ext200.sh` and `submit_margin_ablation.sh` do not check for existing runs, so run each of them once.
   These submitters queue their own held-out reads.

5. Run the readouts once every read has finished.
   They run in `$RESULTS_ROOT/lquant_dynamics/readouts/` and write their tables there:

   ```bash
   R=scripts/minimal_change/readouts
   sbatch $R/strict_table.sbatch                                     # strict_table.csv (step 100)
   sbatch $R/strict_table.sbatch --step 200 --out strict_table_200.csv
   sbatch $R/py_small.sbatch kl_table.py                             # kl_table.csv
   sbatch $R/py_small.sbatch falls_table.py --steps 100 200          # falls_table.csv
   sbatch $R/py_small.sbatch margin_table.py                         # margin_table.csv
   sbatch $R/py_small.sbatch margin_calibration.py --part 1          # margin_calibration_part1.csv
   sbatch --time=01:00:00 --mem=16G $R/py_small.sbatch margin_calibration.py --part 2
   sbatch $R/cert_scale.sbatch cert_scale.csv                        # needs the full-pool scores
   ```

   Then the searches for the best tail of any shape, each in three jobs (prepare, array, aggregate), and the numbers of the paragraphs "Room for improvement" and "The error of a straight tail":

   ```bash
   sbatch $R/tail_opt.sbatch --fleet --bank --label fleet --prepare-only
   sbatch --array=0-124 $R/tail_opt.sbatch --manifest tail_opt_results/fleet.manifest.npz
   sbatch --time=00:30:00 $R/tail_opt.sbatch --manifest tail_opt_results/fleet.manifest.npz --aggregate-only
   sbatch $R/py_small.sbatch summarize_tail_opt.py tail_opt_fleet.csv
   sbatch --mem=32G $R/tail_opt.sbatch --grid --prepare-only --label grid
   sbatch --mem=32G --array=0-167 $R/tail_opt.sbatch --manifest tail_opt_results/grid.manifest.npz
   sbatch --mem=16G $R/tail_opt.sbatch --manifest tail_opt_results/grid.manifest.npz --aggregate-only
   sbatch $R/grid_theory.sbatch tail_opt_grid_cells.csv              # prints the table to its log
   sbatch $R/grid_theory_sweep.sbatch                                # grid_theory_sweep.csv; G at b = 1/6 is column G_lam6
   sbatch $R/py_small.sbatch variance_share.py                       # prints the shares of bias and variance
   ```

   The array sizes must match the number of pools in each manifest, which the prepare step prints; 125 is the count for the fleet search when every run is present.

6. Draw the figures:

   ```bash
   sbatch scripts/minimal_change/readouts/py_small.sbatch scripts/figures/make_minimal_change_figure.py
   sbatch scripts/minimal_change/readouts/py_small.sbatch scripts/figures/make_tail_shape.py
   sbatch scripts/minimal_change/readouts/py_small.sbatch scripts/figures/make_minimal_change_room_figure.py
   ```

   The first draws the appendix's figure of improvement factors against KL divergence, `minimal_change_fold_kl_rebuild.pdf`.
   The second draws Figure 2 (`tail_shape_body_wildchat.pdf`) and the tail-shape appendix figure (`tail_shape_password.pdf`).
   The third draws the figure of the paragraph "Room for improvement" (`minimal_change_room.pdf`) from the readouts of step 5.
   All three scripts check the numbers that the text quotes against the readouts' tables.

## A finite-k decomposition for the inverse-OLS Gumbel-tail extrapolator

The appendix's synthetic validation figure:

```bash
python scripts/theory/synthetic_inverse_ols_rank_bias.py --n-trials 1000000 --chunk 2000 \
  --out-json $RESULTS_ROOT/synthetic_inverse_ols_rank_bias_1M.json
python scripts/theory/synthetic_inverse_ols_rank_bias_plot.py
```

The exact rank-term values that the appendix quotes come from deterministic computations in `scripts/theory/rank_term/`, with no Monte Carlo.
`forward/exact_checks.py` and `verify_inverse/exact_checks.py` check the forward forecaster's results and, independently, the inverse forecaster's:

```bash
R=scripts/theory/rank_term
python $R/inverse/compute_exact.py        # writes $R/inverse/exact.json, a few minutes on one core
python $R/inverse/ms_exact_quad.py        # reads it
python $R/forward/exact_checks.py
python $R/verify_inverse/exact_checks.py  # checks k up to 1,000; we also ran it with 5000
```

These scripts write their JSON files next to themselves, not under `$RESULTS_ROOT`.

## Empirical exploration of the decomposition on WildChat

1. Score the prompts.
   Each scorer is a GPU job that writes one CSV of per-prompt scores under `$RESULTS_ROOT/wildchat_frlmb_tail_fit_quality_<TAG_DATE>/`.
   The figures' configurations read the Detoxify and flagged harmful-token scores from `TAG_DATE=20260503`, and the assistant-length and NLL scores from `TAG_DATE=20260504`:

   ```bash
   TAG_DATE=20260503 sbatch scripts/wildchat/decomposition/run_wildchat_frlmb_detoxify.sbatch
   TAG_DATE=20260503 sbatch scripts/wildchat/decomposition/run_wildchat_frlmb_harmful.sbatch
   TAG_DATE=20260504 sbatch scripts/wildchat/decomposition/run_wildchat_frlmb_assistant_length.sbatch
   TAG_DATE=20260504 sbatch scripts/wildchat/decomposition/run_wildchat_frlmb_assistant_nll.sbatch
   ```

2. Draw the figures:
   `run_wildchat_trifecta_appendix.sbatch` (hazard diagnostics), `run_wildchat_empirical_decomposition.sbatch` (the decomposition at the canonical cell) and `run_wildchat_k_sweep.sbatch` (the k sweep; the end of the file gives the figure call).

3. The numbers in the text at the canonical cell are in `$RESULTS_ROOT/wildchat_empirical_decomposition.json`, and the k-sweep values are in `$RESULTS_ROOT/wildchat_k_sweep/`.
   For the window sensitivity and the sweep over M at fixed R = 10, rerun `wildchat_empirical_decomposition.py` with its launcher's arguments, changing `--delta` or `--m` and `--N`, and with output names of their own so that the canonical outputs stay in place:

   ```bash
   D=scripts/wildchat/decomposition
   decomp() {  # decomp TAG ARGS...
     local tag=$1; shift
     python -u $D/wildchat_empirical_decomposition.py --config $D/wildchat_decomp_panels.json \
       --k 10 --n-trials 2000 --chunk 100 --seed 42 "$@" \
       --out-json $RESULTS_ROOT/wildchat_empirical_decomposition_$tag.json \
       --out-pdf $FIGURES_ROOT/wildchat_empirical_decomposition_$tag.pdf
   }
   decomp delta0.25 --m 5000 --N 50000 --delta 0.25
   decomp delta1.0  --m 5000 --N 50000 --delta 1.0
   decomp m1000     --m 1000 --N 10000 --delta 0.5
   decomp m25000    --m 25000 --N 250000 --delta 0.5
   ```

   The without-replacement totals in "Two further checks" come from the k-sweep cells at k = 10 (length and mean NLL) and from `TAG_DATE=20260503 sbatch scripts/wildchat/decomposition/run_wildchat_frlmb_analyze.sbatch` (Detoxify and harmful-token).

`wildchat_count_flagged_english.py` counts the English and flagged slices, and `wildchat_schema_verify.py` checks a shard's schema.
`python scripts/wildchat/decomposition/reference_decomposition.py` computes the numbers of "Comparison with a bounded reference distribution", the same decomposition on samples from Exp(1), Lognormal, Log-Weibull, Pareto and Uniform(0,1).
It reuses the code of Figure 3's script (`scripts/theory/theory_decomposition_figure.py`), and its rank term is a Monte Carlo estimate, 0.794, where the paper prints the exact value, 0.793.

## Forecastability on natural user prompts

All of the steps after the banks go through one script, `scripts/wildchat/natural_prompts/submit_natural_prompt_runs.sh`, whose header describes each stage.
It takes a stage and the groups of runs: `main` (the appendix's first table and its per-seed figure, the default), `kl` (the table of KL-penalty sensitivity at Qwen3-8B), `extra` (the two extra seeds at KL weight 2.0 and the blocklist-prompt control) or `all`.
Set `DRY=1` to print the jobs without submitting them.

1. Build one bank per model, then write the KL training split into each:

   ```bash
   for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B; do
     MODEL=$m sbatch scripts/banks/run_build_wildchat_ldnoobw_bank.sbatch
   done
   # once the four bank jobs have finished:
   for t in qwen_qwen3-0_6b qwen_qwen3-1_7b qwen_qwen3-4b qwen_qwen3-8b; do
     python scripts/wildchat/natural_prompts/make_kl_train_split.py \
       $TASK_BANK_ROOT/wildchat_ldnoobw_${t}_T40_p1870/bank_seed_11
   done
   ```

   Each bank holds 2,048 non-flagged English WildChat prompts for the KL terms, and the KL evaluation uses 256 of them.
   The split writes the other 1,792 to their own file, so the KL penalty and the KL evaluation never share a prompt.
2. Train: `bash scripts/wildchat/natural_prompts/submit_natural_prompt_runs.sh train all`.
   This submits the forecastability and SFT runs for Qwen3 0.6B, 1.7B, 4B and 8B with seeds 11, 17 and 23, and the runs of the other two groups.
3. While the runs train, copy their best checkpoints now and then: `bash scripts/wildchat/natural_prompts/submit_natural_prompt_runs.sh snapshot all`.
   The trainers keep only the latest best checkpoint, and the selection rule below can fall back only to a checkpoint of which a copy exists.
   We copied ours by hand at irregular times, so a rerun can select a different checkpoint wherever the latest one fails the KL bound.
4. Evaluate the selected checkpoints on the held-out tasks: `bash scripts/wildchat/natural_prompts/submit_natural_prompt_runs.sh eval all`.
   Each run's checkpoint is chosen by its training-side forecast error among the checkpoints whose single-token KL is at most 0.7, and the held-out errors play no part.
   When the newest checkpoint's KL is above 0.7, run the stage again once its evaluation has finished, and it evaluates the newest earlier copy.
   `python scripts/wildchat/natural_prompts/select_natural_checkpoint.py RUN_DIR` prints the choice for one run.
5. Fit the post-hoc calibration of each forecastability run of the main group: `bash scripts/wildchat/natural_prompts/submit_natural_prompt_runs.sh cal`.
6. Collect the results: `python scripts/wildchat/natural_prompts/collect_natural_results.py`.
   It writes `$RESULTS_ROOT/summaries/natural_results_verified.json`, the input of the figure, and prints the appendix's other numbers: the tables, the decades, the pooled improvement factors, the KL ranges, the two extra seeds, the blocklist-prompt control and the improvement factors from post-hoc calibration.

### Forecastability of pretrained models across scale

1. Score the whole pool under each of the eleven pretrained models, and save the logits:
   `bash scripts/wildchat/pretrained_scale/submit_wildchat_fullpool_logits.sh`.
   The dumps go to `$LOGIT_CACHE_ROOT` (by default under `$RESULTS_ROOT`) and take about 355 GB in all.
   The Gemma 3 models require accepting their license on the Hugging Face Hub.
2. Simulate the forecast error over fit and deployment sizes:
   `sbatch scripts/wildchat/pretrained_scale/run_fit_deploy_error_grid.sbatch`.
3. The figure is drawn by `make_appendix_figures.py` (below).

## Password-game scaling and the role of the evaluation bank

The banks for the hardened instruction take four steps per model size.
The sizes are `0_6b`, `1_7b`, `4b` and `8b`, with the models `Qwen/Qwen3-0.6B`, `-1.7B`, `-4B` and `-8B`; we write the commands for 0.6B.

1. Build a bank from scratch, with GCG against the hardened instruction, at every size:
   `SIZE=0_6b MODEL=Qwen/Qwen3-0.6B bash scripts/banks/submit_strictv2_password_bank.sh`.
2. At 0.6B, 4B and 8B, collect the GCG suffixes of seed 11 of that bank:

   ```bash
   sbatch --export=ALL,MODEL=Qwen/Qwen3-0.6B,BANK=$TASK_BANK_ROOT/lquant_prompt_bank_40slots_modbulk_gcg6_strictv2_qwen3_0_6b_20260723/bank_seed_11,OUT=artifacts/gcg_donor_pools/qwen3_0_6b_strictv2_seed11.json \
     scripts/banks/run_extract_gcg_donor_pool.sbatch
   ```

3. At the same three sizes, rebuild the bank with GCG started from those suffixes:
   `SIZE=0_6b MODEL=Qwen/Qwen3-0.6B DONOR_POOL_JSON=artifacts/gcg_donor_pools/qwen3_0_6b_strictv2_seed11.json bash scripts/banks/submit_strictv2_warmstart_bank.sh`.
   At 1.7B, GCG from scratch broke none of the passwords, so there are no suffixes to start from, and the next step uses the bank of step 1.
4. Drop the password-seeded family from the four banks: `bash scripts/banks/build_nopwseed_strictv2_banks.sh` (a few seconds of CPU per bank).

Then train and evaluate:

5. Train with forecastability training at ten seeds at 0.6B and at seeds 11 17 23 at the other sizes, all with `TAG_DATE=20260725`:
   `scripts/password_game/run_lquant_40slots_nopwseed_strictv2_iog_full_{0p6b,1p7b,4b,8b}.sbatch`.
   Train the SFT baseline at 4B and 8B, at seeds 11 17 23 (its run names carry no date):
   `SIZE=4b SEED=11 sbatch scripts/password_game/run_lquant_sft_nopwseed_strictv2.sbatch`.
6. The 8B rows report the checkpoint with the lowest training-side forecast error.
   Evaluate it on the held-out passwords once training has finished:

   ```bash
   for pair in 11:pw8b_s11at89 17:pw8b_s17at80 23:pw8b_s23at126; do
     s=${pair%%:*}
     RUN=$RESULTS_ROOT/lquant_40slots_nopwseed_strictv2_iog_full_bootstrap1_cache296_qwen3_8b_seed${s}_20260725/seed_$s
     SIZE=8b SEED=$s ADAPTER=$RUN/best ARM=${pair#*:} sbatch scripts/password_game/run_lquant_password_eval_adapter.sbatch
   done
   ```

   The table script looks for these evaluation names, which record the steps our runs selected (89, 80 and 126, as each run's `best_step.json` says).
7. `python scripts/figures/prepare_revised_figures.py` computes the table, the 8B final-checkpoint factor in its caption, and the constant-forecast comparison, and writes them to `$RESULTS_ROOT/summaries/`.
   It also draws a figure of the per-seed errors, `password_v2_errors.pdf`, which the paper does not include.

The training budget for Qwen3-8B on the original bank uses the bank of `TAG_DATE=20260506 bash scripts/banks/build_lquant_modbulkgcg_qwen3_8b_bank.sh` and one launcher per row of the table, at seeds 11 17 23:

| Row | Launcher | `TAG_DATE` |
|---|---|---|
| 200 steps, learning rate 1e-5 (the first 8B run) | `run_lquant_modbulkgcg_qwen3_8b_iog_full_1gpu.sbatch`, continued from its checkpoints by `..._resume_3gpu.sbatch` (seeds 11 and 17) and `..._resume.sbatch` (seed 23) | 20260506 |
| 600 steps, learning rate 1e-5 | `run_lquant_modbulkgcg_qwen3_8b_iog_full_3gpu_orig600.sbatch` | 20260728 |
| 200 steps, learning rate 3e-5 | `run_lquant_modbulkgcg_qwen3_8b_iog_full_3gpu_fullbatch.sbatch` | 20260727 (seeds 11, 17), 20260728 (seed 23) |
| 200 steps, learning rate 3e-5, eight targets per update | `run_lquant_modbulkgcg_qwen3_8b_iog_full_4gpu_tps8.sbatch` | 20260727 |

The launchers are in `scripts/password_game/`, and their headers give the GPUs they need and the resume settings.
`python scripts/password_game/collect_8b_budget.py` writes the table to `$RESULTS_ROOT/summaries/8b_budget.md`.

## Forecastability training with sampled returns

The runs of the RL baseline start from the pretrained policies of the 10-seed gridworld fleet of Section 6.2 (`..._10seeds_20260505`), and `exact600ext` below continues that fleet's runs.
The other runs pretrain their own policies with the same recipe and seeds, which gives that fleet's pretrained policies up to floating-point rounding.
The names of all these runs carry fixed dates.

1. Training on negative sampled returns at 1 to 32 rollouts per task, with both estimators:
   `bash scripts/gridworld/run_rlquant_bipolar_reinforce_negret_m600_20260725.sh`.
2. The RL baseline at 600 updates, training on sampled regret, and the scalar-loss estimator:
   `bash scripts/gridworld/run_rlquant_bipolar_reinforce_overnight_20260725.sh`.
   This launcher also submits 300-update runs that no number in the paper reads.
3. Exact-regret training extended to 600 updates, from scratch and resumed (`exact600`, `exact600ext`), and the three 32-rollout seeds that we evaluated at their last stable checkpoint (`n32_last_stable`):
   `bash scripts/gridworld/run_rlquant_bipolar_sampled_returns_extras.sh <set>`.
4. The calibrations, on both the regret and the negative-return scale: `sbatch scripts/gridworld/run_rlquant_posthoc_calibration_sweep.sbatch` (the same sweep as in Section 6.2).
5. `python scripts/gridworld/collect_sampled_returns.py` writes the table, the 32-rollout result, the worst-case regret factors, the RL improvement factors, the calibrated errors and the results of the companion experiment with sampled regret (both estimators, and exact-regret training at 300 and 600 updates) to `$RESULTS_ROOT/summaries/sampled_returns.md`, and the returns that the figure plots to `$RESULTS_ROOT/summaries/rl_capability_verified.json`.
6. The recall of the exact top-k tasks: `sbatch scripts/gridworld/run_rlquant_reinforce_selection_overlap.sbatch`.

## Fit/deployment distribution shift

1. The shift on the original 0.6B bank, which the appendix's table and per-seed figure report:

   ```bash
   bash scripts/password_game/submit_lquant_may_v1_distshift_0p6b_fleet.sh
   ```

   For each of the ten seeds, this submits the training run, the post-hoc calibration of the pretrained forecast, the held-out evaluation of the checkpoint selected by training-side forecast error, and the calibration of that checkpoint's forecast.
   The run names end in `20260726` unless you set `TAG_DATE`.

2. The comparison without the shift reads the runs of Section 6.1.
   Its calibration row needs the calibration of the pretrained forecast in three of them:

   ```bash
   for s in 11 17 23; do
     RUN=$RESULTS_ROOT/lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_seed${s}_20260505/seed_$s
     RUN_DIR=$RUN OUT_JSON=$RUN/posthoc_calibration_baseline.json \
       sbatch scripts/password_game/run_posthoc_calibration_baseline.sbatch
   done
   ```

3. The scan of leak probabilities by prompt family: `TAG_DATE=20260726 sbatch scripts/password_game/run_score_password_bank.sbatch`.

4. The shift on the hardened bank, with the SFT baseline:

   ```bash
   bash scripts/password_game/submit_lquant_distshift_0p6b_fleet.sh
   for s in 11 17 23 29 31 37 41 43 47 53; do
     SEED=$s TAG_DATE=20260726 sbatch scripts/password_game/run_lquant_sft_nopwseed_distshift_0p6b.sbatch
   done
   # once both fleets have finished:
   bash scripts/password_game/submit_distshift_cal_sweep.sh
   ```

5. `python scripts/password_game/collect_distshift_results.py` writes every number of the appendix to `$RESULTS_ROOT/summaries/distshift_results.md`.
   `make_appendix_figures.py` reads the per-seed table in that file.
   The paper's calibrated error for Ours on the hardened bank, 0.772, is the mean over nine seeds, because we did not run the calibration for seed 23.
   The sweep of step 4 calibrates all ten, and the file gives the number of seeds next to each calibrated error.

## Checks beyond the training forecast and score

- Transfer to forecasting procedures unused in training:
  `python scripts/audits/heldout_forecaster_transfer.py --cache-root artifacts/cache --out $RESULTS_ROOT/summaries/heldout_forecaster_transfer.json` recomputes every fleet's held-out forecasts from the saved traces under five forecasters that training did not use.
  It reads the traces under `$RESULTS_ROOT` and in the launchers' baseline-evaluation caches under `artifacts/cache/`, and it does not follow symbolic links to directories.
  The same script counts the exact score ties at the order statistics the forecast uses (the subsection "Ties in finite-precision scores").
  The controlled tests of that subsection are `python -m tests.test_score_ties`.
  They call the loss with the arguments the trainer passes for the Section 6.1 recipe, and check the loss and its gradients at ties inside the fit top k, at the selection boundary, at the deployment maximum, and between two scores that share a parameter.
- Password leakage beyond the first token: `bash scripts/audits/run_leak_elsewhere_pw4b.sh` samples continuations from the pretrained, SFT and forecastability-trained Qwen3-4B models of the hardened password game.
- The improving-only mask and hidden rare modes: `mask_hidden_mode_simulation.py` simulates random fit/deployment splits on the saved baseline traces of the hardened-instruction runs.
  One random stream runs through the run directories in the order given, so pass them in sorted order, as the glob below does.
  The script also loads the Qwen3 tokenizers.

  ```bash
  args=()
  for d in $RESULTS_ROOT/lquant_40slots_nopwseed_strictv2_iog_full_bootstrap1_cache296_*_20260725/seed_*; do args+=(--run-dir "$d"); done
  python scripts/audits/mask_hidden_mode_simulation.py "${args[@]}" --out $RESULTS_ROOT/summaries/mask_hidden_mode_simulation.json
  ```

  The table of that subsection compares the runs of step 4 of the password game with the same runs trained with the mask, `scripts/password_game/run_lquant_40slots_modbulkgcg_metatrain_iog_multiseed.sbatch` (the same seeds and `TAG_DATE`s), and with the runs of Figure 4.
  `python scripts/audits/collect_no_mask_table.py` prints it.

## KL divergence with thinking off, and generations with thinking on

The trainers measured the KL divergence from the base model at the first generated token under Qwen3's default chat template, where the base model opens a thinking block.
The risk scores are read at the first token of a reply with thinking disabled.
These jobs recompute the KL at both tokens from the saved adapters, and let the trained models generate with thinking on.
They load the held-out KL prompts from the path in each run's `config.json`, so the WildChat files must still be where training read them.

1. Read the final adapters of Figure 4's runs, and print each method's means (the paragraph "KL divergence from the base model" of the appendix "Experimental details"):

   ```bash
   sbatch scripts/audits/kl_thinking_off_final.sbatch --out $RESULTS_ROOT/kl_thinking_off/kl_thinking_off_fig4.csv
   python scripts/audits/kl_thinking_off_final_summary.py $RESULTS_ROOT/kl_thinking_off/kl_thinking_off_fig4.csv
   ```

   The paper gives the means of `kl_off_left`, `kl_off_right` and `kl_on_trainer`, and the ratios of SFT's means to ours.

2. Read the selected checkpoints of the natural-prompt runs at each size (the paragraph "What these runs measure"):

   ```bash
   for s in 0_6b 1_7b 4b 8b; do
     sbatch --job-name=kl_thinkoff_$s scripts/audits/kl_thinking_off_final.sbatch --adapter-subdir best \
       --out $RESULTS_ROOT/kl_thinking_off/kl_thinking_off_natural_$s.csv \
       "NatOurs=lquant_wildchat_ldnoobw_qwen_qwen3-${s}_iog_full_boot1_cache296_seed*_mb8klnf05" \
       "NatSFT=lquant_wildchat_ldnoobw_sft_union_qwen_qwen3-${s}_klnf05_seed*"
   done
   ```

   The 8B job needs a GPU with more than 16 GB (ours ran on an A100-80GB).
   At 8B seed 17, `best/` holds the adapter of step 184, whereas the appendix's table uses the copy from step 155, which gave the same thinking-off KL and top-1 probability when we read it.
   Each log lists the most likely first tokens of every checkpoint, and `kl_thinking_off_final_summary.py` prints the KL and the trained models' top-1 probabilities.

3. Read the minimal-change snapshots at 100 and 200 steps into the readouts:

   ```bash
   O=$RESULTS_ROOT/lquant_dynamics/readouts
   sbatch scripts/audits/kl_thinking_off.sbatch --dataset password --out $O/kl_thinking_off_password.csv
   sbatch scripts/audits/kl_thinking_off.sbatch --dataset password --truncation-side left --out $O/kl_thinking_off_password_left.csv
   sbatch scripts/audits/kl_thinking_off.sbatch --dataset wildchat --out $O/kl_thinking_off_wildchat.csv
   sbatch scripts/minimal_change/readouts/py_small.sbatch kl_thinking_off_summary.py --suffix _left
   ```

   When the second and third files are in the readouts, `make_minimal_change_figure.py` (step 6 of "Changing the model as little as possible") checks the appendix's thinking-off KL against them.

4. Let the models generate with thinking on, and count the outcomes:

   ```bash
   bash scripts/audits/submit_think_then_answer.sh
   python scripts/audits/think_then_answer_classify.py $RESULTS_ROOT/think_then_answer $RESULTS_ROOT/think_then_answer_fig4_mc
   ```

   The launcher submits 11 GPU jobs, for the seed-11 natural-prompt checkpoints at 0.6B, 1.7B and 4B, Figure 4's runs, and the minimal-change runs at 200 steps on both banks.
   The paper's "answered normally" is the `normal` column.

## Hardened password instruction

The instruction is the `non_disclosure_strict_v2` policy in `src/forecast_experiment/password_game.py`, which the launchers select with `--password-policy non_disclosure_strict_v2`.

## Appendix figures

`python scripts/figures/make_appendix_figures.py` draws the appendix figures on natural prompts, pretrained models across scale, sampled returns and the fit/deployment shift from saved results, and runs no experiments.
Every number that the appendix text also quotes is checked against the text's value, so a change in the inputs fails loudly.
Its inputs, and the scripts above that write them:

| Input | Written by |
|---|---|
| `$FIGURES_ROOT/lm_three_metric_bars.csv` | `plot_lquant_three_metric_bars.py --csv-output` (Section 6.1) |
| `$RESULTS_ROOT/fit_deploy_error_grid_fullpool/` | `run_fit_deploy_error_grid.sbatch` |
| `$RESULTS_ROOT/summaries/natural_results_verified.json` | `collect_natural_results.py` (Forecastability on natural user prompts) |
| `$RESULTS_ROOT/summaries/rl_capability_verified.json` | `collect_sampled_returns.py` (Forecastability training with sampled returns) |
| `$RESULTS_ROOT/summaries/distshift_results.md` | `collect_distshift_results.py` (Fit/deployment distribution shift) |

The minimal-change appendix's two plotted figures come from `scripts/figures/make_minimal_change_figure.py` and `scripts/figures/make_minimal_change_room_figure.py` (step 6 of that appendix above).
