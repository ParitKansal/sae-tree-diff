# sae-tree-diff

Question: when a vision-language model is instruction-tuned, does it change **parent** concepts (broad, e.g.
"animal") or **child** concepts (narrow, e.g. "husky")? Idea from SAE++ (arXiv 2606.16193), which builds
2-level concept trees with a second SAE trained on the first SAE's decoder columns.

Fast exploratory project: no pre-registration, all results exploratory.

## Setup (week 1)
- Models: `google/gemma-3-4b-pt` vs `google/gemma-3-4b-it` (Qwen3-VL has no public base model).
- Data: COCO val2017, first 5000 images; same raw input for both models (BOS + image, no chat template);
  32 fixed image-token positions per image; LM layers 12, 20, 28.
- Step 1 checks whether the vision tower is identical in pt and it (if yes, only LM layers can differ).
- Tree methods (all trained on the 50/50 pt+it mix): SAE++ joint (alpha=1), flat TopK SAE + post-hoc level-2
  SAE ("meta"), flat SAE + k-means. n1=4096, k1=32, n2=256, k2=2.
- Metric: change = 1 - corr(pt activation, it activation) on the same tokens. Parent activation = sum of its
  children. Null = random groups with the same sizes. Block bootstrap over images (100 blocks).

## How to run
1. Accept the Gemma terms for both repos on huggingface.co; add Colab secret `HF_TOKEN`.
2. Paste `onecell.py` into one Colab cell (A100 or L4), run. Results: `MyDrive/sae_tree_diff/w1/`
   (`step1_vision.json`, `result_L{12,20,28}.json`, `sae_L*.pt`). A rerun skips finished layers.
3. Local test (seconds, CPU): `LOCAL_TEST=1 SYNTH=1 python onecell.py` (tiny random Gemma3 for the plumbing,
   then planted-tree activations for the metric). `SWAP=0` = null test.

## Local test results (2026-10-29)
- Planted re-routing (child re-drawn with prob 0.6, parent kept): flat_meta / flat_kmeans give
  parent - random = -0.065 to -0.080 (CI below 0). Works.
- Null (SWAP=0, amplitude noise only): parent - random = -0.002 to -0.003. So the metric has a small negative
  bias; **a gap is only interesting if it is clearly larger than ~0.01 in size** (and its CI excludes 0).
- SAE++ joint (alpha=1) on planted data: worse reconstruction (FVU 0.24 vs 0.08) and sibling decoder cosine
  0.58 vs 0.15 for the flat methods -> siblings pulled together (the "copies" concern). Check alpha on real data.

## Go / no-go (light rule)
Go if, in at least one layer, parent - random differs from 0 by > 0.01 with CI excluding 0 for both flat
methods (not only SAE++). Otherwise: parents and children change about equally -> stop or change the pair
(e.g. Qwen3-VL Instruct vs Thinking).

## Colab log
- 2026-10-29 w1 (LM site) step 1: vision weights of pt and it are NOT identical (max abs diff 0.18 over vision
  tower + projector; LM layer 0 max abs diff 1.5). So the vision tower can also change -> `cell_vision.py`
  (vision-tower layers 13 and 26; 26 = SAE++ site). Its step 1 also splits the difference by part
  (vision_tower / projector / lm_layer0) with relative change ||w_it - w_pt|| / ||w_pt||.
  Results: `MyDrive/sae_tree_diff/w1_vision/`. Rebuild cells with `python3 make_cells.py`.

- 2026-10-29 w1: activations had to be stored as float32 (float16 overflow), and the first OOM came from the
  lm_head logits (fixed: m.model). `it` max |x|: L12 101k, L20 196k, L28 226k, always dims 443 and 1666
  (median-dim max 42 / 198 / 760) = Gemma massive activations. v2 preprocessing: center with the shared pt+it
  mean, zero dims with std > 10x median dim std, scale. Results named `result_v2_L*.json`.

- 2026-10-29 w1 v2 L12 (Gemma-3-4B pt vs it, LM layer 12; removed dim 443 = 97.5% of centered variance):
  saepp_joint child 0.353 parent 0.300 random 0.346, P-R -0.047 [-0.050,-0.043], sibling cos 0.73, FVU 0.34/0.37;
  flat_meta P-R -0.110 [-0.111,-0.109]; flat_kmeans P-R -0.132 [-0.133,-0.130]; flat sibling cos 0.09-0.11,
  FVU 0.21/0.24. Caveat: TopK switching between similar sibling latents can produce this without any
  fine-tuning effect -> v3 adds a noise-matched null (pt vs pt+noise with the same child change). On planted
  data the noise null gives about half the planted gap (-0.035 vs -0.075), so this control matters.

- 2026-10-29 w1 v2 L20 (dim 443 removed, 94.6%): saepp P-R -0.028 [-0.030,-0.025] sib cos 0.70 FVU 0.39/0.41;
  flat_meta -0.098 [-0.099,-0.097]; flat_kmeans -0.109; child change 0.674 (flat). L28 (dim 443, 68.8%): saepp
  -0.034, sib cos 0.70, FVU 0.43/0.45; flat_meta -0.094; flat_kmeans -0.112; child 0.602. Same pattern at all
  3 layers. v3 null was mis-calibrated (20% subset -> child 0.503 vs target 0.353); v4 calibrates on all tokens
  (exact match on planted data), files `null_v4_L*.json`.

- 2026-10-29 noise null v4 (child change matched exactly), null P-R vs real P-R:
  L12 flat_meta -0.057 vs -0.110, flat_kmeans -0.066 vs -0.132, saepp -0.115 vs -0.047;
  L20 flat_meta -0.053 vs -0.098, flat_kmeans -0.058 vs -0.109, saepp -0.111 vs -0.028;
  L28 saepp -0.010 vs -0.034, flat_meta -0.076 vs -0.094, flat_kmeans -0.076 vs -0.112. Flat excess (real - noise): L12 -0.053/-0.066, L20 -0.045/-0.051, L28 -0.018/-0.036 (shrinks with depth).
  SAE++: noise makes its near-copy groups look MORE stable than real fine-tuning does at L12/L20 (not at L28).
  v5 adds a stronger null: pt + sigma * (it - pt) of another random token (`shuf_v5_L*.json`).

- 2026-10-29 shuffled-difference null v5 (pt + sigma*(it-pt) of a random other token; child change matched):
  null P-R vs real: L12 flat_meta -0.074 vs -0.110, flat_kmeans -0.103 vs -0.132, saepp -0.071 vs -0.047;
  L20 flat_meta -0.051 vs -0.098, flat_kmeans -0.057 vs -0.109, saepp -0.066 vs -0.028;
  L28 flat_meta -0.038 vs -0.094, flat_kmeans -0.063 vs -0.112, saepp +0.014 vs -0.034.

## Week-1 summary (exploratory)
Flat methods, excess = real P-R minus null P-R (negative = parents more stable than the null predicts):

| Layer | noise null (meta / kmeans) | shuffled null (meta / kmeans) |
|---|---|---|
| 12 | -0.053 / -0.066 | -0.036 / -0.029 |
| 20 | -0.045 / -0.051 | -0.047 / -0.052 |
| 28 | -0.018 / -0.036 | -0.056 / -0.049 |

- Robust part: in all 12 cells (3 layers x 2 flat methods x 2 nulls) the real gap is more negative than the null,
  by 0.018-0.066. Instruction tuning moves activation between sibling latents while keeping the parent group
  more stable than size-matched changes do.
- Not robust: the depth trend (it reverses between the two nulls) -> do not claim it.
- SAE++ (joint, alpha=1): siblings are near-copies (cos 0.70-0.73), worse FVU, and its excess flips sign by
  layer (+0.068/+0.083/-0.024 noise; +0.024/+0.038/-0.048 shuffled) -> its groups do not track this structure.
- Limits: one SAE seed per method, one model pair, LM layers only, image tokens only, raw (no chat) input.

Next: SAE seed repeat; vision site (cell_vision.py); second model pair (Qwen3-VL Instruct vs Thinking);
look at what the stable parents are (top images).

- v6: `extra_seeds=[1]` retrains flat SAE + meta with seed 1 at all layers and reruns both nulls
  (`*_s1_L*.json`); `inspect_layers=[20]` writes `parents_L20.json`: the 8 most / least stable flat_meta parents
  (>= 3 children, >= 500 firing tokens) with the COCO labels over-represented in their top-50 images, and each
  child's top label in pt -> it.

- 2026-10-29 v6 seed 1 (flat methods): excess over noise / shuffled null (meta, kmeans):
  L12 -0.053, -0.068 / -0.039, -0.037 (seed 0: -0.053, -0.066 / -0.036, -0.029);
  L20 -0.042, -0.043 / -0.041, -0.036 (seed 0: -0.045, -0.051 / -0.047, -0.052). Replicates across SAE seeds.
- v6 parents L20 (flat_meta seed 0; 8 most / least stable of the candidates): two kinds of groups.
  Stable: parent change 0.06-0.15 while children change 0.36-0.63 (re-routing inside the group); clear
  labels for some: baseball (glove x22, bat x15.5, ball x10.7), vehicles (truck x4.4, car x3.0), train (x7.0, x6.4);
  others only person/dining table near base rate. Unstable: parent 0.89-0.94 with children 0.87-0.95 (whole
  group rewired); labels: food (broccoli x16.9, carrot x14.8), bathroom (sink x10.2, toilet x9.4), tennis x9.6,
  desk (mouse x11.4, laptop x6.0). Hint only (16 groups): outdoor/vehicle/sport stable, food/indoor rewired.
  Caveat: child top-label pt->it is noisy (e.g. children with change 0.02 whose label flips) -> do not over-read.

## Before any public release
Independent code check (blind rebuild in a fresh Desktop folder, then compare).
