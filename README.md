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

## Before any public release
Independent code check (blind rebuild in a fresh Desktop folder, then compare).
