# Stable SAE hierarchies after instruction tuning depend on the dictionary

**Status: stopped (2026-10-29). Exploratory project with a negative / cautionary result.**

## TL;DR

- I asked whether instruction tuning a vision-language model changes *broad* SAE concepts (parents) or *narrow*
  ones (children).
- With the usual setup (one SAE trained on activations from both the base and the tuned model), the answer
  looked clean: parent families were more stable than random groups of the same size in **72 of 72** checks.
  That covered 2 model families, 3 layers, 2 grouping methods, 2 null models and 3 SAE seeds.
- One control removed it. **The same SAE trained on base-model activations only shows no stable-parent effect**
  (10 of 12 checks at or above zero).
- An SAE trained on tuned-model activations only shows a moderate effect. A Matryoshka SAE gives mixed results.
- So "parents stay stable" is not a robust fact about the model in this setup. It depends on which activations
  the dictionary was trained on.
- Practical lesson for SAE model diffing: **always run a base-only dictionary control.**

## 1. Question

Instruction tuning changes a model, but which kind of concept does it change: general ones ("animal", "food")
or specific ones ("husky", "broccoli")? SAE++ (arXiv 2606.16193) builds two-level concept trees by training a
second SAE on the decoder directions of a first SAE. That suggested a direct test: build a concept tree, then
compare how much **parent** concepts and **child** concepts change between the base and the tuned model.

## 2. Setup

| Item | Choice |
|---|---|
| Model pairs | Gemma-3-4B `pt` vs `it`; Qwen2-VL-2B base vs Instruct |
| Data | COCO val2017, first 5000 images; 32 fixed image-token positions per image; same raw input for both models (image block, no chat template) |
| Sites | Language-model residual stream at the image tokens. Gemma layers 12/20/28; Qwen 10/16/23 (same relative depth). |
| Preprocessing | Center with the shared pt+it mean, zero "massive" dimensions (std > 10x the median; Gemma dim 443, Qwen dim 1073), rescale |
| Level-1 SAE | TopK, 4096 latents, k = 32, AuxK for dead latents |
| Parents (256 groups) | (a) "meta": a second TopK SAE (k = 2) trained on the level-1 decoder rows; (b) k-means on decoder rows; (c) SAE++ joint training |
| Change metric | change = 1 - corr(base activation, tuned activation) over the same tokens; a parent's activation = sum of its children |
| Test statistic | P-R = mean parent change minus mean change of random groups with the same sizes. Negative = families more stable than chance. |
| Nulls | Base vs base + sigma·D, with sigma set so the *child* change matches the real one. D = Gaussian noise, or D = the real (tuned - base) difference of a **random other token** ("shuffled"). |
| Excess | real P-R minus null P-R. Negative = more family stability than the null explains. |
| Uncertainty | Block bootstrap over images (100 blocks) |

## 3. The first result looked strong

With one SAE trained on a 50/50 mix of base and tuned activations:

| Model | Child change | Excess over nulls (meta and k-means, both nulls, 3 seeds) |
|---|---|---|
| Gemma-3-4B | 0.35-0.68 | all 36 negative, -0.018 to -0.069 |
| Qwen2-VL-2B | 0.02-0.04 | all 36 negative, -0.004 to -0.009 |

The pattern held in both models, at every layer, for both grouping methods, against both nulls and over 3 SAE
seeds. Examples looked sensible too. At Gemma layer 20, "container" families (children change, parent stays)
included baseball scenes (top-image labels baseball glove x22, bat x15 over base rate), with children re-labelled
inside the family (frisbee -> sports ball, hot dog -> pizza).

SAE++ joint training made sibling decoders near-copies (median sibling cosine 0.70-0.73 vs 0.04-0.11 for
post-hoc groups) and gave worse reconstruction, so I relied on the post-hoc trees.

## 4. The control that changed the conclusion

All SAEs above were trained on activations from **both** models. A dictionary like that has seen both versions
of every token, so it might create the family structure it then reports. The control: the same SAE and grouping,
trained on **one** model's activations only, scored the same way (Gemma, seed 0).

**Base-only (pt) dictionary.** Excess, noise / shuffled:

| Layer | meta | k-means | FVU on pt / it |
|---|---|---|---|
| 12 | +0.028 / +0.027 | +0.003 / +0.019 | 0.19 / 0.42 |
| 20 | +0.029 / +0.015 | +0.028 / +0.023 | 0.21 / 0.50 |
| 28 | +0.020 / -0.007 | +0.007 / -0.006 | 0.19 / 0.43 |

**Tuned-only (it) dictionary:**

| Layer | meta | k-means | FVU on pt / it |
|---|---|---|---|
| 12 | -0.013 / -0.019 | -0.034 / -0.031 | 0.40 / 0.22 |
| 20 | -0.001 / -0.023 | -0.002 / -0.016 | 0.47 / 0.24 |
| 28 | +0.004 / -0.042 | +0.006 / -0.026 | 0.40 / 0.24 |

| Dictionary trained on | Families more stable than chance? |
|---|---|
| base + tuned mix | yes, strongly (72/72) |
| tuned only | moderately (shuffled null 6/6; noise null only at layer 12) |
| base only | no (10/12 at or above zero; the 2 negatives are -0.006 and -0.007) |

The two single-model dictionaries fit the other model about equally badly (FVU 0.40-0.50), so reconstruction
quality alone does not explain why they disagree.

## 5. Two more dictionaries

**Matryoshka SAE** (nested prefixes 256 / 1024 / 4096; parents = the first 256 latents; each later latent joins
the parent it co-fires with most; trained on the mix; seed 0). Excess, noise / shuffled:

| Model | Layers | noise null | shuffled null |
|---|---|---|---|
| Gemma | 12 / 20 / 28 | +0.034 / +0.031 / +0.072 | -0.014 / -0.027 / -0.056 |
| Qwen | 10 / 16 / 23 | -0.003 / -0.005 / -0.004 | -0.002 / -0.003 / -0.002 |

In Gemma, the Matryoshka parents are far more robust to isotropic noise than to real tuning, so the sign flips
against the noise null. The coarse latents (first 256) change about as much under real tuning as under the
shuffled null in both models.

**GemmaScope 2** (pretrained by Google on Gemma-3-4B-pt text; `resid_post_all`, 16k latents, small L0). It
does not reconstruct image-token activations: FVU without the massive dimension is 1.16 / 0.81 / 0.58 at layers
12 / 20 / 28 on base activations. Its family results (mostly zero or positive excess) are therefore
uninformative.

## 6. Why? A mechanism I tested and could not confirm

Hypothesis: a mixed dictionary splits one concept into a base-specific child and a tuned-specific child under a
shared parent, so the children "swap" while the parent stays.

- The mixed dictionary does contain many model-specific children (latents with > 80% of their firings in one
  model): tuned-specific 3-8%, base-specific 3-10%. The base-only dictionary has almost no tuned-specific
  latents (1-9 in total).
- But tuned-specific children do **not** have a base-specific sibling more often than chance. For "meta" groups
  the rate is below the random-group rate at all 3 layers: 0.43 vs 0.53, 0.50 vs 0.80, 0.30 vs 0.40.
- The share of model-specific children in a family does not predict its container score (Spearman -0.14 to
  +0.15).

So the simple twin mechanism is not supported. One untested possibility: model-specific latents change almost
completely and may cluster in a few families, so random groups (which spread them out) look more changed than
real families.

## 7. Side finding: the vision tower does not change

In both pairs, every vision-tower weight is bit-identical between the base and the tuned model (Gemma: 437
tensors, Qwen: 385). Only the projector (relative weight change: Gemma 0.16, Qwen 0.012) and the language model
change. So for these models, instruction tuning changes nothing before the projector, and analyses of
vision-tower layers will find no difference.

## 8. Limitations

- The controls (base-only, tuned-only, Matryoshka, mechanism check) use one SAE seed, and the single-model
  dictionaries were run on Gemma only.
- Small dictionaries (4096 latents, 4000 training steps), small models (2B and 4B), image tokens only, one
  dataset.
- "Change" is a correlation over tokens. Other diffing measures (decoder-direction change, activation
  frequency, causal effects) were not tested.
- No causal test: I did not check whether any of these families matter for model behavior.

## 9. Takeaway

The family structure an SAE reports for a base-vs-tuned comparison can depend strongly on which activations the
dictionary was trained on. This fits known problems with shared dictionaries in model diffing: Minder et al.
(arXiv 2504.02922) showed that crosscoders can report spurious chat-only latents. Here the same issue appears at
the level of concept hierarchies. When diffing with SAEs, compare a mixed dictionary with a base-only one (and
ideally a tuned-only one) before reading structure into the result.

## Related work

- SAE++ / cascaded SAEs: arXiv 2606.16193 (the tree construction that motivated this).
- Crosscoders and model diffing: Lindsey et al. 2024 (transformer-circuits.pub); Minder et al., arXiv 2504.02922;
  Delta-Crosscoder, arXiv 2603.04426; stage-wise model diffing (transformer-circuits.pub/2024/model-diffing).
- Base SAEs on tuned models: Kissane et al. 2024, "SAEs (usually) transfer between base and chat models".
- SAE latents after supervised fine-tuning: Chopra, arXiv 2605.11426.
- Hierarchical SAEs: Matryoshka SAEs, arXiv 2503.17547; H-SAE, arXiv 2506.01197.
- Full list: [references/REFERENCES.md](references/REFERENCES.md).

## Reproduce

| File | What it is |
|---|---|
| `onecell.py` | Whole pipeline as one Colab cell (Gemma; L4 or A100; needs `HF_TOKEN` and the Gemma terms). Results go to Google Drive; finished steps are skipped on a rerun. |
| `cell_qwen2vl.py`, `cell_vision.py` | The same code with other settings, generated by `make_cells.py` |
| `LOG.md` | Full chronological research log (early entries are corrected by later ones) |

Local CPU test with a tiny random model and planted-tree data: `LOCAL_TEST=1 SYNTH=1 python onecell.py`.
Activations take about 17 min per Gemma model on an L4 (about 7 min per Qwen model). Each SAE and its nulls take
a few minutes per layer.
