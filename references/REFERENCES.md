# References (PDFs local only, gitignored)
- SAE++ / Cascaded SAEs — arXiv 2606.16193 — https://arxiv.org/abs/2606.16193
- Crosscoders (Anthropic 2024) — https://transformer-circuits.pub/2024/crosscoders/index.html
- Minder et al., Overcoming Sparsity Artifacts in Crosscoders — arXiv 2504.02922
- Delta-Crosscoder — arXiv 2603.04426
- Chopra, A Mechanistic Investigation of Supervised Fine Tuning — arXiv 2605.11426
- MMDiff, Multimodal Model Diffing — arXiv 2608.09928
- Bussmann et al., SAEs do not find canonical units (meta-SAEs) — arXiv 2502.04878
- Gao et al., Scaling and evaluating sparse autoencoders (TopK, AuxK) — arXiv 2406.04093
- Finer (fine-grained recognition in instruct VLMs) — arXiv 2402.16315

## Literature review 2026-10-29 (round 2)
| Paper | ID / link | Why it matters |
|---|---|---|
| Minder, Dumas, Juang, Chughtai, Nanda - Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning | arXiv 2504.02922 | BatchTopK crosscoders find chat-only latents; patching them closes ~50% of base-chat gap (causal template) |
| Minder et al. - What We Learned Trying to Diff Base and Chat Models | alignmentforum.org/posts/xmpauEXEerzYcJKNm | chat-SAE and diff-SAE alternatives |
| Lindsey et al. 2024 - Sparse crosscoders for cross-layer features and model diffing | transformer-circuits.pub | original crosscoder diffing |
| Kissane, Krzyzanowski, Conmy, Nanda 2024 - SAEs (usually) transfer between base and chat models | lesswrong.com/posts/fmwk6qxrpW8d4jvbd | base SAEs work on chat; fail on Gemma v1 2B |
| Do SAEs transfer across base and finetuned LMs? | lesswrong.com/posts/bsXPTiAhhwt5nwBW3 | ~10% of features keep activation similarity > 0.8 |
| How Fine-Tuning Repurposes Representations in LLMs | EACL 2026 Findings 2026.findings-eacl.96 | "feature drift": recombines existing concepts |
| Chen et al. - SFT Leaves a Reproducible Signature in Activation Geometry | arXiv 2610.17093 | run-to-run noise baseline for base-vs-tuned diffs (Qwen3-1.7B, Llama-3.1-8B) |
| Delta-Crosscoder | arXiv 2603.04426 | crosscoder diffing in narrow fine-tuning |
| Jiralerspong & Bricken - Cross-architecture model diffing | arXiv 2602.11729 | dedicated feature crosscoders |
| Feature Geometry of LoRA Adapters (SAE) | arXiv 2605.28896 | delta-activation SAE analysis |
| Layer-wise evolution in fine-tuned transformers (SAE, BERT) | arXiv 2502.16722 | general early / task-specific late layers |
| Bussmann et al. - Matryoshka SAEs | arXiv 2503.17547 | standard hierarchical SAE baseline (parent/child) |
| H-SAE - Incorporating Hierarchical Semantics in SAE Architectures | arXiv 2506.01197 | hierarchical SAE on Gemma 2-2B |
| Tree SAE | arXiv 2605.07922 | parent-child with reconstruction condition |
| From Atoms to Trees (HSAE) | arXiv 2602.11881 | jointly learned SAE hierarchy |
| Interpreting CLIP with Hierarchical SAEs (MSAE) | arXiv 2502.20578 | hierarchical SAE on vision |
| Feature hedging | arXiv 2505.11756 | absorption / hedging in hierarchies |
| Lin et al. - The Unlocking Spell on Base LLMs (URIAL) | arXiv 2312.01552 | alignment shifts mostly style tokens (output-level) |
| SAE-FT for CLIP | arXiv 2605.15961 | SAE features drift under fine-tuning (vision) |
