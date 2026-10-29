# ===== ONE CELL: does instruction tuning change PARENT concepts or CHILD concepts? (fast week-1 check) =====
# Paste into a Colab GPU cell (A100 or L4) and run it. Needs a Colab secret HF_TOKEN, and the Gemma terms
# accepted on huggingface.co for google/gemma-3-4b-pt and google/gemma-3-4b-it.
# Step 1: are the vision-tower weights of pt and it identical? (if yes, only language-model layers are used)
# Step 2: activations of both models on the same COCO images, at the image tokens, at a few LM layers.
# Step 3: per layer, train on the 50/50 pt+it mix: SAE++ (joint level-2 SAE on decoder columns),
#         a flat TopK SAE + post-hoc level-2 SAE ("meta"), and the flat SAE + k-means groups.
# Step 4: per latent, change = 1 - corr(pt activation, it activation) over the same tokens; compare children,
#         parents (sum of their children), and random groups of the same sizes (null). Block bootstrap over images.
# Results go to Google Drive (finished steps are skipped on a rerun); the GPU is turned off at the end.
import os, json, math, time, gc
VERSION = "v12 (2026-10-29): v11 + control SAE trained on pt only + GemmaScope FVU without massive dims"
print("=" * 80, f"\nonecell.py {VERSION}\n" + "=" * 80, flush=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# site "lm" = language-model layers at the image tokens; "vision" = vision-tower layers (SAE++ used layer 26)
# family "gemma3": prompt = BOS + image block; "qwen2vl": images resized to 448x448 (= 256 image tokens each)
CONFIG = dict(name="w2_qwen2vl", site="lm", family="qwen2vl", pt="Qwen/Qwen2-VL-2B", it="Qwen/Qwen2-VL-2B-Instruct",
              layers=[10, 16, 23], n_images=5000, tok_per_image=32, batch=32, n1=4096, k1=32, n2=256, k2=2, alpha=1.0,
              steps=4000, sae_batch=4096, lr=3e-4, n_blocks=100, n_boot=500, n_rand=5, min_fires=20,
              extra_seeds=[1, 2], inspect_layers=[16],
              gscope_repo="google/gemma-scope-2-4b-pt", gscope_id="16k_l0_small")

LOCAL = os.environ.get("LOCAL_TEST") == "1"
if LOCAL:
    OUT = os.environ.get("OUT", "/tmp/saetree")
    CONFIG["site"] = os.environ.get("SITE", CONFIG["site"])
    CONFIG["family"] = os.environ.get("FAMILY", CONFIG["family"])
    CONFIG.update(layers=[1, 2] if CONFIG["site"] == "vision" else [1, 3], n_images=3000, tok_per_image=4, batch=50, n1=256, k1=4, n2=32, steps=1500,
                  sae_batch=1024, n_blocks=30, n_boot=100)
    CONFIG["inspect_layers"] = [CONFIG["layers"][-1]]
else:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = f"/content/drive/MyDrive/sae_tree_diff/{CONFIG['name']}"
    os.environ["HF_HOME"] = "/content/hf"
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        print("WARNING: no HF_TOKEN secret; the gated Gemma repos will fail to download")
os.makedirs(OUT, exist_ok=True)
ACT = "/tmp/saetree_act" if LOCAL else "/content/act"  # activations: local disk (large), not Drive
os.makedirs(ACT, exist_ok=True)

import numpy as np
import torch
import torch.nn.functional as F
import transformers
print("torch", torch.__version__, "transformers", transformers.__version__, flush=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- models and inputs
def tiny_qwen2vl():
    from transformers import Qwen2VLConfig
    c = Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-2B")
    tc = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
              vocab_size=c.text_config.vocab_size, rope_scaling={"type": "mrope", "mrope_section": [4, 6, 6]})
    vc = dict(depth=2, embed_dim=32, hidden_size=64, num_heads=2, mlp_ratio=2, patch_size=14, spatial_merge_size=2,
              temporal_patch_size=2, in_chans=3)
    return Qwen2VLConfig(text_config=tc, vision_config=vc, image_token_id=c.image_token_id,
                         vision_start_token_id=c.vision_start_token_id, video_token_id=c.video_token_id)


def get_processor(repo):
    from transformers import AutoProcessor
    if CONFIG["family"] == "qwen2vl":
        return AutoProcessor.from_pretrained(repo, min_pixels=448 * 448, max_pixels=448 * 448)
    return AutoProcessor.from_pretrained(repo)


def load_model(which):
    from transformers import AutoModelForImageTextToText
    if LOCAL and CONFIG["family"] == "qwen2vl":  # tiny random Qwen2-VL + the real processor (not gated)
        torch.manual_seed(0)
        m = AutoModelForImageTextToText.from_config(tiny_qwen2vl())
        if which == "it":
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if "language_model" in n:
                        p.add_(0.3 * p.std() * torch.randn_like(p))
        return m.eval(), get_processor("Qwen/Qwen2-VL-2B")
    if LOCAL:  # tiny random Gemma3; "it" = "pt" with its language model perturbed, vision kept equal
        from transformers import Gemma3Config, Gemma3ForConditionalGeneration
        c = Gemma3Config(text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                          num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                                          vocab_size=300, sliding_window=16),
                         vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                            num_attention_heads=2, image_size=56, patch_size=14),
                         mm_tokens_per_image=4, image_token_index=290, boi_token_index=291, eoi_token_index=292)
        torch.manual_seed(0)
        m = Gemma3ForConditionalGeneration(c)
        if which == "it":
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if "language_model" in n:
                        p.add_(0.3 * p.std() * torch.randn_like(p))
        return m.eval(), None
    repo = CONFIG[which]
    v = tuple(int(p) for p in transformers.__version__.split(".")[:2])
    kw = {"dtype": torch.bfloat16} if v >= (4, 56) else {"torch_dtype": torch.bfloat16}  # renamed in 4.56
    m = AutoModelForImageTextToText.from_pretrained(repo, device_map=DEV, **kw).eval()
    return m, get_processor(repo)


def image_list():
    if LOCAL:
        return list(range(CONFIG["n_images"]))
    root = "/content/coco/val2017"
    if not os.path.isdir(root):
        os.makedirs("/content/coco", exist_ok=True)
        os.system("cd /content/coco && wget -q http://images.cocodataset.org/zips/val2017.zip && "
                  "unzip -q val2017.zip && rm val2017.zip")
    files = sorted(os.listdir(root))[:CONFIG["n_images"]]
    return [os.path.join(root, f) for f in files]


def make_inputs(model, proc, items):
    if LOCAL and CONFIG["family"] == "qwen2vl":
        from PIL import Image
        g = np.random.default_rng(int(items[0]))
        imgs = [Image.fromarray((g.random((448, 448, 3)) * 255).astype("uint8")) for _ in items]
        return proc(text=["<|vision_start|><|image_pad|><|vision_end|>"] * len(items), images=imgs, return_tensors="pt")
    if LOCAL:
        c = model.config
        n = c.mm_tokens_per_image
        ids = torch.tensor([[2, c.boi_token_index] + [c.image_token_index] * n + [c.eoi_token_index]] * len(items))
        g = torch.Generator().manual_seed(int(items[0]))
        return dict(input_ids=ids, token_type_ids=(ids == c.image_token_index).long(),
                    pixel_values=torch.randn(len(items), 3, 56, 56, generator=g))
    from PIL import Image
    # the same raw input for both models (no chat template): image block first, so the image-token activations
    # depend only on the image and the weights
    if CONFIG["family"] == "qwen2vl":
        imgs = [Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC) for p in items]
        return proc(text=["<|vision_start|><|image_pad|><|vision_end|>"] * len(items), images=imgs, return_tensors="pt")
    imgs = [[Image.open(p).convert("RGB")] for p in items]
    return proc(text=["<start_of_image>"] * len(items), images=imgs, return_tensors="pt")  # + BOS


# ---------------------------------------------------------------- step 1: vision weights equal?
def step1():
    path = f"{OUT}/step1_vision.json"
    if os.path.exists(path):
        return json.load(open(path))
    part = lambda n: ("projector" if ("multi_modal_projector" in n or "merger" in n) else
                      "vision_tower" if ("vision_tower" in n or ".visual." in n) else
                      "lm_layer0" if "layers.0." in n else None)
    m, _ = load_model("pt")
    ref = {n: p.detach().float().cpu() for n, p in m.named_parameters() if part(n)}
    del m; free()
    m, _ = load_model("it")
    r = {}
    for n, p in m.named_parameters():
        if part(n):
            d = p.detach().float().cpu() - ref[n]
            q = r.setdefault(part(n), dict(max_abs_diff=0.0, d2=0.0, w2=0.0, n_changed=0, n_tensors=0))
            q["max_abs_diff"] = max(q["max_abs_diff"], d.abs().max().item())
            q["d2"] += d.pow(2).sum().item(); q["w2"] += ref[n].pow(2).sum().item()
            q["n_changed"] += int((d != 0).any()); q["n_tensors"] += 1
    del m, p; free()
    for q in r.values():  # relative change = ||w_it - w_pt|| / ||w_pt||
        q["rel_change"] = math.sqrt(q.pop("d2") / max(q.pop("w2"), 1e-30))
    r["vision_identical"] = r["vision_tower"]["max_abs_diff"] == 0.0 and r["projector"]["max_abs_diff"] == 0.0
    save_json(path, r)
    return r


# ---------------------------------------------------------------- step 2: activations
def step2(which):
    paths = {L: f"{ACT}/{which}_{CONFIG['site']}{L}.npy" for L in CONFIG["layers"]}
    if all(os.path.exists(p) for p in paths.values()):
        return
    items = image_list()
    T, B = CONFIG["tok_per_image"], CONFIG["batch"]
    m, proc = load_model(which)
    img_id = m.config.image_token_id if hasattr(m.config, "image_token_id") else m.config.image_token_index
    vision = CONFIG["site"] == "vision"
    width = (m.config.vision_config if vision else m.config.text_config).hidden_size
    bufs = {L: np.lib.format.open_memmap(paths[L] + ".part", "w+", np.float32, (len(items) * T, width))
            for L in CONFIG["layers"]}
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(2)  # decode + preprocess the next batches on CPU while the GPU runs this one
    starts = list(range(0, len(items), B))
    futs = {s: pool.submit(make_inputs, m, proc, items[s:s + B]) for s in starts[:2]}
    with torch.no_grad():
        for k, s in enumerate(starts):
            chunk = items[s:s + B]
            x = {kk: vv.to(DEV) for kk, vv in futs.pop(s).result().items()}
            if k + 2 < len(starts):
                futs[starts[k + 2]] = pool.submit(make_inputs, m, proc, items[starts[k + 2]:starts[k + 2] + B])
            if vision:  # vision tower only; hidden_states[L] = output of vision layer L (patch tokens)
                hs = m.model.vision_tower(pixel_values=x["pixel_values"].to(m.dtype),
                                          output_hidden_states=True).hidden_states
                mask = torch.ones(hs[0].shape[:2], dtype=torch.bool, device=hs[0].device)
            else:
                hs = m.model(**x, output_hidden_states=True).hidden_states  # m.model: no lm_head/logits
                mask = x["input_ids"] == img_id
            n_img = int(mask[0].sum())
            # the same T image-token positions for both models (seeded by the image index); one GPU->CPU copy per layer
            rows = torch.stack([mask[i].nonzero().squeeze(1)[torch.from_numpy(np.sort(
                np.random.default_rng(s + i).choice(n_img, T, replace=False))).to(mask.device)]
                for i in range(len(chunk))])
            bidx = torch.arange(len(chunk), device=mask.device)[:, None]
            for L in CONFIG["layers"]:
                bufs[L][s * T:(s + len(chunk)) * T] = hs[L][bidx, rows].float().reshape(-1, width).cpu().numpy()
            if s % (B * 50) == 0:
                print(f"  {which} {s + len(chunk)}/{len(items)} images, {time.time() - t0:.0f}s", flush=True)
    for L in CONFIG["layers"]:
        a = bufs[L]
        assert np.isfinite(a[::97]).all(), f"non-finite activations at layer {L}"
        top = np.abs(a[::97]).max(0)
        print(f"  {which} L{L}: max |x| {top.max():.0f}, top-3 dims {np.argsort(-top)[:3].tolist()}, "
              f"median-dim max {np.median(top):.1f}", flush=True)
        a.flush(); del bufs[L], a
        os.replace(paths[L] + ".part", paths[L])
    pool.shutdown()
    del m, hs, x, mask; free()
    if LOCAL and os.environ.get("SYNTH") == "1":  # replace with planted-tree data (known answer)
        for L in CONFIG["layers"]:
            np.save(paths[L], planted(which, len(items) * T, L))
    print(f"DONE activations {which}", flush=True)


def planted(which, N, seed, d=64, P=16, C=8, swap=float(os.environ.get("SWAP", "0.6"))):
    """Test data: each token has one parent and one of its children. 'it' keeps the parent and re-draws the
    child with prob `swap` -> children change, parents do not. Expected: parent_minus_random < 0."""
    g = np.random.default_rng(seed)
    U = g.standard_normal((P, d)); V = g.standard_normal((P * C, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True); V /= np.linalg.norm(V, axis=1, keepdims=True)
    t = np.random.default_rng(seed + 1)
    par = t.integers(0, P, N); ch = par * C + t.integers(0, C, N)
    ap, ac, noise = t.uniform(0.5, 1.5, N), t.uniform(0.5, 1.5, N), 0.05 * t.standard_normal((N, d))
    if which == "it":  # SWAP=0 -> null test: only amplitude noise, no child re-routing
        re = t.random(N) < swap
        ch = np.where(re, par * C + t.integers(0, C, N), ch)
        ap, ac = ap * t.uniform(0.7, 1.3, N), ac * t.uniform(0.7, 1.3, N)
    return (ap[:, None] * U[par] + ac[:, None] * V[ch] + noise).astype(np.float16)


# ---------------------------------------------------------------- step 3: SAEs
class TopK(torch.nn.Module):
    """TopK SAE with unit-norm decoder rows and an AuxK loss for dead latents (Gao et al. 2024)."""

    def __init__(self, d, n, k, data_mean=None):
        super().__init__()
        self.k = k
        W = torch.randn(n, d)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(W.clone())
        self.W_enc = torch.nn.Parameter(W.t().clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(n))
        self.b_dec = torch.nn.Parameter(torch.zeros(d) if data_mean is None else data_mean.clone())
        self.register_buffer("since_fired", torch.zeros(n))

    def pre(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def encode(self, x):
        a = self.pre(x)
        v, i = a.topk(self.k, dim=1)
        return torch.zeros_like(a).scatter_(1, i, v)

    def forward(self, x):
        a = self.pre(x)
        v, i = a.topk(self.k, dim=1)
        z = torch.zeros_like(a).scatter_(1, i, v)
        xh = z @ self.W_dec + self.b_dec
        if torch.is_grad_enabled():  # count dead latents during training only
            self.since_fired += 1
            self.since_fired[(z > 0).any(0)] = 0
        dead = self.since_fired > 200
        aux = x.new_zeros(())
        if dead.any():
            e = (x - xh).detach()
            ad = a.masked_fill(~dead, 0)
            kd = min(int(dead.sum()), 4 * self.k)
            v2, i2 = ad.topk(kd, dim=1)
            eh = torch.zeros_like(a).scatter_(1, i2, v2) @ self.W_dec
            aux = (eh - e).pow(2).sum(1).mean() / e.pow(2).sum(1).mean().clamp_min(1e-8)
        return xh, z, aux

    @torch.no_grad()
    def renorm(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)


def level2_loss(sae1, sae2, alive):
    W = sae1.W_dec[alive]
    Wh, _, aux2 = sae2(W)
    return (W - Wh).pow(2).sum(1).mean(), aux2


def train(X, alpha, sae1=None, only_level2=False, seed=0):
    """X: (N, d) float tensor on DEV, already scaled. alpha=0 and not only_level2 -> flat SAE."""
    d, c = X.shape[1], CONFIG
    torch.manual_seed(seed)
    if sae1 is None:
        sae1 = TopK(d, c["n1"], c["k1"], X[:20000].mean(0)).to(DEV)
    sae2 = TopK(d, c["n2"], c["k2"]).to(DEV) if (alpha > 0 or only_level2) else None
    params = (list(sae2.parameters()) if only_level2 else
              list(sae1.parameters()) + (list(sae2.parameters()) if sae2 is not None else []))
    opt = torch.optim.Adam(params, lr=c["lr"])
    var = (X[:20000] - X[:20000].mean(0)).pow(2).sum(1).mean()
    fired_any = torch.ones(c["n1"], dtype=torch.bool, device=DEV)
    for step in range(c["steps"]):
        loss = X.new_zeros(())
        if not only_level2:
            x = X[torch.randint(len(X), (c["sae_batch"],), device=DEV)]
            xh, z, aux = sae1(x)
            loss = (x - xh).pow(2).sum(1).mean() / var + aux / 32
            fired_any = (z > 0).any(0)
        if sae2 is not None:
            # SAE++: level-2 SAE reconstructs the level-1 decoder rows of atoms that fired in this batch;
            # its gradient also flows into the level-1 decoder (joint training)
            alive = fired_any if not only_level2 else (sae1.since_fired <= 200)
            l2, aux2 = level2_loss(sae1, sae2, alive)
            loss = loss + (alpha if not only_level2 else 1.0) * (l2 + aux2 / 32)
        opt.zero_grad(); loss.backward(); opt.step()
        sae1.renorm()
        if sae2 is not None:
            sae2.renorm()
        if step % max(1, c["steps"] // 4) == 0:
            print(f"    step {step} loss {loss.item():.4f}", flush=True)
    return sae1, sae2


def kmeans_groups(W, n2, iters=50):
    Wn = F.normalize(W, dim=1)
    g = torch.Generator(device=W.device).manual_seed(0)
    C = Wn[torch.randperm(len(Wn), generator=g, device=W.device)[:n2]].clone()
    for _ in range(iters):
        lab = (Wn @ C.t()).argmax(1)
        for j in range(n2):
            sel = lab == j
            if sel.any():
                C[j] = F.normalize(Wn[sel].mean(0), dim=0)
    return (Wn @ C.t()).argmax(1)


def parents_from_l2(sae1, sae2):
    with torch.no_grad():
        return sae2.encode(sae1.W_dec).argmax(1)


# ---------------------------------------------------------------- step 4: change metrics
def stats_blocks(sae, Xp, Xi, groups, n_groups):
    """Per image block: sums for corr(pt, it) of children, parents and random groups."""
    c, T = CONFIG, CONFIG["tok_per_image"]
    n1 = sae.W_dec.shape[0]  # 4096 for our SAEs, 16384 for GemmaScope
    n_img = len(Xp) // T
    edges = np.linspace(0, n_img, c["n_blocks"] + 1).astype(int) * T
    M = torch.zeros(n1, n_groups, device=DEV)
    M[torch.arange(n1, device=DEV), groups] = 1
    g = torch.Generator(device="cpu").manual_seed(1)
    rand = [groups[torch.randperm(n1, generator=g).to(DEV)] for _ in range(c["n_rand"])]  # same sizes
    Ms = [M] + [torch.zeros_like(M).index_put_((torch.arange(n1, device=DEV), r), torch.ones(n1, device=DEV))
                for r in rand]
    out = []  # per block: list over [children, parents, rand_1..R] of (7, units) arrays
    with torch.no_grad():
        for b in range(c["n_blocks"]):
            zp = sae.encode(Xp[edges[b]:edges[b + 1]])
            zi = sae.encode(Xi[edges[b]:edges[b + 1]])
            res = []
            for a, bb in [(zp, zi)] + [(zp @ Mk, zi @ Mk) for Mk in Ms]:
                res.append(torch.stack([a.sum(0), bb.sum(0), (a * a).sum(0), (bb * bb).sum(0), (a * bb).sum(0),
                                        (a > 0).sum(0).float(), (bb > 0).sum(0).float(),
                                        torch.full_like(a[0], len(a))]).cpu().double().numpy())
            out.append(res)
    return out


def corr_change(S):
    sx, sy, sxx, syy, sxy, fx, fy, n = S
    cov = sxy / n - sx * sy / n ** 2
    vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
    r = cov / np.sqrt(np.clip(vx * vy, 1e-30, None))
    ok = (fx >= CONFIG["min_fires"]) & (fy >= CONFIG["min_fires"])
    return 1 - r, ok


def summarize(blocks, child_count):
    """Mean change of children / parents (>= 2 alive children) / random groups; block-bootstrap CIs."""
    nb = len(blocks)

    def means(idx):
        tot = [sum(blocks[b][u] for b in idx) for u in range(len(blocks[0]))]
        ch, okc = corr_change(tot[0])
        par, okp = corr_change(tot[1])
        okp &= child_count >= 2
        rnd = []
        for u in range(2, len(tot)):
            cr, okr = corr_change(tot[u])
            rnd.append(cr[okr & (child_count >= 2)].mean())
        return np.array([ch[okc].mean(), par[okp].mean(), np.mean(rnd)]), int(okc.sum()), int(okp.sum())

    point, n_ch, n_par = means(range(nb))
    rng = np.random.default_rng(0)
    boots = np.array([means(rng.integers(0, nb, nb))[0] for _ in range(CONFIG["n_boot"])])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(child_change=point[0], parent_change=point[1], random_group_change=point[2],
                parent_minus_child=point[1] - point[0], parent_minus_child_ci=ci(boots[:, 1] - boots[:, 0]),
                parent_minus_random=point[1] - point[2], parent_minus_random_ci=ci(boots[:, 1] - boots[:, 2]),
                n_children_used=n_ch, n_parents_used=n_par)


def sibling_cos(W, groups):
    Wn = F.normalize(W, dim=1)
    cs = []
    for j in groups.unique():
        idx = (groups == j).nonzero().squeeze(1)
        if len(idx) >= 2:
            S = Wn[idx] @ Wn[idx].t()
            cs.append(S[torch.triu(torch.ones_like(S, dtype=torch.bool), 1)])
    return float(torch.cat(cs).median()) if cs else float("nan")


def fvu(sae, X):
    with torch.no_grad():
        Xs = X[:20000]
        return float((Xs - sae(Xs)[0]).pow(2).sum(1).mean() / (Xs - Xs.mean(0)).pow(2).sum(1).mean())


def prep(L):
    Xp = torch.from_numpy(np.load(f"{ACT}/pt_{CONFIG['site']}{L}.npy")).to(DEV).float()
    Xi = torch.from_numpy(np.load(f"{ACT}/it_{CONFIG['site']}{L}.npy")).to(DEV).float()
    raw_cos = float(F.cosine_similarity(Xp, Xi, dim=1).mean())
    # Gemma has a few "massive" dimensions (values up to ~2e5 vs ~1e2 elsewhere) that would dominate the SAE:
    # center with the shared pt+it mean, zero dims whose std > 10x the median dim std, then scale (same for both)
    mu = (Xp.mean(0) + Xi.mean(0)) / 2
    Xp, Xi = Xp - mu, Xi - mu
    sd = torch.sqrt((Xp.pow(2).mean(0) + Xi.pow(2).mean(0)) / 2)
    big = sd > 10 * sd.median()
    var_big = float(sd[big].pow(2).sum() / sd.pow(2).sum())
    Xp[:, big], Xi[:, big] = 0, 0
    scale = math.sqrt(Xp.shape[1]) / torch.cat([Xp, Xi]).norm(dim=1).mean()
    Xp, Xi = Xp * scale, Xi * scale
    print(f"  layer {L}: removed dims {big.nonzero().squeeze(1).tolist()} (share of centered variance {var_big:.3f})",
          flush=True)
    info = dict(layer=L, raw_cos_pt_it=raw_cos, removed_dims=big.nonzero().squeeze(1).tolist(),
                removed_var_share=var_big, config=CONFIG)
    return Xp, Xi, big, info


def fname(kind, L, seed):
    """seed 0 keeps the original file names; extra seeds (flat methods only) get an _s{seed} name."""
    base = {"result": "result_v2", "sae": "sae_v2", "noise": "null_v4", "shuffle": "shuf_v5"}[kind]
    ext = "pt" if kind == "sae" else "json"
    return f"{OUT}/{base}_L{L}.{ext}" if seed == 0 else f"{OUT}/{base}_s{seed}_L{L}.{ext}"


def load_saes(ck, d):
    st = torch.load(ck, map_location=DEV)
    out = {}
    for key, n, k in [("joint1", "n1", "k1"), ("joint2", "n2", "k2"), ("flat", "n1", "k1"), ("meta2", "n2", "k2")]:
        if key in st:
            out[key] = TopK(d, CONFIG[n], CONFIG[k]).to(DEV)
            out[key].load_state_dict(st[key])
    return out.get("joint1"), out.get("joint2"), out["flat"], out["meta2"]


def get_methods(joint1, joint2, flat, meta2):
    m = {} if joint1 is None else {"saepp_joint": (joint1, parents_from_l2(joint1, joint2))}
    m.update({"flat_meta": (flat, parents_from_l2(flat, meta2)),
              "flat_kmeans": (flat, kmeans_groups(flat.W_dec.data, CONFIG["n2"]))})
    return m


def step34(L, seed=0):
    path = fname("result", L, seed)  # v2 = with massive-dimension removal
    if os.path.exists(path):
        print(f"skip layer {L} seed {seed} (done)")
        return json.load(open(path))
    Xp, Xi, big, res = prep(L)
    res["seed"] = seed
    X = torch.cat([Xp, Xi])  # 50/50 mix
    ck = fname("sae", L, seed)
    if os.path.exists(ck):
        joint1, joint2, flat, meta2 = load_saes(ck, X.shape[1])
    else:
        st = {}
        joint1 = joint2 = None
        if seed == 0:  # extra seeds: flat methods only
            print(f"  layer {L}: SAE++ (joint)", flush=True)
            joint1, joint2 = train(X, CONFIG["alpha"], seed=seed)
            st.update(joint1=joint1.state_dict(), joint2=joint2.state_dict())
        print(f"  layer {L} seed {seed}: flat SAE", flush=True)
        flat, _ = train(X, 0.0, seed=seed)
        print(f"  layer {L} seed {seed}: post-hoc level-2 on flat (meta)", flush=True)
        _, meta2 = train(X, 0.0, sae1=flat, only_level2=True, seed=seed)
        st.update(flat=flat.state_dict(), meta2=meta2.state_dict())
        torch.save(st, ck)
    del X
    methods = get_methods(joint1, joint2, flat, meta2)
    for name, (sae, groups) in methods.items():
        cc = torch.bincount(groups[sae.since_fired <= 200], minlength=CONFIG["n2"]).cpu().numpy()
        blocks = stats_blocks(sae, Xp, Xi, groups, CONFIG["n2"])
        r = summarize(blocks, cc)
        r.update(fvu_pt=fvu(sae, Xp), fvu_it=fvu(sae, Xi), dead_frac=float((sae.since_fired > 200).float().mean()),
                 sibling_cos_median=sibling_cos(sae.W_dec.data, groups), n_parents_2plus=int((cc >= 2).sum()))
        res[name] = r
        print(f"  L{L}{'' if seed == 0 else f' s{seed}'} {name}: child {r['child_change']:.3f} parent {r['parent_change']:.3f} "
              f"random {r['random_group_change']:.3f} | P-R {r['parent_minus_random']:+.3f} "
              f"CI {r['parent_minus_random_ci'][0]:+.3f},{r['parent_minus_random_ci'][1]:+.3f} | "
              f"sib cos {r['sibling_cos_median']:.2f} fvu {r['fvu_pt']:.3f}/{r['fvu_it']:.3f} dead {r['dead_frac']:.2f}",
              flush=True)
    save_json(path, res)
    print(f"DONE layer {L} seed {seed}", flush=True)
    return res


def child_change(sae, A, B):
    """Mean change (1 - corr) over children that fire >= min_fires times in both."""
    S = 0
    with torch.no_grad():
        for i in range(0, len(A), 8192):
            a, b = sae.encode(A[i:i + 8192]), sae.encode(B[i:i + 8192])
            S = S + torch.stack([a.sum(0), b.sum(0), (a * a).sum(0), (b * b).sum(0), (a * b).sum(0),
                                 (a > 0).sum(0).float(), (b > 0).sum(0).float(),
                                 torch.full_like(a[0], len(a))]).double()
    ch, ok = corr_change(S.cpu().numpy())
    return float(ch[ok].mean())


def step5(L, kind="noise", seed=0):
    """Null: pt vs pt + sigma * D, sigma chosen so the CHILD change equals the pt-vs-it child change.
    kind "noise": D = isotropic Gaussian. kind "shuffle": D = (it - pt) of ANOTHER random token (real kinds of
    fine-tuning change, but not tied to this token's content). If parent - random is about as negative here,
    the pt-vs-it gap is an SAE effect, not a token-specific fine-tuning effect."""
    path = fname(kind, L, seed)
    if os.path.exists(path):
        print(f"skip {kind} null layer {L} seed {seed} (done)")
        return json.load(open(path))
    real = json.load(open(fname("result", L, seed)))
    Xp, Xi, big, _ = prep(L)
    joint1, joint2, flat, meta2 = load_saes(fname("sae", L, seed), Xp.shape[1])
    g = torch.Generator(device=DEV).manual_seed(2)
    if kind == "noise":
        noise = torch.randn(Xp.shape, generator=g, device=DEV)
    else:  # it - pt of a random other token (permutation over all tokens; big dims are already 0 in both)
        noise = (Xi - Xp)[torch.randperm(len(Xp), generator=g, device=DEV)]
    del Xi; free()
    noise[:, big] = 0
    # calibrate on ALL tokens: the "fires >= min_fires" filter must select the same children as in the result
    # (v3 used 20% of the images -> rarer children dropped -> child change overshot: 0.503 vs target 0.353)
    sub = slice(None)
    res = dict(layer=L, kind=kind, seed=seed, note="pt vs pt+sigma*D; sigma matched to the pt-vs-it child change per SAE")
    sig_cache = {}
    for name, (sae, groups) in get_methods(joint1, joint2, flat, meta2).items():
        target = real[name]["child_change"]
        key = id(sae)
        if key not in sig_cache:  # bisection in log space
            lo, hi = math.log(1e-3), math.log(10.0 if kind == "noise" else 5.0)
            for _ in range(12):
                mid = (lo + hi) / 2
                c = child_change(sae, Xp[sub], Xp[sub] + math.exp(mid) * noise[sub])
                lo, hi = (mid, hi) if c < target else (lo, mid)
            sig_cache[key] = math.exp((lo + hi) / 2)
        sigma = sig_cache[key]
        Xn = Xp + sigma * noise
        cc = torch.bincount(groups[sae.since_fired <= 200], minlength=CONFIG["n2"]).cpu().numpy()
        r = summarize(stats_blocks(sae, Xp, Xn, groups, CONFIG["n2"]), cc)
        r.update(sigma=sigma, target_child_change=target, real_parent_minus_random=real[name]["parent_minus_random"])
        res[name] = r
        del Xn; free()
        print(f"  {'NULL' if kind == 'noise' else 'SHUF'} L{L}{'' if seed == 0 else f' s{seed}'} {name}: sigma {sigma:.3f} child {r['child_change']:.3f} (target {target:.3f}) "
              f"parent {r['parent_change']:.3f} random {r['random_group_change']:.3f} | "
              f"null P-R {r['parent_minus_random']:+.3f} CI {r['parent_minus_random_ci'][0]:+.3f},"
              f"{r['parent_minus_random_ci'][1]:+.3f} vs real {real[name]['parent_minus_random']:+.3f}", flush=True)
    save_json(path, res)
    del Xp, noise; free()
    print(f"DONE {kind} null layer {L} seed {seed}", flush=True)
    return res


# ---------------------------------------------------------------- step 6: which parents are stable?
def coco_labels(items):
    """Per image: set of COCO object category names (instances_val2017). LOCAL: fake labels."""
    if LOCAL:
        return [{f"c{i % 7}", f"d{i % 3}"} for i in range(len(items))]
    ann = "/content/coco/annotations/instances_val2017.json"
    if not os.path.exists(ann):
        os.system("cd /content/coco && wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
                  " && unzip -q -o annotations_trainval2017.zip annotations/instances_val2017.json"
                  " && rm annotations_trainval2017.zip")
    d = json.load(open(ann))
    names = {c["id"]: c["name"] for c in d["categories"]}
    per = {}
    for a in d["annotations"]:
        per.setdefault(a["image_id"], set()).add(names[a["category_id"]])
    return [per.get(int(os.path.basename(p).split(".")[0]), set()) for p in items]


def top_labels(img_idx, labels, base, k=3):
    """Most over-represented COCO labels among these images: (name, share, lift over all images)."""
    cnt = {}
    for i in img_idx:
        for n in labels[i]:
            cnt[n] = cnt.get(n, 0) + 1
    out = [(n, c / len(img_idx), (c / len(img_idx)) / max(base[n], 1e-9)) for n, c in cnt.items()]
    out = [o for o in out if o[1] >= 0.2]
    return sorted(out, key=lambda o: -o[2])[:k]


def step6(L, n_show=8, n_top=50):
    """flat_meta (seed 0): most and least stable parents (>= 3 children, frequent), with the COCO labels of their
    top-50 images, and the top label of each of their 4 most frequent children (pt and it)."""
    path = f"{OUT}/parents_v8_L{L}.json"
    if os.path.exists(path):
        print(f"skip parents layer {L} (done)")
        return json.load(open(path))
    Xp, Xi, big, _ = prep(L)
    _, _, flat, meta2 = load_saes(fname("sae", L, 0), Xp.shape[1])
    groups = parents_from_l2(flat, meta2)
    T, n1, n2 = CONFIG["tok_per_image"], CONFIG["n1"], CONFIG["n2"]
    n_img = len(Xp) // T
    M = torch.zeros(n1, n2, device=DEV)
    M[torch.arange(n1, device=DEV), groups] = 1
    Sc = Sp = 0
    img_c = {"pt": torch.zeros(n_img, n1, device=DEV), "it": torch.zeros(n_img, n1, device=DEV)}
    with torch.no_grad():
        for i in range(0, len(Xp), 8192):
            zp, zi = flat.encode(Xp[i:i + 8192]), flat.encode(Xi[i:i + 8192])
            rows = torch.arange(i, i + len(zp), device=DEV) // T
            img_c["pt"].index_add_(0, rows, zp); img_c["it"].index_add_(0, rows, zi)
            for a, b, tag in [(zp, zi, "c"), (zp @ M, zi @ M, "p")]:
                S = torch.stack([a.sum(0), b.sum(0), (a * a).sum(0), (b * b).sum(0), (a * b).sum(0),
                                 (a > 0).sum(0).float(), (b > 0).sum(0).float(),
                                 torch.full_like(a[0], len(a))]).double()
                if tag == "c":
                    Sc = Sc + S
                else:
                    Sp = Sp + S
    ch_change, ch_ok = corr_change(Sc.cpu().numpy())
    par_change, par_ok = corr_change(Sp.cpu().numpy())
    fires = Sp[5].cpu().numpy()
    gr = groups.cpu().numpy()
    alive = (flat.since_fired <= 200).cpu().numpy() & ch_ok
    n_kids = np.bincount(gr[alive], minlength=n2)
    kid_mean = np.bincount(gr[alive], weights=ch_change[alive], minlength=n2) / np.maximum(n_kids, 1)
    cand = np.where(par_ok & (n_kids >= 3) & (fires >= 500))[0]
    # v8: "container" = children move but the parent stays (largest child change - parent change);
    # "changed" = largest parent change. (v6 sorted by lowest parent change -> picked untouched groups.)
    by_container = cand[np.argsort(-(kid_mean[cand] - par_change[cand]))]
    by_changed = cand[np.argsort(-par_change[cand])]
    labels = coco_labels(image_list())
    base = {}
    for ls in labels:
        for n in ls:
            base[n] = base.get(n, 0) + 1 / len(labels)
    img_p = {k: (v @ M).cpu().numpy() for k, v in img_c.items()}
    img_c = {k: v.cpu().numpy() for k, v in img_c.items()}
    fmt = lambda tl: ", ".join(f"{n} {s:.2f} (x{l:.1f})" for n, s, l in tl) or "-"
    res = dict(layer=L, method="flat_meta", seed=0, n_candidates=int(len(cand)), container=[], changed=[])
    for tag, sel in [("container", by_container[:n_show]), ("changed", by_changed[:n_show])]:
        for j in sel:
            kids = np.where((gr == j) & alive)[0]
            kids = kids[np.argsort(-Sc[5].cpu().numpy()[kids])][:4]
            top = np.argsort(-img_p["pt"][:, j])[:n_top]
            kid_info = []
            for c in kids:
                tp, ti = np.argsort(-img_c["pt"][:, c])[:n_top], np.argsort(-img_c["it"][:, c])[:n_top]
                kid_info.append(dict(child=int(c), change=float(ch_change[c]), top_pt=top_labels(tp, labels, base, 1),
                                     top_it=top_labels(ti, labels, base, 1),
                                     top_img_overlap=len(set(tp) & set(ti)) / n_top))
            e = dict(parent=int(j), change=float(par_change[j]), n_children=int(n_kids[j]),
                     children_mean_change=float(kid_mean[j]), container_score=float(kid_mean[j] - par_change[j]),
                     top_labels=top_labels(top, labels, base), children=kid_info)
            res[tag].append(e)
            kid_s = " / ".join(f"{(k['top_pt'][0][0] if k['top_pt'] else '-')}"
                               f"->{(k['top_it'][0][0] if k['top_it'] else '-')} {k['change']:.2f}" for k in kid_info)
            print(f"  {tag.upper():9s} L{L} p{j}: change {e['change']:.3f} (children {e['children_mean_change']:.3f}, "
                  f"n={e['n_children']}) | {fmt(e['top_labels'])} | kids pt->it: {kid_s}", flush=True)
    save_json(path, res)
    del Xp, Xi; free()
    print(f"DONE parents layer {L}", flush=True)
    return res


# ---------------------------------------------------------------- step 7: how many families are stable / rewired?
def step7(L, n_rand=20):
    """Per parent (>= 3 alive children, >= 500 firing tokens): container score = mean child change - parent change.
    Null: random groups with the same sizes. 'container' = score above the 95th percentile of the null scores;
    'rewired' = parent change >= 0.8. Seed 0, flat methods, all tokens, no bootstrap."""
    path = f"{OUT}/families_v8_L{L}.json"
    if os.path.exists(path):
        print(f"skip families layer {L} (done)")
        return json.load(open(path))
    Xp, Xi, big, _ = prep(L)
    _, _, flat, meta2 = load_saes(fname("sae", L, 0), Xp.shape[1])
    n1, n2 = CONFIG["n1"], CONFIG["n2"]
    alive = (flat.since_fired <= 200).cpu().numpy()
    res = dict(layer=L)
    g = torch.Generator(device="cpu").manual_seed(3)
    for name, (_, groups) in get_methods(None, None, flat, meta2).items():
        gsets = [groups] + [groups[torch.randperm(n1, generator=g).to(DEV)] for _ in range(n_rand)]
        Ms = []
        for gg in gsets:
            M = torch.zeros(n1, n2, device=DEV)
            M[torch.arange(n1, device=DEV), gg] = 1
            Ms.append(M)
        Sc, Sg = 0, [0] * len(Ms)
        with torch.no_grad():
            for i in range(0, len(Xp), 8192):
                zp, zi = flat.encode(Xp[i:i + 8192]), flat.encode(Xi[i:i + 8192])
                st = lambda a, b: torch.stack([a.sum(0), b.sum(0), (a * a).sum(0), (b * b).sum(0), (a * b).sum(0),
                                               (a > 0).sum(0).float(), (b > 0).sum(0).float(),
                                               torch.full_like(a[0], len(a))]).double()
                Sc = Sc + st(zp, zi)
                for k, M in enumerate(Ms):
                    Sg[k] = Sg[k] + st(zp @ M, zi @ M)
        ch, okc = corr_change(Sc.cpu().numpy())
        use = alive & okc
        scores = []
        for k, gg in enumerate(gsets):
            gp, okp = corr_change(Sg[k].cpu().numpy())
            fires = Sg[k][5].cpu().numpy()
            gr = gg.cpu().numpy()
            kids = np.bincount(gr[use], minlength=n2)
            kid_mean = np.bincount(gr[use], weights=ch[use], minlength=n2) / np.maximum(kids, 1)
            ok = okp & (kids >= 3) & (fires >= 500)
            scores.append((kid_mean[ok] - gp[ok], gp[ok]))
        real_s, real_p = scores[0]
        null_s = np.concatenate([sc for sc, _ in scores[1:]])
        null_p = np.concatenate([p for _, p in scores[1:]])
        thr = float(np.percentile(null_s, 95))
        thr_p = float(np.percentile(null_p, 95))  # v8: 'changed' = parent change above the null's 95th percentile
        r = dict(n_parents=int(len(real_s)), threshold_95=thr, changed_threshold_95=thr_p,
                 changed_share=float((real_p > thr_p).mean()), changed_share_null=float((null_p > thr_p).mean()),
                 container_share=float((real_s > thr).mean()), container_share_null=float((null_s > thr).mean()),
                 rewired_share=float((real_p >= 0.8).mean()), rewired_share_null=float((null_p >= 0.8).mean()),
                 score_median=float(np.median(real_s)), score_median_null=float(np.median(null_s)),
                 parent_change_quartiles=np.percentile(real_p, [25, 50, 75]).tolist())
        res[name] = r
        print(f"  FAMILIES L{L} {name}: n={r['n_parents']} | container {r['container_share']:.2f} (null "
              f"{r['container_share_null']:.2f}) | changed {r['changed_share']:.2f} (null {r['changed_share_null']:.2f}) | "
              f"score median {r['score_median']:+.3f} (null {r['score_median_null']:+.3f})", flush=True)
    save_json(path, res)
    del Xp, Xi; free()
    return res


# ---------------------------------------------------------------- step 8: Matryoshka SAE + pretrained GemmaScope 2
MATRY_SIZES = (CONFIG["n2"], CONFIG["n1"] // 4, CONFIG["n1"])  # nested prefixes (256, 1024, 4096); first n2 = parents


def train_matry(X, seed=0):
    """Matryoshka TopK SAE (Bussmann et al. 2025): every prefix of the dictionary must reconstruct x on its own,
    so early latents learn general concepts and later ones specific ones (no absorption of parents)."""
    d, c = X.shape[1], CONFIG
    torch.manual_seed(seed)
    sae = TopK(d, c["n1"], c["k1"], X[:20000].mean(0)).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=c["lr"])
    var = (X[:20000] - X[:20000].mean(0)).pow(2).sum(1).mean()
    for step in range(c["steps"]):
        x = X[torch.randint(len(X), (c["sae_batch"],), device=DEV)]
        xh, z, aux = sae(x)
        loss = (x - xh).pow(2).sum(1).mean() / var
        for m in MATRY_SIZES[:-1]:
            loss = loss + (x - (z[:, :m] @ sae.W_dec[:m] + sae.b_dec)).pow(2).sum(1).mean() / var
        loss = loss / len(MATRY_SIZES) + aux / 32
        opt.zero_grad(); loss.backward(); opt.step()
        sae.renorm()
        if step % max(1, c["steps"] // 4) == 0:
            print(f"    step {step} loss {loss.item():.4f}", flush=True)
    return sae


def matry_groups(sae, X, n_par=MATRY_SIZES[0]):
    """Parents = latents 0..255 (each is its own group's first member); each later latent joins the parent it
    co-fires with most often (decoder cosine if it never fires). Returns groups and the mean co-fire rate."""
    n1 = sae.W_dec.shape[0]
    co = torch.zeros(n1 - n_par, n_par, device=DEV)
    cnt = torch.zeros(n1 - n_par, device=DEV)
    with torch.no_grad():
        for i in range(0, min(len(X), 200000), 8192):
            f = (sae.encode(X[i:i + 8192]) > 0).float()
            co += f[:, n_par:].t() @ f[:, :n_par]
            cnt += f[:, n_par:].sum(0)
        par = co.argmax(1)
        cos = F.normalize(sae.W_dec[n_par:], dim=1) @ F.normalize(sae.W_dec[:n_par], dim=1).t()
        par = torch.where(cnt > 0, par, cos.argmax(1))
        rate = float((co.max(1).values[cnt > 0] / cnt[cnt > 0]).mean())
    return torch.cat([torch.arange(n_par, device=DEV), par]), rate


class JumpReLU(torch.nn.Module):
    """Pretrained GemmaScope 2 SAE (JumpReLU). sub_bdec: subtract b_dec before encoding (chosen by FVU)."""

    def __init__(self, p, sub_bdec=False):
        super().__init__()
        for k, v in p.items():
            self.register_buffer(k, v.to(DEV).float())
        self.W_dec, self.W_enc = self.w_dec, self.w_enc
        self.sub_bdec = sub_bdec
        self.register_buffer("since_fired", torch.zeros(self.w_dec.shape[0], device=DEV))

    def encode(self, x):
        pre = (x - self.b_dec if self.sub_bdec else x) @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold)

    def forward(self, x):
        z = self.encode(x)
        return z @ self.w_dec + self.b_dec, z, x.new_zeros(())


def load_gemmascope(L):
    """Our layer L = hidden_states[L] = output of decoder block L-1 = GemmaScope 'model.layers.{L-1}.output'."""
    if LOCAL:  # random stand-in with the right shapes (code-path test only)
        g = torch.Generator().manual_seed(L)
        d, n = 64, 512
        return dict(w_enc=torch.randn(d, n, generator=g) / 8, w_dec=torch.randn(n, d, generator=g) / 8,
                    b_enc=torch.zeros(n), b_dec=torch.zeros(d), threshold=torch.full((n,), 0.5))
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    f = hf_hub_download(CONFIG["gscope_repo"], f"resid_post_all/layer_{L - 1}_width_{CONFIG['gscope_id']}/params.safetensors")
    return load_file(f)


def raw_acts(L):
    Xp = torch.from_numpy(np.load(f"{ACT}/pt_{CONFIG['site']}{L}.npy")).to(DEV).float()
    Xi = torch.from_numpy(np.load(f"{ACT}/it_{CONFIG['site']}{L}.npy")).to(DEV).float()
    return Xp, Xi


def coarse_fine(blocks, n_par):
    """Matryoshka-native view: mean change of the first n_par latents (coarse) vs the rest (fine)."""
    ch, ok = corr_change(sum(b[0] for b in blocks))
    idx = np.arange(len(ch))
    return float(ch[ok & (idx < n_par)].mean()), float(ch[ok & (idx >= n_par)].mean())


def eval_pair(tag, name, sae, groups, Xa, Xb, extra=None):
    cc = torch.bincount(groups[sae.since_fired <= 200], minlength=CONFIG["n2"]).cpu().numpy()
    blocks = stats_blocks(sae, Xa, Xb, groups, CONFIG["n2"])
    r = summarize(blocks, cc)
    if name == "matry":
        r["coarse_change"], r["fine_change"] = coarse_fine(blocks, MATRY_SIZES[0])
    r.update(extra or {})
    cf = f" | coarse {r['coarse_change']:.3f} fine {r['fine_change']:.3f}" if name == "matry" else ""
    print(f"  {tag} {name}: child {r['child_change']:.3f} parent {r['parent_change']:.3f} random "
          f"{r['random_group_change']:.3f} | P-R {r['parent_minus_random']:+.3f} CI {r['parent_minus_random_ci'][0]:+.3f},"
          f"{r['parent_minus_random_ci'][1]:+.3f}{cf}", flush=True)
    return r


def nulls_for(L, name, sae, groups, Xp, Xi, big, target, scale):
    """Noise and shuffled-difference nulls (as step 5), sigma matched to the real child change."""
    out = {}
    g = torch.Generator(device=DEV).manual_seed(2)
    for kind in ["noise", "shuffle"]:
        D = (torch.randn(Xp.shape, generator=g, device=DEV) * scale if kind == "noise"
             else (Xi - Xp)[torch.randperm(len(Xp), generator=g, device=DEV)])
        D[:, big] = 0
        lo, hi = math.log(1e-3), math.log(10.0 if kind == "noise" else 5.0)
        for _ in range(12):
            mid = (lo + hi) / 2
            c = child_change(sae, Xp, Xp + math.exp(mid) * D)
            lo, hi = (mid, hi) if c < target else (lo, mid)
        sigma = math.exp((lo + hi) / 2)
        Xn = Xp + sigma * D
        del D; free()
        out[kind] = eval_pair(f"{'NULL' if kind == 'noise' else 'SHUF'} L{L}", name, sae, groups, Xp, Xn,
                              dict(sigma=sigma, target_child_change=target))
        del Xn; free()
    return out


def step8_matry(L, seed=0):
    path = f"{OUT}/matry_v11_L{L}.json" if seed == 0 else f"{OUT}/matry_v11_s{seed}_L{L}.json"
    if os.path.exists(path):
        print(f"skip matryoshka layer {L} seed {seed} (done)")
        return json.load(open(path))
    Xp, Xi, big, res = prep(L)
    ck = path.replace(".json", ".pt")
    sae = TopK(Xp.shape[1], CONFIG["n1"], CONFIG["k1"]).to(DEV)
    if os.path.exists(ck):
        sae.load_state_dict(torch.load(ck, map_location=DEV))
    else:
        print(f"  layer {L} seed {seed}: Matryoshka SAE {MATRY_SIZES}", flush=True)
        sae = train_matry(torch.cat([Xp, Xi]), seed)
        torch.save(sae.state_dict(), ck)
    groups, rate = matry_groups(sae, torch.cat([Xp[:100000], Xi[:100000]]))
    r = eval_pair(f"L{L} s{seed}", "matry", sae, groups, Xp, Xi,
                  dict(fvu_pt=fvu(sae, Xp), fvu_it=fvu(sae, Xi), parent_cofire_rate=rate,
                       sibling_cos_median=sibling_cos(sae.W_dec.data, groups)))
    print(f"    fvu {r['fvu_pt']:.3f}/{r['fvu_it']:.3f} | child fires with its parent {rate:.2f} of the time", flush=True)
    res.update(seed=seed, matry=r, matry_nulls=nulls_for(L, "matry", sae, groups, Xp, Xi, big, r["child_change"], 1.0))
    for k in ["noise", "shuffle"]:
        print(f"  EXCESS L{L} s{seed} matry over {k}: "
              f"{r['parent_minus_random'] - res['matry_nulls'][k]['parent_minus_random']:+.3f}", flush=True)
    save_json(path, res)
    del Xp, Xi; free()
    print(f"DONE matryoshka layer {L} seed {seed}", flush=True)
    return res


def step8_gscope(L):
    """Pretrained GemmaScope 2 SAE (trained on gemma-3-4b-pt text) on RAW activations (no centering / dim
    removal: it was trained on raw residuals). Groups: post-hoc level-2 SAE on its decoder rows, and k-means."""
    path = f"{OUT}/gscope_v11_L{L}.json"
    if os.path.exists(path):
        print(f"skip gemmascope layer {L} (done)")
        return json.load(open(path))
    _, _, big, _ = prep(L)  # only for the massive-dim mask (noise there would dominate)
    free()
    Xp, Xi = raw_acts(L)
    p = load_gemmascope(L)
    fv = {sb: fvu(JumpReLU(p, sb), Xp) for sb in [False, True]}
    sub = fv[True] < fv[False]
    sae = JumpReLU(p, sub)
    print(f"  layer {L}: GemmaScope {CONFIG['gscope_id']} fvu pt {fv[sub]:.3f} it {fvu(sae, Xi):.3f} "
          f"(sub_bdec={sub}; other choice {fv[not sub]:.3f}); image tokens are out of its training data", flush=True)
    Wn = F.normalize(sae.W_dec, dim=1)
    proxy = TopK(Wn.shape[1], Wn.shape[0], 1).to(DEV)  # holder for the level-2 trainer: unit decoder rows, no training
    proxy.W_dec.data = Wn.clone(); proxy.since_fired.zero_()
    proxy.W_dec.requires_grad_(False)
    proxy.renorm = lambda: None
    ck = path.replace(".json", ".pt")
    meta2 = TopK(Wn.shape[1], CONFIG["n2"], CONFIG["k2"]).to(DEV)
    if os.path.exists(ck):
        meta2.load_state_dict(torch.load(ck, map_location=DEV))
    else:
        print(f"  layer {L}: post-hoc level-2 on GemmaScope decoder", flush=True)
        _, meta2 = train(Wn, 0.0, sae1=proxy, only_level2=True, seed=0)
        torch.save(meta2.state_dict(), ck)
    scale = float(torch.cat([Xp, Xi])[:, ~big].std(0).median())  # noise in units of a typical dim's std
    res = dict(layer=L, gscope=f"{CONFIG['gscope_repo']} layer_{L - 1}_width_{CONFIG['gscope_id']}",
               sub_bdec=bool(sub), fvu_pt=fv[sub], fvu_it=fvu(sae, Xi), noise_scale=scale)
    for name, groups in [("gs_meta", parents_from_l2(proxy, meta2)), ("gs_kmeans", kmeans_groups(Wn, CONFIG["n2"]))]:
        r = eval_pair(f"L{L}", name, sae, groups, Xp, Xi, dict(sibling_cos_median=sibling_cos(sae.W_dec, groups)))
        res[name] = r
        res[name + "_nulls"] = nulls_for(L, name, sae, groups, Xp, Xi, big, r["child_change"], scale)
        for k in ["noise", "shuffle"]:
            print(f"  EXCESS L{L} {name} over {k}: "
                  f"{r['parent_minus_random'] - res[name + '_nulls'][k]['parent_minus_random']:+.3f}", flush=True)
    save_json(path, res)
    del Xp, Xi; free()
    print(f"DONE gemmascope layer {L}", flush=True)
    return res


# ---------------------------------------------------------------- step 9: control - SAE trained on pt ONLY
def fvu_nobig(sae, X, big):
    """FVU without the massive dims (they dominate raw variance, so plain FVU on raw data looks too good)."""
    with torch.no_grad():
        Xs = X[:20000]
        e = (Xs - sae(Xs)[0])[:, ~big]
        return float(e.pow(2).sum(1).mean() / (Xs - Xs.mean(0))[:, ~big].pow(2).sum(1).mean())


def step9_ptonly(L):
    """Our SAEs are trained on the 50/50 pt+it mix, so they see both versions of every token and may give the
    pt-vs-it differences their own sibling latents (children swap, parent stays) -> stable parents by
    construction. Control: the same flat SAE + meta / k-means, trained on pt activations only, scored the same."""
    path = f"{OUT}/ptonly_v12_L{L}.json"
    if os.path.exists(path):
        print(f"skip pt-only layer {L} (done)")
        return json.load(open(path))
    Xp, Xi, big, res = prep(L)
    ck = path.replace(".json", ".pt")
    if os.path.exists(ck):
        st = torch.load(ck, map_location=DEV)
        flat, meta2 = TopK(Xp.shape[1], CONFIG["n1"], CONFIG["k1"]).to(DEV), TopK(Xp.shape[1], CONFIG["n2"], CONFIG["k2"]).to(DEV)
        flat.load_state_dict(st["flat"]); meta2.load_state_dict(st["meta2"])
    else:
        print(f"  layer {L}: flat SAE on pt only", flush=True)
        flat, _ = train(Xp, 0.0, seed=0)
        _, meta2 = train(Xp, 0.0, sae1=flat, only_level2=True, seed=0)
        torch.save(dict(flat=flat.state_dict(), meta2=meta2.state_dict()), ck)
    res.update(fvu_pt=fvu(flat, Xp), fvu_it=fvu(flat, Xi))
    print(f"    pt-only SAE fvu pt {res['fvu_pt']:.3f} it {res['fvu_it']:.3f}", flush=True)
    for name, (sae, groups) in get_methods(None, None, flat, meta2).items():
        r = eval_pair(f"PTONLY L{L}", name, sae, groups, Xp, Xi)
        res[name] = r
        res[name + "_nulls"] = nulls_for(L, name, sae, groups, Xp, Xi, big, r["child_change"], 1.0)
        for k in ["noise", "shuffle"]:
            print(f"  EXCESS PTONLY L{L} {name} over {k}: "
                  f"{r['parent_minus_random'] - res[name + '_nulls'][k]['parent_minus_random']:+.3f}", flush=True)
    save_json(path, res)
    del Xp, Xi; free()
    print(f"DONE pt-only layer {L}", flush=True)
    return res


def gscope_fvu(L):
    path = f"{OUT}/gscope_fvu_v12_L{L}.json"
    if os.path.exists(path) or not os.path.exists(f"{OUT}/gscope_v11_L{L}.json"):
        return
    _, _, big, _ = prep(L)
    Xp, Xi = raw_acts(L)
    sae = JumpReLU(load_gemmascope(L), json.load(open(f"{OUT}/gscope_v11_L{L}.json"))["sub_bdec"])
    r = dict(layer=L, fvu_nobig_pt=fvu_nobig(sae, Xp, big), fvu_nobig_it=fvu_nobig(sae, Xi, big))
    print(f"  GemmaScope L{L} fvu without massive dims: pt {r['fvu_nobig_pt']:.3f} it {r['fvu_nobig_it']:.3f}", flush=True)
    save_json(path, r)
    del Xp, Xi, sae; free()


# ---------------------------------------------------------------- run
if __name__ == "__main__" or True:
    s1 = step1()
    print("step 1:", s1, flush=True)
    # projector is outside the vision tower; old step1 files (w1, before v7) have no per-part dict -> treat as changed
    tower_same = s1.get("vision_tower", {}).get("n_changed", 1) == 0
    if not tower_same and CONFIG["site"] == "lm":
        print("NOTE: vision-tower weights differ; also run cell_vision.py (vision-tower layers)")
    run = not (CONFIG["site"] == "vision" and tower_same)
    if not run:
        print("STOP: vision tower is identical in pt and it -> vision activations are identical, nothing to test.")
    if run:
        step2("pt")
        step2("it")
        for L in CONFIG["layers"]:
            step34(L)
            free()
        for L in CONFIG["layers"]:
            step5(L)
        for L in CONFIG["layers"]:
            step5(L, "shuffle")
        for L in CONFIG["inspect_layers"]:
            step6(L)
        for L in CONFIG["layers"]:
            step7(L)
        for seed in CONFIG["extra_seeds"]:
            for L in CONFIG["layers"]:
                step34(L, seed)
                free()
                step5(L, "noise", seed)
                step5(L, "shuffle", seed)
        for L in CONFIG["layers"]:  # v12 control first: needs only our own code, decides how to read everything else
            step9_ptonly(L)
            free()
        for L in CONFIG["layers"]:
            step8_matry(L)
            free()
        if CONFIG["family"] == "gemma3" and CONFIG["site"] == "lm":
            for L in CONFIG["layers"]:
                step8_gscope(L)
                free()
            for L in CONFIG["layers"]:
                gscope_fvu(L)
                free()
    print("\nALL DONE. Results in", OUT)
    print("Read: change = 1 - corr(pt, it). parent_minus_random < 0 with CI below 0 -> real parents are more "
          "stable than random groups of the same size (fine-tuning moves children inside stable parents).")
    if not LOCAL:
        from google.colab import drive, runtime
        drive.flush_and_unmount()
        runtime.unassign()
