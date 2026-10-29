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
VERSION = "v5 (2026-10-29): v4 + shuffled-difference null (pt + it-pt change of another random token)"
print("=" * 80, f"\nonecell.py {VERSION}\n" + "=" * 80, flush=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# site "lm" = language-model layers at the image tokens; "vision" = vision-tower layers (SAE++ used layer 26)
CONFIG = dict(name="w1_vision", site="vision", pt="google/gemma-3-4b-pt", it="google/gemma-3-4b-it", layers=[13, 26],
              n_images=5000, tok_per_image=32, batch=16, n1=4096, k1=32, n2=256, k2=2, alpha=1.0,
              steps=4000, sae_batch=4096, lr=3e-4, n_blocks=100, n_boot=500, n_rand=5, min_fires=20)

LOCAL = os.environ.get("LOCAL_TEST") == "1"
if LOCAL:
    OUT = os.environ.get("OUT", "/tmp/saetree")
    CONFIG["site"] = os.environ.get("SITE", CONFIG["site"])
    CONFIG.update(layers=[1, 2] if CONFIG["site"] == "vision" else [1, 3], n_images=3000, tok_per_image=4, batch=50, n1=256, k1=4, n2=32, steps=1500,
                  sae_batch=1024, n_blocks=30, n_boot=100)
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
def load_model(which):
    from transformers import Gemma3ForConditionalGeneration
    if LOCAL:  # tiny random Gemma3; "it" = "pt" with its language model perturbed, vision kept equal
        from transformers import Gemma3Config
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
    from transformers import AutoProcessor
    repo = CONFIG[which]
    v = tuple(int(p) for p in transformers.__version__.split(".")[:2])
    kw = {"dtype": torch.bfloat16} if v >= (4, 56) else {"torch_dtype": torch.bfloat16}  # renamed in 4.56
    m = Gemma3ForConditionalGeneration.from_pretrained(repo, device_map=DEV, **kw).eval()
    return m, AutoProcessor.from_pretrained(repo)


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
    if LOCAL:
        c = model.config
        n = c.mm_tokens_per_image
        ids = torch.tensor([[2, c.boi_token_index] + [c.image_token_index] * n + [c.eoi_token_index]] * len(items))
        g = torch.Generator().manual_seed(int(items[0]))
        return dict(input_ids=ids, token_type_ids=(ids == c.image_token_index).long(),
                    pixel_values=torch.randn(len(items), 3, 56, 56, generator=g))
    from PIL import Image
    imgs = [[Image.open(p).convert("RGB")] for p in items]
    # the same raw input for both models (no chat template): BOS + image block; image tokens come first, so
    # their activations depend only on the image and the weights
    x = proc(text=["<start_of_image>"] * len(items), images=imgs, return_tensors="pt")
    return {k: v.to(DEV) for k, v in x.items()}


# ---------------------------------------------------------------- step 1: vision weights equal?
def step1():
    path = f"{OUT}/step1_vision.json"
    if os.path.exists(path):
        return json.load(open(path))
    part = lambda n: ("vision_tower" if "vision_tower" in n else "projector" if "multi_modal_projector" in n
                      else "lm_layer0" if "layers.0." in n else None)
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
    with torch.no_grad():
        for s in range(0, len(items), B):
            chunk = items[s:s + B]
            x = make_inputs(m, proc, chunk)
            if vision:  # vision tower only; hidden_states[L] = output of vision layer L (patch tokens)
                hs = m.model.vision_tower(pixel_values=x["pixel_values"].to(m.dtype),
                                          output_hidden_states=True).hidden_states
                mask = torch.ones(hs[0].shape[:2], dtype=torch.bool, device=hs[0].device)
            else:
                hs = m.model(**x, output_hidden_states=True).hidden_states  # m.model: no lm_head/logits
                mask = x["input_ids"] == img_id
            n_img = int(mask[0].sum())
            for i in range(len(chunk)):
                # the same T image-token positions for both models (seeded by the image index)
                pos = np.sort(np.random.default_rng(s + i).choice(n_img, T, replace=False))
                rows = mask[i].nonzero().squeeze(1)[pos]
                for L in CONFIG["layers"]:
                    bufs[L][(s + i) * T:(s + i + 1) * T] = hs[L][i, rows].float().cpu().numpy()
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


def train(X, alpha, sae1=None, only_level2=False):
    """X: (N, d) float tensor on DEV, already scaled. alpha=0 and not only_level2 -> flat SAE."""
    d, c = X.shape[1], CONFIG
    torch.manual_seed(0)
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
    n_img = len(Xp) // T
    edges = np.linspace(0, n_img, c["n_blocks"] + 1).astype(int) * T
    M = torch.zeros(c["n1"], n_groups, device=DEV)
    M[torch.arange(c["n1"], device=DEV), groups] = 1
    g = torch.Generator(device="cpu").manual_seed(1)
    rand = [groups[torch.randperm(c["n1"], generator=g).to(DEV)] for _ in range(c["n_rand"])]  # same sizes
    Ms = [M] + [torch.zeros_like(M).index_put_((torch.arange(c["n1"], device=DEV), r), torch.ones(c["n1"], device=DEV))
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


def load_saes(ck, d):
    st = torch.load(ck, map_location=DEV)
    out = {}
    for key, n, k in [("joint1", "n1", "k1"), ("joint2", "n2", "k2"), ("flat", "n1", "k1"), ("meta2", "n2", "k2")]:
        out[key] = TopK(d, CONFIG[n], CONFIG[k]).to(DEV)
        out[key].load_state_dict(st[key])
    return out["joint1"], out["joint2"], out["flat"], out["meta2"]


def get_methods(joint1, joint2, flat, meta2):
    return {"saepp_joint": (joint1, parents_from_l2(joint1, joint2)),
            "flat_meta": (flat, parents_from_l2(flat, meta2)),
            "flat_kmeans": (flat, kmeans_groups(flat.W_dec.data, CONFIG["n2"]))}


def step34(L):
    path = f"{OUT}/result_v2_L{L}.json"  # v2 = with massive-dimension removal
    if os.path.exists(path):
        print(f"skip layer {L} (done)")
        return json.load(open(path))
    Xp, Xi, big, res = prep(L)
    X = torch.cat([Xp, Xi])  # 50/50 mix
    ck = f"{OUT}/sae_v2_L{L}.pt"
    if os.path.exists(ck):
        joint1, joint2, flat, meta2 = load_saes(ck, X.shape[1])
    else:
        print(f"  layer {L}: SAE++ (joint)", flush=True)
        joint1, joint2 = train(X, CONFIG["alpha"])
        print(f"  layer {L}: flat SAE", flush=True)
        flat, _ = train(X, 0.0)
        print(f"  layer {L}: post-hoc level-2 on flat (meta)", flush=True)
        _, meta2 = train(X, 0.0, sae1=flat, only_level2=True)
        torch.save(dict(joint1=joint1.state_dict(), joint2=joint2.state_dict(), flat=flat.state_dict(),
                        meta2=meta2.state_dict()), ck)
    del X
    methods = get_methods(joint1, joint2, flat, meta2)
    for name, (sae, groups) in methods.items():
        cc = torch.bincount(groups[sae.since_fired <= 200], minlength=CONFIG["n2"]).cpu().numpy()
        blocks = stats_blocks(sae, Xp, Xi, groups, CONFIG["n2"])
        r = summarize(blocks, cc)
        r.update(fvu_pt=fvu(sae, Xp), fvu_it=fvu(sae, Xi), dead_frac=float((sae.since_fired > 200).float().mean()),
                 sibling_cos_median=sibling_cos(sae.W_dec.data, groups), n_parents_2plus=int((cc >= 2).sum()))
        res[name] = r
        print(f"  L{L} {name}: child {r['child_change']:.3f} parent {r['parent_change']:.3f} "
              f"random {r['random_group_change']:.3f} | P-R {r['parent_minus_random']:+.3f} "
              f"CI {r['parent_minus_random_ci'][0]:+.3f},{r['parent_minus_random_ci'][1]:+.3f} | "
              f"sib cos {r['sibling_cos_median']:.2f} fvu {r['fvu_pt']:.3f}/{r['fvu_it']:.3f} dead {r['dead_frac']:.2f}",
              flush=True)
    save_json(path, res)
    print(f"DONE layer {L}", flush=True)
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


def step5(L, kind="noise"):
    """Null: pt vs pt + sigma * D, sigma chosen so the CHILD change equals the pt-vs-it child change.
    kind "noise": D = isotropic Gaussian. kind "shuffle": D = (it - pt) of ANOTHER random token (real kinds of
    fine-tuning change, but not tied to this token's content). If parent - random is about as negative here,
    the pt-vs-it gap is an SAE effect, not a token-specific fine-tuning effect."""
    path = f"{OUT}/null_v4_L{L}.json" if kind == "noise" else f"{OUT}/shuf_v5_L{L}.json"
    if os.path.exists(path):
        print(f"skip null layer {L} (done)")
        return json.load(open(path))
    real = json.load(open(f"{OUT}/result_v2_L{L}.json"))
    Xp, Xi, big, _ = prep(L)
    joint1, joint2, flat, meta2 = load_saes(f"{OUT}/sae_v2_L{L}.pt", Xp.shape[1])
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
    res = dict(layer=L, kind=kind, note="pt vs pt+sigma*D; sigma matched to the pt-vs-it child change per SAE")
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
        print(f"  {'NULL' if kind == 'noise' else 'SHUF'} L{L} {name}: sigma {sigma:.3f} child {r['child_change']:.3f} (target {target:.3f}) "
              f"parent {r['parent_change']:.3f} random {r['random_group_change']:.3f} | "
              f"null P-R {r['parent_minus_random']:+.3f} CI {r['parent_minus_random_ci'][0]:+.3f},"
              f"{r['parent_minus_random_ci'][1]:+.3f} vs real {real[name]['parent_minus_random']:+.3f}", flush=True)
    save_json(path, res)
    del Xp, noise; free()
    print(f"DONE {kind} null layer {L}", flush=True)
    return res


# ---------------------------------------------------------------- run
if __name__ == "__main__" or True:
    s1 = step1()
    print("step 1:", s1, flush=True)
    if not s1["vision_identical"] and CONFIG["site"] == "lm":
        print("NOTE: vision weights differ; also run cell_vision.py (vision-tower layers)")
    step2("pt")
    step2("it")
    for L in CONFIG["layers"]:
        step34(L)
        free()
    for L in CONFIG["layers"]:
        step5(L)
    for L in CONFIG["layers"]:
        step5(L, "shuffle")
    print("\nALL DONE. Results in", OUT)
    print("Read: change = 1 - corr(pt, it). parent_minus_random < 0 with CI below 0 -> real parents are more "
          "stable than random groups of the same size (fine-tuning moves children inside stable parents).")
    if not LOCAL:
        from google.colab import drive, runtime
        drive.flush_and_unmount()
        runtime.unassign()
