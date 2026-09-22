"""
═══════════════════════════════════════════════════════════════════════════
 akarshprompt_boundedcode.py — Fed-CPrompt + BOUNDED PROMPT STORAGE (thesis)
═══════════════════════════════════════════════════════════════════════════

Same faithful Fed-CPrompt as akarshprompt.py (frozen ViT, prompt pool, C2Loss,
FedAvg, iid/label/quantity skew, early stopping) — PLUS a bounded server pool.

When the pool is full (size == pool_cap) and a NEW task arrives, one slot must
be freed DURING TRAINING. Three policies (env AK_POOL_POLICY):

  • evict   : remove a RANDOM prompt (naive baseline — should forget badly)
  • merge   : merge the two most key-similar prompts, MASS-WEIGHTED so a slot
              that already absorbed many tasks isn't diluted (smart policy)
  • hybrid  : if the closest pair is similar enough (cos >= AK_MERGE_THRESH)
              -> merge; else -> evict the LOWEST-UTILITY prompt (the one whose
              removal drops validation accuracy the least, via leave-one-out)

pool_cap = None (default) -> UNBOUNDED = identical to akarshprompt.py (baseline).

LAUNCH (all 3 policies share the SAME config — only the policy differs):
    %env AK_RECIPE=1            # fast recipe (~3-5h/run) — right tool for ablation
    %env AK_AUG=1             # match your reproduction baseline
    %env AK_SPLIT=label_skew   # where forgetting is worst
    %env AK_CLIENTS=10
    # then run each of:
    %env AK_POOL_CAP=                          (unset) -> unbounded baseline
    %env AK_POOL_CAP=2  AK_POOL_POLICY=evict   -> random eviction
    %env AK_POOL_CAP=2  AK_POOL_POLICY=merge   -> mass-weighted merge
    %env AK_POOL_CAP=2  AK_POOL_POLICY=hybrid  -> merge-then-utility-evict
    !python akarshprompt_boundedcode.py

The gap (merge/hybrid vs evict, all vs unbounded) = the thesis result.
═══════════════════════════════════════════════════════════════════════════
"""

import os, copy, math, time, csv, random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import timm

# Share DataLoader tensors via file descriptors instead of /dev/shm, so workers
# don't crash on this cluster's tiny shared memory ("unable to allocate shm").
torch.multiprocessing.set_sharing_strategy("file_system")


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    num_tasks:            int = 6     # MedMNIST cross-domain: 6 different modalities
    classes_per_task:     int = 4     # first 4 classes of each dataset (OCT has only 4)
    seed:                 int = 1993
    data_root:            str = "./data"
    dataset:              str = "medmnist"  # medmnist (cross-domain) | cifar100

    num_clients:          int = 10
    backbone:             str = "vit_base_patch16_224.augreg_in21k"
    embed_dim:            int = 768

    prompts_per_task:     int = 10
    prompt_len:           int = 8
    prompt_attach_layers: List[int] = field(default_factory=lambda: [0,1,2,3,4])
    use_softmax:          bool  = False
    attn_temp:            float = 1.0
    use_ortho:            bool  = True

    classifier:           str   = "linear"
    cos_scale:            float = 16.0

    use_c2l:              bool  = True
    gamma:                float = 0.5
    margin:               float = 0.1
    lambda_c2l:           float = 0.1

    epochs_per_task:      int   = 5
    rounds_per_task:      int   = 40
    lr:                   float = 1e-4
    weight_decay:         float = 0.0
    batch_size:           int   = 128
    warmup_frac:          float = 0.05
    use_amp:              bool  = True

    augment:              bool  = False
    flat_lr:              bool  = True

    early_stop:           bool  = True
    early_stop_patience:  int   = 5
    val_frac:             float = 0.1

    data_split:           str   = "iid"
    beta:                 float = 0.5
    asynchronous:         bool  = False
    async_fraction:       float = 0.5

    # ── BOUNDED STORAGE (thesis extension) ──────────────────────────────────
    pool_cap:             Optional[int] = None    # None = unbounded (baseline)
    pool_policy:          str   = "merge"         # evict | merge | hybrid
    merge_threshold:      float = 0.3             # hybrid: merge if cos>=this, else evict
    util_clean:           bool  = False           # improved utility eviction (per-task loss + tie-break)
    util_eps:             float = 1e-3            # noise band for tie-break among near-equal utilities

    run_tag:              str   = "run"


FAST = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_cfg() -> Config:
    cfg = Config()
    cfg.num_clients  = int(os.environ.get("AK_CLIENTS", cfg.num_clients))
    cfg.classifier   = os.environ.get("AK_CLASSIFIER", cfg.classifier)
    cfg.data_split   = os.environ.get("AK_SPLIT", cfg.data_split)
    cfg.dataset      = os.environ.get("AK_DATASET", cfg.dataset)
    cfg.num_tasks    = int(os.environ.get("AK_TASKS", cfg.num_tasks))
    cfg.classes_per_task = int(os.environ.get("AK_CPT", cfg.classes_per_task))
    cfg.asynchronous = os.environ.get("AK_ASYNC", "0") == "1"
    recipe = os.environ.get("AK_RECIPE", "0") == "1"
    fast   = os.environ.get("AK_FAST", "1" if FAST else "0") == "1"

    if recipe:
        cfg.lr = 1e-3; cfg.augment = True; cfg.flat_lr = False; cfg.use_softmax = True
        fed_rounds, fed_epochs = 10, 3
    else:
        fed_rounds, fed_epochs = cfg.rounds_per_task, cfg.epochs_per_task

    if cfg.num_clients > 1:
        cfg.rounds_per_task = int(os.environ.get("AK_ROUNDS", fed_rounds))
        cfg.epochs_per_task = int(os.environ.get("AK_EPOCHS", fed_epochs))
    else:
        cfg.rounds_per_task = 1
        cfg.epochs_per_task = int(os.environ.get("AK_EPOCHS", 20 if recipe else 5))

    cfg.lr = float(os.environ.get("AK_LR", cfg.lr))
    if "AK_SOFTMAX" in os.environ: cfg.use_softmax = os.environ["AK_SOFTMAX"] == "1"
    if "AK_AUG"     in os.environ: cfg.augment     = os.environ["AK_AUG"] == "1"

    # ── bounded-storage env ──
    if "AK_POOL_CAP" in os.environ and os.environ["AK_POOL_CAP"].strip() != "":
        cfg.pool_cap = int(os.environ["AK_POOL_CAP"])
    cfg.pool_policy     = os.environ.get("AK_POOL_POLICY", cfg.pool_policy)
    cfg.merge_threshold = float(os.environ.get("AK_MERGE_THRESH", cfg.merge_threshold))
    cfg.util_clean      = os.environ.get("AK_UTIL", "") == "clean"

    if fast:
        cfg.num_tasks = 3; cfg.classes_per_task = 4   # 3 modalities, 4 classes each
        cfg.epochs_per_task = 1; cfg.rounds_per_task = 2
        cfg.num_clients = min(cfg.num_clients, 4); cfg.batch_size = 64

    rcp_tag = "_recipe" if recipe else "_paper"
    mode = "fed" if cfg.num_clients > 1 else "central"
    cap_tag = f"_cap{cfg.pool_cap}_{cfg.pool_policy}" if cfg.pool_cap is not None else "_unbounded"
    util_tag = "_cleanutil" if cfg.util_clean else ""
    fast_tag = "_FAST" if fast else ""
    cfg.run_tag = f"akarsh_{cfg.dataset}_{mode}_{cfg.data_split}_{cfg.classifier}{rcp_tag}{cap_tag}{util_tag}{fast_tag}"
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
#  DATA  (identical to akarshprompt.py)
# ═══════════════════════════════════════════════════════════════════════════
def build_transforms(cfg: Config):
    try:
        tmp = timm.create_model(cfg.backbone, pretrained=False, num_classes=0)
        dc = timm.data.resolve_data_config({}, model=tmp)
        mean, std = dc["mean"], dc["std"]; del tmp
    except Exception:
        mean = std = (0.5, 0.5, 0.5)
    test_tf = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])
    if cfg.augment:
        train_tf = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(), T.Normalize(mean, std)])
    else:
        train_tf = test_tf
    return train_tf, test_tf


# ── MedMNIST cross-domain: each task = a DIFFERENT imaging modality ──────────
# Ordered most-diverse-first; first 6 used by default (TissueMNIST is huge, last).
MED_FLAGS = ["pathmnist", "dermamnist", "bloodmnist", "octmnist",
             "organamnist", "retinamnist", "tissuemnist",
             "organcmnist", "organsmnist"]


class MedConcat(torch.utils.data.Dataset):
    """Concatenate several MedMNIST datasets, keep first `keep_c` classes of each,
    and expose GLOBAL contiguous labels (task t -> labels [t*keep_c, (t+1)*keep_c))
    plus a `.targets` array, so the rest of the pipeline is unchanged."""
    def __init__(self, parts, keep_c):
        self.parts = parts
        self.samples = []            # (part_idx, local_idx)
        tg = []
        for p, ds in enumerate(parts):
            labels = np.asarray(ds.labels).reshape(-1)
            off = p * keep_c
            for li, lab in enumerate(labels):
                if lab < keep_c:
                    self.samples.append((p, li)); tg.append(off + int(lab))
        self.targets = np.array(tg)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, li = self.samples[i]
        x, _ = self.parts[p][li]     # transform already applied inside MedMNIST
        return x, int(self.targets[i])


def build_medmnist(cfg: Config):
    """CIFAR-STYLE split of MedMNIST: pool the classes from several datasets into a
    single label space, SHUFFLE them, and chop into disjoint tasks — exactly like
    CIFAR-100 (shuffle 100 classes -> 10 tasks of 10). Each task is a random MIX of
    classes; sampling is BALANCED (K images per class) so every class/task is equal.
    (Contrast with akarshboundedmedical.py, where task = one whole modality.)"""
    import medmnist
    from medmnist import INFO
    train_tf, test_tf = build_transforms(cfg)
    flags = MED_FLAGS[:cfg.num_tasks]
    if len(flags) < cfg.num_tasks:
        raise ValueError(f"only {len(MED_FLAGS)} medmnist datasets available, "
                         f"asked for {cfg.num_tasks} tasks")
    C = cfg.classes_per_task
    def parts(split, tf):
        out = []
        for f in flags:
            DC = getattr(medmnist, INFO[f]["python_class"])
            out.append(DC(split=split, transform=tf, as_rgb=True,
                          download=True, root=cfg.data_root))
        return out
    train = MedConcat(parts("train", train_tf), C)   # pooled: num_tasks*C classes
    test  = MedConcat(parts("test",  test_tf),  C)
    g = np.random.RandomState(cfg.seed)

    # ── CIFAR-style: shuffle all pooled classes, split into disjoint task groups ──
    N = cfg.num_tasks * C
    order = list(range(N)); g.shuffle(order)
    tcls = [order[t*C:(t+1)*C] for t in range(cfg.num_tasks)]           # C classes per task
    global_of = {orig: gi for gi, orig in enumerate(c for tc in tcls for c in tc)}

    # balanced per-class sampling (like CIFAR's 500/class). K = images per class.
    PER_CLASS      = int(os.environ.get("AK_PER_CLASS", "500"))
    TEST_PER_CLASS = int(os.environ.get("AK_TEST_PER_CLASS", "100"))
    def sample(ds, per_class):
        out = []
        for tc in tcls:
            idxs = []
            for c in tc:
                ci = np.where(ds.targets == c)[0]; g.shuffle(ci)
                if per_class > 0: ci = ci[:per_class]          # K per class (or all if fewer)
                idxs.extend(ci.tolist())
            a = np.array(idxs); g.shuffle(a); out.append(a.tolist())
        return out

    train_idx, val_idx = [], []
    for idxs in sample(train, PER_CLASS):
        a = np.array(idxs); g.shuffle(a)
        cut = int((1 - cfg.val_frac) * len(a))
        train_idx.append(a[:cut].tolist()); val_idx.append(a[cut:].tolist())
    test_pt = sample(test, TEST_PER_CLASS)

    print(f"  MedMNIST CIFAR-STYLE split: {N} pooled classes -> {cfg.num_tasks} tasks x {C} "
          f"(shuffled, balanced {PER_CLASS}/class):")
    for t in range(cfg.num_tasks):
        print(f"    task {t+1}: global labels {t*C}-{(t+1)*C-1} (orig classes {tcls[t]}) | "
              f"train={len(train_idx[t])} val={len(val_idx[t])} test={len(test_pt[t])}")
    return train, test, train_idx, val_idx, test_pt, global_of


def build_datasets(cfg: Config):
    if cfg.dataset == "medmnist":
        return build_medmnist(cfg)
    train_tf, test_tf = build_transforms(cfg)
    train = torchvision.datasets.CIFAR100(cfg.data_root, train=True,  download=True, transform=train_tf)
    test  = torchvision.datasets.CIFAR100(cfg.data_root, train=False, download=True, transform=test_tf)
    g = np.random.RandomState(cfg.seed)
    order = list(range(cfg.num_tasks * cfg.classes_per_task)); g.shuffle(order)
    C = cfg.classes_per_task
    tcls = [order[t*C:(t+1)*C] for t in range(cfg.num_tasks)]
    global_of = {orig: gi for gi, orig in enumerate(c for tc in tcls for c in tc)}
    def per_task(ds):
        tg = np.array(ds.targets)
        return [np.where(np.isin(tg, tc))[0].tolist() for tc in tcls]
    tr_full = per_task(train)
    train_idx, val_idx = [], []
    for idxs in tr_full:
        a = np.array(idxs); g.shuffle(a)
        cut = int((1 - cfg.val_frac) * len(a))
        train_idx.append(a[:cut].tolist()); val_idx.append(a[cut:].tolist())
    return train, test, train_idx, val_idx, per_task(test), global_of


class GlobalLabelDS(torch.utils.data.Dataset):
    def __init__(self, base, indices, global_of):
        self.base, self.indices, self.global_of = base, indices, global_of
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, self.global_of[y]


def iid_split(indices, n, seed):
    g = np.random.RandomState(seed); a = np.array(indices); g.shuffle(a)
    return [c.tolist() for c in np.array_split(a, n)]

def quantity_skew_split(indices, n, beta, seed):
    g = np.random.RandomState(seed); a = np.array(indices); g.shuffle(a)
    props = g.dirichlet([beta]*n)
    counts = (props/props.sum() * len(a)).astype(int); counts[-1] = len(a) - counts[:-1].sum()
    out, cur = [], 0
    for c in counts: out.append(a[cur:cur+c].tolist()); cur += c
    return out

def label_skew_split(indices, n, beta, seed, targets):
    g = np.random.RandomState(seed); by_cls = {}
    for i in indices:
        by_cls.setdefault(int(targets[i]), []).append(i)
    out = [[] for _ in range(n)]
    for cls_idx in by_cls.values():
        g.shuffle(np.asarray(cls_idx))
        props = g.dirichlet([beta]*n)
        counts = (props/props.sum() * len(cls_idx)).astype(int); counts[-1] = len(cls_idx) - counts[:-1].sum()
        cur = 0
        for cid, c in enumerate(counts):
            out[cid].extend(cls_idx[cur:cur+c]); cur += c
    return out

def split_clients(cfg, indices, seed, train_ds):
    if cfg.data_split == "iid":            return iid_split(indices, cfg.num_clients, seed)
    if cfg.data_split == "quantity_skew":  return quantity_skew_split(indices, cfg.num_clients, cfg.beta, seed)
    if cfg.data_split == "label_skew":     return label_skew_split(indices, cfg.num_clients, cfg.beta, seed, train_ds.targets)
    raise ValueError(cfg.data_split)


# ═══════════════════════════════════════════════════════════════════════════
#  BACKBONE + PROMPT POOL + C2Loss  (identical to akarshprompt.py)
# ═══════════════════════════════════════════════════════════════════════════
def prefix_attention(attn, x, pk, pv):
    B, N, C = x.shape; H = attn.num_heads; Hd = C // H; Lp = pk.shape[1]
    qkv = attn.qkv(x).reshape(B, N, 3, H, Hd).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0); q, k = attn.q_norm(q), attn.k_norm(k)
    pk = pk.to(k.dtype).reshape(B, Lp, H, Hd).permute(0, 2, 1, 3)
    pv = pv.to(v.dtype).reshape(B, Lp, H, Hd).permute(0, 2, 1, 3)
    k = torch.cat([pk, k], 2); v = torch.cat([pv, v], 2)
    q = q * attn.scale
    aw = (q @ k.transpose(-2, -1)).softmax(-1); aw = attn.attn_drop(aw)
    out = (aw @ v).transpose(1, 2).reshape(B, N, C)
    return attn.proj_drop(attn.proj(out))

def block_forward(blk, x, pk=None, pv=None):
    if pk is None: return blk(x)
    x = x + blk.drop_path1(blk.ls1(prefix_attention(blk.attn, blk.norm1(x), pk, pv)))
    return x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))

class FrozenViT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.vit = timm.create_model(cfg.backbone, pretrained=True, num_classes=0)
        for p in self.vit.parameters(): p.requires_grad_(False)
        self.vit.eval()
        self.prompt_layers = list(cfg.prompt_attach_layers)
    @torch.no_grad()
    def get_query(self, x): return self.vit(x)
    def forward_prompted(self, x, pk_list, pv_list):
        v = self.vit
        x = v.norm_pre(v.patch_drop(v._pos_embed(v.patch_embed(x))))
        pi = 0
        for i, blk in enumerate(v.blocks):
            if i in self.prompt_layers and pi < len(pk_list):
                x = block_forward(blk, x, pk_list[pi], pv_list[pi]); pi += 1
            else:
                x = block_forward(blk, x)
        return v.norm(x)[:, 0]

def _ortho(M, *rest):
    w = torch.empty(M, int(np.prod(rest))); nn.init.orthogonal_(w); return w.reshape(M, *rest)

class TaskPrompt(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        D, Lp, M, nl = cfg.embed_dim, cfg.prompt_len, cfg.prompts_per_task, len(cfg.prompt_attach_layers)
        self.M, self.n_layers = M, nl
        if cfg.use_ortho:
            self.P_K = nn.Parameter(_ortho(M, nl, Lp//2, D).permute(1,0,2,3).contiguous())
            self.P_V = nn.Parameter(_ortho(M, nl, Lp//2, D).permute(1,0,2,3).contiguous())
            self.K   = nn.Parameter(_ortho(M, D)); self.A = nn.Parameter(_ortho(M, D))
        else:
            self.P_K = nn.Parameter(torch.randn(nl, M, Lp//2, D)*0.02)
            self.P_V = nn.Parameter(torch.randn(nl, M, Lp//2, D)*0.02)
            self.K   = nn.Parameter(torch.randn(M, D)*0.02); self.A = nn.Parameter(torch.ones(M, D))
    def cos_scores(self, q):
        qa = q.unsqueeze(1) * self.A.unsqueeze(0)
        return (F.normalize(qa, dim=-1) * F.normalize(self.K, dim=-1).unsqueeze(0)).sum(-1)
    def comps(self): return self.P_K.permute(1,0,2,3), self.P_V.permute(1,0,2,3)
    def prompt_params(self): return torch.cat([self.P_K.flatten(), self.P_V.flatten()])

def pool_prompt(query, prompts, trainable_idx, use_softmax=True, temp=1.0):
    cos_list, pk_c, pv_c = [], [], []
    for ti, p in enumerate(prompts):
        if trainable_idx is not None and ti == trainable_idx:
            cos = p.cos_scores(query); pk, pv = p.comps()
        else:
            with torch.no_grad(): cos = p.cos_scores(query.detach())
            pk, pv = p.comps(); cos, pk, pv = cos.detach(), pk.detach(), pv.detach()
        cos_list.append(cos); pk_c.append(pk); pv_c.append(pv)
    cat = torch.cat(cos_list, 1)
    alpha = torch.softmax(temp * cat, 1) if use_softmax else cat
    pk = torch.einsum('bc,cnpd->bnpd', alpha, torch.cat(pk_c, 0))
    pv = torch.einsum('bc,cnpd->bnpd', alpha, torch.cat(pv_c, 0))
    nl = pk.shape[1]
    return [pk[:, l] for l in range(nl)], [pv[:, l] for l in range(nl)], alpha

class C2Loss(nn.Module):
    def __init__(self, gamma=0.5, margin=0.1):
        super().__init__(); self.gamma, self.margin = gamma, margin
    def forward(self, p_cur, p_prev, others):
        v = p_cur.prompt_params(); vs = p_prev.prompt_params().detach()
        term1 = torch.norm(v - vs, p=2)
        term2 = (torch.stack([torch.norm(v - o.prompt_params().detach(), 2) for o in others]).min()
                 if others else torch.zeros((), device=v.device))
        return torch.clamp(term1 - self.gamma*term2 + self.margin, min=0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
class CosineClassifier(nn.Module):
    def __init__(self, dim, n, scale=16.0):
        super().__init__(); self.W = nn.Parameter(torch.randn(n, dim)*0.02); self.scale = scale
    def forward(self, x): return self.scale * F.linear(F.normalize(x, 1), F.normalize(self.W, 1))

def make_classifier(cfg):
    n = cfg.num_tasks * cfg.classes_per_task
    return CosineClassifier(cfg.embed_dim, n, cfg.cos_scale).to(DEVICE) if cfg.classifier == "cosine" \
        else nn.Linear(cfg.embed_dim, n).to(DEVICE)

def _norm_weights(weights, n):
    if weights is None: w = torch.ones(n, device=DEVICE)
    else: w = torch.tensor([float(x) for x in weights], dtype=torch.float32, device=DEVICE)
    s = w.sum()
    return w / s if s > 0 else torch.ones(n, device=DEVICE) / n


# ═══════════════════════════════════════════════════════════════════════════
#  BOUNDED SERVER  — slot pool with evict / merge / hybrid
#  (uses STABLE slot ids in an ordered list -> no index-shift bugs)
#  pool_cap=None -> never frees -> identical to the unbounded paper baseline.
# ═══════════════════════════════════════════════════════════════════════════
class Server:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cap = cfg.pool_cap
        self.policy = cfg.pool_policy
        self.classifier = make_classifier(cfg)
        self.prompt_of: Dict[int, TaskPrompt] = {}   # slot_id -> prompt
        self.mass_of:   Dict[int, int]        = {}   # slot_id -> #tasks merged in
        self.slot_ids:  List[int]             = []   # ordered active slot ids
        self.task_to_slot: Dict[int, int]     = {}   # task_id -> slot_id (None if evicted)
        self.cur_slot:  Optional[int]         = None
        self._next = 0
        self.merge_log: List[dict] = []
        self.evict_log: List[dict] = []

    # ---- slot helpers ----
    def _new_slot(self) -> int:
        sid = self._next; self._next += 1
        self.prompt_of[sid] = TaskPrompt(self.cfg).to(DEVICE)
        self.mass_of[sid] = 1
        self.slot_ids.append(sid)
        return sid

    def active_prompts(self) -> List[TaskPrompt]:
        return [self.prompt_of[s] for s in self.slot_ids]

    def cur_index(self) -> int:
        return self.slot_ids.index(self.cur_slot)

    def _drop_slot(self, sid, merged_into=None):
        self.slot_ids.remove(sid)
        del self.prompt_of[sid]; del self.mass_of[sid]
        for tid, s in list(self.task_to_slot.items()):
            if s == sid:
                self.task_to_slot[tid] = merged_into

    @torch.no_grad()
    def _most_similar(self):
        keys = [F.normalize(self.prompt_of[s].K.mean(0), dim=0).detach() for s in self.slot_ids]
        best = (-2.0, None, None)
        for a in range(len(keys)):
            for b in range(a+1, len(keys)):
                sim = float((keys[a] * keys[b]).sum())
                if sim > best[0]: best = (sim, self.slot_ids[a], self.slot_ids[b])
        return best

    def _merge_pair(self, i, j, sim):
        mi, mj = self.mass_of[i], self.mass_of[j]          # mass-weighted average
        with torch.no_grad():
            di = dict(self.prompt_of[i].named_parameters())
            dj = dict(self.prompt_of[j].named_parameters())
            for name in di:
                di[name].data.copy_((mi*di[name].data + mj*dj[name].data) / (mi+mj))
        self.mass_of[i] = mi + mj
        self.merge_log.append({"keep": i, "absorbed": j, "sim": round(sim,4), "new_mass": self.mass_of[i]})
        print(f"  [BoundedPool] MERGE slot {j}->{i}  (sim={sim:.3f}, mass {mi}+{mj}->{mi+mj})")
        self._drop_slot(j, merged_into=i)

    def _random_evict(self):
        victim = random.choice(self.slot_ids)
        self.evict_log.append({"victim": victim, "type": "random"})
        print(f"  [BoundedPool] RANDOM-EVICT slot {victim} (knowledge discarded)")
        self._drop_slot(victim, merged_into=None)

    def _utility_evict(self, vit, val_loaders, seen_tasks, sim):
        base = pool_val_acc(vit, self, self.slot_ids, val_loaders, seen_tasks, self.cfg)
        util = {}
        for sid in self.slot_ids:                          # leave-one-out drop
            keep = [s for s in self.slot_ids if s != sid]
            util[sid] = base - pool_val_acc(vit, self, keep, val_loaders, seen_tasks, self.cfg)
        victim = min(util, key=util.get)
        self.evict_log.append({"victim": victim, "type": "utility",
                               "util": round(util[victim],4), "blocked_sim": round(sim,4)})
        print(f"  [BoundedPool] merge blocked (sim={sim:.3f}<{self.cfg.merge_threshold}) "
              f"-> UTILITY-EVICT slot {victim} (drop={util[victim]:.4f})")
        self._drop_slot(victim, merged_into=None)

    def _utility_evict_clean(self, vit, val_loaders, seen_tasks, sim):
        """Improved utility eviction:
          (1) per-task signal  — score each slot by the loss increase on ITS OWN task,
              not global accuracy (decouples from classifier drift on other tasks);
          (2) loss not accuracy — smooth signal, less quantisation noise;
          (3) full val set      — lower-variance estimate;
          (4) noise-aware tie-break — if several slots are within eps of the minimum,
              drop one at RANDOM, so on redundant data it degrades to random (never
              worse) instead of confidently selecting on noise."""
        seen_hi = (max(seen_tasks)+1) * self.cfg.classes_per_task
        owners = {}                                         # slot -> its own task(s)
        for t in seen_tasks:
            s = self.task_to_slot.get(t)
            if s in self.slot_ids: owners.setdefault(s, []).append(t)
        base = {t: pool_task_loss(vit, self, self.slot_ids, val_loaders[t], self.cfg, seen_hi)
                for t in seen_tasks}
        util = {}
        for sid in self.slot_ids:
            keep = [s for s in self.slot_ids if s != sid]
            tasks = owners.get(sid, seen_tasks)             # its own task(s); fallback = all
            inc = [pool_task_loss(vit, self, keep, val_loaders[t], self.cfg, seen_hi) - base[t]
                   for t in tasks]
            util[sid] = sum(inc)/len(inc) if inc else 0.0   # small increase => slot least needed
        mn = min(util.values())
        cands = [s for s in self.slot_ids if util[s] <= mn + self.cfg.util_eps]
        victim = random.choice(cands)                       # (4) tie-break among near-equal
        self.evict_log.append({"victim": victim, "type": "utility_clean",
                               "util": round(util[victim],4), "ties": len(cands),
                               "blocked_sim": round(sim,4)})
        print(f"  [BoundedPool] merge blocked (sim={sim:.3f}<{self.cfg.merge_threshold}) "
              f"-> CLEAN-UTILITY-EVICT slot {victim} (loss-rise={util[victim]:.4f}, ties={len(cands)})")
        self._drop_slot(victim, merged_into=None)

    def register_task(self, task_id, vit=None, val_loaders=None, seen_tasks=None):
        """Make room for task_id (may free a slot via policy), assign it a fresh slot."""
        if self.cap is None or len(self.slot_ids) < self.cap:
            pass                                            # room available
        elif self.policy == "evict":
            self._random_evict()
        elif self.policy == "merge":
            sim, i, j = self._most_similar(); self._merge_pair(i, j, sim)
        elif self.policy == "hybrid":
            sim, i, j = self._most_similar()
            if sim >= self.cfg.merge_threshold:
                self._merge_pair(i, j, sim)
            elif self.cfg.util_clean:
                self._utility_evict_clean(vit, val_loaders, seen_tasks or [], sim)
            else:
                self._utility_evict(vit, val_loaders, seen_tasks or [], sim)
        else:
            raise ValueError(self.policy)
        self.cur_slot = self._new_slot()
        self.task_to_slot[task_id] = self.cur_slot
        return self.cur_slot

    def frozen_snapshot(self):
        snaps = [copy.deepcopy(self.prompt_of[s]) for s in self.slot_ids]
        for s in snaps:
            for pp in s.parameters(): pp.requires_grad_(False)
        return snaps

    def fedavg_prompt(self, task_id, ps, weights=None):
        if not ps: return
        sid = self.task_to_slot[task_id]
        w = _norm_weights(weights, len(ps))
        with torch.no_grad():
            for name, sp in self.prompt_of[sid].named_parameters():
                stack = torch.stack([dict(p.named_parameters())[name].data for p in ps])
                sp.data.copy_((stack * w.view(-1, *([1]*(stack.dim()-1)))).sum(0))

    def fedavg_classifier(self, clfs, weights=None):
        if not clfs: return
        w = _norm_weights(weights, len(clfs))
        with torch.no_grad():
            for name, sp in self.classifier.named_parameters():
                stack = torch.stack([dict(c.named_parameters())[name].data for c in clfs])
                sp.data.copy_((stack * w.view(-1, *([1]*(stack.dim()-1)))).sum(0))

    def pool_status(self):
        return (f"  [Pool] policy={self.policy} cap={self.cap} | active slots={self.slot_ids} | "
                f"mass={[self.mass_of[s] for s in self.slot_ids]}")


# ═══════════════════════════════════════════════════════════════════════════
#  EVAL helpers
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def pool_val_acc(vit, server, slot_ids, val_loaders, seen_tasks, cfg):
    """Validation accuracy using a given set of slots — used for utility eviction."""
    prompts = [server.prompt_of[s] for s in slot_ids]
    if not prompts or not seen_tasks: return 0.0
    vit.eval()
    seen_hi = (max(seen_tasks)+1) * cfg.classes_per_task
    correct = tot = 0
    for t in seen_tasks:
        for xb, yb in val_loaders[t]:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            q = vit.get_query(xb)
            pk, pv, _ = pool_prompt(q, prompts, None, cfg.use_softmax, cfg.attn_temp)
            feats = vit.forward_prompted(xb, pk, pv)
            pred = server.classifier(feats)[:, :seen_hi].argmax(1)
            correct += (pred == yb).sum().item(); tot += yb.size(0)
    return correct/tot if tot else 0.0


@torch.no_grad()
def pool_task_loss(vit, server, slot_ids, val_loader, cfg, seen_hi):
    """Mean cross-entropy loss on ONE task's val set using the given slots.
    Smooth (loss, not accuracy) and task-isolated -> a clean utility signal that
    is not corrupted by classifier drift on other tasks."""
    prompts = [server.prompt_of[s] for s in slot_ids]
    if not prompts: return float("inf")
    vit.eval(); tot_loss = tot = 0
    for xb, yb in val_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        q = vit.get_query(xb)
        pk, pv, _ = pool_prompt(q, prompts, None, cfg.use_softmax, cfg.attn_temp)
        feats = vit.forward_prompted(xb, pk, pv)
        logits = server.classifier(feats)[:, :seen_hi]
        tot_loss += F.cross_entropy(logits, yb, reduction="sum").item(); tot += yb.size(0)
    return tot_loss/tot if tot else float("inf")


@torch.no_grad()
def evaluate(vit, server, loaders, num_seen, cfg, mask_current=False):
    vit.eval()
    seen_hi = num_seen * cfg.classes_per_task
    prompts_seen = server.active_prompts()                 # bounded: only active slots
    accs = {}
    for t in sorted(loaders):
        correct = tot = 0
        for xb, yb in loaders[t]:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            q = vit.get_query(xb)
            pk, pv, _ = pool_prompt(q, prompts_seen, None, cfg.use_softmax, cfg.attn_temp)
            feats = vit.forward_prompted(xb, pk, pv)
            if mask_current:
                lo = t * cfg.classes_per_task
                pred = server.classifier(feats)[:, lo:lo+cfg.classes_per_task].argmax(1) + lo
            else:
                pred = server.classifier(feats)[:, :seen_hi].argmax(1)
            correct += (pred == yb).sum().item(); tot += yb.size(0)
        accs[t] = correct/tot if tot else 0.0
    return accs


# ═══════════════════════════════════════════════════════════════════════════
#  CLIENT TRAIN  (snap_idx = position of current task's slot in the snapshot)
# ═══════════════════════════════════════════════════════════════════════════
def cosine_lr(step, total, base, warmup):
    if step < warmup: return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base * (1 + math.cos(math.pi * p))

def client_train(vit, cfg, c2loss, loader, task_id, snaps, snap_idx, srv_prev, server_clf, scaler):
    local_p = copy.deepcopy(snaps[snap_idx])
    for p in local_p.parameters(): p.requires_grad_(True)
    local_clf = copy.deepcopy(server_clf)
    for p in local_clf.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(list(local_p.parameters()) + list(local_clf.parameters()),
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
    others = [snaps[i] for i in range(len(snaps)) if i != snap_idx]
    C = cfg.classes_per_task; lo, hi = task_id*C, (task_id+1)*C
    total = cfg.epochs_per_task * max(1, len(loader)); warm = int(cfg.warmup_frac*total); step = 0
    n_k = len(loader.dataset)
    correct = tot = 0
    for _ in range(cfg.epochs_per_task):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            if not cfg.flat_lr:
                for g in opt.param_groups: g["lr"] = cosine_lr(step, total, cfg.lr, warm)
            q = vit.get_query(xb)
            src = [local_p if i == snap_idx else snaps[i] for i in range(len(snaps))]
            pk, pv, _ = pool_prompt(q, src, snap_idx, cfg.use_softmax, cfg.attn_temp)
            with torch.autocast(device_type=DEVICE.type, enabled=cfg.use_amp):
                feats = vit.forward_prompted(xb, pk, pv)
                logits = local_clf(feats)[:, lo:hi]
                loss = F.cross_entropy(logits, yb - lo)
                if cfg.use_c2l: loss = loss + cfg.lambda_c2l * c2loss(local_p, srv_prev, others)
            opt.zero_grad()
            if cfg.use_amp: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
            correct += ((logits.argmax(1)+lo) == yb).sum().item(); tot += yb.size(0); step += 1
    return local_p, local_clf, (correct/tot if tot else 0.0), n_k


# ═══════════════════════════════════════════════════════════════════════════
#  CHECKPOINT  (bounded-pool aware — saves slots, masses, mapping, logs)
# ═══════════════════════════════════════════════════════════════════════════
def ckpt_path(cfg): return f"akarsh_ckpt_{cfg.run_tag}.pt"

def save_ckpt(server, best_acc, history, completed, cfg):
    torch.save({
        "prompt_of":    {sid: p.state_dict() for sid, p in server.prompt_of.items()},
        "mass_of":      server.mass_of,    "slot_ids":     server.slot_ids,
        "task_to_slot": server.task_to_slot, "cur_slot":   server.cur_slot,
        "next":         server._next,      "merge_log":    server.merge_log,
        "evict_log":    server.evict_log,  "classifier":   server.classifier.state_dict(),
        "best_acc":     best_acc, "history": history, "completed": completed,
    }, ckpt_path(cfg))
    print(f"  [ckpt] saved through task {completed+1} -> {ckpt_path(cfg)}")

def load_ckpt(server, cfg):
    if not os.path.exists(ckpt_path(cfg)): return None
    ck = torch.load(ckpt_path(cfg), map_location=DEVICE, weights_only=False)
    server.prompt_of = {}
    for sid, sd in ck["prompt_of"].items():
        tp = TaskPrompt(cfg).to(DEVICE); tp.load_state_dict(sd); server.prompt_of[sid] = tp
    server.mass_of      = ck["mass_of"];      server.slot_ids = ck["slot_ids"]
    server.task_to_slot = ck["task_to_slot"]; server.cur_slot = ck["cur_slot"]
    server._next        = ck["next"]
    server.merge_log    = ck["merge_log"];    server.evict_log = ck["evict_log"]
    server.classifier.load_state_dict(ck["classifier"])
    print(f"  [ckpt] resuming: tasks 1-{ck['completed']+1} already done")
    return ck


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    cfg = make_cfg()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    PAPER = {"iid":(79.43,4.75), "label_skew":(65.45,9.15), "quantity_skew":(81.12,7.75)}
    ref = PAPER.get(cfg.data_split, (None,None)) if cfg.dataset == "cifar100" else (None, None)

    print("="*78)
    dname = "MedMNIST cross-domain" if cfg.dataset == "medmnist" else "CIFAR-100"
    print(f"  akarshprompt BOUNDED | Fed-CPrompt | {dname} | {cfg.num_tasks}x{cfg.classes_per_task}")
    mode = "CENTRALIZED" if cfg.num_clients == 1 else f"FEDERATED({cfg.num_clients})"
    print(f"  {mode} | split={cfg.data_split} | classifier={cfg.classifier} | weights={'softmax' if cfg.use_softmax else 'raw-cosine'}")
    print(f"  rounds/task={cfg.rounds_per_task} epochs={cfg.epochs_per_task} lr={cfg.lr}"
          f" ({'flat' if cfg.flat_lr else 'cosine-sched'}) bs={cfg.batch_size} aug={cfg.augment}")
    if cfg.pool_cap is None:
        print(f"  STORAGE: UNBOUNDED (baseline, keeps all prompts)")
    else:
        print(f"  STORAGE: BOUNDED  cap={cfg.pool_cap}  policy={cfg.pool_policy}"
              + (f"  merge_thresh={cfg.merge_threshold}" if cfg.pool_policy=="hybrid" else ""))
    if ref[0]: print(f"  paper reference (unbounded): acc {ref[0]}% / forgetting {ref[1]}%")
    print(f"  Device: {DEVICE} | AMP: {cfg.use_amp}")
    print("="*78)

    NW = int(os.environ.get("AK_WORKERS", "0"))   # 0 avoids /dev/shm crash on this cluster
    # Worker-only speedups (NO effect on results): keep workers alive across rounds
    # and prefetch more batches so the GPU isn't starved. Only when NW>0.
    LKW = dict(num_workers=NW, pin_memory=True)
    if NW > 0:
        LKW["persistent_workers"] = True
        LKW["prefetch_factor"] = int(os.environ.get("AK_PREFETCH", "4"))
    train_ds, test_ds, train_idx, val_idx, test_idx, global_of = build_datasets(cfg)
    test_loaders = {t: DataLoader(GlobalLabelDS(test_ds, test_idx[t], global_of), 256, shuffle=False,
                                  **LKW) for t in range(cfg.num_tasks)}
    val_loaders  = {t: DataLoader(GlobalLabelDS(train_ds, val_idx[t], global_of), 256, shuffle=False,
                                  **LKW) for t in range(cfg.num_tasks)}

    vit = FrozenViT(cfg).to(DEVICE); server = Server(cfg)
    c2loss = C2Loss(cfg.gamma, cfg.margin)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_acc: Dict[int,float] = {}; history: List = []; start = 0
    ck = load_ckpt(server, cfg)
    if ck: best_acc, history, start = ck["best_acc"], ck["history"], ck["completed"]+1

    def _mk_loaders(splits):
        return [DataLoader(GlobalLabelDS(train_ds, ci, global_of), cfg.batch_size, shuffle=True,
                           **LKW) if len(ci) > 0 else None for ci in splits]

    for t in range(start, cfg.num_tasks):
        t0 = time.time()
        print(f"\n{'-'*78}\n  TASK {t+1}/{cfg.num_tasks}  (classes {t*cfg.classes_per_task}-{(t+1)*cfg.classes_per_task-1})\n{'-'*78}")

        # ── bounded storage: register task (may trigger merge / evict) ──
        server.register_task(t, vit=vit, val_loaders=val_loaders, seen_tasks=list(range(t)))
        snap_idx = server.cur_index()
        print(server.pool_status())

        splits = split_clients(cfg, train_idx[t], cfg.seed+t, train_ds)
        loaders_cur = _mk_loaders(splits)

        best_val, patience, best_state = -1.0, 0, None
        for rnd in range(cfg.rounds_per_task):
            srv_prev = copy.deepcopy(server.prompt_of[server.cur_slot])
            for p in srv_prev.parameters(): p.requires_grad_(False)
            snaps = server.frozen_snapshot()

            ps, taccs, w_s, all_clfs, all_w = [], [], [], [], []
            for cid in range(cfg.num_clients):
                if loaders_cur[cid] is None: continue
                lp, lc, ta, nk = client_train(vit, cfg, c2loss, loaders_cur[cid], t,
                                               snaps, snap_idx, srv_prev, server.classifier, scaler)
                ps.append(lp); all_clfs.append(lc); taccs.append(ta); w_s.append(nk); all_w.append(nk)
            server.fedavg_prompt(t, ps, w_s)
            server.fedavg_classifier(all_clfs, all_w)

            line = f"   round {rnd+1:2d}/{cfg.rounds_per_task} | train {(np.mean(taccs)*100 if taccs else 0.0):5.2f}%"
            if cfg.early_stop and cfg.rounds_per_task > 1:
                va = evaluate(vit, server, {t: val_loaders[t]}, t+1, cfg, mask_current=True)[t]
                line += f" | val {va*100:5.2f}%"
                if va > best_val:
                    best_val, patience = va, 0
                    best_state = (copy.deepcopy(server.prompt_of[server.cur_slot].state_dict()),
                                  copy.deepcopy(server.classifier.state_dict()))
                else:
                    patience += 1
                    if patience >= cfg.early_stop_patience: print(line + "  [early stop]"); break
            print(line)

        if best_state:
            server.prompt_of[server.cur_slot].load_state_dict(best_state[0])
            server.classifier.load_state_dict(best_state[1])

        accs = evaluate(vit, server, {k: test_loaders[k] for k in range(t+1)}, t+1, cfg)
        for k, v in accs.items(): best_acc[k] = max(best_acc.get(k, 0.0), v)
        avg = np.mean(list(accs.values()))*100
        forget = np.mean([best_acc[k]-accs[k] for k in range(t)])*100 if t > 0 else 0.0
        history.append((t+1, avg, forget))
        print(f"   >> Avg acc 1-{t+1}: {avg:5.2f}% | Forgetting: {forget:5.2f}% | "
              f"active_slots={len(server.slot_ids)} | {(time.time()-t0)/60:.1f} min")
        save_ckpt(server, best_acc, history, t, cfg)
        torch.cuda.empty_cache()

    final = evaluate(vit, server, {k: test_loaders[k] for k in range(cfg.num_tasks)}, cfg.num_tasks, cfg)
    print("\n" + "="*78)
    store = "unbounded" if cfg.pool_cap is None else f"cap={cfg.pool_cap}/{cfg.pool_policy}"
    print(f"  FINAL SUMMARY  ({cfg.data_split}, storage={store})")
    print("="*78)
    print("   Task | Best Acc | Final Acc | Forgetting")
    print("   " + "-"*46)
    for k in range(cfg.num_tasks):
        print(f"   {k+1:>3}  | {best_acc[k]*100:7.2f}% | {final[k]*100:8.2f}% | {(best_acc[k]-final[k])*100:8.2f}%")
    print("   " + "-"*46)
    print(f"   AVG  | {np.mean(list(best_acc.values()))*100:7.2f}% | {np.mean(list(final.values()))*100:8.2f}% |"
          f" {np.mean([(best_acc[k]-final[k])*100 for k in range(cfg.num_tasks)]):8.2f}%")

    if cfg.pool_cap is not None:
        print(f"\n  STORAGE EVENTS  (cap={cfg.pool_cap}, policy={cfg.pool_policy})")
        print(f"   merges: {len(server.merge_log)} | evictions: {len(server.evict_log)}")
        for m in server.merge_log: print(f"     MERGE {m['absorbed']}->{m['keep']} sim={m['sim']} mass->{m['new_mass']}")
        for e in server.evict_log: print(f"     EVICT slot {e['victim']} ({e['type']})")
        print(f"   final active slots: {server.slot_ids} (mass {[server.mass_of[s] for s in server.slot_ids]})")

    with open(f"{cfg.run_tag}_history.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["task","avg_acc","forgetting"]); w.writerows(history)
    print(f"\n  saved -> {cfg.run_tag}_history.csv")


if __name__ == "__main__":
    main()
