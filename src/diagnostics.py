"""Opt-in SupCon diagnostics (261126): native->mapped label table, pair audit, pair coverage, validation errors.

Nothing here changes a loss, sampler or selection rule; every hook is a no-op unless a PairDiag/diag dir is passed.
Counting rule (identical to SupConLoss, no self-pairs, each unordered pair once): within one SupCon batch, a pair of
rows is POSITIVE iff their (mapped) group labels are equal, NEGATIVE otherwise.
"""
import json
import os
import re

import numpy as np
import torch

# ---------- run/jog and stairs families, matched on NATIVE names (verified against the 36-source pool + 3 targets) ----------
_RUN_EXCLUDE = re.compile(r"shuttle|shuffle|cone|ring|cut|dec")          # 4x10m Shuttle Run, Cone Shuffle Run, RunCut/RunDec ...
_RUN = re.compile(r"(^|[^a-z])(run|running|jog|jogging)([^a-z]|$)")
_STAIRS = re.compile(r"stair|ascend|descend|climb")                         # Climb Stairs, Stairs_up, Upstairs, ascending stairs
_STAIRS_ABBR = {"us": "up", "ds": "down", "stu": "up", "stn": "down"}      # wearable_sensor_dataset (us/ds), mobiact (STU/STN)
# opaque native names (numeric / a<k>): identity verified from the dataset's own documentation / manifest label map
_OVERRIDES = {("wwbs_metrics", 3): ("run_jog", "run"),
              ("smartwatch_gait_dataset", 1): ("stairs", "up"), ("smartwatch_gait_dataset", 2): ("stairs", "down"),
              ("strengthsense", 10): ("stairs", "both")}


def family_of(dataset, native_id, native_name, group_name):
    """(family, sub) for an audited native label, else None."""
    for (pref, nid), v in _OVERRIDES.items():
        if str(dataset).startswith(pref) and int(native_id) == nid:
            return v
    nm = re.sub(r"[_\-]+", " ", str(native_name).lower()).strip()
    low = nm.replace(" ", "")
    if low in _STAIRS_ABBR:
        return ("stairs", _STAIRS_ABBR[low])
    if _STAIRS.search(nm.replace(" ", "")) and "step" not in nm:
        sub = "up" if re.search(r"up|ascend|climb", nm) else "down" if re.search(r"down|descend", nm) else "both"
        return ("stairs", sub)
    if _RUN.search(nm) and not _RUN_EXCLUDE.search(nm):
        return ("run_jog", "jog" if "jog" in nm else "run")
    return None


def native_names(data_root, dataset):
    """{on-disk id: native name} from <data_root>/<dataset>/label_map.json (name -> id), {} if absent."""
    p = os.path.join(data_root or "", dataset, "label_map.json")
    try:
        return {int(v): k for k, v in json.load(open(p)).items()}
    except Exception:
        return {}


def build_label_rows(native, stage, group_to_idx=None, eligible=None, note=None):
    """One row per (dataset, placement, native label). `native` is load_pooled_datasets(return_native=True)[4].
    eligible: {(dataset, native_id): n windows} after the stage's filtering/subsampling (default: loaded windows kept)."""
    rows = []
    g2i = group_to_idx if group_to_idx is not None else native["group_to_idx"]
    for pr in native["pairs"]:
        names = native_names(pr["data_root"], pr["dataset"])
        disk, keep, load_idx = pr["disk_idx"], pr["keep"], pr["loader_idx"]
        seen = {}
        for d, l in zip(disk.tolist(), load_idx.tolist()):
            seen[d] = l
        for d in sorted(set(names) | set(seen)):
            l = seen.get(d)
            nm_raw = names.get(d, str(d))
            if l is None:
                rows.append({"stage": stage, "dataset": pr["dataset"], "placement": pr["placement"], "native_id": d, "native_name": nm_raw,
                             "group_name": None, "group_id": None, "dropped": True,
                             "drop_reason": "no windows in loader output (excluded by USABLE_CLASSES or empty)", "eligible_train_windows": 0})
                continue
            g = pr["idx_to_group"].get(l)
            # opaque on-disk names (numeric activity codes, e.g. PAMAP2 '4') -> use the run's own mapping name as the readable name
            nm = g if (nm_raw.strip().isdigit() and g not in (None, "undefined")) else nm_raw
            n_loaded = int(((disk == d) & keep).sum())
            dropped = g is None or g == "undefined"
            rows.append({"stage": stage, "dataset": pr["dataset"], "placement": pr["placement"], "native_id": d, "native_name": nm, "native_code": nm_raw,
                         "name_source": "label_map.json" if nm == nm_raw else "run label_map (on-disk name is an opaque code)",
                         "loader_idx": l, "remapped_by_usable_classes": pr["remapped_by_usable_classes"],
                         "group_name": g, "group_id": None if dropped else g2i.get(g), "dropped": bool(dropped),
                         "drop_reason": "mapped to 'undefined' (null/transition label)" if dropped else None,
                         "n_loaded_windows": n_loaded if not dropped else int((disk == d).sum()),
                         "eligible_train_windows": 0 if dropped else int((eligible or {}).get((pr["dataset"], d), n_loaded)),
                         **({"note": note} if note else {})})
    return rows


def write_jsonl(path, rows, mode="a"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class PairDiag:
    """Aggregates SupCon pair statistics over batches, dumps one block of rows per epoch/round (end_epoch).

    types: rows of the label table (non-dropped) -> type index per (dataset, native id).
    update(sid, native, group, z, loss_ok) takes ONE SupCon batch: dataset-name ids, native ids, mapped group ids,
    L2-normalised projections (or None), and whether the loss produced a gradient.
    audit_pairs=False skips the T x T matrices (coverage only).
    """

    def __init__(self, out_dir, stage, label_rows, ds_names, audit_pairs=True, device="cpu", extra=None):
        self.dir, self.stage, self.audit, self.dev, self.extra = out_dir, stage, audit_pairs, device, extra or {}
        self.types = [r for r in label_rows if not r["dropped"]]
        self.tid = {(r["dataset"], r["native_id"]): i for i, r in enumerate(self.types)}
        self.ds_names = ds_names                       # {source id (int): dataset name}
        T = len(self.types)
        self.T = T
        self.lut = {}                                  # (source id, native id) -> type index
        for sid, nm in ds_names.items():
            for (d, n), i in self.tid.items():
                if d == nm:
                    self.lut[(sid, n)] = i
        self.fam = [family_of(r["dataset"], r["native_id"], r["native_name"], r["group_name"]) for r in self.types]
        self.reset()

    def reset(self):
        T, dev = self.T, self.dev
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
        self.pos, self.neg, self.negcos = (z(T, T), z(T, T), z(T, T)) if self.audit else (None, None, None)
        self.anchors, self.valid = z(T), z(T)
        self.batch = {"attempted": 0, "zero_positive": 0, "zero_negative": 0, "gradient_producing": 0,
                      "rows": 0, "positive_pairs": 0, "negative_pairs": 0}
        self.ds_batch = {}                              # dataset -> batches containing it / batches where it had no valid anchor

    @torch.no_grad()
    def update(self, sid, native, group, z, loss_ok):
        n = len(group)
        dev = self.dev
        sid, native, group = sid.to(dev), native.to(dev), group.to(dev)
        t = torch.tensor([self.lut.get((int(a), int(b)), -1) for a, b in zip(sid.tolist(), native.tolist())], device=dev)
        same = group[:, None] == group[None, :]
        iu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), 1)
        pos, neg = same & iu, (~same) & iu
        P, N = int(pos.sum()), int(neg.sum())
        b = self.batch
        b["attempted"] += 1; b["rows"] += n; b["positive_pairs"] += P; b["negative_pairs"] += N
        b["zero_positive"] += int(P == 0); b["zero_negative"] += int(N == 0); b["gradient_producing"] += int(bool(loss_ok))
        has_pos = (same & ~torch.eye(n, dtype=torch.bool, device=dev)).any(1)
        ok = t >= 0
        if ok.any():
            O = torch.zeros(n, self.T, dtype=torch.float64, device=dev)
            O[torch.arange(n, device=dev)[ok], t[ok]] = 1.0
            self.anchors += O.sum(0); self.valid += (O * has_pos[:, None].double()).sum(0)
            if self.audit:
                self.pos += O.T @ pos.double() @ O
                self.neg += O.T @ neg.double() @ O
                if z is not None:
                    sim = (z.detach().double() @ z.detach().double().T)
                    self.negcos += O.T @ (neg.double() * sim) @ O
        for s_ in set(sid.tolist()):
            m = sid == s_
            d = self.ds_batch.setdefault(self.ds_names[int(s_)], [0, 0])
            d[0] += 1; d[1] += int(not bool(has_pos[m].any()))

    def end_epoch(self, idx, eligible=None):
        """write pair_audit.jsonl / pair_coverage.jsonl rows for this epoch (or round) and reset."""
        key = "epoch" if "finetune" in self.stage else "round"
        base = {"stage": self.stage, key: idx, **self.extra}
        cov = []
        by_ds = {}
        A, V = self.anchors.cpu().numpy(), self.valid.cpu().numpy()
        for i, r in enumerate(self.types):
            e = by_ds.setdefault(r["dataset"], {"anchors": 0.0, "valid": 0.0, "classes": {}})
            e["anchors"] += float(A[i]); e["valid"] += float(V[i])
            e["classes"][str(r["native_id"])] = {"native_name": r["native_name"], "group": r["group_name"],
                                                 "eligible_train_windows": r.get("eligible_train_windows"),
                                                 "sampled_anchors": float(A[i]), "anchors_with_positive": float(V[i])}
        for d, e in by_ds.items():
            nb = self.ds_batch.get(d, [0, 0])
            cov.append({**base, "source": d, "batches_containing_source": nb[0], "batches_source_without_valid_anchor": nb[1],
                        "total_anchors": e["anchors"], "valid_anchors": e["valid"], "per_native_class": e["classes"]})
        cov.append({**base, "source": "__all_batches__", "batch_counters": dict(self.batch),
                    "note": "batch-level SupCon counters; a pooled batch has ONE shared loss, so gradients are not split per source"})
        write_jsonl(os.path.join(self.dir, "pair_coverage.jsonl"), cov)
        if self.audit:
            P, N, C = self.pos.cpu().numpy(), self.neg.cpu().numpy(), self.negcos.cpu().numpy()
            rows = []
            for a in range(self.T):
                for b_ in range(a, self.T):
                    fa, fb = self.fam[a], self.fam[b_]
                    if fa is None or fb is None or fa[0] != fb[0]:
                        continue                                   # both endpoints must belong to the same audited family
                    if a == b_:
                        p, n_, c = P[a, a], N[a, a], C[a, a]
                    else:
                        p, n_, c = P[a, b_] + P[b_, a], N[a, b_] + N[b_, a], C[a, b_] + C[b_, a]
                    if round(p + n_) == 0:
                        continue                                   # endpoint pair never co-occurred in a batch
                    ra, rb = self.types[a], self.types[b_]
                    rows.append({**base, "family": fa[0], "sub_a": fa[1], "sub_b": fb[1],
                                 "a": {"dataset": ra["dataset"], "placement": ra["placement"], "native_id": ra["native_id"],
                                       "native_name": ra["native_name"], "group": ra["group_name"], "group_id": ra["group_id"]},
                                 "b": {"dataset": rb["dataset"], "placement": rb["placement"], "native_id": rb["native_id"],
                                       "native_name": rb["native_name"], "group": rb["group_name"], "group_id": rb["group_id"]},
                                 "scope": "same_dataset" if ra["dataset"] == rb["dataset"] else "cross_dataset",
                                 "same_native_label": bool(a == b_), "positive_pairs": int(round(p)), "negative_pairs": int(round(n_)),
                                 "negative_cos_sum": float(c), "negative_cos_count": int(round(n_))})
            write_jsonl(os.path.join(self.dir, "pair_audit.jsonl"), rows)
        self.reset()


def validation_errors_payload(tag, epoch, names_by_id, n_classes, confusion, f1_per_class, shared_ids, extra=None, codes_by_id=None):
    """confusion: raw (n_classes x n_classes) rows=true. Support = row sums; zero-support classes are flagged."""
    cm = np.asarray(confusion)
    support = cm.sum(1).astype(int).tolist()
    ids = list(range(n_classes))
    return {"arm": tag, "selected_epoch": epoch,
            "label_ids": ids, "label_names": [names_by_id.get(i, str(i)) for i in ids],
            "label_native_codes": [(codes_by_id or {}).get(i, str(i)) for i in ids],
            "confusion_matrix_rows_true_cols_pred": cm.astype(int).tolist(),
            "per_class_support": support, "per_class_f1": [float(v) for v in f1_per_class],
            "zero_validation_support": [bool(s == 0) for s in support],
            "classes_in_reported_shared_class_f1": shared_ids,
            "note": "full target label list; the reported macro-F1 uses only the shared-class subset, this diagnostic does not filter",
            **(extra or {})}


# ---------- latent-space probe embeddings ----------
@torch.no_grad()
def embed_probe(probe_X, model, proj=None, device="cpu", bs=512):
    """features (model's flattened backbone features) and, if proj is given, the L2-normalised SupCon projection; fp16 numpy."""
    model.eval()
    if proj is not None:
        proj.eval()
    F_, Z_ = [], []
    for j in range(0, len(probe_X), bs):
        x = torch.as_tensor(probe_X[j:j + bs], dtype=torch.float32, device=device)
        f = model(x, return_features=True)[1]
        F_.append(f.float().cpu())
        if proj is not None:
            Z_.append(proj(f).float().cpu())
    out = {"feat": torch.cat(F_).numpy().astype(np.float16)}
    if proj is not None:
        out["proj"] = torch.cat(Z_).numpy().astype(np.float16)
    return out


def save_latent(path, probe_id, emb, **meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, probe_id=np.array(probe_id), **emb, **{k: np.array(v) for k, v in meta.items()})
