from __future__ import annotations

import argparse
import hashlib
import logging
import platform
try:
    import resource
except ImportError:  # Windows
    resource = None
import time
import json
import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import hypergeom, wilcoxon
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from statsmodels.stats.multitest import multipletests

LOGGER = logging.getLogger("pan_gam")


# -----------------------------------------------------------------------------
# Basic I/O helpers
# -----------------------------------------------------------------------------


def configure_logging(outdir: str, verbose: bool = False) -> None:
    ensure_dir(outdir)
    level = logging.DEBUG if verbose else logging.INFO
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream.setLevel(level)
    LOGGER.addHandler(stream)
    fh = logging.FileHandler(os.path.join(outdir, "pan_gam.log"), encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    LOGGER.addHandler(fh)


def file_sha256(path: str, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def peak_rss_gb() -> float:
    """Return peak/current resident memory in GB on Linux, macOS, or Windows."""
    if resource is not None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return float(rss / (1024 ** 3))
        return float(rss / (1024 ** 2))
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3))
    except Exception:
        return float("nan")


def software_manifest() -> dict:
    import scipy
    import sklearn
    import statsmodels
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
    }

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_table_auto(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    sep = "\t" if path.lower().endswith((".tsv", ".tab", ".rtab")) else ","
    return pd.read_csv(path, sep=sep)


def write_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_roary_rtab(path: str) -> pd.DataFrame:
    """Read Roary .Rtab and return isolates x genes binary matrix."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df.fillna(0)
    df[df != 0] = 1
    df = df.astype(np.uint8)
    out = df.T
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    return out


def clean_metadata(path: str) -> pd.DataFrame:
    meta = pd.read_csv(path)
    required = ["isolate_id", "species", "cohort"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")
    for col in required:
        meta[col] = meta[col].astype(str)
    return meta


def align_matrix_and_metadata(x: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ids = [i for i in x.index.astype(str) if i in set(meta["isolate_id"].astype(str))]
    if not ids:
        raise ValueError("No overlapping isolate IDs between Roary matrix and metadata.")
    x = x.loc[ids].copy()
    meta = meta.set_index("isolate_id").loc[ids].reset_index()
    return x, meta


def split_cohorts(x: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_ids = meta.loc[meta["cohort"].str.lower() == "discovery", "isolate_id"].tolist()
    test_ids = meta.loc[meta["cohort"].str.lower() == "external", "isolate_id"].tolist()
    if not train_ids or not test_ids:
        raise ValueError("Both discovery and external cohorts must be present for each species.")
    x_train = x.loc[train_ids]
    x_test = x.loc[test_ids]
    y_train = meta.set_index("isolate_id").loc[train_ids]
    y_test = meta.set_index("isolate_id").loc[test_ids]
    return x_train, x_test, y_train, y_test


# -----------------------------------------------------------------------------
# Phenotypic partitioning
# -----------------------------------------------------------------------------

def phenotype_partition_table(y: pd.DataFrame, drug_cols: Sequence[str], min_group_size: int = 2) -> pd.DataFrame:
    """Create phenotype group table matching the manuscript's C_k grouping logic."""
    y_bin = y[list(drug_cols)].astype(int).copy()
    rows = []
    grouped = y_bin.groupby(list(drug_cols), dropna=False, sort=True)
    group_id = 0
    for signature, idx_df in grouped:
        if not isinstance(signature, tuple):
            signature = (signature,)
        isolate_ids = idx_df.index.astype(str).tolist()
        group_size = len(isolate_ids)
        retained = int(group_size >= min_group_size)
        resistance_count = int(sum(signature))
        rows.append({
            "group_id": f"C_{group_id:04d}",
            "signature": ";".join(f"{d}={v}" for d, v in zip(drug_cols, signature)),
            "n_isolates": group_size,
            "n_resistant_drugs": resistance_count,
            "is_pan_susceptible": int(resistance_count == 0),
            "retained_min_size": retained,
            "isolate_ids": ",".join(isolate_ids),
        })
        group_id += 1
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CTU construction
# -----------------------------------------------------------------------------

def ctu_cluster(x_train: pd.DataFrame, eps: float = 0.05, method: str = "average") -> pd.Series:
    """Cluster genes into CTUs using Jaccard distance among gene columns."""
    genes = list(x_train.columns.astype(str))
    arr = x_train.values.astype(bool)
    if arr.shape[1] == 0:
        raise ValueError("No gene columns available for CTU clustering.")
    if arr.shape[1] < 2:
        return pd.Series(np.ones(arr.shape[1], dtype=int), index=genes, name="ctu_cluster")
    d = pdist(arr.T, metric="jaccard")
    z = linkage(d, method=method)
    labels = fcluster(z, t=eps, criterion="distance")
    return pd.Series(labels.astype(int), index=genes, name="ctu_cluster")


def ctu_membership_table(ctu_map: pd.Series, species: str) -> pd.DataFrame:
    rows = []
    for gene, cid in ctu_map.items():
        rows.append({
            "species": species,
            "gene": str(gene),
            "ctu_cluster": int(cid),
            "ctu": f"{species}_CTU_{int(cid):05d}",
        })
    return pd.DataFrame(rows)


def collapse_to_ctu(x: pd.DataFrame, ctu_map: pd.Series, prefix: str) -> pd.DataFrame:
    """Collapse gene matrix to CTU matrix by OR/max over CTU member genes."""
    ctu_map = ctu_map.copy()
    ctu_map.index = ctu_map.index.astype(str)
    cols = [c for c in ctu_map.index if c in x.columns]
    x = x.reindex(columns=cols, fill_value=0)
    out = {}
    for cid, gene_index in ctu_map.loc[cols].groupby(ctu_map.loc[cols]).groups.items():
        name = f"{prefix}_CTU_{int(cid):05d}"
        out[name] = x.loc[:, list(gene_index)].max(axis=1).astype(np.uint8)
    return pd.DataFrame(out, index=x.index)


# -----------------------------------------------------------------------------
# Association testing
# -----------------------------------------------------------------------------

def wilson_lower_bound(pos: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = pos / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    adj = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return float((centre - adj) / denom)


def pan_gam_association(
    x_ctu: pd.DataFrame,
    y: pd.DataFrame,
    drug_cols: Sequence[str],
    alpha: float = 0.05,
    ppv_lower_min: float = 0.90,
    nmin: int = 10,
) -> pd.DataFrame:
    rows = []
    y_drug = y[list(drug_cols)].astype(int)
    for drug in drug_cols:
        target = y_drug[drug].values
        resistant = target == 1
        specific_control = (target == 0) & (y_drug.drop(columns=[drug]).sum(axis=1).values > 0)
        fallback_used = 0
        if specific_control.sum() < nmin:
            pan_sus = y_drug.sum(axis=1).values == 0
            specific_control = specific_control | pan_sus
            fallback_used = 1
        n_r = int(resistant.sum())
        n_s = int(specific_control.sum())
        if n_r == 0 or n_s == 0:
            continue
        pvals = []
        stat_rows = []
        for ctu in x_ctu.columns:
            v = x_ctu[ctu].values.astype(int)
            n11 = int(v[resistant].sum())
            n01 = int(v[specific_control].sum())
            n10 = n_r - n11
            n00 = n_s - n01
            n_total = n11 + n10 + n01 + n00
            n_present = n11 + n01
            # Upper-tail hypergeometric probability P[X >= n11]
            p = float(hypergeom.sf(n11 - 1, n_total, n_present, n_r))
            ppv = n11 / (n11 + n01) if (n11 + n01) > 0 else 0.0
            ppv_lb = wilson_lower_bound(n11, n11 + n01)
            pvals.append(p)
            stat_rows.append((drug, ctu, n11, n10, n01, n00, p, ppv, ppv_lb, n_r, n_s, fallback_used))
        qvals = multipletests(pvals, alpha=alpha, method="fdr_bh")[1] if pvals else []
        for r, q in zip(stat_rows, qvals):
            drug, ctu, n11, n10, n01, n00, p, ppv, ppv_lb, n_r, n_s, fallback_used = r
            keep = (q < alpha) and (ppv_lb >= ppv_lower_min)
            rows.append({
                "drug": drug,
                "ctu": ctu,
                "n_resistant": n_r,
                "n_specific_control": n_s,
                "fallback_pan_susceptible_used": fallback_used,
                "n11_present_resistant": n11,
                "n10_absent_resistant": n10,
                "n01_present_control": n01,
                "n00_absent_control": n00,
                "pvalue": p,
                "qvalue": float(q),
                "ppv": float(ppv),
                "ppv_lower": float(ppv_lb),
                "selected": int(keep),
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Model training, external validation, bootstrap CIs
# -----------------------------------------------------------------------------

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": safe_auc(y_true, y_prob) if y_prob is not None else np.nan,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def bootstrap_metric_ci(
    pred_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 7,
    metric_cols: Sequence[str] = ("accuracy", "sensitivity", "specificity", "ppv", "f1", "auc"),
) -> pd.DataFrame:
    """Bootstrap 95% CIs within each species/drug prediction table."""
    if pred_df.empty or n_bootstrap <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    group_cols = [c for c in ["species", "drug"] if c in pred_df.columns]
    for keys, g in pred_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        n = len(g)
        vals = {m: [] for m in metric_cols}
        y = g["y_true"].values.astype(int)
        pred = g["y_pred"].values.astype(int)
        prob = g["y_prob"].values.astype(float) if "y_prob" in g else None
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            yy = y[idx]
            pp = pred[idx]
            pr = prob[idx] if prob is not None else None
            met = binary_metrics(yy, pp, pr)
            for m in metric_cols:
                vals[m].append(met.get(m, np.nan))
        for m in metric_cols:
            arr = np.asarray(vals[m], dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                lo = hi = mean = np.nan
            else:
                lo, hi = np.percentile(arr, [2.5, 97.5])
                mean = np.mean(arr)
            rows.append({**key_dict, "metric": m, "bootstrap_mean": mean, "ci_lower": lo, "ci_upper": hi, "n_bootstrap": n_bootstrap})
    return pd.DataFrame(rows)


def tune_gbdt_model(x: np.ndarray, y: np.ndarray, seed: int = 7) -> Tuple[GradientBoostingClassifier, dict]:
    counts = np.bincount(y.astype(int), minlength=2)
    min_class = int(counts.min())
    if min_class < 2:
        clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=seed)
        clf.fit(x, y)
        return clf, {"tuned": False, "reason": "not enough samples per class"}
    n_splits = min(10, min_class)
    clf = GradientBoostingClassifier(random_state=seed)
    param_grid = {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.10],
        "max_depth": [2, 3],
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(clf, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=1, error_score="raise")
    search.fit(x, y)
    return search.best_estimator_, {"tuned": True, "best_params": search.best_params_, "best_cv_auc": float(search.best_score_), "cv_splits": n_splits}


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    assoc: pd.DataFrame,
    drug_cols: Sequence[str],
    species: str,
    tune: bool = True,
    seed: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, dict]]:
    results = []
    pred_rows = []
    models = {}
    for drug in drug_cols:
        if assoc.empty or "selected" not in assoc:
            continue
        selected = assoc.loc[(assoc["drug"] == drug) & (assoc["selected"] == 1), "ctu"].dropna().unique().tolist()
        selected = [c for c in selected if c in x_train.columns]
        if len(selected) == 0:
            continue
        yt = y_train[drug].astype(int).values
        yv = y_test[drug].astype(int).values
        if len(np.unique(yt)) < 2 or len(np.unique(yv)) < 2:
            continue
        if tune:
            clf, tuning_info = tune_gbdt_model(x_train[selected].values, yt, seed=seed)
        else:
            clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=seed)
            clf.fit(x_train[selected].values, yt)
            tuning_info = {"tuned": False, "fixed_params": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3}}
        pred = clf.predict(x_test[selected].values)
        try:
            prob = clf.predict_proba(x_test[selected].values)[:, 1]
        except Exception:
            prob = np.full(len(pred), np.nan)
        met = binary_metrics(yv, pred, prob)
        results.append({
            "species": species,
            "drug": drug,
            "n_features": len(selected),
            **met,
            "tuning_info": json.dumps(tuning_info),
        })
        for iso, yy, pp, pr in zip(x_test.index.astype(str), yv, pred, prob):
            pred_rows.append({"species": species, "drug": drug, "isolate_id": iso, "y_true": int(yy), "y_pred": int(pp), "y_prob": float(pr) if not np.isnan(pr) else np.nan})
        models[drug] = {"model": clf, "features": selected, "tuning_info": tuning_info}
    return pd.DataFrame(results), pd.DataFrame(pred_rows), models


# -----------------------------------------------------------------------------
# Functional audit and curated driver recovery
# -----------------------------------------------------------------------------

def normalize_gene_annotations(ann: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if ann is None:
        return None
    ann = ann.copy()
    ann.columns = [c.strip() for c in ann.columns]
    required = ["species", "gene"]
    missing = [c for c in required if c not in ann.columns]
    if missing:
        raise ValueError(f"Gene annotation table is missing required columns: {missing}")
    if "category" not in ann.columns:
        ann["category"] = "unclassified"
    if "target_drug" not in ann.columns:
        ann["target_drug"] = ""
    ann["species"] = ann["species"].astype(str)
    ann["gene"] = ann["gene"].astype(str)
    ann["category"] = ann["category"].astype(str).str.lower().str.replace(" ", "_")
    ann["target_drug"] = ann["target_drug"].astype(str)
    return ann


def functional_audit(
    assoc: pd.DataFrame,
    membership: pd.DataFrame,
    annotations: Optional[pd.DataFrame],
    species: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Audit selected CTUs for target drivers, passenger genes, MGE/backbone genes, and non-target AMR genes."""
    if annotations is None or assoc.empty or membership.empty:
        return pd.DataFrame(), pd.DataFrame()
    ann = normalize_gene_annotations(annotations)
    ann_s = ann.loc[ann["species"].astype(str) == str(species)].copy()
    mem = membership.merge(ann_s, on=["species", "gene"], how="left")
    mem["category"] = mem["category"].fillna("unannotated")
    mem["target_drug"] = mem["target_drug"].fillna("")
    selected = assoc.loc[assoc["selected"] == 1, ["drug", "ctu", "qvalue", "ppv"]].copy()
    rows = []
    gene_rows = []
    for _, hit in selected.iterrows():
        drug = str(hit["drug"])
        ctu = str(hit["ctu"])
        genes = mem.loc[mem["ctu"] == ctu].copy()
        if genes.empty:
            continue
        # target-concordant if explicitly categorized or target_drug matches current drug
        target_match = genes["target_drug"].astype(str).str.lower().eq(drug.lower())
        cat = genes["category"].astype(str).str.lower()
        is_target_driver = cat.str.contains("target_concordant") | (cat.str.contains("amr") & target_match)
        is_regulatory = cat.str.contains("regulatory")
        is_mobility = cat.str.contains("mobility") | cat.str.contains("plasmid") | cat.str.contains("backbone") | cat.str.contains("transpos") | cat.str.contains("insertion")
        is_passenger = cat.str.contains("passenger")
        is_amr = cat.str.contains("amr") | cat.str.contains("resistance")
        is_non_target_amr = is_amr & (~target_match) & (~cat.str.contains("target_concordant"))
        genes = genes.assign(
            audited_drug=drug,
            is_target_concordant_driver=is_target_driver.astype(int),
            is_resistance_regulatory=is_regulatory.astype(int),
            is_mobility_or_backbone=is_mobility.astype(int),
            is_same_mge_passenger=is_passenger.astype(int),
            is_non_target_amr=is_non_target_amr.astype(int),
        )
        gene_rows.append(genes)
        rows.append({
            "species": species,
            "drug": drug,
            "ctu": ctu,
            "n_genes": int(len(genes)),
            "n_target_concordant_drivers": int(is_target_driver.sum()),
            "n_resistance_regulatory": int(is_regulatory.sum()),
            "n_mobility_or_backbone": int(is_mobility.sum()),
            "n_same_mge_passengers": int(is_passenger.sum()),
            "has_same_mge_passenger": int(is_passenger.any()),
            "n_non_target_amr": int(is_non_target_amr.sum()),
            "has_non_target_amr": int(is_non_target_amr.any()),
            "qvalue": hit.get("qvalue", np.nan),
            "ppv": hit.get("ppv", np.nan),
        })
    ctu_audit = pd.DataFrame(rows)
    gene_audit = pd.concat(gene_rows, ignore_index=True) if gene_rows else pd.DataFrame()
    return ctu_audit, gene_audit


def audit_summary(ctu_audit: pd.DataFrame) -> pd.DataFrame:
    if ctu_audit.empty:
        return pd.DataFrame()
    total_ctus = len(ctu_audit)
    total_genes = int(ctu_audit["n_genes"].sum())
    rows = [{
        "n_significant_ctus": total_ctus,
        "ctus_with_same_mge_passenger": int(ctu_audit["has_same_mge_passenger"].sum()),
        "pct_ctus_with_same_mge_passenger": float(100 * ctu_audit["has_same_mge_passenger"].mean()),
        "ctus_with_non_target_amr": int(ctu_audit["has_non_target_amr"].sum()),
        "pct_ctus_with_non_target_amr": float(100 * ctu_audit["has_non_target_amr"].mean()),
        "n_genes_in_significant_ctus": total_genes,
        "n_same_mge_passenger_genes": int(ctu_audit["n_same_mge_passengers"].sum()),
        "pct_same_mge_passenger_genes": float(100 * ctu_audit["n_same_mge_passengers"].sum() / total_genes) if total_genes else np.nan,
    }]
    return pd.DataFrame(rows)


def curated_driver_recovery(assoc: pd.DataFrame, membership: pd.DataFrame, curated: Optional[pd.DataFrame], species: str) -> pd.DataFrame:
    if curated is None or assoc.empty or membership.empty:
        return pd.DataFrame()
    curated = curated.copy()
    required = ["species", "gene", "drug"]
    missing = [c for c in required if c not in curated.columns]
    if missing:
        raise ValueError(f"Curated driver table is missing required columns: {missing}")
    curated["species"] = curated["species"].astype(str)
    curated["gene"] = curated["gene"].astype(str)
    curated["drug"] = curated["drug"].astype(str)
    cur_s = curated.loc[curated["species"] == str(species)].copy()
    if cur_s.empty:
        return pd.DataFrame()
    gene_to_ctu = membership.set_index("gene")["ctu"].to_dict()
    selected_pairs = set(zip(assoc.loc[assoc["selected"] == 1, "drug"].astype(str), assoc.loc[assoc["selected"] == 1, "ctu"].astype(str)))
    rows = []
    for _, r in cur_s.iterrows():
        ctu = gene_to_ctu.get(str(r["gene"]), "")
        recovered = int((str(r["drug"]), ctu) in selected_pairs)
        rows.append({"species": species, "drug": r["drug"], "gene": r["gene"], "ctu": ctu, "recovered": recovered})
    return pd.DataFrame(rows)


def driver_recovery_summary(rec: pd.DataFrame) -> pd.DataFrame:
    if rec.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in rec.groupby(["species", "drug"]):
        species, drug = keys
        rows.append({
            "species": species,
            "drug": drug,
            "n_curated": int(len(g)),
            "n_recovered": int(g["recovered"].sum()),
            "retention_rate_pct": float(100 * g["recovered"].mean()) if len(g) else np.nan,
        })
    all_row = {
        "species": "ALL",
        "drug": "ALL",
        "n_curated": int(len(rec)),
        "n_recovered": int(rec["recovered"].sum()),
        "retention_rate_pct": float(100 * rec["recovered"].mean()) if len(rec) else np.nan,
    }
    rows.append(all_row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Baseline comparison and Wilcoxon tests
# -----------------------------------------------------------------------------

def compare_with_baselines(pangam_perf: pd.DataFrame, baseline_results: Optional[pd.DataFrame], outdir: str) -> pd.DataFrame:
    if baseline_results is None or baseline_results.empty:
        return pd.DataFrame()
    base = baseline_results.copy()
    if "method" not in base.columns or "drug" not in base.columns:
        raise ValueError("Baseline results must contain at least columns: method, drug")
    # Pan-GAM summary by drug across species
    pan = pangam_perf.groupby("drug", as_index=False)[["sensitivity", "specificity", "ppv", "f1", "auc", "accuracy"]].mean()
    pan["method"] = "Pan-GAM"
    all_res = pd.concat([base, pan], ignore_index=True, sort=False)
    rows = []
    metrics = [m for m in ["sensitivity", "specificity", "ppv", "f1", "auc", "accuracy", "false_positive_hits_per_drug", "significant_hits_per_drug"] if m in all_res.columns]
    for method in sorted(set(base["method"].astype(str))):
        b = base.loc[base["method"].astype(str) == method]
        merged = pan.merge(b, on="drug", suffixes=("_pangam", "_baseline"))
        for m in metrics:
            a_col = f"{m}_pangam"
            b_col = f"{m}_baseline"
            if a_col not in merged.columns or b_col not in merged.columns:
                continue
            a = merged[a_col].astype(float).values
            bvals = merged[b_col].astype(float).values
            mask = ~(np.isnan(a) | np.isnan(bvals))
            if mask.sum() < 2:
                p = np.nan
                stat = np.nan
            else:
                try:
                    stat, p = wilcoxon(a[mask], bvals[mask])
                except Exception:
                    stat, p = np.nan, np.nan
            rows.append({"baseline_method": method, "metric": m, "n_paired_drugs": int(mask.sum()), "wilcoxon_statistic": stat, "pvalue": p})
    comp = pd.DataFrame(rows)
    all_res.to_csv(os.path.join(outdir, "method_comparison_table.tsv"), sep="\t", index=False)
    comp.to_csv(os.path.join(outdir, "wilcoxon_baseline_comparisons.tsv"), sep="\t", index=False)
    return comp


# -----------------------------------------------------------------------------
# Optional external-tool command templates
# -----------------------------------------------------------------------------

def write_external_tool_templates(outdir: str, drug_cols: Sequence[str]) -> None:
    """Write command templates for pyseer/DBGWAS/SNP-LMM; actual execution requires installed tools and covariates."""
    lines = [
        "# External baseline command templates", "",
        "# These commands are templates only. They require properly formatted phenotype, covariate,", 
        "# kinship/core-distance, k-mer/unitig, and Roary gene presence/absence files.", "",
        "# pyseer-Gene/LMM example for each drug:",
    ]
    for drug in drug_cols:
        lines.append(
            f"pyseer --phenotypes phenotypes_{drug}.tsv --pres gene_presence_absence.Rtab "
            f"--similarity core_distance.tsv --lmm --output-patterns pyseer_gene_{drug}.tsv"
        )
    lines.extend([
        "", "# DBGWAS is normally run outside Python using its own graph/unitig workflow.",
        "# Example:",
        "# dbgwas -strains strains.txt -newick tree.nwk -vcf unitigs.vcf -phenotypes phenotypes.tsv -out dbgwas_out", "",
        "# SNP-LMM can be run with pyseer SNP input or GEMMA/limix-style workflows depending on SNP matrix format.",
    ])
    with open(os.path.join(outdir, "external_baseline_command_templates.sh"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Species run and complete pipeline
# -----------------------------------------------------------------------------

@dataclass
class SpeciesRunOutput:
    perf: pd.DataFrame
    predictions: pd.DataFrame
    assoc: pd.DataFrame
    summary: dict
    membership: pd.DataFrame
    phenotype_groups: pd.DataFrame
    audit_ctu: pd.DataFrame
    audit_gene: pd.DataFrame
    driver_recovery: pd.DataFrame


def run_species(
    species: str,
    rtab_path: str,
    meta: pd.DataFrame,
    drug_cols: Sequence[str],
    outdir: str,
    eps: float,
    annotations: Optional[pd.DataFrame] = None,
    curated_drivers: Optional[pd.DataFrame] = None,
    tune_gbdt: bool = True,
    seed: int = 7,
) -> SpeciesRunOutput:
    ensure_dir(outdir)
    x = read_roary_rtab(rtab_path)
    meta_s = meta.loc[meta["species"] == species].copy()
    x, meta_s = align_matrix_and_metadata(x, meta_s)
    x_train, x_test, y_train, y_test = split_cohorts(x, meta_s)

    phenotype_groups = phenotype_partition_table(y_train, drug_cols)
    phenotype_groups.insert(0, "species", species)

    profile_rows = []

    t0 = time.perf_counter()
    ctu_map = ctu_cluster(x_train, eps=eps)
    profile_rows.append({
        "species": species,
        "stage": "jaccard_and_hierarchical_clustering",
        "wall_seconds": time.perf_counter() - t0,
        "peak_rss_gb": peak_rss_gb(),
        "n_isolates": int(x_train.shape[0]),
        "n_features": int(x_train.shape[1]),
        "n_pairwise_distances": int(x_train.shape[1] * (x_train.shape[1] - 1) // 2),
    })
    membership = ctu_membership_table(ctu_map, species)

    t0 = time.perf_counter()
    x_train_ctu = collapse_to_ctu(x_train, ctu_map, species)
    x_test_ctu = collapse_to_ctu(x_test, ctu_map, species)
    profile_rows.append({
        "species": species,
        "stage": "ctu_matrix_construction",
        "wall_seconds": time.perf_counter() - t0,
        "peak_rss_gb": peak_rss_gb(),
        "n_isolates": int(x_train.shape[0]),
        "n_features": int(x_train.shape[1]),
        "n_pairwise_distances": None,
    })
    x_test_ctu = x_test_ctu.reindex(columns=x_train_ctu.columns, fill_value=0)

    t0 = time.perf_counter()
    assoc = pan_gam_association(x_train_ctu, y_train, drug_cols=drug_cols, alpha=0.05, ppv_lower_min=0.90, nmin=10)
    profile_rows.append({
        "species": species,
        "stage": "association_testing",
        "wall_seconds": time.perf_counter() - t0,
        "peak_rss_gb": peak_rss_gb(),
        "n_isolates": int(x_train.shape[0]),
        "n_features": int(x_train_ctu.shape[1]),
        "n_pairwise_distances": None,
    })

    t0 = time.perf_counter()
    perf, predictions, models = train_models(x_train_ctu, y_train, x_test_ctu, y_test, assoc, drug_cols, species=species, tune=tune_gbdt, seed=seed)
    profile_rows.append({
        "species": species,
        "stage": "model_training_and_external_validation",
        "wall_seconds": time.perf_counter() - t0,
        "peak_rss_gb": peak_rss_gb(),
        "n_isolates": int(x_train.shape[0] + x_test.shape[0]),
        "n_features": int(x_train_ctu.shape[1]),
        "n_pairwise_distances": None,
    })

    audit_ctu, audit_gene = functional_audit(assoc, membership, annotations, species)
    recovery = curated_driver_recovery(assoc, membership, curated_drivers, species)

    species_dir = os.path.join(outdir, species)
    ensure_dir(species_dir)
    membership.to_csv(os.path.join(species_dir, f"{species}_ctu_membership.tsv"), sep="\t", index=False)
    ctu_map.to_csv(os.path.join(species_dir, f"{species}_ctu_map.tsv"), sep="\t", header=["ctu_cluster"])
    phenotype_groups.to_csv(os.path.join(species_dir, f"{species}_phenotype_groups.tsv"), sep="\t", index=False)
    x_train_ctu.to_csv(os.path.join(species_dir, f"{species}_discovery_ctu_matrix.tsv"), sep="\t")
    x_test_ctu.to_csv(os.path.join(species_dir, f"{species}_external_ctu_matrix.tsv"), sep="\t")
    assoc.to_csv(os.path.join(species_dir, f"{species}_pangam_association.tsv"), sep="\t", index=False)
    perf.to_csv(os.path.join(species_dir, f"{species}_external_performance.tsv"), sep="\t", index=False)
    predictions.to_csv(os.path.join(species_dir, f"{species}_external_predictions.tsv"), sep="\t", index=False)
    if not audit_ctu.empty:
        audit_ctu.to_csv(os.path.join(species_dir, f"{species}_functional_ctu_audit.tsv"), sep="\t", index=False)
    if not audit_gene.empty:
        audit_gene.to_csv(os.path.join(species_dir, f"{species}_functional_gene_audit.tsv"), sep="\t", index=False)
    if not recovery.empty:
        recovery.to_csv(os.path.join(species_dir, f"{species}_curated_driver_recovery.tsv"), sep="\t", index=False)
    dump(models, os.path.join(species_dir, f"{species}_models.joblib"))
    pd.DataFrame(profile_rows).to_csv(
        os.path.join(species_dir, f"{species}_runtime_profile.tsv"),
        sep="\t",
        index=False,
    )

    summary = {
        "species": species,
        "n_discovery": int(x_train.shape[0]),
        "n_external": int(x_test.shape[0]),
        "genes_train": int(x_train.shape[1]),
        "ctus_train": int(x_train_ctu.shape[1]),
        "feature_reduction": float(1 - x_train_ctu.shape[1] / x_train.shape[1]),
        "epsilon": float(eps),
        "n_phenotypic_groups": int(len(phenotype_groups)),
        "n_retained_phenotypic_groups": int(phenotype_groups["retained_min_size"].sum()),
        "n_selected_ctu_drug_hits": int(assoc["selected"].sum()) if not assoc.empty else 0,
    }
    write_json(summary, os.path.join(species_dir, f"{species}_summary.json"))
    return SpeciesRunOutput(perf, predictions, assoc, summary, membership, phenotype_groups, audit_ctu, audit_gene, recovery)


def run_all_species(
    meta: pd.DataFrame,
    species_rtabs: Dict[str, str],
    drug_cols: Sequence[str],
    outdir: str,
    eps: float,
    annotations: Optional[pd.DataFrame] = None,
    curated_drivers: Optional[pd.DataFrame] = None,
    tune_gbdt: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 7,
    baseline_results: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    ensure_dir(outdir)
    outputs: List[SpeciesRunOutput] = []
    for species, rtab_path in species_rtabs.items():
        outputs.append(run_species(species, rtab_path, meta, drug_cols, outdir, eps, annotations, curated_drivers, tune_gbdt, seed))

    perf = pd.concat([o.perf for o in outputs if not o.perf.empty], ignore_index=True) if any(not o.perf.empty for o in outputs) else pd.DataFrame()
    predictions = pd.concat([o.predictions for o in outputs if not o.predictions.empty], ignore_index=True) if any(not o.predictions.empty for o in outputs) else pd.DataFrame()
    assoc = pd.concat([o.assoc.assign(species=o.summary["species"]) for o in outputs if not o.assoc.empty], ignore_index=True) if any(not o.assoc.empty for o in outputs) else pd.DataFrame()
    membership = pd.concat([o.membership for o in outputs], ignore_index=True)
    phenotype_groups = pd.concat([o.phenotype_groups for o in outputs], ignore_index=True)
    audit_ctu = pd.concat([o.audit_ctu for o in outputs if not o.audit_ctu.empty], ignore_index=True) if any(not o.audit_ctu.empty for o in outputs) else pd.DataFrame()
    audit_gene = pd.concat([o.audit_gene for o in outputs if not o.audit_gene.empty], ignore_index=True) if any(not o.audit_gene.empty for o in outputs) else pd.DataFrame()
    recovery = pd.concat([o.driver_recovery for o in outputs if not o.driver_recovery.empty], ignore_index=True) if any(not o.driver_recovery.empty for o in outputs) else pd.DataFrame()
    summaries = pd.DataFrame([o.summary for o in outputs])

    perf.to_csv(os.path.join(outdir, "all_species_external_performance.tsv"), sep="\t", index=False)
    predictions.to_csv(os.path.join(outdir, "all_species_external_predictions.tsv"), sep="\t", index=False)
    assoc.to_csv(os.path.join(outdir, "all_species_pangam_association.tsv"), sep="\t", index=False)
    membership.to_csv(os.path.join(outdir, "all_species_ctu_membership.tsv"), sep="\t", index=False)
    phenotype_groups.to_csv(os.path.join(outdir, "all_species_phenotype_groups.tsv"), sep="\t", index=False)
    summaries.to_csv(os.path.join(outdir, "species_summary.tsv"), sep="\t", index=False)

    ci = bootstrap_metric_ci(predictions, n_bootstrap=n_bootstrap, seed=seed) if not predictions.empty else pd.DataFrame()
    if not ci.empty:
        ci.to_csv(os.path.join(outdir, "bootstrap_external_metric_ci.tsv"), sep="\t", index=False)

    audit_sum = audit_summary(audit_ctu)
    if not audit_ctu.empty:
        audit_ctu.to_csv(os.path.join(outdir, "functional_ctu_audit_all_species.tsv"), sep="\t", index=False)
        audit_gene.to_csv(os.path.join(outdir, "functional_gene_audit_all_species.tsv"), sep="\t", index=False)
        audit_sum.to_csv(os.path.join(outdir, "functional_audit_summary.tsv"), sep="\t", index=False)

    rec_sum = driver_recovery_summary(recovery)
    if not recovery.empty:
        recovery.to_csv(os.path.join(outdir, "curated_driver_recovery_all_species.tsv"), sep="\t", index=False)
        rec_sum.to_csv(os.path.join(outdir, "curated_driver_recovery_summary.tsv"), sep="\t", index=False)

    baseline_comp = compare_with_baselines(perf, baseline_results, outdir) if baseline_results is not None else pd.DataFrame()
    write_external_tool_templates(outdir, drug_cols)

    return {
        "performance": perf,
        "predictions": predictions,
        "association": assoc,
        "membership": membership,
        "phenotype_groups": phenotype_groups,
        "summary": summaries,
        "bootstrap_ci": ci,
        "functional_audit_summary": audit_sum,
        "driver_recovery_summary": rec_sum,
        "baseline_comparison": baseline_comp,
    }


# -----------------------------------------------------------------------------
# Epsilon sensitivity scan
# -----------------------------------------------------------------------------

def run_epsilon_scan(
    meta: pd.DataFrame,
    species_rtabs: Dict[str, str],
    drug_cols: Sequence[str],
    outdir: str,
    eps_values: Sequence[float],
    annotations: Optional[pd.DataFrame] = None,
    curated_drivers: Optional[pd.DataFrame] = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Run threshold analysis using discovery data only.

    The external cohort is never projected, predicted, or evaluated in this
    function, preventing threshold-selection leakage.
    """
    rows = []
    ensure_dir(outdir)
    for eps in eps_values:
        LOGGER.info("Discovery-only epsilon scan: epsilon=%s", eps)
        genes_total = 0
        ctus_total = 0
        selected_total = 0
        all_audit = []
        all_recovery = []

        for species, rtab_path in species_rtabs.items():
            x = read_roary_rtab(rtab_path)
            meta_s = meta.loc[meta["species"] == species].copy()
            x, meta_s = align_matrix_and_metadata(x, meta_s)
            x_train, _, y_train, _ = split_cohorts(x, meta_s)

            ctu_map = ctu_cluster(x_train, eps=eps)
            membership = ctu_membership_table(ctu_map, species)
            x_train_ctu = collapse_to_ctu(x_train, ctu_map, species)
            assoc = pan_gam_association(
                x_train_ctu,
                y_train,
                drug_cols=drug_cols,
                alpha=0.05,
                ppv_lower_min=0.90,
                nmin=10,
            )
            audit_ctu, _ = functional_audit(assoc, membership, annotations, species)
            recovery = curated_driver_recovery(
                assoc, membership, curated_drivers, species
            )

            genes_total += int(x_train.shape[1])
            ctus_total += int(x_train_ctu.shape[1])
            selected_total += int(assoc["selected"].sum()) if not assoc.empty else 0
            if not audit_ctu.empty:
                all_audit.append(audit_ctu)
            if not recovery.empty:
                all_recovery.append(recovery)

        row = {
            "epsilon": float(eps),
            "equivalent_similarity_pct": float((1.0 - eps) * 100.0),
            "genes": genes_total,
            "ctus": ctus_total,
            "feature_reduction_pct": float(100 * (1 - ctus_total / genes_total)) if genes_total else np.nan,
            "n_selected_ctu_drug_hits": selected_total,
        }

        audit_all = pd.concat(all_audit, ignore_index=True) if all_audit else pd.DataFrame()
        audit_sum = audit_summary(audit_all)
        if not audit_sum.empty:
            row.update(audit_sum.iloc[0].to_dict())

        rec_all = pd.concat(all_recovery, ignore_index=True) if all_recovery else pd.DataFrame()
        rec_sum = driver_recovery_summary(rec_all)
        if not rec_sum.empty:
            overall = rec_sum.loc[(rec_sum["species"] == "ALL") & (rec_sum["drug"] == "ALL")]
            if not overall.empty:
                row.update({
                    "curated_drivers": int(overall.iloc[0]["n_curated"]),
                    "curated_drivers_recovered": int(overall.iloc[0]["n_recovered"]),
                    "driver_retention_rate_pct": float(overall.iloc[0]["retention_rate_pct"]),
                })
        rows.append(row)

    scan = pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)
    scan.to_csv(os.path.join(outdir, "epsilon_sensitivity_summary.tsv"), sep="\t", index=False)
    return scan


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_species_rtabs(args: argparse.Namespace) -> Dict[str, str]:
    species_rtabs = {}
    if args.kp_rtab:
        species_rtabs["Kp"] = args.kp_rtab
    if args.sa_rtab:
        species_rtabs["Sa"] = args.sa_rtab
    if args.species_rtabs:
        # Format: Kp:path/to/rtab,Sa:path/to/rtab or Species=path
        for item in args.species_rtabs.split(","):
            if not item.strip():
                continue
            if ":" in item:
                sp, path = item.split(":", 1)
            elif "=" in item:
                sp, path = item.split("=", 1)
            else:
                raise ValueError("--species-rtabs entries must be Species:path or Species=path")
            species_rtabs[sp.strip()] = path.strip()
    if not species_rtabs:
        raise ValueError("Provide --kp-rtab/--sa-rtab or --species-rtabs.")
    return species_rtabs


def main() -> None:
    parser = argparse.ArgumentParser(description="Pan-GAM pipeline with analyses.")
    parser.add_argument("--metadata", required=True, help="Metadata CSV with isolate_id, species, cohort, and drug phenotype columns.")
    parser.add_argument("--kp-rtab", default=None, help="Roary .Rtab for K. pneumoniae with species label Kp.")
    parser.add_argument("--sa-rtab", default=None, help="Roary .Rtab for S. aureus with species label Sa.")
    parser.add_argument("--species-rtabs", default=None, help="Alternative species map, e.g. Kp:data/kp.rtab,Sa:data/sa.rtab")
    parser.add_argument("--drugs", required=True, help="Comma-separated drug phenotype columns.")
    parser.add_argument("--outdir", default="results", help="Output directory.")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Jaccard distance threshold for CTU clustering.")
    parser.add_argument("--epsilon-scan", default=None, help="Comma-separated epsilon values, e.g. 0.01,0.03,0.05,0.10,0.15,0.20")
    parser.add_argument("--gene-annotations", default=None, help="Optional CSV/TSV with species,gene,category,target_drug for AMR/MGE/passenger audit.")
    parser.add_argument("--curated-drivers", default=None, help="Optional CSV/TSV with species,gene,drug for driver retention analysis.")
    parser.add_argument("--baseline-results", default=None, help="Optional CSV/TSV with external baseline results for pyseer/DBGWAS/SNP-LMM comparisons.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap iterations for 95%% external metric CIs. Use 0 to skip.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-tune-gbdt", action="store_true", help="Use fixed GBDT parameters instead of 10-fold CV grid search.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    ensure_dir(args.outdir)
    configure_logging(args.outdir, args.verbose)
    LOGGER.info("Starting Pan-GAM analysis")
    write_json(software_manifest(), os.path.join(args.outdir, "software_manifest.json"))

    meta = clean_metadata(args.metadata)
    drug_cols = [x.strip() for x in args.drugs.split(",") if x.strip()]
    missing_drugs = [d for d in drug_cols if d not in meta.columns]
    if missing_drugs:
        raise ValueError(f"Drug columns not found in metadata: {missing_drugs}")

    species_rtabs = parse_species_rtabs(args)
    input_manifest = {
        "metadata": {"path": args.metadata, "sha256": file_sha256(args.metadata)},
        "species_rtabs": {sp: {"path": path, "sha256": file_sha256(path)} for sp, path in species_rtabs.items()},
    }
    write_json(input_manifest, os.path.join(args.outdir, "input_manifest.json"))
    annotations = read_table_auto(args.gene_annotations)
    curated = read_table_auto(args.curated_drivers)
    baseline_results = read_table_auto(args.baseline_results)
    tune_gbdt = not args.no_tune_gbdt

    if args.epsilon_scan:
        eps_values = [float(x.strip()) for x in args.epsilon_scan.split(",") if x.strip()]
        scan = run_epsilon_scan(
            meta=meta,
            species_rtabs=species_rtabs,
            drug_cols=drug_cols,
            outdir=args.outdir,
            eps_values=eps_values,
            annotations=annotations,
            curated_drivers=curated,
            seed=args.seed,
        )
        print("\nEpsilon sensitivity summary")
        print(scan)
    else:
        res = run_all_species(
            meta=meta,
            species_rtabs=species_rtabs,
            drug_cols=drug_cols,
            outdir=args.outdir,
            eps=args.epsilon,
            annotations=annotations,
            curated_drivers=curated,
            tune_gbdt=tune_gbdt,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            baseline_results=baseline_results,
        )
        print("\nSpecies summary")
        print(res["summary"])
        print("\nExternal performance")
        if not res["performance"].empty:
            print(res["performance"].groupby("species")[["accuracy", "sensitivity", "specificity", "ppv", "f1", "auc"]].mean())
        else:
            print("No evaluable drug model was produced.")
        if not res["functional_audit_summary"].empty:
            print("\nFunctional audit summary")
            print(res["functional_audit_summary"])
        if not res["driver_recovery_summary"].empty:
            print("\nCurated driver recovery summary")
            print(res["driver_recovery_summary"])

    LOGGER.info("Pan-GAM analysis completed successfully")


if __name__ == "__main__":
    main()
