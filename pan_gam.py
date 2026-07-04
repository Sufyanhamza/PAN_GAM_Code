import argparse
import os
import json
import warnings

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import hypergeom

from statsmodels.stats.multitest import multipletests

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
     accuracy_score,
     sensitivity_score if False else recall_score,
     precision_score,
     f1_score,
     roc_auc_score,
     confusion_matrix
)

from joblib import dump


def read_roary_rtab(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df.fillna(0)
    df[df != 0] = 1
    df = df.astype(np.uint8)
    return df.T


def clean_metadata(path):
    meta = pd.read_csv(path)
    meta["isolate_id"] = meta["isolate_id"].astype(str)
    meta["species"] = meta["species"].astype(str)
    meta["cohort"] = meta["cohort"].astype(str)
    return meta


def align_matrix_and_metadata(x, meta):
    ids = [i for i in x.index if i in set(meta["isolate_id"])]
    x = x.loc[ids].copy()
    meta = meta.set_index("isolate_id").loc[ids].reset_index()
    return x, meta


def split_cohorts(x, meta):
    train_ids = meta.loc[meta["cohort"] == "discovery", "isolate_id"].tolist()
    test_ids = meta.loc[meta["cohort"] == "external", "isolate_id"].tolist()

    x_train = x.loc[train_ids]
    x_test = x.loc[test_ids]

    y_train = meta.set_index("isolate_id").loc[train_ids]
    y_test = meta.set_index("isolate_id").loc[test_ids]

    return x_train, x_test, y_train, y_test


def ctu_cluster(x_train, eps=0.05, method="average"):
    genes = list(x_train.columns)
    arr = x_train.values.astype(bool)

    if arr.shape[1] < 2:
        return pd.Series(np.ones(arr.shape[1], dtype=int), index=genes)

    d = pdist(arr.T, metric="jaccard")
    z = linkage(d, method=method)
    labels = fcluster(z, t=eps, criterion="distance")

    return pd.Series(labels, index=genes)


def collapse_to_ctu(x, ctu_map, prefix):
    ctu_map = ctu_map.copy()
    ctu_map.index = ctu_map.index.astype(str)

    cols = [c for c in ctu_map.index if c in x.columns]
    x = x.reindex(columns=cols, fill_value=0)

    out = {}
    for cid, genes in ctu_map.loc[cols].groupby(ctu_map.loc[cols]).groups.items():
        name = f"{prefix}_CTU_{int(cid):05d}"
        out[name] = x.loc[:, list(genes)].max(axis=1).astype(np.uint8)

    return pd.DataFrame(out, index=x.index)


def wilson_lower_bound(pos, total, z=1.96):
    if total == 0:
        return 0.0

    p = pos / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    adj = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total)

    return (centre - adj) / denom


def pan_gam_association(x_ctu, y, drug_cols, alpha=0.05, ppv_lower_min=0.90, nmin=10):
    rows = []

    y_drug = y[drug_cols].astype(int)

    for drug in drug_cols:
        target = y_drug[drug].values

        resistant = target == 1
        specific_control = (target == 0) & (y_drug.drop(columns=[drug]).sum(axis=1).values > 0)

        if specific_control.sum() < nmin:
            pan_sus = y_drug.sum(axis=1).values == 0
            specific_control = specific_control | pan_sus

        n_r = int(resistant.sum())
        n_s = int(specific_control.sum())

        if n_r == 0 or n_s == 0:
            continue

        pvals = []
        stat_rows = []

        for ctu in x_ctu.columns:
            v = x_ctu[ctu].values.astype(int)

            a = int(v[resistant].sum())
            c = int(v[specific_control].sum())

            b = n_r - a
            d = n_s - c

            n_total = a + b + c + d
            k_present = a + c

            p = hypergeom.sf(a - 1, n_total, k_present, n_r)

            ppv = a / (a + c) if (a + c) > 0 else 0.0
            ppv_lb = wilson_lower_bound(a, a + c)

            pvals.append(p)
            stat_rows.append((drug, ctu, a, b, c, d, p, ppv, ppv_lb))

        padj = multipletests(pvals, alpha=alpha, method="fdr_bh")[1]

        for r, q in zip(stat_rows, padj):
            drug, ctu, a, b, c, d, p, ppv, ppv_lb = r
            keep = (q < alpha) and (ppv_lb >= ppv_lower_min)

            rows.append({
                "drug": drug,
                "ctu": ctu,
                "present_resistant": a,
                "absent_resistant": b,
                "present_control": c,
                "absent_control": d,
                "pvalue": p,
                "qvalue": q,
                "ppv": ppv,
                "ppv_lower": ppv_lb,
                "selected": int(keep)
            })

    return pd.DataFrame(rows)


def train_models(x_train, y_train, x_test, y_test, assoc, drug_cols):
    results = []
    models = {}

    for drug in drug_cols:
        selected = assoc.loc[
            (assoc["drug"] == drug) & (assoc["selected"] == 1),
            "ctu"
        ].unique().tolist()

        selected = [c for c in selected if c in x_train.columns]

        if len(selected) == 0:
            continue

        yt = y_train[drug].astype(int).values
        yv = y_test[drug].astype(int).values

        if len(np.unique(yt)) < 2 or len(np.unique(yv)) < 2:
            continue

        clf = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=7
        )

        clf.fit(x_train[selected].values, yt)

        pred = clf.predict(x_test[selected].values)

        try:
            prob = clf.predict_proba(x_test[selected].values)[:, 1]
            auc = roc_auc_score(yv, prob)
        except Exception:
            auc = np.nan

        tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()

        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan

        results.append({
            "drug": drug,
            "n_features": len(selected),
            "accuracy": accuracy_score(yv, pred),
            "sensitivity": sens,
            "specificity": spec,
            "ppv": precision_score(yv, pred, zero_division=0),
            "f1": f1_score(yv, pred, zero_division=0),
            "auc": auc,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        })

        models[drug] = {
            "model": clf,
            "features": selected
        }

    return pd.DataFrame(results), models


def run_species(species, rtab_path, meta, drug_cols, outdir, eps):
    x = read_roary_rtab(rtab_path)

    meta_s = meta.loc[meta["species"] == species].copy()
    x, meta_s = align_matrix_and_metadata(x, meta_s)

    x_train, x_test, y_train, y_test = split_cohorts(x, meta_s)

    ctu_map = ctu_cluster(x_train, eps=eps)
    x_train_ctu = collapse_to_ctu(x_train, ctu_map, species)
    x_test_ctu = collapse_to_ctu(x_test, ctu_map, species)

    x_test_ctu = x_test_ctu.reindex(columns=x_train_ctu.columns, fill_value=0)

    assoc = pan_gam_association(
        x_train_ctu,
        y_train,
        drug_cols=drug_cols,
        alpha=0.05,
        ppv_lower_min=0.90,
        nmin=10
    )

    perf, models = train_models(
        x_train_ctu,
        y_train,
        x_test_ctu,
        y_test,
        assoc,
        drug_cols
    )

    os.makedirs(outdir, exist_ok=True)

    ctu_map.to_csv(os.path.join(outdir, f"{species}_ctu_map.tsv"), sep="\t", header=["ctu_id"])
    assoc.to_csv(os.path.join(outdir, f"{species}_pangam_association.tsv"), sep="\t", index=False)
    perf.to_csv(os.path.join(outdir, f"{species}_external_performance.tsv"), sep="\t", index=False)

    dump(models, os.path.join(outdir, f"{species}_models.joblib"))

    summary = {
        "species": species,
        "n_train": int(x_train.shape[0]),
        "n_external": int(x_test.shape[0]),
        "genes_train": int(x_train.shape[1]),
        "ctus_train": int(x_train_ctu.shape[1]),
        "feature_reduction": float(1 - x_train_ctu.shape[1] / x_train.shape[1]),
        "epsilon": eps
    }

    with open(os.path.join(outdir, f"{species}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return perf, assoc, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--kp-rtab", required=True)
    parser.add_argument("--sa-rtab", required=True)
    parser.add_argument("--drugs", required=True)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--epsilon", type=float, default=0.05)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    meta = clean_metadata(args.metadata)
    drug_cols = [x.strip() for x in args.drugs.split(",") if x.strip()]

    all_perf = []
    all_assoc = []
    summaries = []

    perf_kp, assoc_kp, sum_kp = run_species(
        "Kp",
        args.kp_rtab,
        meta,
        drug_cols,
        args.outdir,
        args.epsilon
    )

    perf_sa, assoc_sa, sum_sa = run_species(
        "Sa",
        args.sa_rtab,
        meta,
        drug_cols,
        args.outdir,
        args.epsilon
    )

    all_perf.append(perf_kp.assign(species="Kp"))
    all_perf.append(perf_sa.assign(species="Sa"))

    all_assoc.append(assoc_kp.assign(species="Kp"))
    all_assoc.append(assoc_sa.assign(species="Sa"))

    summaries.extend([sum_kp, sum_sa])

    perf = pd.concat(all_perf, ignore_index=True)
    assoc = pd.concat(all_assoc, ignore_index=True)

    perf.to_csv(os.path.join(args.outdir, "all_species_external_performance.tsv"), sep="\t", index=False)
    assoc.to_csv(os.path.join(args.outdir, "all_species_pangam_association.tsv"), sep="\t", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(args.outdir, "species_summary.tsv"), sep="\t", index=False)

    print("\nSpecies summary")
    print(summary_df)

    print("\nExternal performance")
    if not perf.empty:
        print(perf.groupby("species")[["accuracy", "sensitivity", "specificity", "ppv", "f1", "auc"]].mean())
    else:
        print("No evaluable drug model was produced.")


if __name__ == "__main__":
    main()