import os
import subprocess
import pandas as pd

eps_list = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]

base_cmd = [
    "python", "pan_gam.py",
    "--metadata", "data/metadata.csv",
    "--kp-rtab", "data/roary_kp.rtab",
    "--sa-rtab", "data/roary_sa.rtab",
    "--drugs", "methicillin,carbapenem,ciprofloxacin,gentamicin,erythromycin,clindamycin,tetracycline"
]

rows = []

for eps in eps_list:
    outdir = f"results_eps_{eps:.2f}"

    cmd = base_cmd + [
        "--outdir", outdir,
        "--epsilon", str(eps)
    ]

    subprocess.run(cmd, check=True)

    s = pd.read_csv(os.path.join(outdir, "species_summary.tsv"), sep="\t")
    p = pd.read_csv(os.path.join(outdir, "all_species_external_performance.tsv"), sep="\t")

    rows.append({
        "epsilon": eps,
        "genes": s["genes_train"].sum(),
        "ctus": s["ctus_train"].sum(),
        "feature_reduction": 1 - s["ctus_train"].sum() / s["genes_train"].sum(),
        "accuracy": p["accuracy"].mean(),
        "sensitivity": p["sensitivity"].mean(),
        "specificity": p["specificity"].mean(),
        "ppv": p["ppv"].mean(),
        "f1": p["f1"].mean(),
        "auc": p["auc"].mean()
    })

res = pd.DataFrame(rows)
res.to_csv("epsilon_sensitivity_summary.tsv", sep="\t", index=False)
print(res)