"""Predict vp with frozen per-kind binary logistic and plot PCA + approximate boundary."""
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import train_nipslmcol as m

warnings.filterwarnings("ignore")

VP = ("data/truth.vp.chr12.fasta", "data/truth.vp.chr12.gff3")
MAX_POSITIVES = 500
SEED = 0
COLORS = {0: "#1f77b4", 1: "#d62728"}
MARKERS = {0: "o", 1: "^"}
NAMES = {0: "negative", 1: "positive"}

model = np.load("logistic_model.npz")

codes = m.read_fasta(VP[0])
name = next(iter(codes))
masks = m.positive_masks(m.read_annotation(VP[1]), name, len(codes[name]))
positions, strands, kinds, labels = m.build_candidates(codes[name], masks)
rng = np.random.default_rng(SEED)
index = m.sample_candidates(kinds, labels, MAX_POSITIVES, 1.0, rng)
bank = m.CwtFeatureBank(codes[name], m.PERIODS, m.W0S, 100_000)
offsets = np.arange(-m.RADIUS, m.RADIUS + 1)
matrix = bank.gather(
    (positions[index][:, None] + offsets[None, :]).ravel(),
    np.repeat(strands[index], len(offsets)),
).reshape(len(index), -1)
strands = strands[index]
labels = labels[index]
kinds = kinds[index]

figure, axes = plt.subplots(2, 4, figsize=(24, 12))
for ax, (kind_index, kind) in zip(axes.ravel(), enumerate(m.KINDS)):
    rows = kinds == kind_index
    x = matrix[rows]
    y = labels[rows]
    coef = model[f"{kind}_coef"]
    intercept = model[f"{kind}_intercept"]

    pca = PCA(2, random_state=SEED).fit(x)
    z = pca.transform(x)

    decision = x @ coef + intercept
    pred = (decision > 0).astype(int)

    # Approximate the full-feature classifier's decision in the 2D PCA plane.
    approx = LogisticRegression().fit(z, pred)
    w2 = approx.coef_[0]
    b2 = approx.intercept_[0]

    pad = 0.5
    lo = z.min(axis=0) - pad
    hi = z.max(axis=0) + pad
    if abs(w2[1]) > 1e-12:
        xs = np.array([lo[0], hi[0]])
        ys = -(w2[0] * xs + b2) / w2[1]
        ax.plot(xs, ys, "k-", linewidth=1.2)
    else:
        ax.axvline(-b2 / w2[0], color="k", linewidth=1.2)

    for value in (0, 1):
        mask = y == value
        if mask.any():
            ax.scatter(
                z[mask, 0], z[mask, 1], c=COLORS[value], marker=MARKERS[value],
                s=10, alpha=0.7, edgecolors="none",
            )

    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, decision)
    except ValueError:
        auc = float("nan")
    approx_acc = approx.score(z, pred)
    ax.set_title(f"vp {kind}  acc={acc:.3f}  AUC={auc:.3f}  2D~{approx_acc:.3f}")
    print(f"vp {kind}: acc={acc:.3f} AUC={auc:.3f} 2D-approx={approx_acc:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])

handles = [
    Line2D([], [], marker=MARKERS[v], linestyle="", color=COLORS[v], label=NAMES[v])
    for v in (0, 1)
] + [Line2D([], [], color="k", linestyle="-", label="approx. boundary")]
figure.legend(handles=handles, loc="lower center", ncol=3)
figure.tight_layout(rect=(0, 0.04, 1, 1))
figure.savefig("vp_logistic.png", dpi=200)
plt.close(figure)
print("wrote vp_logistic.png")
