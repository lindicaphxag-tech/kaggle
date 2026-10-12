"""Reproducible journal vector figure from immutable V24 Weibo2014 receipt.

Never reads new target EEG or modifies the precommitted selector.
Output is descriptive paired outcomes for all ten persons, NOT CIs.
"""
import argparse
import io
import json
import pathlib
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTCOME = "WEIBO2014_10_PERSON_FIRST_RUN_EXTERNAL_TARGET_AUC.json"


def run(archive, out):
    with zipfile.ZipFile(io.BytesIO(pathlib.Path(archive).read_bytes())) as z:
        keys = [n for n in z.namelist() if n.endswith(OUTCOME)]
        if len(keys) != 1:
            raise AssertionError("exact original first-run target outcome required")
        data = json.loads(z.read(keys[0]))
    rows=sorted(data["per_person_target_outcomes"],key=lambda x:x["held_person"])
    if len(rows)!=10 or [r["held_person"] for r in rows]!=list(range(1,11)):
        raise AssertionError("must display every original external person")
    macro=np.array([r["auc"]["source_macro_grid"] for r in rows])
    new=np.array([r["auc"]["v24_anchored_continuous"] for r in rows])
    proper=np.array([r["auc"]["source_correct_predictive_nll"] for r in rows])
    historic=np.array([r["auc"]["historical_validation_class_centred"] for r in rows])
    if not np.allclose(macro,new,rtol=0,atol=1e-12):
        raise AssertionError("cannot publish nonzero V24 advantage over original grid")
    d_proper=new-proper
    d_historic=new-historic

    plt.rcParams.update({
        "font.family":"DejaVu Sans",
        "font.size":8.5,"axes.titlesize":9,
        "axes.labelsize":8.5,"xtick.labelsize":8,"ytick.labelsize":8,
        "pdf.fonttype":42,"ps.fonttype":42,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":.65,"savefig.bbox":"tight",
    })
    fig,ax=plt.subplots(figsize=(6.8,3.25))
    x=np.arange(1,11)
    ax.axhline(0,color=".25",lw=.8,zorder=1)
    ax.axhline(.02,color=".55",lw=.55,ls="--")
    ax.axhline(-.02,color=".55",lw=.55,ls="--")
    ax.axhspan(-.02,.02,color=".94",zorder=0)
    ax.plot(x-.09,d_proper,marker="o",ls="none",ms=5.4,
            color="#205071",label="V24 minus source-trained-mean Gaussian NLL",zorder=3)
    ax.plot(x+.09,d_historic,marker="s",ls="none",ms=4.6,
            color="#bc6e32",label="V24 minus historical class-centred score",zorder=3)
    ax.set_xticks(x)
    ax.set_xlabel("Held-out person ID (each counted once)")
    ax.set_ylabel("Paired target ROC-AUC difference")
    ax.set_xlim(.45,10.55)
    ymax=max(.023,float(np.max(d_historic)+.005),float(np.max(d_proper)+.005))
    ymin=min(-.023,float(np.min(d_historic)-.005),float(np.min(d_proper)-.005))
    ax.set_ylim(ymin,ymax)
    ax.grid(axis="y",color=".88",lw=.45)
    ax.legend(loc="upper right",fontsize=7,frameon=False)
    fig.text(.12,.005,
        "All 10 Weibo2014 persons; original grid and V24 selected identical lambda in 10/10 folds.",
        ha="left",va="bottom",fontsize=7.2,color=".3")
    out=pathlib.Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,format="pdf",pad_inches=.13)
    fig.savefig(out.with_suffix(".png"),dpi=230,pad_inches=.13)
    plt.close(fig)
    print(json.dumps({
        "status":"V24_VECTOR_FIGURE_FROM_IMMUTABLE_FIRST_RUN",
        "n_people":10,
        "mean_v24_minus_proper":float(d_proper.mean()),
        "mean_v24_minus_historical":float(d_historic.mean()),
        "n_same_grid_new":int(np.sum(np.isclose(macro,new,rtol=0,atol=1e-12))),
        "figure_pdf":str(out),
    }))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("original_zip")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(a.original_zip,a.out)
