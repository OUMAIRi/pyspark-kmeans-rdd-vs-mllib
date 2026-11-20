"""Benchmark K-Means : version RDD (L1 / L2) vs PySpark MLlib KMeans (DataFrame, k-means||).

Sorties : results.json + figures/*.png
"""
import argparse
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

from data_utils import load_data, preprocess
from custom_kmeans_rdd import kmeans_rdd, dist

FIG = "figures"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def labels_of(X, centers, metric):
    D = np.stack([dist(X, c, metric) for c in centers], 1)
    return D.argmin(1)


def mllib_fit(df, k, max_iter, seed):
    km = KMeans(k=k, maxIter=max_iter, tol=0.0, seed=seed, initMode="k-means||",
                featuresCol="features", distanceMeasure="euclidean")
    t0 = time.time()
    model = km.fit(df)
    return model, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    X, ztrue, src = load_data(a.data)
    Xs = preprocess(X)
    n, d = Xs.shape
    spark = (SparkSession.builder.master("local[*]").appName("KMeansBenchmark")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    rdd = sc.parallelize(list(Xs), 4).cache(); rdd.count()
    df = spark.createDataFrame([(i, Vectors.dense(r.tolist())) for i, r in enumerate(Xs)], ["idx", "features"]).repartition(4).cache()
    df.count()
    res = {"source": src, "n": n, "d": d, "k": a.k}

    # ---------- 1. Convergence RDD L2 / L1 ----------
    rdd_hist = {}
    rdd_centers = {}
    for m in ("l2", "l1"):
        c, h = kmeans_rdd(sc, rdd, a.k, m, a.max_iter, tol=1e-6, seed=a.seed)
        rdd_hist[m], rdd_centers[m] = h, c
    res["rdd"] = rdd_hist

    # ---------- 2. Convergence MLlib : fit avec maxIter = 1..T ----------
    ml_wsse, ml_time = [], []
    for it in range(1, a.max_iter + 1):
        model, t = mllib_fit(df, a.k, it, a.seed)
        ml_wsse.append(float(model.summary.trainingCost))
        ml_time.append(t)
    model, t_full = mllib_fit(df, a.k, a.max_iter, a.seed)
    res["mllib"] = {"wsse": ml_wsse, "fit_time_by_maxiter": ml_time,
                    "numIter": int(model.summary.numIter), "final_wsse": float(model.summary.trainingCost)}

    # ---------- 3. Temps (3 répétitions, 10 itérations fixes) ----------
    T = 10
    t_rdd, t_ml = [], []
    for r in range(3):
        t0 = time.time(); kmeans_rdd(sc, rdd, a.k, "l2", T, tol=0.0, seed=a.seed); t_rdd.append(time.time() - t0)
        _, t = mllib_fit(df, a.k, T, a.seed); t_ml.append(t)
    res["timing_10_iter"] = {"rdd_l2": t_rdd, "mllib": t_ml}

    # ---------- 4. Qualité : silhouette + WSSE final ----------
    lab = {m: labels_of(Xs, rdd_centers[m], m) for m in rdd_centers}
    pred = model.transform(df)
    ml_sil = float(ClusteringEvaluator(metricName="silhouette", distanceMeasure="squaredEuclidean").evaluate(pred))
    ml_lab = np.array([r.prediction for r in pred.orderBy("idx").select("prediction").collect()])  # ordre aligné sur Xs
    ss = np.arange(n)  # silhouette sur toutes les données (n ~ 4600)
    res["silhouette"] = {
        "rdd_l2": float(silhouette_score(Xs[ss], lab["l2"][ss])),
        "rdd_l1": float(silhouette_score(Xs[ss], lab["l1"][ss])),
        "mllib": float(silhouette_score(Xs[ss], ml_lab[ss])),
        "rdd_l1_manhattan": float(silhouette_score(Xs[ss], lab["l1"][ss], metric="manhattan")),
        "mllib_evaluator_sqeuclid": ml_sil}
    res["final_wsse"] = {"rdd_l2": rdd_hist["l2"][-1]["wsse"], "rdd_l1_wsse_l2sq": rdd_hist["l1"][-1]["wsse"],
                         "rdd_l1_cost_L1": rdd_hist["l1"][-1]["cost_metric"], "mllib": res["mllib"]["final_wsse"]}
    from sklearn.metrics import adjusted_rand_score
    if ztrue is not None:
        res["ari_vs_truth"] = {"rdd_l2": float(adjusted_rand_score(ztrue, lab["l2"])),
                               "rdd_l1": float(adjusted_rand_score(ztrue, lab["l1"])),
                               "mllib": float(adjusted_rand_score(ztrue, ml_lab))}
    res["ari_rdd_l2_vs_mllib"] = float(adjusted_rand_score(lab["l2"], ml_lab))

    res["n_labels_differ_l1_l2"] = int((lab["l1"] != lab["l2"]).sum())

    # ---------- 4b. Robustesse aux initialisations (5 graines) ----------
    seeds = [1, 2, 3, 4, 5]
    rob = {"rdd_l2": [], "mllib": []}
    for sd in seeds:
        _, h = kmeans_rdd(sc, rdd, a.k, "l2", a.max_iter, tol=1e-6, seed=sd)
        rob["rdd_l2"].append(h[-1]["wsse"])
        m, _ = mllib_fit(df, a.k, a.max_iter, sd)
        rob["mllib"].append(float(m.summary.trainingCost))
    res["robustness_seeds"] = {"seeds": seeds, **rob}

    # ---------- 5. Coude + silhouette (MLlib) ----------
    ks = list(range(2, 11)); elbow, sils = [], []
    ev = ClusteringEvaluator(metricName="silhouette", distanceMeasure="squaredEuclidean")
    for k in ks:
        best = min((mllib_fit(df, k, 20, sd)[0] for sd in (1, 2, 3)), key=lambda mm: mm.summary.trainingCost)  # meilleur de 3 graines
        elbow.append(float(best.summary.trainingCost)); sils.append(float(ev.evaluate(best.transform(df))))
    res["elbow"] = {"k": ks, "wsse": elbow, "silhouette": sils}

    json.dump(res, open("results.json", "w"), indent=2)

    # ---------- Figures ----------
    it_ = np.arange(1, a.max_iter + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    h2 = [h["wsse"] for h in rdd_hist["l2"]]
    h1 = [h["wsse"] for h in rdd_hist["l1"]]
    ax[0].plot(range(1, len(h2) + 1), h2, "o-", label="RDD L2")
    ax[0].plot(range(1, len(h1) + 1), h1, "s-", label="RDD L1 (WSSE en L2²)")
    ax[0].plot(it_, ml_wsse, "^--", label="MLlib")
    ax[0].set_xlabel("Itération"); ax[0].set_ylabel("WSSE"); ax[0].set_title("Convergence du WSSE"); ax[0].legend()
    ax[1].plot(range(1, len(h1) + 1), [h["cost_metric"] for h in rdd_hist["l1"]], "s-", color="C1")
    ax[1].set_xlabel("Itération"); ax[1].set_ylabel("Coût L1 (somme des distances)")
    ax[1].set_title("RDD L1 : coût Manhattan")
    plt.tight_layout(); plt.savefig(f"{FIG}/convergence_wsse.png", dpi=200); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ks, elbow, "o-"); ax[0].set_title("Méthode du coude (MLlib)")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("WSSE")
    ax[1].plot(ks, sils, "o-", color="C3"); ax[1].set_title("Silhouette (MLlib)")
    ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette")
    plt.tight_layout(); plt.savefig(f"{FIG}/elbow_silhouette.png", dpi=200); plt.close()

    # temps par itération RDD vs temps moyen MLlib
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar(["RDD (L2)", "MLlib"], [np.mean(t_rdd) / T * 1000, np.mean(t_ml) / T * 1000], color=["C0", "C2"])
    ax.set_ylabel("ms / itération"); ax.set_title("Temps moyen par itération (10 it.)")
    plt.tight_layout(); plt.savefig(f"{FIG}/timing.png", dpi=200); plt.close()

    pca = PCA(2, random_state=0).fit(Xs)
    P = pca.transform(Xs)
    res["pca_var"] = [float(v) for v in pca.explained_variance_ratio_]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), sharex=True, sharey=True)
    for axx, (lbl, title) in zip(ax, [(lab["l2"], "RDD – L2"), (lab["l1"], "RDD – L1"), (ml_lab, "MLlib")]):
        axx.scatter(P[:, 0], P[:, 1], c=lbl, s=4, cmap="tab10", alpha=0.7); axx.set_title(title)
        axx.set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    plt.tight_layout(); plt.savefig(f"{FIG}/pca_clusters.png", dpi=200); plt.close()

    json.dump(res, open("results.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("rdd", "elbow")}, indent=1))
    spark.stop()


if __name__ == "__main__":
    main()
