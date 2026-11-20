"""K-Means "from scratch" sur RDDs Spark, distances L1 (Manhattan) et L2 (Euclidienne).

Étape Map    : assignation de chaque point au centroïde le plus proche
               (centroïdes diffusés par broadcast).
Étape Reduce : reduceByKey pour sommer vecteurs / effectifs / coûts par cluster,
               puis calcul des nouveaux centroïdes (moyennes).
Coût WSSE calculé à chaque itération.
"""
import argparse
import time
import numpy as np
from pyspark import SparkContext, SparkConf

from data_utils import load_data, preprocess


def dist(X, c, metric):
    """Distances (n, ) de chaque ligne de X au centre c."""
    diff = X - c
    return np.abs(diff).sum(1) if metric == "l1" else np.sqrt((diff ** 2).sum(1))


def assign_partition(metric, bc_centers):
    """Fonction Map (par partition, vectorisée numpy).
    Émet (cluster_id, (somme_vecteurs, effectif, coût_métrique, coût_L2_carré))."""
    def _f(rows):
        pts = np.array(list(rows))
        if len(pts) == 0:
            return iter([])
        C = bc_centers.value
        if metric == "l1":
            D = np.abs(pts[:, None, :] - C[None, :, :]).sum(2)
        else:
            D = np.sqrt(((pts[:, None, :] - C[None, :, :]) ** 2).sum(2))
        a = D.argmin(1)
        dmin = D[np.arange(len(pts)), a]
        sq = ((pts - C[a]) ** 2).sum(1)   # contribution au WSSE classique (L2^2)
        out = []
        for j in np.unique(a):
            m = a == j
            out.append((int(j), (pts[m].sum(0), int(m.sum()), float(dmin[m].sum()), float(sq[m].sum()))))
        return iter(out)
    return _f


def init_kmeanspp(sample, k, metric, rng):
    """Initialisation k-means++ (sur un échantillon collecté au driver)."""
    centers = [sample[rng.integers(len(sample))]]
    for _ in range(1, k):
        d = np.min([dist(sample, c, metric) for c in centers], axis=0) ** 2
        centers.append(sample[rng.choice(len(sample), p=d / d.sum())])
    return np.array(centers)


def kmeans_rdd(sc, rdd, k, metric="l2", max_iter=20, tol=1e-4, seed=42, init_centers=None):
    rng = np.random.default_rng(seed)
    if init_centers is None:
        sample = np.array(rdd.takeSample(False, 1000, seed))
        centers = init_kmeanspp(sample, k, metric, rng)
    else:
        centers = init_centers.copy()
    history = []
    for it in range(1, max_iter + 1):
        t0 = time.time()
        bc = sc.broadcast(centers)
        stats = (rdd.mapPartitions(assign_partition(metric, bc))
                    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]))
                    .collect())
        bc.unpersist()
        new = centers.copy()
        cost_metric = wsse = 0.0
        for j, (s, n, cm, sq) in stats:
            new[j] = s / n
            cost_metric += cm
            wsse += sq
        shift = float(np.abs(new - centers).max())
        history.append({"iter": it, "wsse": wsse, "cost_metric": cost_metric,
                        "time": time.time() - t0, "shift": shift})
        centers = new
        if shift < tol:
            break
    return centers, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="chemin spambase.data (optionnel)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--metric", choices=["l1", "l2"], default="l2")
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--partitions", type=int, default=4)
    a = ap.parse_args()

    X, _, src = load_data(a.data)
    Xs = preprocess(X)
    sc = SparkContext(conf=SparkConf().setAppName("KMeansRDD").setMaster("local[*]")
                      .set("spark.ui.showConsoleProgress", "false"))
    sc.setLogLevel("ERROR")
    rdd = sc.parallelize(list(Xs), a.partitions).cache()
    print(f"Données : {src}, n={rdd.count()}, d={Xs.shape[1]}")
    centers, hist = kmeans_rdd(sc, rdd, a.k, a.metric, a.max_iter)
    for h in hist:
        print(f"iter {h['iter']:2d}  WSSE={h['wsse']:12.2f}  coût_{a.metric}={h['cost_metric']:12.2f}  "
              f"t={h['time']*1000:7.1f} ms  shift={h['shift']:.5f}")
    sc.stop()


if __name__ == "__main__":
    main()
