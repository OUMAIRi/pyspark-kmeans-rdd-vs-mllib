"""Chargement / génération des vecteurs documents 58D (format Spambase).

Si un fichier spambase.data (UCI) est fourni, il est utilisé tel quel.
Sinon, un jeu synthétique de même structure est généré (4601 x 58) :
48 fréquences de mots, 6 fréquences de caractères, 3 statistiques de
majuscules, + 1 colonne d'étiquette (58 colonnes au total).
"""
import os
import numpy as np


def generate_synthetic(n=4601, seed=42):
    rng = np.random.default_rng(seed)
    k_true = 4
    d_words, d_chars, d_caps = 48, 6, 3
    centers_w = rng.gamma(0.6, 0.5, size=(k_true, d_words)) * (rng.random((k_true, d_words)) < 0.35)
    centers_c = rng.gamma(0.8, 0.15, size=(k_true, d_chars)) * (rng.random((k_true, d_chars)) < 0.5)
    caps_mu = np.array([[1.6, 5.0, 60], [3.5, 25, 600], [2.2, 10, 150], [5.0, 60, 1500]])
    z = rng.choice(k_true, size=n, p=[0.35, 0.25, 0.25, 0.15])
    W = rng.gamma(1.0, 1.0, size=(n, d_words)) * centers_w[z] * (rng.random((n, d_words)) < 0.6)
    C = rng.gamma(1.0, 1.0, size=(n, d_chars)) * centers_c[z] * (rng.random((n, d_chars)) < 0.5)
    caps = np.exp(rng.normal(np.log(caps_mu[z]), 0.5))
    label = (z >= 2).astype(float)
    X = np.hstack([np.round(W, 2), np.round(C, 3), np.round(caps, 2), label[:, None]])
    return X, z


def load_data(path=None, seed=42):
    """Retourne (X brut 58D, labels connus ou None, source)."""
    if path and os.path.exists(path):
        X = np.loadtxt(path, delimiter=",")
        return X, None, f"spambase ({path})"
    X, z = generate_synthetic(seed=seed)
    return X, z, "synthétique type Spambase (4601 x 58)"


def preprocess(X):
    """log1p puis standardisation (z-score) : évite que les colonnes
    de majuscules (échelle 1e3) dominent les distances."""
    Xl = np.log1p(X)
    mu, sd = Xl.mean(0), Xl.std(0)
    sd[sd == 0] = 1.0
    return (Xl - mu) / sd
