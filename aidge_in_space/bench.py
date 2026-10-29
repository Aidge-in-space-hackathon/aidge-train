"""! @brief Campagnes de benchmark : codecs x niveaux x images -> un CSV par campagne dans outputs/bench.

Exemple (sur la machine d'entraînement, sans interface) :
@code{.sh}
python -m aidge_in_space.bench --codecs jpeg,jpeg2000,bmshj2018-factorized --images 24 --mode rgb
@endcode
"""
import argparse
import datetime
import platform
import re
import sys

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import backends
from .codecs import all_codecs, get
from .config import BENCH_DIR
from .images import datasets, download_kodak, load_image
from .metrics import bd_rate, ms_ssim, ms_ssim_db, psnr

## Colonnes qui identifient un point de courbe (les mesures sont moyennées sur les images).
GROUP_COLS = ["campagne", "codec", "codec_label", "famille", "niveau", "niveau_label", "mode", "moteur", "débit"]
## Métriques tracées : colonne -> libellé d'axe.
METRICS = {"psnr": "PSNR (dB)", "ms_ssim_db": "MS-SSIM (dB)", "ms_ssim": "MS-SSIM"}


def pick_levels(levels, n):
    """! @brief n niveaux répartis régulièrement, extrêmes compris (tous s'il y en a moins).

    @param levels  Niveaux du codec, du plus bas au plus haut débit.
    @param n       Nombre de niveaux voulus.
    @return Sous-liste de @p levels.
    """
    if n >= len(levels):
        return list(levels)
    return [levels[i] for i in np.unique(np.round(np.linspace(0, len(levels) - 1, n)).astype(int))]


def evaluate(codec, img, level, mode, backend="torch", real_rate=True):
    """! @brief Une mesure : compression, décompression et métriques.

    @param codec      Codec.
    @param img        Image uint8.
    @param level      Level du codec.
    @param mode       @c "gris" ou @c "rgb".
    @param backend    Moteur de l'encodeur (ignoré par les codecs sans moteur).
    @param real_rate  Débit réel ou estimé (voir Codec.run()).
    @return Couple (ligne de résultats sous forme de dictionnaire, Result).
    """
    use_backend = backend if codec.backends else None
    res = codec.run(img, level, mode, backend=use_backend or "torch", real_rate=real_rate)
    H, W = img.shape[:2]
    row = {
        "codec": codec.key, "codec_label": codec.label, "famille": codec.family,
        "niveau": str(level.value), "niveau_label": level.label, "mode": mode,
        "moteur": use_backend or "-", "débit": res.rate_kind, "H": H, "W": W,
        "bits": res.bits, "bpp": res.bits / (H * W), "taux": img.size * 8 / res.bits,
        "psnr": psnr(img, res.recon), "ms_ssim": ms_ssim(img, res.recon),
        "t_encodage_ms": 1e3 * res.timings.get("encodage", np.nan),
        "t_analyse_ms": 1e3 * res.timings.get("analyse", np.nan),
        "t_decodage_ms": 1e3 * res.timings.get("décodage", np.nan),
    }
    return row, res


def campaign_name(name=None):
    """! @brief Nom de fichier d'une nouvelle campagne, sans écraser une campagne existante.

    @param name  Nom souhaité (caractères spéciaux remplacés par « _ ») ; date et heure si vide.
    @return Nom libre dans BENCH_DIR, suffixé par _2, _3... si nécessaire.
    """
    base = re.sub(r"[^\w\-]+", "_", name.strip()) if name and name.strip() else \
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate, i = base, 1
    while (BENCH_DIR / f"{candidate}.csv").exists():
        i += 1
        candidate = f"{base}_{i}"
    return candidate


def run_benchmark(codec_keys, image_paths, mode, points=6, backend="torch", real_rate=False,
                  name=None, progress=None):
    """! @brief Lance une campagne et l'enregistre en CSV.

    Une mesure en erreur n'arrête pas la campagne : elle est ajoutée à la liste des erreurs.
    @param codec_keys   Identifiants des codecs.
    @param image_paths  Chemins des images.
    @param mode         @c "gris" ou @c "rgb" (les codecs qui ne l'acceptent pas sont sautés).
    @param points       Nombre de niveaux par codec (voir pick_levels()).
    @param backend      Moteur de l'encodeur des codecs appris.
    @param real_rate    Débit réel (plus lent) ou estimé.
    @param name         Nom de la campagne (voir campaign_name()).
    @param progress     Fonction facultative progress(fraction, texte).
    @return Triplet (DataFrame des mesures, chemin du CSV ou None si aucune mesure, liste des erreurs).
    """
    name = campaign_name(name)
    codecs = [get(k) for k in codec_keys]
    jobs = [(c, lvl) for c in codecs if mode in c.modes for lvl in pick_levels(c.levels(mode), points)]
    images = [(p, load_image(p, mode)) for p in image_paths]
    total = max(1, len(jobs) * len(images))
    rows, errors, done = [], [], 0
    meta = {"campagne": name, "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "machine": platform.node(), "threads": backends.THREADS}
    for codec, level in jobs:
        for path, img in images:
            if progress:
                progress(done / total, f"{codec.label} · {level.label} · {path.name}")
            try:
                row, _ = evaluate(codec, img, level, mode, backend, real_rate)
                rows.append({**meta, "image": path.name, **row})
            except Exception as e:
                errors.append(f"{codec.label} · {level.label} · {path.name} : {e}")
            done += 1
    df = pd.DataFrame(rows)
    path = None
    if not df.empty:
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        path = BENCH_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
    if progress:
        progress(1.0, "terminé")
    return df, path, errors


def list_runs():
    """! @brief Noms des campagnes enregistrées, de la plus récente à la plus ancienne."""
    if not BENCH_DIR.exists():
        return []
    return [p.stem for p in sorted(BENCH_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)]


def load_runs(names):
    """! @brief Mesures de plusieurs campagnes concaténées (les noms inconnus sont ignorés).

    @param names  Noms de campagnes (voir list_runs()).
    @return DataFrame, vide si aucune campagne n'est trouvée.
    """
    frames = [pd.read_csv(BENCH_DIR / f"{n}.csv") for n in names if (BENCH_DIR / f"{n}.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate(df):
    """! @brief Moyenne des mesures sur les images : une ligne par point de courbe.

    La colonne @c courbe est le libellé de légende : nom du codec, complété par le moteur si un
    même codec a été mesuré avec plusieurs moteurs, et par la campagne si plusieurs sont affichées.
    @param df  Mesures brutes (run_benchmark() ou load_runs()).
    @return DataFrame trié par courbe puis par débit.
    """
    if df.empty:
        return df
    agg = (df.groupby(GROUP_COLS, dropna=False)
             .agg(images=("image", "count"), bpp=("bpp", "mean"), psnr=("psnr", "mean"),
                  ms_ssim=("ms_ssim", "mean"), t_encodage_ms=("t_encodage_ms", "mean"),
                  t_analyse_ms=("t_analyse_ms", "mean"), t_decodage_ms=("t_decodage_ms", "mean"))
             .reset_index())
    agg["ms_ssim_db"] = agg["ms_ssim"].map(ms_ssim_db)
    several_runs = agg["campagne"].nunique() > 1
    several_engines = agg[agg["moteur"] != "-"].groupby("codec")["moteur"].nunique().max() > 1 \
        if (agg["moteur"] != "-").any() else False

    def label(r):
        s = r["codec_label"]
        if several_engines and r["moteur"] != "-":
            s += f" · {r['moteur']}"
        if several_runs:
            s += f" [{r['campagne']}]"
        return s
    agg["courbe"] = agg.apply(label, axis=1)
    return agg.sort_values(["courbe", "bpp"]).reset_index(drop=True)


def rd_figure(agg, metric="psnr"):
    """! @brief Courbes débit / distorsion (codecs appris en traits pleins, classiques en pointillés).

    @param agg     Points agrégés (aggregate()).
    @param metric  Colonne de METRICS en ordonnée.
    @return matplotlib.figure.Figure.
    """
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    if agg.empty:
        ax.text(0.5, 0.5, "aucun résultat", ha="center", va="center", transform=ax.transAxes)
        return fig
    for courbe, g in agg.groupby("courbe", sort=False):
        g = g.sort_values("bpp")
        learned = g["famille"].iloc[0] == "Appris"
        ax.plot(g["bpp"], g[metric], "o-" if learned else "s--", lw=2 if learned else 1.3, ms=5, label=courbe)
    ax.set_xlabel("débit (bits par pixel)")
    ax.set_ylabel(METRICS.get(metric, metric))
    n_img = int(agg["images"].max())
    ax.set_title(f"Débit / distorsion, moyenne sur {n_img} image(s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def time_figure(agg):
    """! @brief Temps d'encodage moyen par courbe (encodage complet et transformée d'analyse), échelle log.

    @param agg  Points agrégés (aggregate()).
    @return matplotlib.figure.Figure.
    """
    fig = Figure(figsize=(8, 3.8))
    ax = fig.subplots()
    if agg.empty:
        return fig
    t = agg.groupby("courbe", sort=False)[["t_analyse_ms", "t_encodage_ms"]].mean()
    t = t.sort_values("t_encodage_ms")
    y = np.arange(len(t))
    ax.barh(y, t["t_encodage_ms"], color="tab:blue", alpha=0.35, label="encodage complet (bord)")
    ax.barh(y, t["t_analyse_ms"].fillna(0), color="tab:blue", label="dont transformée d'analyse")
    ax.set_yticks(y, t.index, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("ms par image (1 thread, PC : indicatif)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def bd_table(agg, reference, metric="psnr"):
    """! @brief BD-rate de chaque courbe par rapport à une courbe de référence.

    @param agg        Points agrégés (aggregate()).
    @param reference  Valeur de la colonne @c courbe qui sert de référence.
    @param metric     Qualité utilisée (@c psnr ou @c ms_ssim_db).
    @return DataFrame : courbe, points, bpp min, bpp max et BD-rate en % (négatif = mieux) ;
            vide si la référence est absente.
    """
    if agg.empty or reference not in set(agg["courbe"]):
        return pd.DataFrame()
    ref = agg[agg["courbe"] == reference]
    rows = []
    for courbe, g in agg.groupby("courbe", sort=False):
        rows.append({"courbe": courbe, "points": len(g),
                     "bpp min": g["bpp"].min(), "bpp max": g["bpp"].max(),
                     f"BD-rate vs {reference} (%)": 0.0 if courbe == reference else
                     bd_rate(ref["bpp"], ref[metric], g["bpp"], g[metric])})
    return pd.DataFrame(rows)


def main(argv=None):
    """! @brief Point d'entrée de la ligne de commande (@c python -m aidge_in_space.bench --help).

    @param argv  Arguments (ceux de la ligne de commande si None).
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Benchmark des codecs d'image du projet AIDGE IN SPACE")
    ap.add_argument("--list", action="store_true", help="liste les codecs, jeux d'images et moteurs")
    ap.add_argument("--codecs", default="", help="identifiants séparés par des virgules (défaut : tous)")
    ap.add_argument("--dataset", default="kodak")
    ap.add_argument("--images", type=int, default=24, help="nombre d'images du jeu à utiliser")
    ap.add_argument("--mode", choices=["gris", "rgb"], default="gris")
    ap.add_argument("--points", type=int, default=6, help="niveaux par codec")
    ap.add_argument("--backend", choices=list(backends.BACKENDS), default="torch")
    ap.add_argument("--real-rate", action="store_true", help="vrai codeur arithmétique (plus lent)")
    ap.add_argument("--name", default=None, help="nom de la campagne (défaut : date ; jamais écrasée)")
    args = ap.parse_args(argv)

    download_kodak()
    if args.list:
        for c in all_codecs():
            ok, why = c.available()
            print(f"{c.key:28} {c.label:42} modes={','.join(c.modes):9} {'' if ok else '(indisponible : ' + why + ')'}")
        print("jeux d'images :", {k: len(v) for k, v in datasets().items()})
        print("moteurs :", backends.available_backends())
        return
    keys = [k.strip() for k in args.codecs.split(",") if k.strip()] or \
        [c.key for c in all_codecs() if c.available()[0] and args.mode in c.modes]
    paths = datasets()[args.dataset][:args.images]

    def show(frac, text):
        print(f"\r[{100 * frac:5.1f} %] {text[:90]:90}", end="", flush=True)

    df, path, errors = run_benchmark(keys, paths, args.mode, args.points, args.backend, args.real_rate,
                                     args.name, progress=show)
    print()
    for e in errors:
        print("ERREUR", e)
    if df.empty:
        print("Aucune mesure : rien n'a été enregistré.")
        return
    cols = ["courbe", "niveau_label", "bpp", "psnr", "ms_ssim", "t_encodage_ms"]
    print(aggregate(df)[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nRésultats : {path}")


if __name__ == "__main__":
    main()
