"""! @brief Lecture des résultats d'optimisation : rapports, séries et lignes du tableau des variantes.

Bibliothèque standard uniquement : l'interface et le plugin de codecs découvrent les variantes sans importer Aidge.
Le dossier d'un job en cours porte le suffixe @c .tmp-PID et reste invisible jusqu'à la fin du job.
"""
import glob
import hashlib
import json
import shutil
from pathlib import Path

from .. import config

## Version du format de report.json.
REPORT_SCHEMA = 1
## Nom du rapport dans un dossier de résultats.
REPORT_NAME = "report.json"
## Marque d'un dossier de résultats inachevé.
TMP_MARK = ".tmp-"
## Colonnes du tableau des variantes.
TABLE_COLUMNS = ["job", "nom", "recette", "statut", "N", "λ", "tuile", "pic / BRAM", "poids (Kio)", "poids conv / BRAM", "Δbpp (%)",
                 "ΔPSNR (dB)", "binaire (Kio)", "export", "identiques Aidge (%)", "identiques PyTorch (%)", "LoCod"]
## Libellés des statuts d'une variante.
STATUS_LABELS = {"ok": "OK", "sans_effet": "SANS EFFET", "echec": "ÉCHEC"}

## Empreintes déjà calculées : (chemin, date, taille) -> SHA-256.
_sha_cache = {}


def file_sha256(path):
    """! @brief SHA-256 d'un fichier, mis en cache tant que sa date et sa taille ne changent pas.

    @param path  Fichier.
    @return Empreinte hexadécimale, ou None si le fichier est absent.
    """
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _sha_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _sha_cache[key] = h.hexdigest()
    return _sha_cache[key]


def remove_temporary(run_id, optim_dir=None):
    """! @brief Supprime les dossiers temporaires d'un résultat, laissés par un processus tué.

    @param run_id     Nom du dossier de résultats.
    @param optim_dir  Dossier des résultats ; None : config.OPTIM_DIR lu à l'appel.
    """
    root = Path(config.OPTIM_DIR if optim_dir is None else optim_dir)
    for path in root.glob(f"{glob.escape(str(run_id))}{TMP_MARK}*"):
        shutil.rmtree(path, ignore_errors=True)


def read_reports(optim_dir=None):
    """! @brief Lit les rapports terminés, du plus ancien au plus récent.

    Un rapport illisible est signalé et ignoré, comme un checkpoint illisible.
    @param optim_dir  Dossier des résultats ; None : config.OPTIM_DIR lu à l'appel.
    @return Liste de rapports, chacun avec la clé @c _dir (dossier du rapport).
    """
    root = Path(config.OPTIM_DIR if optim_dir is None else optim_dir)
    if not root.is_dir():
        return []
    reports = []
    finished = [p for p in root.glob(f"*/{REPORT_NAME}") if TMP_MARK not in p.parent.name]
    for path in sorted(finished, key=lambda p: p.stat().st_mtime):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("schema") != REPORT_SCHEMA:
                raise ValueError(f"format {report.get('schema')} inconnu")
        except (OSError, ValueError) as e:
            print(f"Rapport ignoré ({path}) : {e}")
            continue
        report["_dir"] = str(path.parent)
        reports.append(report)
    return reports


def series_key(report, recipe):
    """! @brief Clé de regroupement d'une variante en courbe : nom du job, architecture, recette, tuile (fonction pure).

    @param report  Rapport.
    @param recipe  Recette.
    @return Tuple hachable.
    """
    s = report["source"]
    return report["name"], s["in_ch"], s["N"], s.get("gdn_last", True), recipe, tuple(report["tile"])


def _kib(v):
    """! @brief Octets vers Kio arrondis, ou None."""
    return None if v is None else round(v / 1024)


def _round(v, digits):
    """! @brief Arrondi tolérant à None."""
    return None if v is None else round(v, digits)


def variant_rows(reports):
    """! @brief Lignes du tableau des variantes (fonction pure).

    @param reports  Rapports de read_reports().
    @return Liste de dictionnaires aux clés TABLE_COLUMNS, du plus récent au plus ancien.
    """
    rows = []
    for report in reversed(reports):
        src = report["source"]
        for recipe, v in report["variants"].items():
            mem, rd, ex = v.get("memory") or {}, v.get("rd") or {}, v.get("export") or {}
            locod = v.get("locod") or {}
            bram = report.get("bram_bytes") or config.BRAM_BYTES
            rows.append({"job": report["id"], "nom": report["name"], "recette": recipe,
                         "statut": STATUS_LABELS.get(v["status"], v["status"]), "N": src["N"], "λ": src["lmbda"],
                         "tuile": report["tile"][1],
                         "pic / BRAM": f"{mem['peak_bytes'] / bram * 100:.0f} %" if mem else None,
                         "poids (Kio)": _kib(mem.get("weights_bytes")),
                         "poids conv / BRAM": f"{mem['conv_weights_bytes'] / bram * 100:.0f} %" if mem else None,
                         "Δbpp (%)": _round(rd.get("delta_bpp_pct"), 2), "ΔPSNR (dB)": _round(rd.get("delta_psnr_db"), 3),
                         "binaire (Kio)": _kib(ex.get("text_bytes")), "export": ex.get("status"),
                         "identiques Aidge (%)": ex.get("symbols_identical_pct"),
                         "identiques PyTorch (%)": ex.get("symbols_identical_vs_pytorch_pct"),
                         "LoCod": locod.get("status")})
    return rows


def no_effect_alerts(reports):
    """! @brief Messages des variantes sans effet ou en échec, à afficher (fonction pure).

    @param reports  Rapports de read_reports().
    @return Liste de phrases, sans doublon.
    """
    messages = []
    for report in reversed(reports):
        for recipe, v in report["variants"].items():
            if v["status"] != "ok":
                text = f"n°{report['id']} · {report['name']} · {recipe} : {v.get('reason') or v['status']}"
                if text not in messages:
                    messages.append(text)
    return messages


def report_for_job(job_id, optim_dir=None):
    """! @brief Rapport d'un job, ou None.

    @param job_id     Numéro du job.
    @param optim_dir  Dossier des résultats ; None : config.OPTIM_DIR lu à l'appel.
    @return Rapport avec la clé @c _dir (dossier du rapport), comme read_reports(), ou None s'il est absent ou illisible.
    """
    path = Path(config.OPTIM_DIR if optim_dir is None else optim_dir) / str(job_id) / REPORT_NAME
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(report, dict):
        return None
    report["_dir"] = str(path.parent)
    return report


def summary_markdown(report):
    """! @brief Résumé d'un rapport, pour le suivi d'un job dans l'onglet « 5. Optimisation Aidge » (fonction pure).

    @param report  Rapport, ou None.
    @return Texte Markdown.
    """
    if report is None:
        return "Rapport pas encore disponible."
    lines = []
    for recipe, v in report["variants"].items():
        ex = v.get("export") or {}
        line = f"- **{recipe}** : {STATUS_LABELS.get(v['status'], v['status'])}"
        if v.get("reason"):
            line += f". {v['reason']}"
        if ex.get("status"):
            line += f" Export : {ex['status']}."
        lines.append(line)
    return "\n".join(lines)
