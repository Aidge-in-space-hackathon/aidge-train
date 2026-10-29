#!/bin/sh
# Lance la plateforme AIDGE (worker + interface) dans le venv, sous Linux et macOS.
# Ne pas mélanger avec Docker : les bases de la file d'entraînement sont différentes.
# Ctrl+C arrête les deux processus.
ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 1
PY="$ROOT/.venv/bin/python"
URL="http://127.0.0.1:7860"
# Durée maximale d'attente de l'interface, en secondes (sonde toutes les 2 s)
ATTENTE_MAX="${AIDGE_ATTENTE_MAX:-120}"
export PYTHONIOENCODING=utf-8

if [ ! -x "$PY" ]; then
    echo "ERREUR : l'environnement Python (.venv) n'existe pas encore."
    echo "Installez-le d'abord, une seule fois : sh scripts/setup_venv.sh"
    exit 1
fi

cd "$ROOT" || exit 1
"$PY" -m aidge_in_space.worker &
WORKER_PID=$!
"$PY" -m aidge_in_space.app &
APP_PID=$!

# À l'arrêt (Ctrl+C ou fin du script), on termine proprement les deux processus
trap 'kill "$WORKER_PID" "$APP_PID" 2>/dev/null; wait; exit 0' INT TERM
trap 'kill "$WORKER_PID" "$APP_PID" 2>/dev/null' EXIT

# Sonde : succès dès que l'interface répond (curl, sinon wget)
sonde() {
    if command -v curl >/dev/null 2>&1; then
        curl -s -o /dev/null --max-time 2 "$URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O /dev/null -T 2 -t 1 "$URL"
    else
        return 1
    fi
}

echo "Attente de l'interface sur $URL (au plus $ATTENTE_MAX s)..."
ecoule=0
pret=0
while [ "$ecoule" -lt "$ATTENTE_MAX" ]; do
    if sonde; then pret=1; break; fi
    sleep 2
    ecoule=$((ecoule + 2))
done

if [ "$pret" -eq 1 ]; then
    echo "Interface prête : $URL  (Ctrl+C pour tout arrêter)"
    if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1
    elif command -v open >/dev/null 2>&1; then open "$URL"
    fi
else
    echo "ATTENTION : l'interface ne répond toujours pas après $ATTENTE_MAX secondes."
    echo "Regardez les messages d'erreur ci-dessus, puis ouvrez $URL à la main."
    echo "(Sans curl ni wget, la sonde ne peut pas fonctionner : ouvrez l'adresse à la main.)"
fi
wait
