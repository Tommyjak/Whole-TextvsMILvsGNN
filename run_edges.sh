#!/usr/bin/env bash
#
# run_edges.sh — Ablation sugli ARCHI della GNN (knn e both) per i dataset corti.
#
# Aggiunge le varianti di grafo mancanti — 'knn' (vicini semantici) e 'both'
# (sequenziali + semantici) — SOLO per la GNN e SOLO sui dataset corti
# (daic, imcs21, mtsamples). Il 'sequential' esiste gia' dai run precedenti.
#
# PERCHE' funziona senza toccare i risultati esistenti:
#   il run_name della GNN include l'edge_mode -> daic_gnn_g64_seed42_knn e'
#   una cartella distinta da daic_gnn_g64_seed42_sequential. I run gia' fatti
#   non vengono sfiorati; analyze.py mostra knn/both/sequential come setup separati.
#
# PERCHE' solo i corti e non ECtHR:
#   il kNN costruisce la matrice di similarita' N×N sui chunk di ogni documento.
#   Sui documenti lunghi di ECtHR (~100+ chunk) questo e' O(N^2) pesante su 9000
#   documenti -> va lanciato a parte, con tempi da valutare. Qui restiamo sui corti.
#
# PERCHE' due protocolli:
#   --NATURAL (split ufficiale, 5 seed)  e  --CV (Monte Carlo, 10 fold), coerenti
#   con il resto del progetto. Attivabili separatamente (vedi USO).
#
# USO:
#   ./run_edges.sh                 # tutti e 3 i corti, sia natural sia cv
#   ./run_edges.sh daic            # solo daic (natural + cv)
#   MODE=natural ./run_edges.sh    # solo split ufficiale
#   MODE=cv ./run_edges.sh         # solo Monte Carlo cv
#
# NOTE: virtualenv attivo; GPU consigliata. run.py e' idempotente.

set -uo pipefail

# ------------------------------------------------------------------ config ----

EDGE_MODES="${EDGE_MODES:-knn both}"        # le varianti da aggiungere (sequential gia' fatto)
SEEDS_NATURAL="${SEEDS_NATURAL:-42 43 44 45 46}"
SEEDS_CV="${SEEDS_CV:-42 43 44 45 46 47 48 49 50 51}"
GRANS_SHORT="${GRANS_SHORT:-64 128}"
MODE="${MODE:-both}"                        # both | natural | cv
PY="${PY:-python3}"

ENC_DAIC="mental/mental-bert-base-uncased"
ENC_IMCS="bert-base-chinese"
ENC_MTS="emilyalsentzer/Bio_ClinicalBERT"

# ---------------------------------------------------- individua la root ----

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SELF_DIR/scripts/run.py" ]; then
    ROOT="$SELF_DIR"
elif [ -f "$SELF_DIR/../scripts/run.py" ]; then
    ROOT="$(cd "$SELF_DIR/.." && pwd)"
else
    echo "ERRORE: non trovo scripts/run.py. Metti run_edges.sh nella root del progetto."
    exit 1
fi
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/edges_$STAMP"
mkdir -p "$LOG_DIR"

declare -a OK_STEPS=()
declare -a FAIL_STEPS=()

# --------------------------------------------------------------- helper ----

run_step() {
    local label="$1"; shift
    local logfile="$LOG_DIR/${label}.log"
    echo ""
    echo "======================================================================"
    echo ">> [$label]  $(date '+%H:%M:%S')"
    echo "   $*"
    echo "======================================================================"
    local start end elapsed
    start="$(date +%s)"
    if "$@" 2>&1 | tee "$logfile"; then
        end="$(date +%s)"; elapsed=$(( end - start ))
        echo "   -> OK ($label) in ${elapsed}s"
        OK_STEPS+=("$label (${elapsed}s)")
    else
        end="$(date +%s)"; elapsed=$(( end - start ))
        echo "   -> FALLITO ($label) dopo ${elapsed}s — vedi $logfile"
        FAIL_STEPS+=("$label")
    fi
}

# gnn_edges <label> <dataset> <task> <num_classes|-> <encoder> <seeds> <cv_flag>
gnn_edges() {
    local label="$1" dataset="$2" task="$3" nc="$4" enc="$5" seeds="$6" cvflag="$7"
    local nc_arg=""
    [ "$nc" != "-" ] && nc_arg="--num-classes $nc"
    run_step "$label" \
        "$PY" scripts/run.py --dataset "$dataset" --task "$task" $nc_arg --encoder "$enc" \
        --models gnn --granularities $GRANS_SHORT --edge-modes $EDGE_MODES \
        --seeds $seeds $cvflag
}

# ------------------------------------------------------ blocchi dataset ----

run_daic() {
    echo ""; echo "########################  DAIC-WOZ — GNN edges  ########################"
    [ "$MODE" != "cv" ]      && gnn_edges "daic_edges_natural" daic binary - "$ENC_DAIC" "$SEEDS_NATURAL" ""
    [ "$MODE" != "natural" ] && gnn_edges "daic_edges_cv"      daic binary - "$ENC_DAIC" "$SEEDS_CV" "--cv"
}

run_imcs() {
    echo ""; echo "########################  IMCS-21 — GNN edges  ########################"
    [ "$MODE" != "cv" ]      && gnn_edges "imcs_edges_natural" imcs21 multiclass 10 "$ENC_IMCS" "$SEEDS_NATURAL" ""
    [ "$MODE" != "natural" ] && gnn_edges "imcs_edges_cv"      imcs21 multiclass 10 "$ENC_IMCS" "$SEEDS_CV" "--cv"
}

run_mtsamples() {
    echo ""; echo "########################  MTSamples — GNN edges  ########################"
    [ "$MODE" != "cv" ]      && gnn_edges "mtsamples_edges_natural" mtsamples multiclass 9 "$ENC_MTS" "$SEEDS_NATURAL" ""
    [ "$MODE" != "natural" ] && gnn_edges "mtsamples_edges_cv"      mtsamples multiclass 9 "$ENC_MTS" "$SEEDS_CV" "--cv"
}

# ---------------------------------------------------- selezione dataset ----

if [ "$#" -eq 0 ]; then
    TARGETS=(daic mtsamples imcs)
else
    TARGETS=("$@")
fi

echo "Progetto     : $ROOT"
echo "Edge modes   : $EDGE_MODES   (sequential gia' presente dai run precedenti)"
echo "Modalita'    : $MODE   (natural: $SEEDS_NATURAL | cv: $SEEDS_CV)"
echo "Granularita' : $GRANS_SHORT"
echo "Dataset      : ${TARGETS[*]}"
echo "Log          : $LOG_DIR"
echo ""
echo "NB: solo GNN, solo dataset corti. ECtHR escluso (kNN O(N^2) sui doc lunghi)."

if ! "$PY" -c "import torch" 2>/dev/null; then
    echo "ATTENZIONE: import torch fallisce. Virtualenv attivo?"
fi

GLOBAL_START="$(date +%s)"
for t in "${TARGETS[@]}"; do
    case "$t" in
        daic)          run_daic ;;
        imcs|imcs21)   run_imcs ;;
        mtsamples|mts) run_mtsamples ;;
        *) echo "Dataset sconosciuto: '$t' (usa: daic mtsamples imcs)" ;;
    esac
done
GLOBAL_END="$(date +%s)"

# ------------------------------------------------------------ riepilogo ----

echo ""
echo "######################################################################"
echo "RIEPILOGO EDGES  (durata totale: $(( GLOBAL_END - GLOBAL_START ))s)"
echo "######################################################################"
echo "OK (${#OK_STEPS[@]}):"
for s in "${OK_STEPS[@]}"; do echo "   ✓ $s"; done
if [ "${#FAIL_STEPS[@]}" -gt 0 ]; then
    echo "FALLITI (${#FAIL_STEPS[@]}):"
    for s in "${FAIL_STEPS[@]}"; do echo "   ✗ $s"; done
    exit 1
else
    echo ""
    echo "Fatto. Rilancia analyze.py: gnn_*_knn e gnn_*_both compaiono come"
    echo "setup separati accanto a gnn_*_sequential, dentro il rispettivo protocollo."
fi