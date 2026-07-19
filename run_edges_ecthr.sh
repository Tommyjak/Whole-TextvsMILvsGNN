#!/usr/bin/env bash
#
# run_edges_ecthr.sh — Ablation sugli ARCHI della GNN (knn e both) su ECtHR.
#
# Complemento di run_edges.sh, che copre la stessa ablation sui dataset corti
# (daic, imcs21, mtsamples) ed esclude ECtHR di proposito. Qui si chiude il buco:
# le varianti di grafo 'knn' (vicini semantici sul cosine) e 'both' (sequenziali +
# semantici) vengono aggiunte SOLO per la GNN e SOLO su ECtHR. Il 'sequential'
# esiste gia' dai run di run_all.sh.
#
# PERCHE' e' uno script separato da run_edges.sh:
#   il kNN costruisce la matrice di similarita' N×N sui chunk di OGNI documento.
#   Su ECtHR i documenti sono lunghi (~100 chunk a granularita' 512) e sono ~9000:
#   il costo O(N^2) per documento e' di un altro ordine rispetto ai dataset corti.
#   Tenerlo a parte permette di lanciarlo quando c'e' GPU e tempo, senza bloccare
#   l'ablation sui corti.
#
# PERCHE' non c'e' la modalita' CV:
#   ECtHR ha uno split UFFICIALE cronologico ed e' l'unico protocollo previsto per
#   questo dataset (vedi run_mccv.sh, che lo esclude). Niente --cv qui: mescolare
#   i due protocolli e' esattamente cio' che il progetto evita.
#
# PERCHE' funziona senza toccare i risultati esistenti:
#   il run_name della GNN include l'edge_mode -> ecthr_gnn_g512_seed42_knn e' una
#   cartella distinta da ecthr_gnn_g512_seed42_sequential. I run gia' fatti non
#   vengono sfiorati; analyze.py mostra knn/both/sequential come setup separati.
#
# PARAMETRI ECtHR (identici a run_all.sh, per non introdurre confounder):
#   task multilabel, 10 classi, encoder dei chunk bert-base-uncased,
#   granularita' 512 (rapporto 8:1 col whole-text Longformer@4096).
#
# USO:
#   ./run_edges_ecthr.sh                      # precompute (se serve) + knn e both
#   EDGE_MODES="knn" ./run_edges_ecthr.sh     # solo una variante
#   SEEDS="42 43" ./run_edges_ecthr.sh        # override seed (default 42..46)
#   GRANS_ECTHR="256 512" ./run_edges_ecthr.sh
#   SKIP_PRECOMPUTE=1 ./run_edges_ecthr.sh    # cache gia' popolata, vai diretto
#
# NOTE: virtualenv ATTIVO; GPU fortemente consigliata. run.py e
#       precompute_embeddings.py sono idempotenti: rilanciando si riprende.

set -uo pipefail

# ------------------------------------------------------------------ config ----

EDGE_MODES="${EDGE_MODES:-knn both}"    # 'sequential' gia' fatto da run_all.sh
SEEDS="${SEEDS:-42 43 44 45 46}"
GRANS_ECTHR="${GRANS_ECTHR:-512}"       # stessa granularita' di run_all.sh
SKIP_PRECOMPUTE="${SKIP_PRECOMPUTE:-0}"
PY="${PY:-python3}"

ENC_ECTHR_CHUNK="bert-base-uncased"     # encoder dei chunk MIL/GNN su ecthr

# ---------------------------------------------------- individua la root ----

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SELF_DIR/scripts/run.py" ]; then
    ROOT="$SELF_DIR"
elif [ -f "$SELF_DIR/../scripts/run.py" ]; then
    ROOT="$(cd "$SELF_DIR/.." && pwd)"
else
    echo "ERRORE: non trovo scripts/run.py. Metti run_edges_ecthr.sh nella root del progetto."
    exit 1
fi
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/edges_ecthr_$STAMP"
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
    echo "   log: $logfile"
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

# ------------------------------------------------------------- esecuzione ----

echo "Progetto     : $ROOT"
echo "Dataset      : ecthr  (multilabel, 10 classi, split UFFICIALE)"
echo "Edge modes   : $EDGE_MODES   (sequential gia' presente dai run precedenti)"
echo "Seed         : $SEEDS"
echo "Granularita' : $GRANS_ECTHR"
echo "Encoder      : $ENC_ECTHR_CHUNK"
echo "Log          : $LOG_DIR"
echo ""
echo "NB: solo GNN. MIL e whole-text ignorano gli archi, rilanciarli non ha senso."
echo "NB: il kNN e' O(N^2) sui chunk di ogni documento -> su ECtHR e' lento."

if ! "$PY" -c "import torch" 2>/dev/null; then
    echo "ATTENZIONE: '$PY -c \"import torch\"' fallisce. Hai attivato il virtualenv?"
fi

GLOBAL_START="$(date +%s)"

# 1. PRECOMPUTE — la GNN legge gli embedding frozen dalla cache. Idempotente:
#    se la cache e' gia' popolata da run_all.sh questo step costa pochi secondi.
if [ "$SKIP_PRECOMPUTE" != "1" ]; then
    run_step "ecthr_precompute" \
        "$PY" scripts/precompute_embeddings.py \
        --dataset ecthr --encoder "$ENC_ECTHR_CHUNK" --granularities $GRANS_ECTHR
fi

# 2. TRAINING — run.py fa il prodotto seed x granularita x edge_mode.
run_step "ecthr_edges_natural" \
    "$PY" scripts/run.py --dataset ecthr --task multilabel --num-classes 10 \
    --encoder "$ENC_ECTHR_CHUNK" --models gnn \
    --granularities $GRANS_ECTHR --edge-modes $EDGE_MODES --seeds $SEEDS

GLOBAL_END="$(date +%s)"

# ------------------------------------------------------------ riepilogo ----

echo ""
echo "######################################################################"
echo "RIEPILOGO EDGES ECtHR  (durata totale: $(( GLOBAL_END - GLOBAL_START ))s)"
echo "######################################################################"
echo "OK (${#OK_STEPS[@]}):"
for s in "${OK_STEPS[@]}"; do echo "   ✓ $s"; done
if [ "${#FAIL_STEPS[@]}" -gt 0 ]; then
    echo "FALLITI (${#FAIL_STEPS[@]}):"
    for s in "${FAIL_STEPS[@]}"; do echo "   ✗ $s   (log: $LOG_DIR/$s.log)"; done
    echo ""
    echo "run.py e' idempotente: rilancia lo stesso comando per riprendere."
    exit 1
else
    echo ""
    echo "Fatto. Rilancia analyze.py: ecthr_gnn_*_knn e ecthr_gnn_*_both compaiono"
    echo "come setup separati accanto a ecthr_gnn_*_sequential, nel protocollo [ufficiale]."
fi
