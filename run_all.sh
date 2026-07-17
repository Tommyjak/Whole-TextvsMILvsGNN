#!/usr/bin/env bash
#
# run_all.sh — Orchestratore di tutti gli esperimenti del progetto
#              "Side by Side: MIL vs GNN vs Whole-Text".
#
# Per ogni dataset:
#   1. PRECOMPUTE degli embedding frozen (MIL/GNN) a piu' granularita (64, 128)
#   2. TRAINING con run.py SEPARATO per ciascun modello (mil, gnn, whole_text)
#
# Granularita:
#   - mil/gnn  -> ablation su GRANS (64 128): run.py fa il prodotto seed x granularita
#   - whole_text -> UNA sola granularita (GRAN_WT): il whole-text ignora il chunking,
#     quindi lanciarlo su piu' granularita creerebbe run IDENTICI con nomi diversi
#     (spreco puro). Un valore solo basta.
#
# Logica di scala (vedi NOTA su ECtHR piu' sotto):
#   - dataset corti (daic, imcs21, mtsamples):
#       whole-text = BERT@512 ; chunk MIL/GNN = 64/128 ; stesso encoder -> confronto pulito
#   - dataset lungo (ecthr):
#       whole-text = Longformer@4096 ; chunk MIL/GNN = vedi GRANS_ECTHR
#
# USO:
#   ./run_all.sh                      # tutti i dataset
#   ./run_all.sh daic ecthr           # solo quelli elencati
#   SEEDS="42 43" ./run_all.sh        # override seed (default 42..46)
#   GRANS_SHORT="64 128 256" ./run_all.sh   # override granularita
#
# NOTE:
#   - virtualenv ATTIVO; idealmente GPU (ECtHR + Longformer e' pesante su CPU).
#   - run.py e' idempotente: rilanciando, salta le celle gia' calcolate.
#   - lo script NON si ferma al primo errore: logga, continua, riepiloga alla fine.

set -uo pipefail

# ------------------------------------------------------------------ config ----

SEEDS="${SEEDS:-42 43 44 45 46}"
PY="${PY:-python3}"

# granularita dell'ablation (precompute + mil/gnn)
GRANS_SHORT="${GRANS_SHORT:-64 128}"   # daic, imcs, mtsamples
GRANS_ECTHR="${GRANS_ECTHR:-512}"      # ecthr: rapporto 8:1 col whole-text @4096
# granularita SINGOLA per il whole-text (chunking ignorato: un solo valore)
GRAN_WT="${GRAN_WT:-64}"

# learning rate del whole-text (fine-tuning dell'encoder -> lr basso)
LR_WT="${LR_WT:-2e-5}"

# encoder per dataset
ENC_DAIC="mental/mental-bert-base-uncased"
ENC_IMCS="bert-base-chinese"              # MedBERT cinese: trueto/medbert-base-wwm-chinese
ENC_MTS="emilyalsentzer/Bio_ClinicalBERT"
ENC_ECTHR_CHUNK="bert-base-uncased"       # encoder dei chunk MIL/GNN su ecthr
ENC_ECTHR_WHOLE="allenai/longformer-base-4096"  # whole-text a contesto lungo

# ---------------------------------------------------- individua la root ----

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SELF_DIR/scripts/run.py" ]; then
    ROOT="$SELF_DIR"
elif [ -f "$SELF_DIR/../scripts/run.py" ]; then
    ROOT="$(cd "$SELF_DIR/.." && pwd)"
else
    echo "ERRORE: non trovo scripts/run.py. Metti run_all.sh nella root del"
    echo "        progetto (o in scripts/) e rilancia da lì."
    exit 1
fi
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/run_$STAMP"
mkdir -p "$LOG_DIR"

declare -a OK_STEPS=()
declare -a FAIL_STEPS=()

# --------------------------------------------------------------- helper ----

# run_step <label> <comando...>
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
        return 0
    else
        end="$(date +%s)"; elapsed=$(( end - start ))
        echo "   -> FALLITO ($label) dopo ${elapsed}s — vedi $logfile"
        FAIL_STEPS+=("$label")
        return 1
    fi
}

# ------------------------------------------------------ blocchi dataset ----

run_daic() {
    echo ""; echo "########################  DAIC-WOZ  ########################"
    run_step "daic_precompute" \
        "$PY" scripts/precompute_embeddings.py \
        --dataset daic --encoder "$ENC_DAIC" --granularities $GRANS_SHORT
    run_step "daic_mil" \
        "$PY" scripts/run.py --dataset daic --task binary --encoder "$ENC_DAIC" \
        --models mil --granularities $GRANS_SHORT --seeds $SEEDS
    run_step "daic_gnn" \
        "$PY" scripts/run.py --dataset daic --task binary --encoder "$ENC_DAIC" \
        --models gnn --granularities $GRANS_SHORT --seeds $SEEDS
    run_step "daic_whole" \
        "$PY" scripts/run.py --dataset daic --task binary --encoder "$ENC_DAIC" \
        --models whole_text --granularities $GRAN_WT --max-length 512 \
        --seeds $SEEDS --lr "$LR_WT"
}

run_imcs() {
    echo ""; echo "########################  IMCS-21  ########################"
    run_step "imcs_precompute" \
        "$PY" scripts/precompute_embeddings.py \
        --dataset imcs21 --encoder "$ENC_IMCS" --granularities $GRANS_SHORT
    run_step "imcs_mil" \
        "$PY" scripts/run.py --dataset imcs21 --task multiclass --num-classes 10 --encoder "$ENC_IMCS" \
        --models mil --granularities $GRANS_SHORT --seeds $SEEDS
    run_step "imcs_gnn" \
        "$PY" scripts/run.py --dataset imcs21 --task multiclass --num-classes 10 --encoder "$ENC_IMCS" \
        --models gnn --granularities $GRANS_SHORT --seeds $SEEDS
    run_step "imcs_whole" \
        "$PY" scripts/run.py --dataset imcs21 --task multiclass --num-classes 10 --encoder "$ENC_IMCS" \
        --models whole_text --granularities $GRAN_WT --max-length 512 \
        --seeds $SEEDS --lr "$LR_WT"
}

run_mtsamples() {
    echo ""; echo "########################  MTSamples  ########################"
    run_step "mtsamples_precompute" \
        "$PY" scripts/precompute_embeddings.py \
        --dataset mtsamples --encoder "$ENC_MTS" --granularities $GRANS_SHORT
    run_step "mtsamples_mil" \
        "$PY" scripts/run.py --dataset mtsamples --task multiclass --num-classes 9 --encoder "$ENC_MTS" \
        --models mil --granularities $GRANS_SHORT --seeds $SEEDS
    run_step "mtsamples_gnn" \
        "$PY" scripts/run.py --dataset mtsamples --task multiclass --num-classes 9 --encoder "$ENC_MTS" \
        --models gnn --granularities $GRANS_SHORT --seeds $SEEDS
    run_step "mtsamples_whole" \
        "$PY" scripts/run.py --dataset mtsamples --task multiclass --num-classes 9 --encoder "$ENC_MTS" \
        --models whole_text --granularities $GRAN_WT --max-length 512 \
        --seeds $SEEDS --lr "$LR_WT"
}

run_ecthr() {
    echo ""; echo "########################  ECtHR (lungo)  ########################"
    run_step "ecthr_precompute" \
        "$PY" scripts/precompute_embeddings.py \
        --dataset ecthr --encoder "$ENC_ECTHR_CHUNK" --granularities $GRANS_ECTHR
    run_step "ecthr_mil" \
        "$PY" scripts/run.py --dataset ecthr --task multilabel --num-classes 10 --encoder "$ENC_ECTHR_CHUNK" \
        --models mil --granularities $GRANS_ECTHR --seeds $SEEDS
    run_step "ecthr_gnn" \
        "$PY" scripts/run.py --dataset ecthr --task multilabel --num-classes 10 --encoder "$ENC_ECTHR_CHUNK" \
        --models gnn --granularities $GRANS_ECTHR --seeds $SEEDS
    # whole-text: Longformer@4096; granularita singola (ignorata dal modello)
    run_step "ecthr_whole" \
        "$PY" scripts/run.py --dataset ecthr --task multilabel --num-classes 10 --encoder "$ENC_ECTHR_WHOLE" \
        --models whole_text --granularities $GRAN_WT --max-length 4096 \
        --seeds $SEEDS --lr "$LR_WT"
}

# ---------------------------------------------------- selezione dataset ----

if [ "$#" -eq 0 ]; then
    TARGETS=(daic imcs mtsamples ecthr)   # ecthr per ultimo (il piu' pesante)
else
    TARGETS=("$@")
fi

echo "Progetto      : $ROOT"
echo "Seed          : $SEEDS"
echo "Gran. corti   : $GRANS_SHORT   | ECtHR: $GRANS_ECTHR   | whole-text: $GRAN_WT"
echo "Dataset       : ${TARGETS[*]}"
echo "Log           : $LOG_DIR"

if ! "$PY" -c "import torch" 2>/dev/null; then
    echo ""
    echo "ATTENZIONE: '$PY -c \"import torch\"' fallisce. Hai attivato il virtualenv?"
    echo "            (proseguo comunque, ma probabilmente fallira')"
fi

GLOBAL_START="$(date +%s)"

for t in "${TARGETS[@]}"; do
    case "$t" in
        daic)          run_daic ;;
        imcs|imcs21)   run_imcs ;;
        mtsamples|mts) run_mtsamples ;;
        ecthr)         run_ecthr ;;
        *) echo "Dataset sconosciuto: '$t' (usa: daic imcs mtsamples ecthr)" ;;
    esac
done

GLOBAL_END="$(date +%s)"

# ------------------------------------------------------------ riepilogo ----

echo ""
echo "######################################################################"
echo "RIEPILOGO  (durata totale: $(( GLOBAL_END - GLOBAL_START ))s)"
echo "######################################################################"
echo ""
echo "OK (${#OK_STEPS[@]}):"
for s in "${OK_STEPS[@]}"; do echo "   ✓ $s"; done
echo ""
if [ "${#FAIL_STEPS[@]}" -gt 0 ]; then
    echo "FALLITI (${#FAIL_STEPS[@]}):"
    for s in "${FAIL_STEPS[@]}"; do echo "   ✗ $s   (log: $LOG_DIR/$s.log)"; done
    echo ""
    echo "run.py e' idempotente: rilancia lo stesso comando per riprendere."
    exit 1
else
    echo "Tutti gli step completati. Riassunti in results/summary_*.json"
fi

# ======================================================================
# NOTA SCALA
# ======================================================================
# Dataset corti (daic/imcs/mtsamples): whole-text BERT@512, chunk 64/128 -> 8:1.
# ECtHR: whole-text Longformer@4096, chunk 512 -> 8:1 (stesso rapporto).
#
# Cosi la "scala" (rapporto finestra whole-text : chunk) e' COSTANTE fra tutti i
# dataset, come da design. Su ECtHR il chunk grande (512) tiene anche il costo
# gestibile: ~100 chunk per doc lungo, contro le centinaia che si avrebbero a 64.
#
# Se un giorno volessi l'ablation di granularita ANCHE su ECtHR (per studiare
# l'effetto della grana sui documenti lunghi), imposta p.es. GRANS_ECTHR="256 512".