## Cos'è questo progetto

**"Side by Side: MIL vs GNN vs Whole-Text for Long Medical Text"** — progetto per il corso
*AI in Bioinformatics* (UniMORE, proff. Ficarra e Lovino).

È uno **studio comparativo controllato** su testo medico/clinico lungo. Non chiede "quale modello
classifica meglio", ma:

> Per il testo lungo, **frammentare conviene**? E se sì, con quale **aggregazione** — MIL o GNN?
> Oppure è meglio processare tutto il testo in modo **olistico** (whole-text)?

Tre modelli a confronto sullo stesso testo:
1. **MIL** — frammenta in chunk e aggrega con gated attention pooling (insieme non ordinato).
2. **GNN** — stessi chunk, ma li tratta come nodi di un grafo con message passing (struttura relazionale).
3. **Whole-text** — non frammenta: tronca il testo alla finestra dell'encoder e lo processa in un blocco.

### Il principio guida: eliminare i confounder

Tutto il design serve a garantire che **l'unica variabile che cambia tra i modelli sia l'aggregazione**.
Conseguenze architetturali fondamentali:

- **MIL e GNN condividono lo stesso encoder frozen** e leggono gli **stessi identici embedding cachati** su disco (`.pt`). È materialmente impossibile che vedano chunk o vettori diversi → l'unica differenza tra loro è come aggregano.
- **MIL, GNN e whole-text condividono la stessa testa di classificazione, la stessa loss e le stesse metriche** (`src/models/heads.py`, `src/training/metrics.py`).
- Confounder **dichiarato**: il whole-text fa **fine-tuning** dell'encoder (il `[CLS]` è un buon vettore-documento solo se addestrato), mentre MIL/GNN usano l'encoder **frozen**. Il backbone di partenza è lo stesso.

### La differenza vera MIL vs GNN

Non è "attenzione sì / attenzione no" (il MIL *usa* l'attention come pooling). È la **struttura tra segmenti**:
- **MIL** pesa ogni segmento indipendentemente, senza scambio di informazione tra segmenti.
- **GNN** fa scambiare informazione ai segmenti lungo gli archi (message passing) prima del readout.

Se la GNN batte il MIL, il merito è degli **archi** — ed è questo il risultato interpretabile che il confronto isola.

### La strategia di scaling (chiave del design)

Si **cappa il whole-text a 512 token** con un BERT standard, così che anche i dataset medio-corti
(IMCS ~580 token, MTSamples spesso >512) **sforino** la finestra e diventino casi in cui la frammentazione
può mostrare il suo valore. "Lungo" è ridefinito *in relazione alla capacità del modello*, non in assoluto.
Ciò che conta è mantenere costante il **rapporto** finestra-whole-text : chunk (≈8:1). ECtHR usa invece
Longformer@4096 come cella di controllo a contesto lungo, con chunk a 512 (stesso rapporto 8:1).

Perché BERT@512 e non Longformer@512 per l'esperimento principale: sotto i 512 token l'attenzione sparsa
del Longformer non dà vantaggi e lavora fuori dal suo regime. E usare lo stesso BERT dei chunk riduce ancora
il confounder di encoder.

## Struttura del repository

```
src/
├── data/                    # CONDIVISO — la parte agnostica al dataset a valle dei loader
│   ├── canonical.py         # Document(doc_id, text, label, meta): IL CONTRATTO dati
│   ├── chunking.py          # chunk_text(): finestre fisse di token con overlap (granularità = ablation)
│   └── loaders/             # UN loader per dataset → produce list[Document]. Qui vive TUTTA
│       ├── daic.py          #   l'eterogeneità dei dataset; a valle nessun modulo sa la provenienza.
│       ├── imcs21.py
│       ├── mtsamples.py
│       └── ecthr.py
├── encoders/                # CONDIVISO
│   ├── backbone.py          # FrozenEncoder: carica il BERT, lo congela (eval + no_grad), masked mean-pooling
│   └── cache.py             # precompute_embeddings(): loader→chunk→encode→salva .pt (idempotente)
├── models/                  # QUI E SOLO QUI i tre modelli divergono
│   ├── heads.py             # ClassificationHead (MLP) + build_loss + to_target + class weights: CONDIVISE
│   ├── mil.py               # GatedAttentionPooling (Ilse et al. 2018) + testa condivisa
│   ├── gnn.py               # build_edges + GATConv multi-layer + global attention pooling + testa condivisa
│   └── whole_text.py        # encoder fine-tunato sul testo intero troncato → [CLS] → testa condivisa
└── training/                # CONDIVISO
    ├── datasets.py          # CachedBagDataset (MIL/GNN, legge .pt) + RawTextDataset (whole-text, testo grezzo)
    ├── splits.py            # resolve_splits (ufficiale o stratificato) + resolve_splits_montecarlo (CV)
    ├── metrics.py           # build_metrics (torchmetrics), IDENTICHE per i tre modelli
    ├── lit_module.py        # BaseLitModule + MIL/GNN/WholeText LitModule (differenza confinata a _forward_batch)
    └── loop.py              # ExperimentConfig + train_one(): una cella della matrice, end-to-end

scripts/
├── precompute_embeddings.py # entrypoint: popola la cache degli embedding frozen
├── run.py                   # orchestratore matrice (modelli × granularità × seed), aggrega su seed, idempotente
└── analyze.py               # aggrega results/**/result.json → tabelle, McNemar, boxplot

run_all.sh                   # pipeline completa a SPLIT UFFICIALE (tutti i dataset, ECtHR incluso)
run_mccv.sh                  # pipeline in MONTE CARLO CV (--cv, 10 fold), solo daic/imcs/mtsamples
cache/                       # embedding .pt: cache/{dataset}/{encoder_slug}/g{chunk}_o{overlap}/{doc_id}.pt
datasets/                    # daic-woz, imcs21, mtsamples, ecthr (dati grezzi)
results/                     # un cartella per run: result.json (metriche + predictions + timing)
```

**Regola strutturale d'oro:** ciò che deve essere identico vive in un posto solo; diverge solo l'aggregazione
(`models/`), e persino lì `heads.py`, la loss e le metriche sono condivise. Il modello riceve un `Document`
canonico e non sa mai da quale dataset provenga. La struttura rende **auto-evidente** che "cambia solo l'aggregazione".

## Flusso dati

```
loader → Document(text, label, meta)          # data/loaders/*.py
   │
   ├─ [MIL/GNN] chunk_text → FrozenEncoder → cache .pt     # una volta, condiviso
   │      └─ CachedBagDataset → (embeddings N×H) → MIL o GNN
   │
   └─ [whole-text] RawTextDataset → testo grezzo → encoder fine-tunato (troncato a max_length)
```

- Il precompute degli embedding gira **una volta** per `(dataset, encoder, chunk_size, overlap)`. Poi ogni
  training MIL/GNN parte in secondi caricando i tensori. È questo che rende praticabile la matrice.
- La label viaggia dentro il `.pt` così il training non deve riconsultare il loader.
- Lo `split` di ogni documento vive in `meta` (il precompute embedda tutto; il filtro train/dev/test è a valle).

## I dataset

| Dataset | Dominio / lingua | Geometria | Task | Encoder di default | Ruolo |
|---|---|---|---|---|---|
| **DAIC-WOZ** | mental health / EN | conversazionale, medio-corto | binary (depressione PHQ8) | `mental/mental-bert-base-uncased` | banco sviluppo, small-data |
| **IMCS-21** | clinico / **ZH** | conversazionale corto (~580 tok) | multiclass (10 malattie) | `bert-base-chinese` | bio primario (multi-classe, cross-lingua) |
| **MTSamples** | clinico / EN | referti, spesso >512 tok | multiclass (9 specialità) | `emilyalsentzer/Bio_ClinicalBERT` | bio non conversazionale |
| **ECtHR** (LexGLUE) | legale / EN | molto lungo (migliaia di tok) | **multilabel** (10 articoli) | chunk `bert-base-uncased`, whole `allenai/longformer-base-4096` | regime "lungo" + controllo Longformer@4096 |

Note operative dai loader:
- **DAIC**: solo `*_TRANSCRIPT.csv` come testo; il pairing Q&A è abbandonato (trascrizione ricostruita in ordine temporale). Sessioni escluse per convenzione AVEC2017: 342/394/398/460 (tecnici), 451/458/480 (Ellie mancante). Il file di test usa nomi colonna incoerenti (`PHQ_Binary` vs `PHQ8_Binary`) — gestito da `_find_col`.
- **IMCS-21**: testo = `self_report` + turni (医生→DOCTOR / 患者→PATIENT). Split ufficiale train/dev/test, tutti con label. Serve encoder cinese. Vocab malattia→indice persistito su disco.
- **MTSamples**: nessuno split ufficiale (tutti `meta['split']='all'`, split generato a valle). Pulizia: drop transcription vuote, dedup (evita leakage), rimozione non-specialità, soglia ≥40 campioni/classe, cap 300/classe → ~1132 doc, 9 classi.
- **ECtHR**: i paragrafi vengono **concatenati** in una stringa unica (la segmentazione la fa il chunker generico, per coerenza col confronto). Label multi-label; lista vuota valida e frequente (~900 casi senza violazione). Split ufficiale cronologico.

**Confounder di portfolio da dichiarare:** "lunghezza" e "non-bio" coincidono solo in ECtHR — se lo split
vince lì e non altrove, non si distingue se conta la lunghezza o il dominio legale. Lo scaling a 512 mitiga
(variabilità di lunghezza within-dataset anche nei bio), ma il confound va comunque esplicitato.

## Task, loss, metriche

Tre tipi di task, un'unica astrazione parametrizzata (in `heads.py` e `metrics.py`):
- **binary** (DAIC) → 1 logit, `BCEWithLogitsLoss` con `pos_weight`, soglia 0.5.
- **multiclass** (IMCS 10, MTSamples 9) → C logit, `CrossEntropyLoss` con class weights, argmax.
- **multilabel** (ECtHR) → C logit, `BCEWithLogitsLoss` per-classe, soglia 0.5 per-classe.

Il sigmoid/softmax **non** è nel forward: si lavora sui logit. I class weights si calcolano **solo dalle label
di training** (mai dev/test → niente leakage). Metriche primarie (robuste allo sbilanciamento): F1 macro,
AUROC, **AUPRC**, MCC, balanced accuracy; accuracy solo come riferimento.

**Robustezza statistica:** run **multi-seed** (≈5 su split ufficiale, 10 fold in Monte Carlo CV); il test di
**McNemar** su predizioni per-documento richiede che `result.json` salvi le `predictions` (fa `PredictionCollector` in `loop.py`).

## Come si eseguono gli esperimenti

Ordine: **prima si precomputano gli embedding, poi si allena.** MIL/GNN dipendono dalla cache.

```bash
# 1. Popola la cache degli embedding frozen (una volta per dataset/encoder/granularità)
python scripts/precompute_embeddings.py --dataset daic \
    --encoder mental/mental-bert-base-uncased --granularities 64 128

# 2. Lancia la matrice modelli × granularità × seed (aggrega su seed, idempotente)
python scripts/run.py --dataset daic --task binary \
    --encoder mental/mental-bert-base-uncased \
    --models mil gnn whole_text --granularities 64 128 --seeds 42 43 44 45 46

# 3. Aggrega risultati, test di significatività, figure
python scripts/analyze.py                      # tutti i dataset
python scripts/analyze.py --datasets daic ecthr
```

Oppure gli orchestratori completi (attivare prima il virtualenv, idealmente GPU):

```bash
./run_all.sh                 # tutti i dataset, split UFFICIALE (ECtHR incluso, per ultimo)
./run_all.sh daic ecthr      # solo alcuni
SEEDS="42 43" ./run_all.sh   # override seed
./run_mccv.sh                # Monte Carlo CV (flag --cv, 10 fold), solo daic/imcs/mtsamples
```

**Note operative importanti:**
- `run.py` e `precompute_embeddings.py` sono **idempotenti**: saltano le celle il cui `result.json` / `.pt`
  esiste già. Rilanciare riprende una matrice interrotta; usa `--overwrite` per forzare.
- Il **whole-text ignora il chunking** → si lancia su **una sola granularità** (più valori darebbero run
  identici con nomi diversi). Nei run è `--max-length 512` (BERT) o `--max-length 4096` (Longformer/ECtHR) con lr basso (`2e-5`, fine-tuning).
- `edge_modes` (GNN) si applica **solo alla GNN**: MIL e whole-text ignorano gli archi.

### Split ufficiale vs Monte Carlo CV — non mescolare mai

Sono **due protocolli distinti**. `analyze.py` li tiene separati sempre: i run CV finiscono in cartelle
`*_cv` (`config['cv']==True`) e non si confrontano mai con i run a split ufficiale. Per i dataset che hanno
entrambi (daic, imcs21, mtsamples) si producono due analisi separate `[ufficiale]` e `[cv]`. ECtHR ha solo lo
split ufficiale. La chiave di gruppo include `_cv` così i seed collassano nel gruppo giusto senza mai fondere i due mondi.

## Configurazione di un esperimento

Tutto ciò che definisce un esperimento vive in `ExperimentConfig` (dataclass in `src/training/loop.py`),
esplicito e serializzabile (finisce nei risultati per riproducibilità). Campi chiave: `dataset`, `model`,
`task`, `num_classes`, `encoder`, `chunk_size`, `overlap`, `cv`/`n_folds`, `seed`, `max_epochs`, `patience`,
`batch_size`, `lr`, `max_length`, `edge_mode`/`knn_k` (solo GNN). Il `monitor` di default è `val/f1`, corretto
automaticamente a `val/f1_macro` per multiclass/multilabel.

## Ablation previste

- **granularità** dei chunk (64 / 128 / 256) a whole-text fisso — l'ablation centrale.
- **archi del grafo**: `sequential` (i→i+1) / `knn` (kNN sul cosine) / `both`.
- **frozen vs fine-tuned** encoder per il MIL (la vecchia pipeline DAIC end-to-end diventa ablation).
- **mean-pool vs CLS** per il pooling dell'encoder frozen.

## Stack tecnologico

PyTorch + **Lightning** (loop, logging, riproducibilità), **HuggingFace Transformers** (`AutoModel`/`AutoTokenizer`),
**torch-geometric** (`GATConv`, global pooling), **torchmetrics** (metriche), scipy (McNemar), pandas/pyarrow (dati,
parquet ECtHR), matplotlib (figure). Vedi `requirements.txt`.

## Convenzioni da rispettare quando si lavora qui

- Il **`Document` canonico** (`data/canonical.py`) è il contratto: `text` è il documento intero come singola
  stringa (la segmentazione **non** avviene nel loader — è ablation, vive nel chunker). `label` è `int` per
  single-label, `list[int]` per multi-label. `meta` è la via di fuga dataset-specifica; nessun modello deve dipenderne.
- Il **tokenizer** passato al chunker DEVE essere quello dell'encoder che poi encoderà i chunk, altrimenti i
  confini dei chunk si disallineano. `chunk_size` conta i token di contenuto (`add_special_tokens=False`).
- **Non rompere l'equità sperimentale:** qualsiasi cosa condivisa (encoder frozen, cache, testa, loss, metriche,
  split) deve restare identica tra i tre modelli. Toccare solo `models/` per cambiare l'aggregazione.
- I commenti e le docstring del codice sono in italiano e molto dettagliati: mantenere lo stile.
