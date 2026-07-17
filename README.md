# Side by Side: MIL vs GNN vs Whole-Text for Long Medical Text

**Corso:** AI in Bioinformatics — UniMORE (proff. Ficarra, Lovino)
**Documento:** sintesi di scelte, motivazioni e piano implementativo *prima* della codifica.

---

## 1. Il progetto, come è stato impostato

L'obiettivo è uno **studio comparativo controllato** su dati testuali medici molto lunghi. Si raccolgono più dataset che forniscono testo medico esteso e si applicano **tre modelli**:

1. **MIL** — frammenta il testo e aggrega le parti;
2. **GNN** — frammenta il testo e aggrega le parti reintroducendo struttura relazionale;
3. **Whole-text** — non frammenta, processa il testo in un blocco unico.

È ammesso e previsto che ogni dataset abbia **codice di preprocessing separato** (non serve una pipeline unica: i dataset sono eterogenei internamente e il codice non viene valutato). Il training gira su **server esterni potenti**, quindi è accettabile un carico computazionale alto e l'uso massiccio di strutture e modelli pre-esistenti. È ammesso includere un dataset **non-bio** per verificare se il dominio clinico cambia l'esito.

**Scadenza:** esame 21/22 luglio, progetto pronto qualche giorno prima (~18 luglio).

---

## 2. La domanda di ricerca (riformulazione)

Il progetto **non** chiede "quale modello classifica meglio". Chiede:

> Per il testo lungo, **frammentare conviene**? E se sì, con quale **aggregazione** (MIL o GNN)? Oppure è meglio processare tutto il testo **in modo olistico** (whole-text)?

Questa riformulazione governa ogni scelta successiva, perché il nemico numero uno diventano i **confounder**: se i tre modelli differiscono in più di una cosa alla volta, non si può attribuire una differenza di risultato alla frammentazione. Tutto il design serve a garantire che **l'unica variabile che cambia tra i modelli sia l'aggregazione**.

---

## 3. Principi di design

### 3.1 Rappresentazione canonica condivisa

Anche se ogni dataset ha preprocessing diverso, tutti i loader producono la **stessa struttura intermedia**:

```
Document(doc_id, segments: list[str], label)
```

La logica *dataset-specifica* (come si legge e si pulisce ogni fonte) vive nei loader; da lì in poi i tre modelli consumano una struttura identica. Così i loader restano separati e liberi (come richiesto), ma il resto del codice è condiviso e la comparazione è pulita.

### 3.2 Encoder frozen condiviso tra MIL e GNN

MIL e GNN devono usare lo **stesso encoder congelato**, che produce gli **stessi embedding**, salvati su disco in cache (`.pt`). L'identità è garantita a livello di filesystem: gli embedding si calcolano **una volta**, e i due modelli leggono gli stessi file. L'**unica** differenza tra MIL e GNN diventa così l'aggregazione.

### 3.3 La struttura delle cartelle rende visibile l'argomento

Tutto ciò che deve essere identico vive in un posto solo; diverge **solo** ciò che è la variabile sperimentale. Il layout stesso deve rendere auto-evidente l'equità dell'esperimento (vedi §11).

---

## 4. Le tre pipeline, definite con precisione

### Whole-text (non frammenta)
Prende la stringa unica del documento, la passa nell'encoder fino al suo limite di token (`max_length`, con `truncation=True`), estrae un vettore-documento dal token `[CLS]`, lo classifica. **Fa fine-tuning** dell'encoder (il `[CLS]` è un buon vettore-documento solo se l'encoder viene addestrato; su encoder congelato è debole).

### MIL (insieme non ordinato)
Frammenta in chunk, encoda ogni chunk con l'encoder **frozen** condiviso → embedding cachati, poi aggrega via **gated attention pooling** (Ilse et al. 2018). I pesi di attenzione danno interpretabilità (quali chunk pesano).

### GNN (grafo)
Usa gli **stessi identici chunk e gli stessi embedding cachati** del MIL. Costruisce un **grafo**: nodi = chunk, feature = embedding, archi = **sequenziali** (`i→i+1`) e/o **semantici** (kNN sul cosine). Applica **GAT** (message passing) + readout a livello di grafo. La costruzione degli archi è essa stessa un'**ablation**.

### La differenza vera tra MIL e GNN

> **Attenzione all'equivoco:** il MIL **usa** l'attention (è il suo pooling). La differenza tra MIL e GNN **non** è "attenzione sì / attenzione no".

La differenza è la **struttura tra i segmenti**:

- **MIL** tratta i segmenti come un **insieme non ordinato**: pesa ogni segmento *indipendentemente*, senza scambio di informazione tra segmenti prima dell'aggregazione.
- **GNN** fa **scambiare informazione** ai segmenti lungo gli archi (message passing): ogni nodo si aggiorna guardando i vicini, *prima* del readout.

Se la GNN batte il MIL, il merito è degli **archi** (la struttura relazionale). Questo è il risultato interpretabile che il confronto MIL-vs-GNN è progettato per isolare.

---

## 5. La segmentazione è chunking generico (non coppie Q&A)

La segmentazione **non** è più il pairing domanda-risposta (era una scelta specifica di DAIC, abbandonata). È un **chunking a finestra fissa** agnostico rispetto al dataset: finestra di N token con un piccolo **overlap** (stride) per mitigare la perdita ai confini.

Conseguenze:
- generalizza a **qualsiasi** testo lungo (note cliniche, sentenze, articoli), non solo alle interviste;
- la **granularità** del chunk diventa una **variabile misurata**, non una costante fissata a priori;
- la domanda "conta la coerenza semantica del chunk?" si **misura empiricamente** confrontando le granularità, invece di assumerla.

MIL e GNN chiamano lo **stesso** chunker con gli **stessi** parametri → chunk byte-identici.

---

## 6. La strategia di scaling verso il basso

Idea centrale: **riscalare l'intero setup** in modo che il **rapporto** tra la finestra del whole-text e la finestra dei chunk resti realistico, portando la soglia di "documento lungo" abbastanza in basso da far **sforare anche i dataset medio-corti** (IMCS, MTSamples).

- Si **cappa il whole-text a 512 token**. Così un documento IMCS da ~580 token *è* un documento lungo: sfora, viene troncato, e diventa un caso in cui la frammentazione può mostrare il suo valore. Si ottiene **variabilità di lunghezza *dentro* i dataset bio**, senza dipendere solo da ECtHR per il regime "lungo". "Lungo" è ridefinito **in relazione alla capacità del modello**, non in valore assoluto.

- **Ciò che conta è la proporzione** whole-text : chunk. A piena scala, Longformer@4096 con chunk da 256 dà un rapporto 16:1. Riscalando il whole-text a 512, per mantenere un rapporto interessante il chunk scende.

- **Il chunk a 32 è troppo aggressivo.** Riscalando il chunk **non si riscala l'encoder**: un BERT che riceve 32 token produce embedding degradati e spezza il testo a metà frase. Si misurerebbe "chunk troppo piccoli producono embedding poveri" (artefatto), non "lo split aiuta a questa scala". **Pavimento ragionevole: 64–128 token.** Rapporti più modesti (8:1, 4:1) ma reali e difendibili; a 512 IMCS e MTSamples sforano comunque.

- **Meglio ancora:** rendere la **granularità l'ablation centrale** (64 / 128 / 256) a whole-text fisso a 512. Non si indovina "la scala giusta": si spazza, e il grafico diventa *"come cambia il vantaggio dello split al variare della granularità, a parità di finestra whole-text"*. È esattamente la domanda di ricerca.

- **Coerenza informativa:** nel setup scalato il whole-text vede **meno testo per costruzione** (perde la coda oltre i 512). Questo **è il fenomeno studiato**, non un difetto — ma va **dichiarato** esplicitamente. MIL e GNN vedono tutti i chunk (documento intero).

- Il **Longformer@4096 non sparisce**: resta come **cella di controllo a contesto lungo** su ECtHR (e volendo IMCS), per misurare il costo del troncamento quando il no-split ha davvero tutto il testo.

---

## 7. Perché BERT e non Longformer per il whole-text@512

Il **Longformer cappato a 512 è la scelta di encoder sbagliata**, per ragioni meccaniche:

- La sua **attenzione sparsa a finestra scorrevole** serve ad ammortizzare sequenze **lunghe**; sotto i 512 token non offre alcun vantaggio su un BERT normale.
- I suoi pesi pre-addestrati sono ottimizzati per documenti lunghi con **global attention** attiva. Usarlo a 512 lo fa lavorare **fuori dal suo regime**: più lento e potenzialmente più debole.

**Encoder corretto per il whole-text@512:** un **BERT standard** — anzi, lo **stesso** encoder che produce gli embedding dei chunk in MIL/GNN. Proprietà preziosa: se whole-text@512 e i chunk condividono lo stesso backbone, l'**unica** differenza tra i tre modelli torna a essere *solo* l'aggregazione, non anche l'encoder. Il confounder si riduce ulteriormente.

Per dataset: **BERT cinese** per IMCS, **BERT inglese (clinico dove possibile)** per MTSamples e DAIC.

Il **Longformer** trova il suo posto giusto come encoder della **cella di controllo whole-text@4096** (ECtHR, opz. IMCS), dove lavora nel suo regime.

> BERT@512 nell'esperimento principale, Longformer@4096 nel controllo. Longformer@512 nel principale sarebbe il peggio dei due mondi.

**Nota sul chunk a 64:** è il *pavimento* del ragionevole per un BERT, non un default comodo. Va bene come estremo basso dell'ablation, ma 128 e 256 devono essere nella spazzata (l'optimum probabilmente sta lì; 64 può già mostrare degrado).

---

## 8. Strategia encoder: frozen vs fine-tuned

- **MIL e GNN → encoder frozen** (setting principale). Isola l'effetto dell'aggregazione, riduce il costo, rende i due perfettamente confrontabili. Embedding **cachati** su disco.
- **Whole-text → fine-tuning** dell'encoder (setting naturale; il `[CLS]` frozen è debole).
- Il backbone (modello base) è lo **stesso** tra i tre; MIL/GNN lo **congelano**, il whole-text lo **addestra**. Condividono il punto di partenza ma divergono sul fatto che l'encoder venga allenato: questo riduce il confounder di *famiglia* dell'encoder, ma la differenza frozen-vs-finetuned **resta** ed è **dichiarata**.
- Il **fine-tuning dell'encoder per il MIL** (la vecchia pipeline DAIC end-to-end) diventa un'**ablation** ("il fine-tuning aiuta il MIL?"), non si butta.

**Pooling:** con encoder frozen, **masked mean-pooling** dei token è preferibile al `[CLS]` grezzo (debole senza fine-tuning). Si testa su DAIC e si usa il vincitore ovunque per MIL/GNN. Il whole-text (fine-tunato) usa il `[CLS]`.

---

## 9. Loss, metriche, robustezza statistica

**Loss (condivisa, parametrizzata per task):**
- binario (DAIC) e multi-label (ECtHR) → `BCEWithLogitsLoss` con `pos_weight`;
- multi-classe (IMCS 10 malattie, MTSamples specialità) → `CrossEntropyLoss`.
- Il sigmoid **non** sta nel `forward` (si lavora sui logit). L'harness è parametrizzato da `num_classes` / tipo di task.

**Metriche (torchmetrics, identiche per tutti):** F1 (macro + classe positiva), **AUROC**, **AUPRC** (metrica primaria su dati sbilanciati), **MCC**, balanced accuracy, confusion matrix. Multi-classe: micro/macro F1, AUROC/AUPRC one-vs-rest. Accuracy solo come riferimento.

**Efficienza (asse di prima classe):** tempo di training totale e per epoca, latenza di inferenza per documento, throughput, picco di memoria GPU, numero di parametri (trainable vs totali), opz. FLOPs. Va contabilizzato il **costo di embedding/preprocessing** che MIL/GNN pagano e il whole-text no. Figura chiave: **accuratezza vs costo**.

**Robustezza statistica:** **multi-seed** (≈5) anche con split fisso; intervalli di confidenza via bootstrap; test appaiati per affermare "X batte Y" — **McNemar** sulle predizioni dello stesso test set, **Wilcoxon** tra fold/dataset. Con test piccoli (DAIC n=47) senza questo le differenze sono rumore.

---

## 10. Il framework dei regimi (chiave interpretativa)

L'esito dipende da **quattro proprietà** del documento:
1. **lunghezza** vs finestra di contesto dell'encoder;
2. **sparsità del segnale** (localizzato vs diffuso);
3. **dipendenza tra passaggi** (conta solo *quali* passaggi, o anche *come si relazionano*?);
4. **quantità di dati etichettati** (pochi → favorisce frozen + testa leggera; molti → il long-encoder fine-tunato può competere).

**Tre regimi prevedibili:**
- documento **entro** il contesto + segnale **diffuso** → **whole-text** dovrebbe vincere;
- documento **oltre** il contesto + segnale **sparso/localizzato** → **MIL** (niente troncamento, l'attention localizza);
- documento **lungo** + segnale **relazionale** → **GNN** > MIL (gli archi codificano ciò che l'insieme non ordinato scarta).

**Bio vs non-bio:** verifica se l'esito dipende dal **dominio clinico** o dalla **geometria del testo** (lunghezza/sparsità). Se un legale-lungo e un clinico-lungo si comportano uguale mentre un conversazionale-corto differisce, allora conta la geometria, non la natura — il risultato più forte e meno ovvio.

**Caveat DAIC:** transcript di lunghezza moderata, per lo più entro il contesto, segnale diffuso → DAIC **da solo** potrebbe essere un caso in cui il whole-text è competitivo e lo split non stacca. Non è un fallimento, è un punto nello spazio dei regimi; ma è la ragione per cui serve almeno un dataset lungo e strutturato accanto.

---

## 11. I dataset (portfolio, accessi, ruoli)

**Regola d'accesso (deadline-driven):** nessun dataset *core* dipende da una richiesta d'accesso con esito incerto. Esclusi per lead-time: **MIMIC, i2b2/n2c2, RSDD, SMHD**.

| Dataset | Dominio / lingua | Geometria | Task / label | Accesso | Ruolo |
|---|---|---|---|---|---|
| **DAIC-WOZ** | mental health / EN | conversazionale, medio-corto | depressione binaria (PHQ8) | già in mano | banco sviluppo, punto *small-data* |
| **IMCS-21** | clinico / **ZH** | conversazionale, corto (~580 tok) | 10 malattie pediatriche (multi-classe) | libero (GitHub) | **bio primario** (scala, multi-classe, cross-lingua) |
| **MTSamples** | clinico / EN | non conversazionale, spesso >512 tok | specialità medica (multi-classe) | Kaggle CC0 | bio non conversazionale medio |
| **ECtHR** (LexGLUE) | legale / EN | **molto lungo**, migliaia di token | violazione articoli (multi-**label**) | libero (HF) | regime "lungo" + controllo Longformer@4096 |

Note operative:
- **IMCS-21**: usare la versione 2.0 (`imcs21-cblue`); le label del test set sono sul leaderboard TIANCHI/CBLUE → si lavora su train/dev. Struttura: `self_report` + `dialogue` (turni con speaker, atto dialogico, entità BIO, sintomi normalizzati + stato 0/1/2) + `report` (2 referti) + diagnosi. Richiede **encoder cinesi**.
- **MTSamples**: `tboyle10/medicaltranscriptions`, `mtsamples.csv`. Colonne utili: `transcription`, `medical_specialty`. Da gestire: righe con `transcription` vuoto (filtrare), ~40 classi sbilanciate (decidere politica top-N / accorpamento), il target "specialità" è un po' artificiale (meno clinicamente significativo della diagnosi IMCS).
- **ECtHR**: `coastalcph/lex_glue`, config `ecthr_a` (o `ecthr_b`). Il campo `text` è **già una lista di paragrafi** → segmentazione pronta. `labels` **multi-label** → obbliga la modalità BCE nell'harness.

### Il confounder di portfolio (limite da dichiarare)

Nel portfolio a tre celle principali, **"lunghezza" e "non-bio" coincidono in un solo dataset** (ECtHR). Se lo split vince su ECtHR e non sugli altri, non si distingue se è per la **lunghezza** o per il **dominio legale**: i due fattori sono confusi. La strategia di scaling (§6) **mitiga parzialmente** il problema, perché cappando il whole-text a 512 anche i dataset bio spaziano tra "entro" e "oltre" la finestra (variabilità di lunghezza *within-dataset*, che controlla il dominio). Ma il confound domini×lunghezza a livello di portfolio **va comunque dichiarato**: con 3 celle su 4, alcune conclusioni restano correlazionali. Cella aggiuntiva ideale per romperlo: un **bio-lungo** (full-text PMC/PubMed con label derivata) o un **non-bio-corto**.

### Traduzione in lingua franca: perché no

Tradurre tutto in inglese è **scartato come spina dorsale**: la MT è *lossy* proprio sulle due cose misurate — altera la **lunghezza** (il cinese è denso; i conteggi di token cambiano in modo non uniforme → contamina l'asse lunghezza) e **lava via il segnale clinico** sottile (disfluenze, ripetizioni, scelte lessicali) che in mental-health/dialogo porta l'informazione. Inoltre **non serve**: il confronto è **interno a ciascun dataset** (mai F1 assoluto tra DAIC e IMCS; solo split-vs-whole *dentro* ogni dataset), quindi la lingua è costante dove conta. Se si volesse uno spazio condiviso, la via principiata è un **encoder multilingue** (XLM-R + long-encoder multilingue), non la traduzione. La traduzione ha un solo posto legittimo: **ablation su un singolo dataset** (in appendice).

---

## 12. Tecnologie

- **PyTorch + PyTorch Lightning** — loop, logging, riproducibilità, multi-GPU.
- **HuggingFace Transformers** (`AutoModel`/`AutoTokenizer`) — encoder.
- **masked mean-pooling** (o sentence-transformers) — embedding di segmento frozen.
- **torch_geometric** — `GATConv`, global pooling, `Data`/`DataLoader` per grafi.
- **torchmetrics** — metriche identiche.
- **scikit-learn** — split stratificati, alcune metriche.
- Encoder per dataset: **BERT cinese** (`bert-base-chinese` / `hfl/chinese-roberta-wwm-ext` / MC-BERT) per IMCS; **BERT clinico EN** (ClinicalBERT / BioBERT / PubMedBERT) per MTSamples; **MentalBERT** per DAIC.
- Controllo lungo: **Longformer / Clinical-Longformer** (EN), **Longformer cinese** (es. `schen/longformer-chinese-base-4096` o Lawformer — *verificare disponibilità*).
- **W&B o MLflow** — tracking dei run; **Hydra** — gestione della matrice di esperimenti.

---

## 13. Struttura del progetto

```
~/Documents/universita/new_AI4Bio/
├── datasets/
│   ├── daic-woz/
│   ├── imcs21/
│   ├── mtsamples/
│   └── ecthr/
├── notebooks/          # analisi finale, figure, test di significatività
├── results/            # log, checkpoint, metriche per run
├── scripts/            # entrypoint: precompute_embeddings.py, run.py
└── src/
    ├── config/         # matrice esperimenti: dataset × modello × granularità × seed
    ├── data/           # CONDIVISO
    │   ├── canonical.py    # Document(doc_id, segments, label) — il contratto
    │   ├── loaders/        # un file per dataset → produce Document
    │   │   ├── daic.py
    │   │   ├── imcs21.py
    │   │   ├── mtsamples.py
    │   │   └── ecthr.py
    │   └── chunking.py     # chunker parametrico (granularità = iperparametro)
    ├── encoders/       # CONDIVISO
    │   ├── backbone.py     # carica il BERT (it/zh/clinico) da config
    │   └── cache.py        # calcola e salva gli embedding frozen .pt
    ├── models/         # QUI E SOLO QUI i tre modelli divergono
    │   ├── heads.py        # testa MLP + loss: IDENTICHE, condivise
    │   ├── mil.py          # attention pooling
    │   ├── gnn.py          # grafo + GAT + readout
    │   └── whole_text.py   # forward su testo troncato
    └── training/       # CONDIVISO
        ├── lit_module.py   # LightningModule generico
        ├── metrics.py      # torchmetrics, identiche
        └── loop.py         # train/eval, seed, early stopping
```

**Principio:** ciò che deve essere identico vive in un posto solo; diverge solo l'aggregazione (`models/`), e persino lì `heads.py` + loss sono condivise. Tutta l'eterogeneità dei dataset è confinata in `data/loaders/`: **il modello riceve un `Document` canonico e non sa mai da quale dataset provenga**. La struttura rende **auto-evidente** l'argomento "cambia solo l'aggregazione".

*Promemoria tecnico:* servono i file `__init__.py` nelle sottocartelle di `src/` appena iniziano gli import tra moduli.

---

## 14. Piano temporale (≈ 1 → 18 luglio)

1. Rappresentazione canonica + layer di **caching** degli embedding.
2. Refactor del MIL a **encoder frozen** dentro un `LightningModule` + **harness di valutazione condiviso** (la vecchia versione fine-tunata → ablation).
3. Pipeline **GNN** (costruzione grafo, GAT, readout) sulla stessa cache.
4. Pipeline **whole-text** (BERT@512) + cella di controllo **Longformer@4096**.
5. Run **multi-seed** dei tre modelli.
6. **Ablation**: granularità (64/128/256), archi del grafo (sequenziali / semantici), frozen vs fine-tuned, mean-pool vs CLS.
7. Test di **significatività**, figure di confronto, tabella di **efficienza**.
8. Stesura relazione + buffer.

DAIC e IMCS come banco di sviluppo; MTSamples ed ECtHR aggiunti **prima della fine** perché danno la varianza di lunghezza e dominio su cui poggia la domanda di ricerca.

---

## 15. Prossimo passo

**`src/data/canonical.py`** — la dataclass `Document`, contratto tra loader (che la producono) e modelli (che la consumano). È la fondazione da cui dipende tutto il resto.
