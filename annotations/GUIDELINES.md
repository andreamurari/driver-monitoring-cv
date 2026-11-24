# Guida di annotazione

Riquadri (bounding box) su immagini State Farm per un detector a 4 classi. Strumento: Label
Studio in locale, partendo da pre-etichette automatiche (`scripts/prelabel.py`) corrette a mano.
Nel repo ci sono solo la selezione (`selection.csv`), la configurazione di Label Studio e questa
guida. Né le immagini né le etichette YOLO sono pubblicate: le etichette sono dati derivati dal
dataset, che le regole della competizione non permettono di ridistribuire. Si rigenerano con
`scripts/select_images.py` → `scripts/prelabel.py` → annotazione in Label Studio →
`scripts/ls_to_yolo.py`, e stanno in `annotations/labels/` (esclusa da Git).

## Classi

| id | Classe | Cosa si annota |
|---|---|---|
| 0 | `phone` | telefono, in mano, all'orecchio o appoggiato |
| 1 | `cup` | bicchiere, tazza, lattina o bottiglia, in mano o no |
| 2 | `hand` | ogni mano del conducente |
| 3 | `face` | il volto del conducente |

## Regole

- **Annotare tutto ciò che è visibile**, anche solo in parte. Il riquadro copre la parte
  visibile, non quella stimata dietro un ostacolo.
- **Mani.** Si annotano tutte le mani del conducente, comprese quelle nascoste in parte (dietro
  il volante, dietro il corpo, tagliate dal bordo). La mano sinistra sul volante va annotata
  anche se è piccola e in ombra.
- **Mano e oggetto sovrapposti.** Due riquadri separati, uno per la mano e uno per il telefono o
  il bicchiere, anche se si sovrappongono quasi del tutto.
- **Telefono all'orecchio.** Si annota solo se se ne vede almeno una parte; se è del tutto
  coperto dalla mano o dalla testa, si annota solo la mano.
- **Bicchieri e bottiglie.** Si annotano anche quando non sono in mano (portabicchieri,
  cruscotto): il detector deve imparare l'oggetto, non il gesto.
- **Volti.** Si annotano anche girati di profilo o di tre quarti, dal mento alla fronte, capelli
  esclusi dove non serve.
- **Persone diverse dal conducente.** Non si annotano (nelle immagini State Farm a volte si vede
  il braccio o la gamba di chi sta sul sedile del passeggero).

## Primo lotto: 146 immagini

| Classe | Pre-etichette | Dopo la correzione | Cosa è cambiato di più |
|---|---|---|---|
| phone | 57 | 60 | alcuni "phone" erano bicchieri (c6); altri telefoni mancavano |
| cup | 46 | 67 | molti bicchieri e bottiglie bianche su sfondo chiaro non trovati da YOLO-COCO; 35 in classi diverse da c6 (portabicchieri) |
| hand | 177 | 273 | aggiunte circa 96 mani, soprattutto la sinistra sul volante |
| face | 140 | 147 | pochi volti mancanti |

Nessuna immagine è risultata davvero ambigua.

## Limiti delle pre-etichette

- **Oggetti chiari su sfondo chiaro.** YOLO-COCO perde spesso bottiglie e bicchieri bianchi
  contro il finestrino o la plastica chiara dell'abitacolo.
- **Mano lontana dalla telecamera.** MediaPipe trova di solito la mano destra, vicina alla
  telecamera, e perde la sinistra sul volante, piccola e in ombra.
- **Differenze tra conducenti.** Le pre-etichette di mani e volto sono state sensibilmente
  peggiori per i conducenti con la pelle più scura e per quelli di corporatura più robusta
  (osservazione qualitativa, non misurata). È un limite noto dei modelli pre-addestrati e un
  rischio concreto per un sistema di monitoraggio, che deve funzionare allo stesso modo per
  tutti: da misurare sul detector addestrato, con le metriche per conducente.
