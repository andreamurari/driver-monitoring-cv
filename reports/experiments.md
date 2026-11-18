# Log degli esperimenti

Ogni riga rimanda a una config, a un commit e a un seed. Metriche su **validazione**
(4 conducenti), che è servita anche a scegliere l'epoca migliore: sono quindi un po'
ottimistiche. Il test set si usa una volta sola, alla fine del progetto.

| Data | Run | Config | Commit | Seed | Split | Epoca migliore | val log-loss | val acc | Note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | resnet18_driver_s0 | configs/resnet18.yaml | 1b2228e | 0 | per conducente | 7 | 0.4195 | 0.866 | baseline |
| 2026-11-18 | resnet18_random_s0 | configs/resnet18.yaml + `split: random` | 1b2228e | 0 | casuale per immagine | n.d. | 0.0165 | 0.996 | solo per il leakage |
| 2026-11-18 | resnet18_driver_s1 | configs/resnet18_s1.yaml | a243f03 | 1 | per conducente | 8 | 0.5156 | 0.844 | rumore tra seed |
| 2026-11-18 | resnet18_aug-none_s0 | configs/resnet18_aug-none.yaml | a243f03 | 0 | per conducente | 7 | 0.6820 | 0.778 | ablazione: senza augmentation |
| 2026-11-18 | resnet18_aug-flip_s0 | configs/resnet18_aug-flip.yaml | a243f03 | 0 | per conducente | 8 | 0.4904 | 0.834 | ablazione: flip, non conclusivo |
| 2026-11-18 | resnet18_320_s0 | configs/resnet18_320.yaml | a243f03 | 0 | per conducente | 7 | 0.2763 | 0.912 | migliore finora, nuova baseline |
| 2026-11-18 | resnet50_s0 | configs/resnet50.yaml | a243f03 | 0 | per conducente | 4 | 0.5255 | 0.820 | overfitting dopo l'epoca 4 |

Ambiente: Kaggle, 1× Tesla T4, batch 64, 224 letterbox, 3 epoche head (lr 1e-3) +
5 epoche fine-tuning (lr 1e-4), AdamW, coseno, nessun label smoothing.
Metriche complete in `reports/metrics/`.

## Giorno 2: baseline e leakage

- **Leakage:** stesso modello e stessa config, cambia solo lo split. Con lo split per
  conducente (18 conducenti in training, 4 mai visti in validazione) l'accuratezza è
  0.866; con lo split casuale per immagine arriva a 0.996. Il secondo numero è gonfiato:
  ogni conducente della validazione compare anche nel training, con fotogrammi quasi
  identici, quindi il modello riconosce invece di generalizzare. In un'applicazione
  reale il sistema vede persone nuove: il numero da riportare è 0.866.
- **Fase head contro fine-tuning:** con il backbone congelato la val accuracy si ferma a
  circa 0.51; dopo il fine-tuning arriva a 0.866. Le feature di ImageNet da sole non
  bastano a distinguere posture molto simili (telefono a destra o a sinistra, telefono
  o bere): serve adattare il backbone perché impari a guardare mani e oggetti.
- **Overfitting:** train accuracy 1.000 contro val 0.866. La val loss è minima
  all'epoca 7 (0.4195) e risale all'epoca 8 (0.4404); il checkpoint salvato è quello
  dell'epoca 7. Con la train accuracy già al 100% altre epoche non aiuterebbero: servono
  più regolarizzazione e augmentation (giorno 3), non un training più lungo.
- **Rumore:** la prova veloce con 1+1 epoche aveva dato 0.885, più del run completo
  (0.866). Con soli 4 conducenti in validazione una differenza di 2 punti può dipendere
  da chi sono quei conducenti: un singolo numero non basta per confrontare due
  configurazioni. Servono più seed e, più avanti, una cross-validation per gruppi.
- **Tempi:** circa 60 s per epoca con il backbone congelato e 68 s con il fine-tuning
  completo, nonostante il secondo aggiorni migliaia di volte più parametri. Il collo di
  bottiglia non è la GPU ma il caricamento dei dati sulla CPU (lettura e decodifica di
  JPEG 640x480, resize, augmentation). Possibile rimedio: pre-ridimensionare le immagini.

## Giorno 3: ablazioni

Ogni run cambia una sola cosa rispetto alla baseline `resnet18_driver_s0`.
Metrica principale: val log-loss (l'accuracy come riferimento).

| Run | Cosa cambia | Previsione | Risultato |
|---|---|---|---|
| resnet18_driver_s1 | seed 1 | simile | simile: misura il rumore |
| resnet18_aug-none_s0 | nessuna augmentation | peggiore | peggiore |
| resnet18_aug-flip_s0 | standard + flip con scambio etichette | migliore | non conclusivo |
| resnet18_320_s0 | risoluzione 320 | migliore | migliore |
| resnet50_s0 | ResNet50 | migliore | smentita: non migliore |

- **Rumore (seed 0 contro seed 1):** stessa config, cambiano solo i pesi iniziali della
  testa, l'ordine dei batch e le estrazioni dell'augmentation. La log-loss passa da 0.420
  a 0.516 e l'accuratezza da 0.866 a 0.844. Una differenza di questo ordine (circa 0.1 di
  log-loss, 2 punti di accuratezza) non basta a dire che una configurazione è migliore.
  Due seed sottostimano il rumore vero: per le decisioni importanti ne servono di più.
- **Augmentation:** senza augmentation la log-loss sale a 0.682 e l'accuratezza scende a
  0.778, ben oltre il rumore: con soli 18 conducenti in training il modello impara a
  memoria (train accuracy 1.000) e l'augmentation di colore e geometria è necessaria.
  Il flip con scambio delle etichette dà 0.490 di log-loss (migliore del seed 1) ma 0.834
  di accuratezza (un punto sotto entrambi i seed): le due metriche non sono d'accordo e la
  differenza è piccola, quindi il risultato non è conclusivo.
- **Risoluzione:** a 320 (letterbox, lato lungo 320) la log-loss scende a 0.276 e
  l'accuratezza sale a 0.912, ben oltre il rumore. È il risultato più netto della
  giornata: dettagli piccoli come telefono, mani e bicchiere occupano pochi pixel e a 224
  si perdono. Costo: circa il doppio dei pixel per immagine, quindi training e inferenza
  più lenti (da misurare la latenza su CPU).
- **ResNet50:** il miglior checkpoint è all'epoca 4, la prima del fine-tuning (0.526);
  dopo la val loss sale fino a 0.612 mentre la train accuracy va verso 1.0. È overfitting:
  23 milioni di parametri contro 11, con gli stessi 18 conducenti e lo stesso learning
  rate. Rispetto al seed 1 la differenza è dentro il rumore: la previsione "migliore" è
  smentita, ma non è dimostrato che sia peggiore. Con questa ricetta un modello più
  grande non aiuta; servirebbero lr più basso, meno epoche o più regolarizzazione.

## Prossimi passi

- [x] Giorno 3: ResNet50, ablazione augmentation, 224 contro 320
- [x] Salvare in `metrics.json` anche l'epoca migliore e lo storico per epoca
- [x] Stampare il device all'avvio di `train.py`
- [ ] Usare 320 come nuova baseline per i confronti successivi
- [ ] ResNet50 con lr di fine-tuning più basso (es. 3e-5) o meno epoche
- [ ] Più seed per il flip prima di decidere se tenerlo
- [ ] Misurare la latenza su CPU a 224 e a 320
