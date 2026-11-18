# Log degli esperimenti

Ogni riga rimanda a una config, a un commit e a un seed. Metriche su **validazione**
(4 conducenti), che è servita anche a scegliere l'epoca migliore: sono quindi un po'
ottimistiche. Il test set si usa una volta sola, alla fine del progetto.

| Data | Run | Config | Commit | Seed | Split | Epoca migliore | val log-loss | val acc | Note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | resnet18_driver_s0 | configs/resnet18.yaml | 1b2228e | 0 | per conducente | 7 | 0.4195 | 0.866 | baseline |
| 2026-11-18 | resnet18_random_s0 | configs/resnet18.yaml + `split: random` | 1b2228e | 0 | casuale per immagine | n.d. | 0.0165 | 0.996 | solo per il leakage |

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

## Prossimi passi

- [ ] Giorno 3: ResNet50, ablazione augmentation, 224 contro 320
- [ ] Salvare in `metrics.json` anche l'epoca migliore e lo storico per epoca
- [ ] Stampare il device all'avvio di `train.py`
