# 🎓 Student Dropout Prediction: Un approccio di Machine Learning

Questo progetto esplora l'applicazione di algoritmi di classificazione per prevedere il successo accademico o l'abbandono (*dropout*) degli studenti universitari, basandosi esclusivamente su dati disponibili alla fine del primo semestre. Il lavoro è stato sviluppato come progetto finale per il corso di **Machine Learning** (A.A. 2025/2026) presso l'Università degli Studi di Salerno.

## 🧠 Motivazione e Obiettivo
Il tasso di abbandono universitario rappresenta una sfida critica per gli atenei. L'obiettivo di questo studio non è solo predittivo, ma **di supporto decisionale**: identificare tempestivamente gli studenti "a rischio" permette di attivare interventi mirati (tutorato, orientamento, supporto didattico) prima che l'abbandono diventi definitivo.

## 📊 Dataset: Caratteristiche e Preprocessing
Il dataset di partenza proviene dalla UCI Machine Learning Repository ed è composto da 4.424 istanze[cite: 6]. Il lavoro di preparazione dei dati è stato fondamentale per garantire la validità scientifica della previsione:

* **Riformulazione del Task**: Il problema originale era una classificazione multiclasse; è stato convertito in **binario** (Graduate vs Dropout) rimuovendo la classe *Enrolled*.
* **Prevenzione del Data Leakage**: Sono state rimosse tutte le feature relative al secondo semestre, poiché contenevano informazioni "future" rispetto al momento della previsione (ovvero la fine del primo semestre).
* **Gestione delle Variabili**: È stata effettuata una distinzione rigorosa tra variabili numeriche (come l'età o i crediti) e variabili categoriche codificate numericamente (come la facoltà o lo status sociale), applicando *One-Hot Encoding* solo a queste ultime.

## 🚀 Metodologia Sperimentale
Il progetto ha previsto il confronto di tre architetture classiche per valutarne l'efficacia in contesti di sbilanciamento delle classi (la classe *Graduate* è maggioritaria rispetto al *Dropout*):

1. **Gaussian Naive Bayes (Baseline)**: Utilizzato come riferimento statistico probabilistico.
2. **Decision Tree**: Scelto per l'elevata interpretabilità delle regole decisionali.
3. **Random Forest**: Utilizzato come modello di ensemble per ridurre l'overfitting e migliorare la stabilità delle predizioni.

Tutti i modelli sono stati integrati in **Pipeline di Scikit-learn** per automatizzare lo scaling delle feature (ove necessario) e prevenire errori di manipolazione dei dati durante la cross-validation[cite: 5].

## 📈 Risultati Chiave
La valutazione è andata oltre la semplice *Accuracy*, focalizzandosi sulla capacità del modello di intercettare i casi di *Dropout* (la classe di interesse).

| Modello | Accuracy | F1-Score (Dropout) |
| :--- | :--- | :--- |
| Gaussian Naive Bayes | 41.05% | 55.44% |
| Decision Tree | 85.54% | 80.75% |
| **Random Forest** | **88.84%** | **84.63%** |

*La **Random Forest** si è confermata la scelta vincente, garantendo il miglior bilanciamento tra la precisione delle segnalazioni e la capacità di non "perdere" studenti a rischio (recall).*

## 🛠️ Come riprodurre il progetto
Il codice è strutturato per essere modulare. Per eseguire l'analisi:
1. Assicurati di avere `pandas`, `scikit-learn` e `numpy` installati.
2. Esegui il file `dropout_prediction.py` per avviare l'addestramento e visualizzare le metriche di valutazione direttamente nel terminale[cite: 5].

---
*Progetto accademico realizzato da Francesco Di Giovanni | Dipartimento di Informatica, UNISA*
