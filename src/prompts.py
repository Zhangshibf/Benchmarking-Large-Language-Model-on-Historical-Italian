from langchain_core.messages import HumanMessage, SystemMessage

sys_message = "Sei un modello linguistico di grandi dimensioni specializzato nel trattamento dell'italiano storico."

PROMPT_TIER1 = """
Il tuo compito è risolvere un esercizio di disambiguazione del significato delle parole (Word Sense Disambiguation). Dato il contesto e la frase target, seleziona il significato corretto tra i candidati forniti.

**Istruzioni:**
1. Analizza attentamente il contesto e la frase target.
2. Scegli l'unico significato più appropriato tra i candidati.
3. Restituisci ESCLUSIVAMENTE il numero corrispondente (senza testo, spiegazioni o formattazione).

**Input:**
Contesto: {context}
Frase target: {target}
Candidati: {candidate_str}

**Formato di output:** Restituisci solo il numero (es. "1", "2", "3").
"""

PROMPT_TIER2 = """
Analizza la relazione di dipendenza sintattica (dependency relation) per la parola target nel contesto fornito.

**Istruzioni:**
1. Esamina il target e la sua funzione sintattica all'interno della frase (contesto).
2. Confronta la funzione individuata con la lista delle etichette 'deprel' candidate.
3. Seleziona il numero corrispondente alla relazione corretta secondo lo standard Universal Dependencies.

**Input:**
Parole target: {target}
Contesto: {context}
Candidati deprels: {answers_str}

**Vincoli:**
Restituisci ESCLUSIVAMENTE il numero della risposta corretta. Non aggiungere commenti, etichette testuali o punteggiatura.

**Formato di output:** Restituisci solo il numero (es. "1", "2", "3").
"""


PROMPT_TIER3 = """
Esegui il riconoscimento delle entità nominate (Named Entity Recognition) sul testo fornito.

**Compito:**
1. Identifica nel testo tutte le entità che appartengono ai tipi specificati.
2. Estrai l'entità esattamente come appare nel testo.
3. Associa a ogni entità il suo tipo corretto.

**Input:**
Testo: {testo}
Tipi di entità ammessi: {tipi_entita}

**Formato di Output richiesto:**
Restituisci ESCLUSIVAMENTE una lista di tuple nel formato: [(entità1:tipo),(entità2:tipo),...]
Non aggiungere spiegazioni, introduzioni o punteggiatura extra. Se non trovi entità, restituisci [].

**Esempio di output:**
[(Leonardo da Vinci:PER),(Firenze:LOC)]
"""

# prompts.py

PROMPT_TIER4 = """
Analizza il testo fornito e determina con quale delle due fonti proposte esiste una relazione di intertestualità (citazione, influenza stilistica, parodia o riferimento diretto).

**Compito:**
1. Esamina il testo principale cercando riferimenti a opere precedenti.
2. Confronta il testo con la Fonte 1 e la Fonte 2.
3. Identifica quale delle due fonti ha un legame testuale dimostrabile con il testo principale.

**Input:**
Testo: {testo}
Fonte 1: {fonte_a}
Fonte 2: {fonte_b}

**Vincoli:**
Restituisci ESCLUSIVAMENTE il numero del della fonte vera del testo (es. "1", or "2"). Non aggiungere spiegazioni.

**Output:**
"""


PROMPT_TIER5_AUTHORSHIP = """
Analizza i quattro frammenti di testo forniti. Tre di essi appartengono allo stesso autore, mentre uno è di un autore diverso (l'intruso).

**Compito:**
1. Confronta lo stile, il lessico, la sintassi e l'uso della punteggiatura dei quattro frammenti.
2. Identifica quale dei quattro frammenti è stato scritto dall'autore differente.
**Input:**
Frammenti:
{snippets_str}

**Vincoli:**
Restituisci ESCLUSIVAMENTE il numero del frammento intruso (es. "1", "2", "3" o "4"). Non aggiungere spiegazioni.

**Output:**
"""


PROMPT_TIER5_RANKING = """
Analizza i quattro frammenti di testo tratti da diverse epoche della storia italiana.

**Compito:**
1. Confronta lo stile, il lessico, la sintassi e l'uso della punteggiatura dei quattro frammenti.
2. Ordina i frammenti dal più RECENTE al più ANTICO.

**Input:**
Frammenti:
{snippets_str}

**Formato di Output richiesto:**
Restituisci ESCLUSIVAMENTE solo i numeri dei frammenti come una list python, in ordine cronologico decrescente (dal più moderno al più arcaico). Non aggiungere spiegazioni.
Esempio: [3, 1, 4, 2]

**Output:**
"""