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
Candidati: {answers_str}

**Formato di output:** Restituisci solo il numero (es. "1", "2", "3").
"""

PROMPT_TIER2_DEPREL = """
Analizza la relazione di dipendenza sintattica (dependency relation) per la parola target nel contesto fornito.

**Istruzioni:**
1. Esamina il target e la sua funzione sintattica all'interno della frase (contesto).
2. Confronta la funzione individuata con la lista dei andidati deprels.
3. Seleziona il numero corrispondente alla relazione corretta secondo lo standard Universal Dependencies.

**Input:**
Parole target: {target}
Contesto: {context}
Candidati deprels: {answers_str}

**Vincoli:**
Restituisci ESCLUSIVAMENTE il numero della risposta corretta. Non aggiungere commenti, etichette testuali o punteggiatura.

**Formato di output:** Restituisci solo il numero (es. "1", "2", "3").
"""

PROMPT_TIER2_HEAD="""
Identifica la testa sintattica della parola target nella frase data, scegliendo tra le quattro parole candidate.

**Istruzioni:**
1. Analizza la frase e individua la parola target: {target}.
2. Determina quale delle quattro parole candidate è la testa sintattica di {target} secondo le relazioni di dipendenza Universal Dependencies.
3. Seleziona il numero corrispondente alla parola corretta.

**Input**:
Parola target: {target}
Contesto: {context}
Candidati testa sintattica: {answers_str}

**Vincoli**:
Restituisci ESCLUSIVAMENTE il numero della risposta corretta. Non aggiungere commenti, etichette testuali o punteggiatura.

**Formato di output**: Restituisci solo il numero (es. "1", "2", "3").
"""
PROMPT_TIER3_BELLINI = """
Esegui il riconoscimento delle entità nominate (Named Entity Recognition) sul testo fornito.

**Compito:**
1. Identifica nel testo tutte le entità che appartengono ai tipi specificati.
2. Estrai l'entità esattamente come appare nel testo.
3. Associa a ogni entità il suo tipo corretto.

**Input:**
Testo: {testo}
Tipi di entità ammessi: ["PER", "PER_GROUP", "LOC", "ORG", "WORK", "MUSIC_TERM", "MISC"]

**Formato di Output richiesto:**
Restituisci ESCLUSIVAMENTE una lista di tuple nel formato: [(entità1:tipo),(entità2:tipo),...]
Non aggiungere spiegazioni, introduzioni o punteggiatura extra. Se non trovi entità, restituisci [].

## Tipi di Entità

- **PER** - Riferimenti a persone:
    - Nomi: Bellini, Rossini, Mercadante
    - Con titoli: Signor Beltrame, Signora Contessa
    - Indiretti: Signore (allocutivo formale), Ella, amico, zio, papà, sua moglie
- **PER_GROUP** - Gruppi di persone: figli, famiglia, genitori
- **LOC** - Luoghi: Parigi, Napoli, Milano, Catania
- **ORG** - Organizzazioni: Casa (editrice), Teatro, Società
- **WORK** - Titoli di opere: Puritani, Norma, Sonnambula
- **MUSIC_TERM** - Termini musicali: opera, libretto, aria, spartito, duetto
- **MISC** - Solo altri nomi propri non classificabili sopra

**Esempio di output:**
[(Leonardo da Vinci:PER),(Firenze:LOC)]
"""

PROMPT_TIER3_CLASSENSE = """
Esegui il riconoscimento delle entità nominate (Named Entity Recognition) sul testo fornito.

**Compito:**
1. Identifica nel testo tutte le entità che appartengono ai tipi specificati.
2. Estrai l'entità esattamente come appare nel testo.
3. Associa a ogni entità il suo tipo corretto.

**Input:**
Testo: {testo}
Tipi di entità ammessi: ["PER", "LOC", "ORG", "WORK"]

**Formato di Output richiesto:**
Restituisci ESCLUSIVAMENTE una lista di tuple nel formato: [(entità1:tipo),(entità2:tipo),...]
Non aggiungere spiegazioni, introduzioni o punteggiatura extra. Se non trovi entità, restituisci [].

## Tipi di Entità

- **PER**: Nomi di persona (es. "Padre Mariangelo Fiacchi", "Ambrogio Traversari", "Frate Leandro Alberti"). 
  Includi titoli completi e onorifici.

- **LOC**: Nomi di luogo (es. "Ravenna", "Camaldoli", "Venezia", "Roma", "Bologna"). 
  Città, regioni, monasteri, chiese, edifici.

- **ORG**: Organizzazioni e istituzioni (es. "Accademia della Crusca", "Ordine Camaldolese", "Studio Sbaraglia"). 
  Accademie, ordini religiosi, studi, biblioteche.

- **WORK**: Opere e pubblicazioni (es. "Istoria di Bologna", "Epistole", "Calendario Benedettino"). 
  Libri, manoscritti, lettere, cronache, qualsiasi opera citata.

**Esempio di output:**
[(Leonardo da Vinci:PER),(Firenze:LOC)]
"""

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