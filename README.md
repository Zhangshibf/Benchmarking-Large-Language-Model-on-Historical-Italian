# Benchmarking-Large-Language-Model-on-Historical-Italian

This repository contains code and resources for benchmarking Large Language Models (LLMs) on Historical Italian texts. The project evaluates how well modern LLMs understand and process historical variants of the Italian language.

## Repository Structure
Benchmarking-Large-Language-Model-on-Historical-Italian/
```
├── benchmark/ # Contains the benchmark dataset
├── output/ # Stores LLM outputs after inference
├── eval_results/ # Stores evaluation results
├── ner_mapback/ # Legacy folder for old free-text NER parsing (no longer used)
├── src/
│ ├── settings.py
│ ├── benchmark_loader.py # Handles loading of the benchmark
│ ├── prompts.py # Contains all prompt templates
│ ├── main.py # Main script for loading benchmark and querying LLMs
│ └── evaluation.py # Script for evaluating LLM outputs
└── README.md
```

## Components

### Benchmark
The `benchmark/` directory contains the benchmark for Historical Italian.

### Source Code (`src/`)

- **`benchmark_loader.py`**: Handles loading of the benchmark dataset.

- **`prompts.py`**: Stores all prompt templates used for querying the LLMs. One prompt per task.

- **`main.py`**: The main entry point for running inference. This script:
  - Loads the benchmark using the benchmark loader
  - Queries Claude, DeepSeek, and GPT with prompts from prompts.py
  - Saves model outputs to the `output/` directory

- **`evaluation.py`**: Handles evaluation of LLM outputs. This script:
  - Reads model outputs from the `output/` directory
  - Computes evaluation metrics
  - Saves evaluation results to the `eval_results/` directory

## Usage

### Running Inference
To load the benchmark and query LLMs:

```bash
python -m src.main
```
This will process the benchmark through the configured LLMs and save the outputs in the output/ directory.

### Running Evaluation
To evaluate the LLM outputs:

```bash
python -m src.evaluation
```
This will saves evaluation results to the `eval_results/` directory

## UPDATE: NER Task Scripts

### How inference works

For the two NER tasks (`tier3_bellini` and `tier3_classense`), the pipeline uses LangChain's `with_structured_output()` instead of free-text generation. This means the model returns a structured JSON object directly, validated against a Pydantic schema:

```python
class Entity(BaseModel):
    text: str   # exact surface form as it appears in the text
    type: str   # entity type (e.g. "PER", "LOC", "ORG", "WORK")

class EntityExtraction(BaseModel):
    entities: List[Entity]
```

After the model returns the entities, `resolve_offsets()` finds the character position (`start`/`end`) of each entity in the original text using `str.find`. This produces the format expected by the evaluation script.

### Output format

Each line in the NER output JSONL file looks like:
```json
{
  "id": "DLCL_CF_E10001",
  "letter_id": "DLCL_CF_E10001",
  "entities": [
    {"text": "Roma", "type": "LOC", "start": 12, "end": 16},
    {"text": "Padre Mariangelo Fiacchi", "type": "PER", "start": 45, "end": 69}
  ]
}
```

### Backward compatibility

`evaluation.py` auto-detects the format: if the JSONL already has an `entities` key (new structured format), it uses it directly. If not (old free-text format from previous runs), it falls back to the legacy `create_mapped_ner` parser. Old results from `round1`–`round3` still evaluate correctly.

### Note on `ner_mapback/`

This folder was used in previous runs to store the parsed entities from free-text model responses. With structured output it is no longer needed and can be ignored.
