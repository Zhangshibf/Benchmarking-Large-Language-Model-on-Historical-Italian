# Benchmarking-Large-Language-Model-on-Historical-Italian

This repository contains code and resources for benchmarking Large Language Models (LLMs) on Historical Italian texts. The project evaluates how well modern LLMs understand and process historical variants of the Italian language.

## Repository Structure
Benchmarking-Large-Language-Model-on-Historical-Italian/
```
├── benchmark/ # Contains the benchmark dataset
├── output/ # Stores LLM outputs after inference
├── eval_results/ # Stores evaluation results
├── ner_mapback # This is for the evaluation of NER task. Stores the mapping from LLMs output to entities in text
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
