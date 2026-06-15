# Benchmarking-Large-Language-Model-on-Historical-Italian

This repository contains code and resources for benchmarking Large Language Models (LLMs) on Historical Italian texts. The project evaluates how well modern LLMs understand and process historical variants of the Italian language.

## Repository Structure
Benchmarking-Large-Language-Model-on-Historical-Italian/
```
├── benchmark/ # Contains the benchmark dataset
├── output/ # Stores LLM outputs after inference
├── eval_results/ # Stores evaluation results
├── ner_mapback/ # map LLM outputs back to NER dataset
├── src/
│ ├── settings.py
│ ├── benchmark_loader.py # Handles loading of the benchmark
│ ├── prompts.py # Contains all prompt templates
│ ├── eval_llm.py # script for loading benchmark and querying Claude, DeepSeek, and GPT
│ ├── eval_smaller_llm.py # script for loading benchmark and querying llama and Minerva
│ └── evaluation.py # Script for evaluating LLM outputs
└── README.md
```

## Components

### Benchmark
The `benchmark/` directory contains the benchmark for Historical Italian.

### Source Code (`src/`)

- **`benchmark_loader.py`**: Handles loading of the benchmark dataset.

- **`prompts.py`**: Stores all prompt templates used for querying the LLMs. One prompt per task.

- **`eval_llm.py`**: The entry point for running inference using Claude, DeepSeek and GPT. This script:
  - Loads the benchmark using the benchmark loader
  - Queries Claude, DeepSeek, and GPT with prompts from prompts.py
  - Saves model outputs to the `output/` directory

- **`eval_llm.py`**: The entry point for running inference using Minerva and Llama. This script:
  - Loads the benchmark using the benchmark loader
  - Queries Minerva and Llama with prompts from prompts.py. Minerva is loaded locally on GPU, Llama uses Openrouter API
  - Saves model outputs to the `output/` directory

- **`evaluation.py`**: Handles evaluation of LLM outputs. This script:
  - Reads model outputs from the `output/` directory
  - Computes evaluation metrics
  - Saves evaluation results to the `eval_results/` directory

## Usage
### Set up environment and API keys
Clone the repo
```angular2html
git clone https://github.com/Zhangshibf/Benchmarking-Large-Language-Model-on-Historical-Italian.git
```

Install the dependencies
```angular2html
pip install -r requirements.txt
```

Create a `.env` file in the project folder to set up API keys in the following format
```angular2html
#for GPT, Claude and DeepSeek
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
#for Llama
OPENROUTER_API_KEY=...
```

### Running Inference
To load the benchmark and query LLMs:

```bash
python -m src.eval_llm
python -m src.eval_smaller_llm
```
This will process the benchmark through the configured LLMs and save the outputs in the output/ directory.

### Running Evaluation
To evaluate the LLM outputs:

```bash
python -m src.evaluation
```
This will saves evaluation results to the `eval_results/` directory
