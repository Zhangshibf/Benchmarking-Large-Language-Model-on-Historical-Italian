from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
BENCHMARK_DIR = ROOT_DIR / "benchmark"
OUTPUT_DIR = ROOT_DIR / "output"
RESULTS_DIR = ROOT_DIR / "eval_results"
NER_DIR = ROOT_DIR / "ner_mapback"

BELLINI_FILE = BENCHMARK_DIR / "tier3" / "bellini.json"
CLASSENSE_FILE = BENCHMARK_DIR / "tier3" / "classense.json"