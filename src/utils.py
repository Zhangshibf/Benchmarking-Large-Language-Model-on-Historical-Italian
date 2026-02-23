from src.settings import BENCHMARK_DIR
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Dict, Any
import json
import random

from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkInstance1:
    id: str
    target: str
    context: str
    candidates: List[str]
    gold: str
    gold_index: int

class BaseBenchmarkLoader(ABC):
    def __init__(self, benchmark_dir: Path):
        self.benchmark_dir = benchmark_dir

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Yield one processed instance at a time."""
        pass


class Tier1Loader(BaseBenchmarkLoader):
    def __init__(self, benchmark_dir: Path):
        super().__init__(benchmark_dir)
        self.file_path = self.benchmark_dir / "tier1" / "manzoni.json"

    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for raw_idx, instance in data.items():
            target = instance["target"]
            context = instance["context"]
            candidates = list(instance["candidates"])  # copy before shuffle
            gold = instance["gold"]
            random.shuffle(candidates)
            gold_index = candidates.index(gold)

            yield BenchmarkInstance1(
                id=raw_idx,
                target=target,
                context=context,
                candidates=candidates,
                gold=gold,
                gold_index=gold_index,
            )

class Tier2Loader(BaseBenchmarkLoader):
    def __init__(self, benchmark_dir: Path):
        super().__init__(benchmark_dir)
        self.file_path = self.benchmark_dir / "tier2" / "cavalcanti.json"

    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for raw_idx, instance in data.items():
            target = instance["selected_token_form"]
            context = instance["sentence_text"]
            candidates = list(instance["distractors_full"].extend(instance["deprel_full_name"]))
            gold = instance["deprel_full_name"]
            random.shuffle(candidates)
            gold_index = candidates.index(gold)

            yield BenchmarkInstance1(
                id=raw_idx,
                target=target,
                context=context,
                candidates=candidates,
                gold=gold,
                gold_index=gold_index,
            )


class Tier5Loader(BaseBenchmarkLoader):
    def __init__(self, benchmark_dir: Path):
        super().__init__(benchmark_dir)
        self.narrativa_authorship = self.benchmark_dir / "tier5" / "narrativa_authorship.json"
        self.poesia_authorship = self.benchmark_dir / "tier5" / "poesia_authorship.json"
        self.narrativa_ranking = self.benchmark_dir / "tier5" / "narrativa_temporal_ranking.json"
        self.poesia_ranking = self.benchmark_dir / "tier5" / "poesia_temporal_ranking.json"
        self.f_paths = [self.narrativa_authorship,self.poesia_authorship,self.narrativa_ranking,self.poesia_ranking]
    def __iter__(self):
        data=[]
        for f_path in self.f_paths:
            with open(f_path, "r", encoding="utf-8") as f:
                data.append(json.load(f_path))
        for dataset in data:
            task = dataset["file_info"]["type"]
            instances = dataset["tasks"]
            if task=="ranking":
                for instance in instances:
                    snippets = [i['text'] for i in instance["snippets"]]
                    years = [i['year'] for i in instance["snippets"]]
