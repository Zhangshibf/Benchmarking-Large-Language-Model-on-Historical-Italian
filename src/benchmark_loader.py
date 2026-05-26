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

@dataclass
class BenchmarkInstanceRanking:
    id: int
    snippets: List[str]
    order: list[int]

@dataclass
class BenchmarkInstanceNERLetter:
    dataset:str
    id: str
    text:str
    entities:list[[str]]
    entites_type:list[[str]]



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
            gold_index = candidates.index(gold)+1
            # print(candidates)
            # print(gold)
            # print(gold_index)

            yield BenchmarkInstance1(
                id=raw_idx,
                target=target,
                context=context,
                candidates=candidates,
                gold=gold,
                gold_index=gold_index,
            )

class Tier2LoaderDeprel(BaseBenchmarkLoader):
    def __init__(self, benchmark_dir: Path):
        super().__init__(benchmark_dir)
        self.file_path = self.benchmark_dir / "tier2" / "cavalcanti_deprel_prediction.json"

    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for raw_idx, instance in data.items():
            target = instance["selected_token_form"]
            context = instance["sentence_text"]
            candidates = instance["distractors_full"]
            candidates.append(instance["deprel_full_name"])
            gold = instance["deprel_full_name"]
            random.shuffle(candidates)
            gold_index = candidates.index(gold)+1
            # print(candidates)
            # print(gold)
            # print(gold_index)

            yield BenchmarkInstance1(
                id=raw_idx,
                target=target,
                context=context,
                candidates=candidates,
                gold=gold,
                gold_index=gold_index,
            )

class Tier2LoaderHead(BaseBenchmarkLoader):
    def __init__(self, benchmark_dir: Path):
        super().__init__(benchmark_dir)
        self.file_path = self.benchmark_dir / "tier2" / "cavalcanti_head_prediction.json"

    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for raw_idx, instance in data.items():
            target = instance["selected_token_form"]
            context = instance["sentence_text"]
            candidates = [i["form"] for i in instance["distractors"]]
            candidates.append(instance["true_head_form"])
            gold = instance["true_head_form"]
            random.shuffle(candidates)
            gold_index = candidates.index(gold)+1
            # print(candidates)
            # print(gold)
            # print(gold_index)

            yield BenchmarkInstance1(
                id=raw_idx,
                target=target,
                context=context,
                candidates=candidates,
                gold=gold,
                gold_index=gold_index,
            )

class Tier3LoaderBELLINI(BaseBenchmarkLoader):
    def __init__(self, benchmark_dir: Path):
        super().__init__(benchmark_dir)

        self.bellini_path = self.benchmark_dir / "tier3" / "bellini.json"

    def __iter__(self):
        with open(self.bellini_path,"r",encoding='utf-8') as f:
            dataset = json.load(f)
            dataset_name = "bellini"
            for letter in dataset:
                id = letter['letter_id']
                text=letter['text']
                entites = [i['text'] for i in letter['entities']]
                entites_type = [i['type'] for i in letter['entities']]

                yield BenchmarkInstanceNERLetter(dataset=dataset_name,id=id,text = text,entities=entites,entites_type=entites_type)

class Tier3LoaderCLASSENSE(BaseBenchmarkLoader):
    def __init__(self, benchmark_dir: Path):
        super().__init__(benchmark_dir)
        self.classense_path = self.benchmark_dir / "tier3" / "classense.json"

    def __iter__(self):
        with open(self.classense_path,"r",encoding='utf-8') as f:
            dataset = json.load(f)
            dataset_name =  "classense"
            for letter in dataset:
                id = letter['letter_id']
                text=letter['text']
                entites = [i['text'] for i in letter['entities']]
                entites_type = [i['type'] for i in letter['entities']]

                yield BenchmarkInstanceNERLetter(dataset=dataset_name,id=id,text = text,entities=entites,entites_type=entites_type)

class Tier4Loader(BaseBenchmarkLoader):
    def __init__(self, benchmark_dir: Path):
        super().__init__(benchmark_dir)
        self.file_path = self.benchmark_dir / "tier4" / "dante.json"
    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for instance in data:
                id = instance['id']
                target = instance["dante_text"]
                candidates = [instance['chunk_a']['text'],instance['chunk_b']['text']]
                correct = instance["correct_answer"]
                if correct=="A":
                    gold_idx = 0
                elif correct=="B":
                    gold_idx=1
                gold = candidates[gold_idx]
                gold_idx+=1
                # print(candidates)
                # print(gold)
                # print(gold_idx)
                yield BenchmarkInstance1(
                    id=id,
                    target=target,
                    context="",
                    candidates=candidates,
                    gold=gold,
                    gold_index=gold_idx,
                )

class Tier5LoaderAuthorship(BaseBenchmarkLoader):
    def __init__(self, benchmark_dir: Path):
        super().__init__(benchmark_dir)
        self.narrativa_authorship = self.benchmark_dir / "tier5" / "narrativa_authorship.json"
        self.poesia_authorship = self.benchmark_dir / "tier5" / "poesia_authorship.json"

        self.f_paths = [self.narrativa_authorship,self.poesia_authorship]
    def __iter__(self):
        data=[]
        for f_path in self.f_paths:
            with open(f_path, "r", encoding="utf-8") as f:
                data.append(json.load(f))
        for dataset in data:
            instances = dataset["tasks"]

            for instance in instances:
                id = instance['task_id']
                snippets = [i['text'] for i in instance["snippets"]]
                gold_index = instance['solution']+1
                # print(id)
                # print(snippets)
                # print(gold_index)

                yield BenchmarkInstance1(
                    id=id,
                    target="",
                    context="",
                    candidates=snippets,
                    gold="",
                    gold_index=gold_index,
                )


class Tier5LoaderRanking(BaseBenchmarkLoader):
    def __init__(self, benchmark_dir: Path):
        super().__init__(benchmark_dir)
        self.narrativa_ranking = self.benchmark_dir / "tier5" / "narrativa_temporal_ranking.json"
        self.poesia_ranking = self.benchmark_dir / "tier5" / "poesia_temporal_ranking.json"
        self.f_paths = [self.narrativa_ranking,self.poesia_ranking]
    def __iter__(self):
        data=[]
        for f_path in self.f_paths:
            with open(f_path, "r", encoding="utf-8") as f:
                data.append(json.load(f))
        for dataset in data:

            instances = dataset["tasks"]

            for instance in instances:
                id = instance['task_id']
                raw_snippets = [i['text'] for i in instance["snippets"]]
                raw_years = [int(i['year']) for i in instance["snippets"]]

                # Zip them so we can shuffle and keep track of original snippet content
                combined = list(zip(raw_snippets, raw_years))
                random.shuffle(combined)
                shuffled_snippets, shuffled_years = zip(*combined)

                # We need the 1-based indices of the snippets sorted by year (Newest to Oldest)
                # 1. Create list of (index+1, year)
                indexed_years = list(enumerate(shuffled_years, start=1))
                # 2. Sort by year descending (Most Recent -> Oldest)
                indexed_years.sort(key=lambda x: x[1], reverse=True)
                # 3. Extract the ordered indices
                gold_sequence = [item[0] for item in indexed_years]
                # print(shuffled_snippets)
                # print(shuffled_years)
                # print(gold_sequence)
                yield BenchmarkInstanceRanking(
                    id=id,
                    snippets=list(shuffled_snippets),
                    order=gold_sequence
                )