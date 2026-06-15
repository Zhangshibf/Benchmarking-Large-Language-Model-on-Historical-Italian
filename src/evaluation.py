import json
from pathlib import Path
from src.settings import OUTPUT_DIR, RESULTS_DIR,NER_DIR, BELLINI_FILE,CLASSENSE_FILE
import pandas as pd
from collections import defaultdict
def evaluate_all_tiers(output_base_dir=OUTPUT_DIR, results_base_path=RESULTS_DIR,
                       round_names=("round1", "round2", "round3")):
    model_groups = defaultdict(list)

    for round_name in round_names:
        round_dir = output_base_dir / round_name
        if not round_dir.is_dir():
            print(f"[warn] round folder not found, skipping: {round_dir}")
            continue

        for folder in round_dir.rglob('**/'):
            jsonl_files = list(folder.glob('*.jsonl'))
            if len(jsonl_files) == 8:
                model_name = folder.name
                overall_result = {}
                print(f"Found match: {folder} contains exactly 8 JSONs")
                overall_result = {}
                for jsonl_file in jsonl_files:
                    relative_path = folder.relative_to(output_base_dir)
                    target_folder = results_base_path / relative_path
                    target_folder.mkdir(parents=True, exist_ok=True)

                    output_file_path = target_folder / jsonl_file.name
                    overall_path = target_folder / "overall.json"
                    preds = []
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                preds.append(json.loads(line))
                    tier = jsonl_file.stem.replace("_results", "")
                    if tier in ["tier1", "tier2_deprel", "tier2_head", "tier4", "tier5_auth"]:
                        evaluation_result = calculate_accuracy(preds)
                        overall_result[tier] = evaluation_result
                    elif "tier3" in tier:
                        # If predictions already have 'entities' (structured output), use directly.
                        # Otherwise fall back to create_mapped_ner for legacy free-text responses.
                        if preds and "entities" in preds[0]:
                            ner_preds = [{"letter_id": p["letter_id"], "entities": p["entities"]} for p in preds]
                        else:
                            ner_mapped_file = NER_DIR / relative_path / jsonl_file.name
                            if ner_mapped_file.is_file():
                                ner_preds = json.load(open(ner_mapped_file, "r", encoding="utf-8"))
                            else:
                                create_mapped_ner(preds, ner_mapped_file)
                                ner_preds = json.load(open(ner_mapped_file, "r", encoding="utf-8"))
                        evaluation_result = evaluate_NER_result(ner_preds)
                        overall_result[tier] = evaluation_result
                    elif "rank" in tier:
                        evaluation_result = evaluate_rank_result(preds)
                        overall_result[tier] = evaluation_result

                    with open(output_file_path, "w", encoding="utf-8") as f:
                        json.dump(evaluation_result, f)
                with open(overall_path, "w", encoding="utf-8") as f:
                    json.dump(overall_result, f, indent=2)
                model_groups[model_name].append(overall_result)

    for model_name, rounds in model_groups.items():
        if not rounds:
            continue

        # Flatten all rounds, calculate mean, and unflatten
        # We use pd.json_normalize to turn nested dicts into flat columns (e.g., "tier3_bellini.strict.micro.f1")
        df = pd.json_normalize(rounds)
        mean_series = df.mean()

        # Reconstruct the nested dictionary
        final_mean_dict = {}
        for keys, value in mean_series.items():
            parts = keys.split('.')
            d = final_mean_dict
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value

        # Save the result
        summary_filename = f"{model_name}_final_mean.json"
        summary_path = results_base_path / summary_filename
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(final_mean_dict, f, indent=2)

        print(f"Generated mean results for {model_name} at {summary_path}")


from scipy.stats import spearmanr

import json
import ast
import re

import numpy as np
from difflib import SequenceMatcher


def get_overlap_ratio(s1, s2):

    match = SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))
    overlap_len = match.size

    denom = (len(s1) + len(s2) - overlap_len)
    return overlap_len / denom if denom > 0 else 0


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


from collections import defaultdict  # add near the top with your other imports

# admitted label sets per dataset
ADMITTED_TYPES = {
    "bellini":   {"PER", "PER_GROUP", "LOC", "ORG", "WORK", "MUSIC_TERM", "MISC"},
    "classense": {"PER", "LOC", "ORG", "WORK"},
}


def evaluate_NER_result(ner_preds):

    first_id = ner_preds[0]['letter_id']
    if "LL" in first_id:
        benchmark = json.load(open(BELLINI_FILE, "r", encoding='utf-8'))
        admitted_types = ADMITTED_TYPES["bellini"]
    elif "DLCL" in first_id:
        benchmark = json.load(open(CLASSENSE_FILE, "r", encoding='utf-8'))
        admitted_types = ADMITTED_TYPES["classense"]

    gold_map = {item['letter_id']: item['entities'] for item in benchmark}

    strict_doc_scores = []
    fuzzy_doc_scores = []

    total_strict = {"tp": 0, "fp": 0, "fn": 0}
    total_fuzzy = {"tp": 0, "fp": 0, "fn": 0}

    # per-class corpus-level counts: per_counts[mode][type] = {tp, fp, fn}
    per_counts = {
        "strict": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        "fuzzy":  defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
    }
    support = defaultdict(int)  # gold count per type (mode-independent)

    for pred_doc in ner_preds:
        doc_id = pred_doc['letter_id']

        # relabel any predicted type outside the admitted set as "hallucination".
        # this never creates a new match (gold types are all admitted), so it only
        # reroutes false positives into a single bucket; aggregate numbers unchanged.
        pred_ents = [
            {**p, "type": p["type"] if p["type"] in admitted_types else "hallucination"}
            for p in pred_doc['entities']
        ]
        gold_ents = gold_map.get(doc_id, [])

        for mode in ['strict', 'fuzzy']:
            tp, fp, fn = 0, 0, 0
            matched_gold_indices = set()

            for p in pred_ents:
                found_match = False
                for i, g in enumerate(gold_ents):
                    if i in matched_gold_indices: continue

                    # Type must always match
                    if p['type'] != g['type']: continue

                    if mode == 'strict':
                        is_match = (p['start'] == g['start'] and p['end'] == g['end'])
                    else:
                        is_match = get_overlap_ratio(p['text'], g['text']) > 0.5

                    if is_match:
                        tp += 1
                        per_counts[mode][p['type']]["tp"] += 1
                        matched_gold_indices.add(i)
                        found_match = True
                        break

                if not found_match:
                    fp += 1
                    per_counts[mode][p['type']]["fp"] += 1

            # unmatched gold -> false negatives, attributed to the gold type
            for i, g in enumerate(gold_ents):
                if i not in matched_gold_indices:
                    per_counts[mode][g['type']]["fn"] += 1

            fn = len(gold_ents) - len(matched_gold_indices)

            if mode == 'strict':
                total_strict["tp"] += tp;
                total_strict["fp"] += fp;
                total_strict["fn"] += fn
                strict_doc_scores.append(calculate_metrics(tp, fp, fn))
            else:
                total_fuzzy["tp"] += tp;
                total_fuzzy["fp"] += fp;
                total_fuzzy["fn"] += fn
                fuzzy_doc_scores.append(calculate_metrics(tp, fp, fn))

        # count gold support once per doc (outside the mode loop)
        for g in gold_ents:
            support[g['type']] += 1

    mi_s_p, mi_s_r, mi_s_f1 = calculate_metrics(total_strict["tp"], total_strict["fp"], total_strict["fn"])
    mi_f_p, mi_f_r, mi_f_f1 = calculate_metrics(total_fuzzy["tp"], total_fuzzy["fp"], total_fuzzy["fn"])

    ma_s_p, ma_s_r, ma_s_f1 = np.mean(strict_doc_scores, axis=0)
    ma_f_p, ma_f_r, ma_f_f1 = np.mean(fuzzy_doc_scores, axis=0)

    # ---- build per-class (corpus-level / micro within each class) ----
    def build_per_class(mode):
        out = {}
        types = set(per_counts[mode].keys()) | set(support.keys())
        for t in types:
            c = per_counts[mode][t]
            p, r, f1 = calculate_metrics(c["tp"], c["fp"], c["fn"])
            out[t] = {
                "precision": p, "recall": r, "f1": f1,
                "support": support.get(t, 0),
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
            }
        return out

    per_class = {"strict": build_per_class("strict"), "fuzzy": build_per_class("fuzzy")}

    return {
        "strict": {
            "micro": {"precision": mi_s_p, "recall": mi_s_r, "f1": mi_s_f1},
            "macro": {"precision": ma_s_p, "recall": ma_s_r, "f1": ma_s_f1}
        },
        "fuzzy": {
            "micro": {"precision": mi_f_p, "recall": mi_f_r, "f1": mi_f_f1},
            "macro": {"precision": ma_f_p, "recall": ma_f_r, "f1": ma_f_f1}
        },
        "per_class": per_class,
    }
def create_mapped_ner(preds, output_path):
    print(preds)
    if "LL" in preds[0]['id']:
        benchmark = json.load(open(BELLINI_FILE, "r", encoding='utf-8'))
    elif "DLCL" in preds[0]['id']:
        benchmark = json.load(open(CLASSENSE_FILE, "r", encoding='utf-8'))

    benchmark_map = {item['letter_id']: item for item in benchmark}

    mapped_results = []

    for entry in preds:
        letter_id = entry['id']
        raw_response = entry['model_response']
        try:
            inner = raw_response.strip().strip('[]')
            entities_list = []
            for item in inner.split(','):
                item = item.strip()
                if not item:
                    continue
                text, _, label = item.rpartition(':')  # split on the LAST colon
                entities_list.append((text.strip(), label.strip()))
        except Exception as e:
            print(f"Skipping {letter_id} due to parsing error: {e}")
            continue

        if letter_id not in benchmark_map:
            continue

        full_text = benchmark_map[letter_id]['text']
        current_search_pos = 0
        extracted_entities = []


        for text_snippet, ent_type in entities_list:

            start_idx = full_text.find(text_snippet, current_search_pos)

            if start_idx != -1:
                end_idx = start_idx + len(text_snippet)
                extracted_entities.append({
                    "text": text_snippet,
                    "type": ent_type,
                    "start": start_idx,
                    "end": end_idx
                })
                current_search_pos = end_idx
            else:
                start_idx = full_text.find(text_snippet)
                if start_idx != -1:
                    extracted_entities.append({
                        "text": text_snippet,
                        "type": ent_type,
                        "start": start_idx,
                        "end": start_idx + len(text_snippet)
                    })

        mapped_results.append({
            "letter_id": letter_id,
            "text": full_text,
            "entities": extracted_entities
        })

    output_path_obj = Path(output_path)

    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapped_results, f, indent=2, ensure_ascii=False)

    print(f"Mapped {len(mapped_results)} entries to {output_path}")



def evaluate_rank_result(preds):
    parsed_preds = [ast.literal_eval(i["model_response"]) for i in preds]
    gt = [i["gold_order"] for i in preds]

    rho_scores = []
    for p, g in zip(parsed_preds, gt):

        if len(list(p))!=len(g):
            uncommon = set(p)^set(g)
            p=list(p)
            p.extend(list(uncommon))
        p = tuple(p)
        try:
            score, _ = spearmanr(p, g)
        except:
            print(p)
            print(g)
        rho_scores.append(score)

    avg_rho = np.nanmean(rho_scores)

    return {
        "mean_spearman_rho": round(avg_rho, 4),
    }




def calculate_accuracy(preds):
    model_responses=[]
    for i in preds:
        try:
            model_responses.append(int(i["model_response"]))
        except:
            model_responses.append(int(i["model_response"][0]))
    gt = [int(i["gold_index"]) for i in preds]

    correct=0
    for a,b in zip(model_responses,gt):
        if a==b:
            correct+=1

    return {"accuracy":correct/len(gt)}


import json
from pathlib import Path



import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

if __name__ == "__main__":
    evaluate_all_tiers()