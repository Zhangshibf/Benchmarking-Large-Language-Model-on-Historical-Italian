import json
from pathlib import Path
from itertools import islice
from typing import List
from dotenv import load_dotenv
from pathlib import Path as _Path
load_dotenv(dotenv_path=_Path(__file__).resolve().parents[1] / ".env", override=True)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from src.settings import OUTPUT_DIR, BENCHMARK_DIR
from src.prompts import (
    sys_message, PROMPT_TIER1, PROMPT_TIER2_DEPREL,
   PROMPT_TIER4,PROMPT_TIER2_HEAD,
    PROMPT_TIER5_RANKING, PROMPT_TIER5_AUTHORSHIP, PROMPT_TIER3_BELLINI, PROMPT_TIER3_CLASSENSE
)
from src.benchmark_loader import (
    Tier1Loader, Tier2LoaderDeprel, Tier3LoaderCLASSENSE, Tier3LoaderBELLINI,
    Tier4Loader, Tier5LoaderAuthorship, Tier5LoaderRanking,Tier2LoaderHead
)


class Entity(BaseModel):
    text: str
    type: str


class EntityExtraction(BaseModel):
    entities: List[Entity]


def resolve_offsets(entities: list, full_text: str) -> list:
    """Find character offsets for each entity via sequential str.find."""
    resolved = []
    current_pos = 0
    for e in entities:
        start = full_text.find(e["text"], current_pos)
        if start != -1:
            end = start + len(e["text"])
            resolved.append({**e, "start": start, "end": end})
            current_pos = end
        else:
            start = full_text.find(e["text"])
            if start != -1:
                resolved.append({**e, "start": start, "end": start + len(e["text"])})
    return resolved


def format_candidates(candidates: list) -> str:
    return "\n".join([f"{i + 1}. {c}" for i, c in enumerate(candidates)])


def run_benchmark(experiment_name: str, toy_mode: bool = False):
    base_save_path = OUTPUT_DIR / experiment_name
    base_save_path.mkdir(parents=True, exist_ok=True)

    models = {
        "deepseek-chat": ChatDeepSeek(model="deepseek-chat", temperature=0),
        "gpt": ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=0),
        "claude": ChatAnthropic(model="claude-opus-4-6", temperature=0)
    }

    tasks = [
        {"name": "tier1", "loader": Tier1Loader(BENCHMARK_DIR), "prompt": PROMPT_TIER1},
        {"name": "tier2_head", "loader": Tier2LoaderHead(BENCHMARK_DIR), "prompt": PROMPT_TIER2_HEAD},
        {"name": "tier2_deprel", "loader": Tier2LoaderDeprel(BENCHMARK_DIR), "prompt": PROMPT_TIER2_DEPREL},
        {"name": "tier3_bellini", "loader": Tier3LoaderBELLINI(BENCHMARK_DIR), "prompt": PROMPT_TIER3_BELLINI},
        {"name": "tier3_classense", "loader": Tier3LoaderCLASSENSE(BENCHMARK_DIR), "prompt": PROMPT_TIER3_CLASSENSE},
        {"name": "tier4", "loader": Tier4Loader(BENCHMARK_DIR), "prompt": PROMPT_TIER4},
        {"name": "tier5_auth", "loader": Tier5LoaderAuthorship(BENCHMARK_DIR), "prompt": PROMPT_TIER5_AUTHORSHIP},
        {"name": "tier5_rank", "loader": Tier5LoaderRanking(BENCHMARK_DIR), "prompt": PROMPT_TIER5_RANKING},
    ]

    for model_name, llm in models.items():
        print(f"\n🚀 Modello: {model_name}")

        for task in tasks:
            task_name = task["name"]
            loader = task["loader"]
            is_ner = task_name in ("tier3_classense", "tier3_bellini")

            # We save everything for this task/model combo in one file
            task_output_dir = base_save_path / model_name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = task_output_dir / f"{task_name}_results.jsonl"

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", sys_message),
                ("human", task["prompt"]),
            ])

            if is_ner:
                # Use structured output for NER — more reliable than free-text parsing
                structured_llm = llm.with_structured_output(EntityExtraction)
                ner_chain = prompt_template | structured_llm
            else:
                chain = prompt_template | llm | StrOutputParser()

            instances = islice(loader, 2) if toy_mode else loader
            print(f"  📂 Task: {task_name}")

            prompt_saved = False

            # Open file in 'append' mode ('a')
            with open(output_file, "a", encoding="utf-8") as f_out:
                for instance in instances:
                    inputs = {}

                    if task_name in ["tier1","tier2_head","tier2_deprel"]:
                        inputs = {"context": getattr(instance, 'context', ""), "target": instance.target,
                                  "answers_str": format_candidates(instance.candidates)}
                    elif task_name in ("tier3_bellini", "tier3_classense"):
                        inputs = {"testo": instance.text}
                    elif task_name =="tier4":
                        inputs = {
                            "testo": instance.target,
                            "fonte_a": instance.candidates[0],
                            "fonte_b": instance.candidates[1]
                        }
                    elif task_name == "tier5_auth":
                        inputs = {"snippets_str": format_candidates(instance.candidates)}
                    elif task_name == "tier5_rank":
                        inputs = {"snippets_str": format_candidates(instance.snippets)}

                    # Debug prompt logic
                    if "test" in experiment_name.lower() and not prompt_saved:
                        debug_path = task_output_dir / f"{task_name}_prompts_debug.txt"
                        with open(debug_path, "w", encoding="utf-8") as f_debug:
                            f_debug.write(f"{prompt_template.format(**inputs)}")
                        prompt_saved = True

                    try:
                        if is_ner:
                            result = ner_chain.invoke(inputs)
                            entities_text = [{"text": e.text, "type": e.type} for e in result.entities]
                            entities_with_offsets = resolve_offsets(entities_text, instance.text)
                            result_data = {
                                "id": instance.id,
                                "letter_id": instance.id,
                                "entities": entities_with_offsets,
                            }
                        else:
                            response = chain.invoke(inputs)
                            result_data = {
                                "id": instance.id,
                                "model_response": response.strip(),
                                "gold_index": getattr(instance, 'gold_index', None),
                                "gold_order": getattr(instance, 'order', None),
                            }

                        f_out.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                        f_out.flush()  # Force write to disk immediately

                    except Exception as e:
                        print(f"    ⚠️ Errore istanza {instance.id}: {e}")

if __name__ == "__main__":
    run_benchmark(experiment_name="round1", toy_mode=False)

