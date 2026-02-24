import json
from pathlib import Path
from itertools import islice
from dotenv import load_dotenv
load_dotenv()
# LangChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Project Imports
from src.settings import OUTPUT_DIR, BENCHMARK_DIR
from src.prompts import (
    sys_message, PROMPT_TIER1, PROMPT_TIER2,
    PROMPT_TIER3_CLASSENSE, PROMPT_TIER3_BELLINI,PROMPT_TIER4,
    PROMPT_TIER5_RANKING, PROMPT_TIER5_AUTHORSHIP, PROMPT_TIER3_BELLINI, PROMPT_TIER3_CLASSENSE
)
from src.benchmark_loader import (
    Tier1Loader, Tier2Loader, Tier3LoaderCLASSENSE,Tier3LoaderBELLINI,
    Tier4Loader, Tier5LoaderAuthorship, Tier5LoaderRanking
)


def format_candidates(candidates: list) -> str:
    return "\n".join([f"{i + 1}. {c}" for i, c in enumerate(candidates)])


def run_benchmark(experiment_name: str, toy_mode: bool = False):
    base_save_path = OUTPUT_DIR / experiment_name
    base_save_path.mkdir(parents=True, exist_ok=True)

    models = {
        "deepseek-chat": ChatDeepSeek(model="deepseek-chat", temperature=0),
        "gpt": ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=0),
        "claude-3-5-sonnet": ChatAnthropic(model="claude-opus-4-6", temperature=0)
    }

    tasks = [
        {"name": "tier1", "loader": Tier1Loader(BENCHMARK_DIR), "prompt": PROMPT_TIER1},
        {"name": "tier2", "loader": Tier2Loader(BENCHMARK_DIR), "prompt": PROMPT_TIER2},
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

            # We save everything for this task/model combo in one file
            task_output_dir = base_save_path / model_name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = task_output_dir / f"{task_name}_results.jsonl"

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", sys_message),
                ("human", task["prompt"]),
            ])
            chain = prompt_template | llm | StrOutputParser()

            instances = islice(loader, 2) if toy_mode else loader
            print(f"  📂 Task: {task_name}")

            prompt_saved = False

            # Open file in 'append' mode ('a')
            with open(output_file, "a", encoding="utf-8") as f_out:
                for instance in instances:
                    inputs = {}

                    if task_name in ["tier1", "tier2"]:
                        inputs = {"context": getattr(instance, 'context', ""), "target": instance.target,
                                  "answers_str": format_candidates(instance.candidates)}
                    elif task_name == "tier3_bellini":
                        inputs = {"testo": instance.text}
                    elif task_name == "tier3_classense":
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
                            f_debug.write(
                                f"{prompt_template.format(**inputs)}")
                        prompt_saved = True

                    try:
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
    #run_benchmark(experiment_name="test_rapido_v1", toy_mode=True)
    #run_benchmark(experiment_name="round1", toy_mode=False)

    run_benchmark(experiment_name="round2", toy_mode=False)
    run_benchmark(experiment_name="round3", toy_mode=False)
