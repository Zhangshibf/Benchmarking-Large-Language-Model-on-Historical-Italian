import json
from pathlib import Path
from itertools import islice

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
    PROMPT_TIER3, PROMPT_TIER4,
    PROMPT_TIER5_RANKING, PROMPT_TIER5_AUTHORSHIP
)
from src.benchmark_loader import (
    Tier1Loader, Tier2Loader, Tier3Loader,
    Tier4Loader, Tier5LoaderAuthorship, Tier5LoaderRanking
)


def run_benchmark(experiment_name: str, toy_mode: bool = False):
    """
    Esegue il benchmark e salva i prompt se 'test' è presente nel nome dell'esperimento.
    """
    base_save_path = OUTPUT_DIR / experiment_name
    base_save_path.mkdir(parents=True, exist_ok=True)

    # Configurazione Modelli
    models = {
        "gpt": ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=0),
        # "claude-3-5-sonnet": ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0),
        "deepseek-chat": ChatDeepSeek(model="deepseek-chat", temperature=0)
    }

    tasks = [
        {"name": "tier1", "loader": Tier1Loader(BENCHMARK_DIR), "prompt": PROMPT_TIER1},
        {"name": "tier2", "loader": Tier2Loader(BENCHMARK_DIR), "prompt": PROMPT_TIER2},
        {"name": "tier3", "loader": Tier3Loader(BENCHMARK_DIR), "prompt": PROMPT_TIER3},
        {"name": "tier4", "loader": Tier4Loader(BENCHMARK_DIR), "prompt": PROMPT_TIER4},
        {"name": "tier5_auth", "loader": Tier5LoaderAuthorship(BENCHMARK_DIR), "prompt": PROMPT_TIER5_AUTHORSHIP},
        {"name": "tier5_rank", "loader": Tier5LoaderRanking(BENCHMARK_DIR), "prompt": PROMPT_TIER5_RANKING},
    ]

    def format_candidates(candidates: list) -> str:
        return "\n".join([f"{i + 1}. {c}" for i, c in enumerate(candidates)])

    for model_name, llm in models.items():
        print(f"\n🚀 Modello: {model_name}")

        for task in tasks:
            task_name = task["name"]
            loader = task["loader"]
            task_output_dir = base_save_path / model_name / task_name
            task_output_dir.mkdir(parents=True, exist_ok=True)

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", sys_message),
                ("human", task["prompt"]),
            ])
            chain = prompt_template | llm | StrOutputParser()

            instances = islice(loader, 2) if toy_mode else loader
            print(f"  📂 Task: {task_name}")

            # Flag per salvare il prompt una sola volta per task
            prompt_saved = False

            for instance in instances:
                # Mapping input (come visto precedentemente)
                inputs = {}
                if task_name in ["tier1", "tier2", "tier4"]:
                    inputs = {"context": getattr(instance, 'context', ""), "target": instance.target,
                              "answers_str": format_candidates(instance.candidates)}
                elif task_name == "tier3":
                    inputs = {"testo": instance.text, "tipi_entita": ", ".join(set(instance.entites_type))}
                elif task_name == "tier5_auth":
                    inputs = {"snippets_str": format_candidates(instance.candidates)}
                elif task_name == "tier5_rank":
                    inputs = {"snippets_str": format_candidates(instance.snippets)}

                # --- LOGICA SALVATAGGIO PROMPT (DEBUG) ---
                if "test" in experiment_name.lower() and not prompt_saved:
                    # Formattiamo il prompt con i dati della prima istanza
                    debug_prompt = prompt_template.format(**inputs)
                    with open(task_output_dir / "prompts_debug.txt", "w", encoding="utf-8") as f:
                        f.write(f"=== SYSTEM MESSAGE ===\n{sys_message}\n\n")
                        f.write(f"=== HUMAN PROMPT (EXAMPLE) ===\n{debug_prompt}")
                    prompt_saved = True

                try:
                    response = chain.invoke(inputs)
                    result_data = {
                        "id": instance.id,
                        "model_response": response.strip(),
                        "gold_index": getattr(instance, 'gold_index', None),
                        "gold_order": getattr(instance, 'order', None),
                    }
                    with open(task_output_dir / f"{instance.id}.json", "w", encoding="utf-8") as f:
                        json.dump(result_data, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    print(f"    ⚠️ Errore istanza {instance.id}: {e}")

# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    # Esegui un test rapido
    run_benchmark(experiment_name="test_rapido_v1", toy_mode=True)

    # Esegui l'esperimento completo (commentato per sicurezza)
    # run_benchmark(experiment_name="esperimento_finale_marzo", toy_mode=False)