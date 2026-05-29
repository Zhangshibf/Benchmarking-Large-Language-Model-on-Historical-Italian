"""
two additional smaller scale LLMs
Minerva-7B-instruct-v1.0
meta-llama/llama-3.1-8b-instruct
"""

import json
import os
import re
from itertools import islice
from pathlib import Path as _Path
from dotenv import load_dotenv
from openai import api_key

load_dotenv(dotenv_path=_Path(__file__).resolve().parents[1] / ".env", override=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.settings import OUTPUT_DIR, BENCHMARK_DIR
from src.prompts import (
    sys_message,
    PROMPT_TIER1,
    PROMPT_TIER2_DEPREL,
    PROMPT_TIER2_HEAD,
    PROMPT_TIER3_BELLINI,
    PROMPT_TIER3_CLASSENSE,
    PROMPT_TIER4,
    PROMPT_TIER5_AUTHORSHIP,
    PROMPT_TIER5_RANKING,
)
from src.benchmark_loader import (
    Tier1Loader,
    Tier2LoaderDeprel,
    Tier2LoaderHead,
    Tier3LoaderBELLINI,
    Tier3LoaderCLASSENSE,
    Tier4Loader,
    Tier5LoaderAuthorship,
    Tier5LoaderRanking,
)


def build_minerva_llm(
    model_id: str = "sapienzanlp/Minerva-7B-instruct-v1.0",
    max_new_tokens: int = 1024,
    device: str | int | None = None,
):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device is None else None,
    )
    if device is not None and device != "auto":
        model = model.to(device)

    gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    hf_llm = HuggingFacePipeline(pipeline=gen_pipe)

    return ChatHuggingFace(llm=hf_llm, tokenizer=tokenizer)


def build_llama_openrouter_llm(
    model_id: str = "meta-llama/llama-3.1-8b-instruct",
    temperature: float = 0.0,api_key=None
) -> ChatOpenAI:

    api_key = api_key
    default_headers = {}
    if os.getenv("OPENROUTER_HTTP_REFERER"):
        default_headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER")
    if os.getenv("OPENROUTER_X_TITLE"):
        default_headers["X-Title"] = os.getenv("OPENROUTER_X_TITLE")

    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers=default_headers or None,
    )


def _format_candidates(candidates: list) -> str:
    return "\n".join([f"{i + 1}. {c}" for i, c in enumerate(candidates)])


def _extract_ner_freetext(response: str) -> str:
    txt = response.strip()# strip ```...``` fences
    fence = re.search(r"```(?:python|json)?\s*(.*?)```", txt, flags=re.S)
    if fence:
        txt = fence.group(1).strip()# grab from first '[' to last ']'
    first = txt.find("[")
    last = txt.rfind("]")
    if first != -1 and last != -1 and last > first:
        txt = txt[first : last + 1]
    return txt



def run_extra_benchmark(
    experiment_name: str = "round1",
    toy_mode: bool = False,
    include_minerva: bool = True,
    include_llama: bool = True,
    minerva_device: str | int | None = None,
        api_key:str = None
):

    base_save_path = OUTPUT_DIR / experiment_name
    base_save_path.mkdir(parents=True, exist_ok=True)

    models: dict[str, object] = {}
    if include_minerva:
        models["minerva-7b"] = build_minerva_llm(device=minerva_device)
    if include_llama:
        models["llama-3.1-8b"] = build_llama_openrouter_llm(api_key=api_key)

    tasks = [
        {"name": "tier1",            "loader": Tier1Loader(BENCHMARK_DIR),            "prompt": PROMPT_TIER1},
        {"name": "tier2_head",       "loader": Tier2LoaderHead(BENCHMARK_DIR),        "prompt": PROMPT_TIER2_HEAD},
        {"name": "tier2_deprel",     "loader": Tier2LoaderDeprel(BENCHMARK_DIR),      "prompt": PROMPT_TIER2_DEPREL},
        {"name": "tier3_bellini",    "loader": Tier3LoaderBELLINI(BENCHMARK_DIR),     "prompt": PROMPT_TIER3_BELLINI},
        {"name": "tier3_classense",  "loader": Tier3LoaderCLASSENSE(BENCHMARK_DIR),   "prompt": PROMPT_TIER3_CLASSENSE},
        {"name": "tier4",            "loader": Tier4Loader(BENCHMARK_DIR),            "prompt": PROMPT_TIER4},
        {"name": "tier5_auth",       "loader": Tier5LoaderAuthorship(BENCHMARK_DIR),  "prompt": PROMPT_TIER5_AUTHORSHIP},
        {"name": "tier5_rank",       "loader": Tier5LoaderRanking(BENCHMARK_DIR),     "prompt": PROMPT_TIER5_RANKING},
    ]

    for model_name, llm in models.items():
        print(f"Modello: {model_name}")

        for task in tasks:
            task_name = task["name"]
            loader = task["loader"]
            is_ner = task_name in ("tier3_classense", "tier3_bellini")

            task_output_dir = base_save_path / model_name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = task_output_dir / f"{task_name}_results.jsonl"

            prompt_template = ChatPromptTemplate.from_messages(
                [("system", sys_message), ("human", task["prompt"])]
            )


            chain = prompt_template | llm | StrOutputParser()

            instances = islice(loader, 2) if toy_mode else loader
            print(f"Task: {task_name}")

            prompt_saved = False
            with open(output_file, "a", encoding="utf-8") as f_out:
                for instance in instances:
                    if task_name in ["tier1", "tier2_head", "tier2_deprel"]:
                        inputs = {
                            "context": getattr(instance, "context", ""),
                            "target": instance.target,
                            "answers_str": _format_candidates(instance.candidates),
                        }
                    elif is_ner:
                        inputs = {"testo": instance.text}
                    elif task_name == "tier4":
                        inputs = {
                            "testo": instance.target,
                            "fonte_a": instance.candidates[0],
                            "fonte_b": instance.candidates[1],
                        }
                    elif task_name == "tier5_auth":
                        inputs = {"snippets_str": _format_candidates(instance.candidates)}
                    elif task_name == "tier5_rank":
                        inputs = {"snippets_str": _format_candidates(instance.snippets)}
                    else:
                        continue

                    if "test" in experiment_name.lower() and not prompt_saved:
                        debug_path = task_output_dir / f"{task_name}_prompts_debug.txt"
                        with open(debug_path, "w", encoding="utf-8") as f_debug:
                            f_debug.write(prompt_template.format(**inputs))
                        prompt_saved = True

                    try:
                        response = chain.invoke(inputs)
                        response = response.strip()

                        if is_ner:
                            # Save in the legacy free-text shape that evaluation.py
                            # → create_mapped_ner consumes.
                            result_data = {
                                "id": instance.id,
                                "letter_id": instance.id,
                                "model_response": _extract_ner_freetext(response),
                            }
                        else:
                            result_data = {
                                "id": instance.id,
                                "model_response": response,
                                "gold_index": getattr(instance, "gold_index", None),
                                "gold_order": getattr(instance, "order", None),
                            }

                        f_out.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                        f_out.flush()

                    except Exception as e:
                        print(f"    ⚠️ Errore istanza {instance.id}: {e}")


if __name__ == "__main__":
    run_extra_benchmark(
        experiment_name="round1",
        toy_mode=True,
        include_minerva=True,
        include_llama=False,
    )
    # run_extra_benchmark(
    #     experiment_name="round2",
    #     toy_mode=False,
    #     include_minerva=True,
    #     include_llama=False,
    # )
    # run_extra_benchmark(
    #     experiment_name="round3",
    #     toy_mode=False,
    #     include_minerva=True,
    #     include_llama=False,
    # )