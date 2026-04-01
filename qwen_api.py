from together import Together
from datasets import load_dataset
import json
import re
import random


client = Together()

MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"  


# =========================
# Load datasets
# =========================
dataset_en = load_dataset("li-lab/MMLU-ProX-Lite", "en")
dataset_zh = load_dataset("li-lab/MMLU-ProX-Lite", "zh")
dataset_es = load_dataset("li-lab/MMLU-ProX-Lite", "es")

test_en = dataset_en["test"]
test_zh = dataset_zh["test"]
test_es = dataset_es["test"]


# =========================
# Random sample
# =========================
seed = random.randint(0, 100000)
print("Random seed:", seed)

samples_en = test_en.shuffle(seed=seed).select(range(30))

question_ids = [ex["question_id"] for ex in samples_en]

samples_zh = test_zh.filter(lambda x: x["question_id"] in question_ids)
samples_es = test_es.filter(lambda x: x["question_id"] in question_ids)

zh_dict = {ex["question_id"]: ex for ex in samples_zh}
es_dict = {ex["question_id"]: ex for ex in samples_es}


# =========================
# Extract options
# =========================
def extract_options(example):

    options = []

    for i in range(10):
        key = f"option_{i}"
        if key in example and example[key] is not None:
            options.append(example[key])

    return options


# =========================
# Convert options to dict
# =========================
def options_to_dict(options):

    letters = "ABCDEFGHIJ"
    return {letters[i]: opt for i, opt in enumerate(options)}


# =========================
# Build prompt
# =========================
def build_prompt(question, options, lang):

    letters = "ABCDEFGHIJ"

    option_text = ""
    for i, opt in enumerate(options):
        option_text += f"{letters[i]}. {opt}\n"

    if lang == "en":
        return f"""
Solve the following multiple choice question.

Question:
{question}

Options:
{option_text}

Think step by step and answer the question in 2048 tokens.

You MUST output in this format:

Final Answer: X
"""

    elif lang == "zh":
        return f"""
请回答以下选择题。

问题:
{question}

选项:
{option_text}

请一步一步推理并且把输出的长度限制在2048个次元的范围内。

最后必须输出：

Final Answer: X
"""

    else:
        return f"""
Resuelve la siguiente pregunta de opción múltiple. Responde a la pregunta en 2048 tokens.

Pregunta:
{question}

Opciones:
{option_text}

Piensa paso a paso.

Debes terminar con:

Final Answer: X
"""


def ask_model(prompt):

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful reasoning assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,         # 更稳定
            max_tokens=4096
        )

        return response.choices[0].message.content

    except Exception as e:
        print("API error:", e)
        return ""


# =========================
# Extract answer
# =========================
def extract_answer(text):

    match = re.search(r"Final Answer:\s*([A-J])", text, re.IGNORECASE)

    if match:
        return match.group(1)

    return None


# =========================
# Run experiment
# =========================
results = []

correct_en = 0
correct_zh = 0
correct_es = 0

for i, ex_en in enumerate(samples_en):

    print(f"\nRunning question {i+1}")

    qid = ex_en["question_id"]

    ex_zh = zh_dict[qid]
    ex_es = es_dict[qid]

    options_en = extract_options(ex_en)
    options_zh = extract_options(ex_zh)
    options_es = extract_options(ex_es)

    # 调用模型
    reasoning_en = ask_model(build_prompt(ex_en["question"], options_en, "en"))
    reasoning_zh = ask_model(build_prompt(ex_zh["question"], options_zh, "zh"))
    reasoning_es = ask_model(build_prompt(ex_es["question"], options_es, "es"))

    pred_en = extract_answer(reasoning_en)
    pred_zh = extract_answer(reasoning_zh)
    pred_es = extract_answer(reasoning_es)

    gold = ex_en["answer"]

    if pred_en == gold:
        correct_en += 1
    if pred_zh == gold:
        correct_zh += 1
    if pred_es == gold:
        correct_es += 1

    results.append({

        "question_id": qid,

        "question_en": ex_en["question"],
        "question_zh": ex_zh["question"],
        "question_es": ex_es["question"],

        "options_en": options_to_dict(options_en),
        "options_zh": options_to_dict(options_zh),
        "options_es": options_to_dict(options_es),

        "gold_answer": gold,

        "prediction_en": pred_en,
        "prediction_zh": pred_zh,
        "prediction_es": pred_es,

        "reasoning_en": reasoning_en,
        "reasoning_zh": reasoning_zh,
        "reasoning_es": reasoning_es
    })


# =========================
# Accuracy
# =========================
n = len(samples_en)

print("\n===== Accuracy =====")
print("EN:", correct_en / n)
print("ZH:", correct_zh / n)
print("ES:", correct_es / n)


# =========================
# Save results
# =========================
output_file = "together_results3.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "seed": seed,
        "accuracy_en": correct_en / n,
        "accuracy_zh": correct_zh / n,
        "accuracy_es": correct_es / n,
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"Saved to {output_file}")