import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# ---- 1. Load data from JSON file ----
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data/data.json")

with open(data_path, "r") as f:
    data = json.load(f)

# ---- 2. Create output directories if they don't exist ----
base_images_dir = os.path.join(os.path.dirname(current_dir), "images")
os.makedirs(base_images_dir, exist_ok=True)

# Create language-specific directories
langs = ["en", "pt", "es"]
for lang in langs:
    lang_dir = os.path.join(base_images_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)

# ---- 3. Create scatter plots (Cost vs each benchmark) with language options ----
df = pd.DataFrame(data)

# Labels for different languages
labels = {
    "en": {
        "cost_label": "Cost per 1K tokens (USD, log scale)",
        "score_label": "Score",
        "vs_label": "Cost vs",
        "gen_msg": "Generated",
        "data_title": "LLM Cost/Performance Data:",
        "charts_title": "Generated charts:"
    },
    "pt": {
        "cost_label": "Custo por 1K tokens (USD, escala logarítmica)",
        "score_label": "Pontuação",
        "vs_label": "Custo vs",
        "gen_msg": "Gerado",
        "data_title": "Dados de Custo/Desempenho de LLM:",
        "charts_title": "Gráficos gerados:"
    },
    "es": {
        "cost_label": "Costo por 1K tokens (USD, escala logarítmica)",
        "score_label": "Puntuación",
        "vs_label": "Costo vs",
        "gen_msg": "Generado",
        "data_title": "Datos de Costo/Rendimiento de LLM:",
        "charts_title": "Gráficos generados:"
    }
}

def scatter(metric, y_label, lang="pt"):
    plt.figure(figsize=(9,5))
    plt.scatter(df["cost"], df[metric])
    for _, row in df.iterrows():
        plt.text(row["cost"], row[metric], row["model"])
    plt.xscale("log")
    plt.xlabel(labels[lang]["cost_label"])
    plt.ylabel(f"{labels[lang]['score_label']} {y_label} (%)")
    plt.title(f"{labels[lang]['vs_label']} {y_label}")
    plt.tight_layout()
    
    # Save to language-specific directory
    lang_dir = os.path.join(base_images_dir, lang)
    filename = os.path.join(lang_dir, f"cost_vs_{metric}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"{labels[lang]['gen_msg']} {filename}")
    return filename

# Generate charts for all languages
all_files = {}
for lang in langs:
    print(f"\n--- Generating charts for {lang} ---")
    lang_files = {
        f"{lang}_mmlu_chart": scatter("mmlu", "MMLU", lang),
        f"{lang}_hellaswag_chart": scatter("hellaswag", "HellaSwag", lang),
        f"{lang}_humaneval_chart": scatter("humaneval", "HumanEval", lang)
    }
    all_files.update(lang_files)

# ---- 4. Display the dataframe for quick inspection ----
# Use the language from command line if provided, otherwise default to Portuguese
display_lang = "pt"
if len(sys.argv) > 1 and sys.argv[1] in langs:
    display_lang = sys.argv[1]

print(f"\n{labels[display_lang]['data_title']}")
print(df)

print(f"\n{labels[display_lang]['charts_title']}")
for chart_name, chart_path in all_files.items():
    print(f"- {chart_name}: {chart_path}")
