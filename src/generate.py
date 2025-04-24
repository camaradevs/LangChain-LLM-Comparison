import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from matplotlib.lines import Line2D

# ---- 1. Load data from JSON file ----
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data/data.json")
benchmarks_path = os.path.join(current_dir, "data/benchmarks.json")

with open(data_path, "r") as f:
    data = json.load(f)

with open(benchmarks_path, "r") as f:
    benchmarks_data = json.load(f)

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

# Get all benchmark names from the data columns
benchmark_columns = [col for col in df.columns if col not in ["model", "cost"]]

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

# Create a mapping from benchmark code to full name
benchmark_names = {}
for benchmark in benchmarks_data["text_model_benchmarks"]:
    benchmark_names[benchmark["name"].lower()] = benchmark["full_name"]

def scatter(metric, lang="pt"):
    plt.figure(figsize=(10, 7))
    
    # Load freedom data if available (from data_updated.json)
    freedom_data_path = os.path.join(current_dir, "data/data_updated.json")
    try:
        with open(freedom_data_path, "r") as f:
            updated_data = json.load(f)
        
        # Create a dictionary mapping model names to freedom scores
        freedom_dict = {}
        for item in updated_data:
            if "model" in item and "freedom" in item:
                freedom_dict[item["model"]] = item["freedom"]
        
        # Check if we have freedom scores for all models
        has_freedom_data = all(model in freedom_dict for model in df["model"])
    except (FileNotFoundError, json.JSONDecodeError):
        has_freedom_data = False
    
    # Define colors for each model family
    color_map = {
        "GPT": "red",
        "Claude": "blue",
        "DeepSeek": "green",
        "Gemini": "purple",
        "O1": "orange"
    }
    
    # Determine color for each model based on its name
    colors = []
    for model_name in df["model"]:
        if "GPT" in model_name:
            colors.append(color_map["GPT"])
        elif "Claude" in model_name:
            colors.append(color_map["Claude"])
        elif "DeepSeek" in model_name:
            colors.append(color_map["DeepSeek"])
        elif "Gemini" in model_name:
            colors.append(color_map["Gemini"])
        elif "O1" in model_name:
            colors.append(color_map["O1"])
        else:
            colors.append("gray")
    
    # Calculate bubble sizes based on cost (direct relationship - more expensive models get bigger bubbles)
    # Use area-based scaling for proper visual proportions
    max_size = 600
    min_size = 150
    
    # Get min and max cost values to normalize
    min_cost = min(df["cost"])
    max_cost = max(df["cost"])
    cost_range = max_cost - min_cost
    
    # Set bubble sizes with special override for O1 model
    bubble_sizes = []
    for _, row in df.iterrows():
        if "O1" in row["model"]:
            # Force O1 to have largest bubble
            bubble_size = max_size * 2  # Make O1 extra large
        else:
            # For other models, scale based on cost
            if cost_range > 0:
                normalized_cost = (row["cost"] - min_cost) / cost_range
                bubble_size = min_size + (max_size - min_size) * np.sqrt(normalized_cost)
            else:
                bubble_size = (min_size + max_size) / 2
        bubble_sizes.append(bubble_size)
    
    # Always use freedom data as the x-axis if available
    if has_freedom_data:
        x_values = [freedom_dict[model] for model in df["model"]]
        x_label = "Freedom Score (%)" if lang == "en" else "Pontuação de Liberdade (%)" if lang == "pt" else "Puntuación de Libertad (%)"
        title_prefix = "Freedom vs" if lang == "en" else "Liberdade vs" if lang == "pt" else "Libertad vs"
        filename_prefix = "freedom_vs"
    else:
        # Fallback to cost as x-axis if freedom data is not available
        x_values = df["cost"]
        x_label = labels[lang]["cost_label"]
        title_prefix = labels[lang]["vs_label"]
        filename_prefix = "cost_vs"
    
    # Create bubble plot
    scatter = plt.scatter(
        x_values, 
        df[metric], 
        s=bubble_sizes,
        c=colors,
        alpha=0.6,
        edgecolors="black",
        linewidth=1
    )
    
    # Add model names as labels
    for i, (_, row) in enumerate(df.iterrows()):
        x_value = x_values[i]
        plt.annotate(
            row["model"],
            (x_value, row[metric]),
            fontsize=8,
            ha='center',
            va='center'
        )
    
    # Create legend for model families
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                      label=family, markerfacecolor=color, markersize=8)
                      for family, color in color_map.items()]
    
    # Add a legend for bubble size
    bubble_size_text = {
        "en": "Higher cost = Larger bubble", 
        "pt": "Maior custo = Maior bolha", 
        "es": "Mayor costo = Burbuja más grande"
    }
    bubble_size_legend = plt.Line2D([0], [0], marker='o', color='w', 
                        label=bubble_size_text.get(lang, bubble_size_text["en"]), 
                        markerfacecolor='gray', markersize=10)
    legend_elements.append(bubble_size_legend)
    
    legend_title = {
        "en": "Model Family",
        "pt": "Família de Modelos",
        "es": "Familia de Modelos"
    }
    plt.legend(handles=legend_elements, loc='lower right', title=legend_title.get(lang, legend_title["en"]))
    
    # Set appropriate x-scale if using cost
    if not has_freedom_data:
        plt.xscale("log")
    
    plt.xlabel(x_label)
    
    # Use full name if available, otherwise use the metric name
    y_label = benchmark_names.get(metric, metric.upper())
    
    plt.ylabel(f"{labels[lang]['score_label']} {y_label} (%)")
    plt.title(f"{title_prefix} {y_label}")
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    
    # Save to language-specific directory
    lang_dir = os.path.join(base_images_dir, lang)
    filename = os.path.join(lang_dir, f"{filename_prefix}_{metric}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"{labels[lang]['gen_msg']} {filename}")
    return filename

# Generate charts for all languages and all benchmarks
all_files = {}
for lang in langs:
    print(f"\n--- Generating charts for {lang} ---")
    lang_files = {}
    
    # Generate a chart for each benchmark
    for metric in benchmark_columns:
        chart_path = scatter(metric, lang)
        lang_files[f"{lang}_{metric}_chart"] = chart_path
        
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
