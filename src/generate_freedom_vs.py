import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

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
languages = ["en", "es", "pt"]

for lang in languages:
    lang_dir = os.path.join(base_images_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Define colors for different model families
color_map = {
    "OpenAI": "#00A67E",          # Teal
    "Anthropic": "#5436DA",       # Purple
    "Google": "#4285F4",          # Google Blue
    "DeepSeek": "#FF7518",        # Orange
    "Meta": "#0668E1",            # Facebook Blue
    "Other": "#888888"            # Gray
}

# Assign colors based on model family
def get_model_family(model_name):
    if "GPT" in model_name:
        return "OpenAI"
    elif "Claude" in model_name or "O1" in model_name:
        return "Anthropic"
    elif "Gemini" in model_name:
        return "Google"
    elif "DeepSeek" in model_name:
        return "DeepSeek"
    elif "Llama" in model_name:
        return "Meta"
    else:
        return "Other"
        
df["model_family"] = df["model"].apply(get_model_family)
colors = [color_map[family] for family in df["model_family"]]

# Create a function to generate legend elements
def create_legend(color_map, size_label, lang):
    legend_elements = []
    
    # Add color legend for model families
    for family, color in color_map.items():
        if family in df["model_family"].values:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                          markersize=8, label=family)
            )
    
    return legend_elements

# Multi-language labels
labels = {
    "en": {
        "model_family": "Model Family",
        "cost_label": "Cost (USD per 1K tokens)",
        "freedom_label": "Freedom Score",
        "higher_mmlu_label": "Higher MMLU Score"
    },
    "pt": {
        "model_family": "Família de Modelos",
        "cost_label": "Custo (USD por 1K tokens)",
        "freedom_label": "Pontuação de Liberdade",
        "higher_mmlu_label": "Pontuação MMLU mais alta"
    },
    "es": {
        "model_family": "Familia de Modelos",
        "cost_label": "Costo (USD por 1K tokens)",
        "freedom_label": "Puntuación de Libertad",
        "higher_mmlu_label": "Puntuación MMLU más alta"
    }
}

# Get the benchmarks we want to plot against freedom
benchmarks = [b["name"].lower() for b in benchmarks_data["text_model_benchmarks"] if b["name"].lower() != "freedom"]

# Generate charts for each benchmark vs freedom
for benchmark in benchmarks:
    print(f"\n--- Generating Freedom vs {benchmark.upper()} charts ---")
    
    for lang in languages:
        plt.figure(figsize=(10, 8))
        
        # Create bubble sizes based on cost (inverse, so cheaper models are bigger)
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
                # For other models, use inverse cost relationship (cheaper = bigger)
                if cost_range > 0:
                    # Normalize and invert (smaller cost = higher value)
                    normalized_inverse_cost = (max_cost - row["cost"]) / cost_range
                    bubble_size = min_size + (max_size - min_size) * np.sqrt(normalized_inverse_cost)
                else:
                    bubble_size = (min_size + max_size) / 2
            bubble_sizes.append(bubble_size)
        
        # Create bubble plot
        scatter = plt.scatter(
            df["freedom"], 
            df[benchmark], 
            s=bubble_sizes,
            c=colors,
            alpha=0.6,
            edgecolors="black",
            linewidth=1
        )
        
        # Add model names as labels
        for i, (_, row) in enumerate(df.iterrows()):
            plt.annotate(
                row["model"],
                (row["freedom"], row[benchmark]),
                fontsize=8,
                ha='center',
                va='center'
            )
        
        # Create legend for model families and bubble size
        legend_elements = create_legend(color_map, labels[lang]["higher_mmlu_label"], lang)
        
        plt.legend(handles=legend_elements, loc='lower right', title=labels[lang]["model_family"])
        
        plt.xlabel(labels[lang]["freedom_label"])
        
        # Find the benchmark friendly name
        benchmark_name = benchmark.upper()
        for b in benchmarks_data["text_model_benchmarks"]:
            if b["name"] == benchmark:
                benchmark_name = b["full_name"]
                break
        
        plt.ylabel(benchmark_name)
        
        if lang == "en":
            title = f"Freedom vs {benchmark_name}"
        elif lang == "pt":
            title = f"Liberdade vs {benchmark_name}"
        else:  # es
            title = f"Libertad vs {benchmark_name}"
            
        plt.title(title)
        
        # Add a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.4)
        
        # Set x axis limits with some padding
        x_min, x_max = df["freedom"].min() - 5, df["freedom"].max() + 5
        plt.xlim(x_min, x_max)
        
        # Set y axis limits with some padding (usually scores are percentages)
        y_min, y_max = df[benchmark].min() - 5, df[benchmark].max() + 5
        y_min = max(0, y_min)  # Don't go below 0
        y_max = min(100, y_max)  # Don't go above 100
        plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        
        # Save to language-specific directory
        lang_dir = os.path.join(base_images_dir, lang)
        filename = os.path.join(lang_dir, f"freedom_vs_{benchmark}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        
        msg = "Generated" if lang == "en" else "Gerado" if lang == "pt" else "Generado"
        print(f"{msg} {filename}")

print("\nAll freedom vs benchmark charts generated!")
