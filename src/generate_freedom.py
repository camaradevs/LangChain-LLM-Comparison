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

# ---- 3. Create DataFrame from the data ----
df = pd.DataFrame(data)

# ---- 4. Labels for different languages ----
labels = {
    "en": {
        "freedom_label": "Freedom Score (%)",
        "power_label": "MMLU Score (%) - Model Power",
        "title": "Model Power vs Freedom",
        "gen_msg": "Generated",
        "vs_label": "Power vs Freedom",
        "cheaper_label": "Higher cost = Larger bubble",
        "higher_mmlu_label": "Higher MMLU = Larger bubble",
        "model_family": "Model Family",
        "cost_label": "Cost per 1K tokens (USD, log scale)"
    },
    "pt": {
        "freedom_label": "Pontuação de Liberdade (%)",
        "power_label": "Pontuação MMLU (%) - Potência do Modelo",
        "title": "Potência vs Liberdade do Modelo",
        "gen_msg": "Gerado",
        "vs_label": "Potência vs Liberdade",
        "cheaper_label": "Maior custo = Maior bolha",
        "higher_mmlu_label": "Maior MMLU = Maior bolha",
        "model_family": "Família de Modelos",
        "cost_label": "Custo por 1K tokens (USD, escala logarítmica)"
    },
    "es": {
        "freedom_label": "Puntuación de Libertad (%)",
        "power_label": "Puntuación MMLU (%) - Potencia del Modelo",
        "title": "Potencia vs Libertad del Modelo",
        "gen_msg": "Generado",
        "vs_label": "Potencia vs Libertad",
        "cheaper_label": "Mayor costo = Burbuja más grande",
        "higher_mmlu_label": "Mayor MMLU = Burbuja más grande",
        "model_family": "Familia de Modelos",
        "cost_label": "Costo por 1K tokens (USD, escala logarítmica)"
    }
}

# Helper function to get model colors
def get_model_colors(model_names):
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
    for model_name in model_names:
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
            
    return colors, color_map

# Helper function to create a legend for model families and bubble size
def create_legend(color_map, size_label, lang="en"):
    # Create legend for model families
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                      label=family, markerfacecolor=color, markersize=8)
                      for family, color in color_map.items()]
    
    # Add a legend for bubble size
    bubble_size_legend = Line2D([0], [0], marker='o', color='w', 
                        label=size_label, 
                        markerfacecolor='gray', markersize=10)
    legend_elements.append(bubble_size_legend)
    
    return legend_elements

# ---- 5. Create scatter plots for Power (MMLU) vs Freedom ----
def scatter_freedom_power(lang="en"):
    plt.figure(figsize=(10, 7))
    
    # Get colors for model families
    colors, color_map = get_model_colors(df["model"])
    
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
    
    # Create bubble plot
    scatter = plt.scatter(
        df["freedom"], 
        df["mmlu"], 
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
            (row["freedom"], row["mmlu"]),
            fontsize=8,
            ha='center',
            va='center'
        )
    
    # Create legend for model families and bubble size
    legend_elements = create_legend(color_map, labels[lang]["cheaper_label"], lang)
    
    plt.legend(handles=legend_elements, loc='lower right', title=labels[lang]["model_family"])
    
    plt.xlabel(labels[lang]["freedom_label"])
    plt.ylabel(labels[lang]["power_label"])
    plt.title(labels[lang]["title"])
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Set axes limits with some padding
    x_min, x_max = df["freedom"].min() - 5, df["freedom"].max() + 5
    y_min, y_max = df["mmlu"].min() - 5, df["mmlu"].max() + 5
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save to language-specific directory
    lang_dir = os.path.join(base_images_dir, lang)
    filename = os.path.join(lang_dir, f"power_vs_freedom.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"{labels[lang]['gen_msg']} {filename}")
    return filename

# Generate charts for all languages
all_files = {}
for lang in langs:
    print(f"\n--- Generating Power vs Freedom chart for {lang} ---")
    chart_path = scatter_freedom_power(lang)
    all_files[f"{lang}_power_freedom_chart"] = chart_path

print("\nGenerated charts:")
for chart_name, chart_path in all_files.items():
    print(f"- {chart_name}: {chart_path}")

# Also create a scatter plot for Cost vs Freedom
def scatter_freedom_cost(lang="en"):
    plt.figure(figsize=(10, 7))
    
    # Get colors for model families
    colors, color_map = get_model_colors(df["model"])
    
    # Calculate bubble sizes based on performance (MMLU score)
    # Higher MMLU score = bigger bubble
    min_size = 150
    max_size = 600
    
    # Get min and max MMLU values to normalize
    min_mmlu = min(df["mmlu"])
    max_mmlu = max(df["mmlu"])
    mmlu_range = max_mmlu - min_mmlu
    
    # Set bubble sizes with special override for O1 model
    bubble_sizes = []
    for _, row in df.iterrows():
        if "O1" in row["model"]:
            # Force O1 to have largest bubble
            bubble_size = max_size * 2  # Make O1 extra large
        else:
            # For other models, scale based on mmlu
            if mmlu_range > 0:
                normalized_mmlu = (row["mmlu"] - min_mmlu) / mmlu_range
                bubble_size = min_size + (max_size - min_size) * np.sqrt(normalized_mmlu)
            else:
                bubble_size = (min_size + max_size) / 2
        bubble_sizes.append(bubble_size)
    
    # Create bubble plot
    scatter = plt.scatter(
        df["freedom"], 
        df["cost"], 
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
            (row["freedom"], row["cost"]),
            fontsize=8,
            ha='center',
            va='center'
        )
    
    # Create legend for model families and bubble size
    legend_elements = create_legend(color_map, labels[lang]["higher_mmlu_label"], lang)
    
    plt.legend(handles=legend_elements, loc='lower right', title=labels[lang]["model_family"])
    
    plt.xlabel(labels[lang]["freedom_label"])
    plt.ylabel(labels[lang]["cost_label"])
    plt.title("Cost vs Freedom" if lang == "en" else
             "Custo vs Liberdade" if lang == "pt" else
             "Costo vs Libertad")
    
    # Use log scale for cost axis
    plt.yscale("log")
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Set x axis limits with some padding
    x_min, x_max = df["freedom"].min() - 5, df["freedom"].max() + 5
    plt.xlim(x_min, x_max)
    
    plt.tight_layout()
    
    # Save to language-specific directory
    lang_dir = os.path.join(base_images_dir, lang)
    filename = os.path.join(lang_dir, f"cost_vs_freedom.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"{labels[lang]['gen_msg']} {filename}")
    return filename

# Generate cost vs freedom charts for all languages
for lang in langs:
    print(f"\n--- Generating Cost vs Freedom chart for {lang} ---")
    chart_path = scatter_freedom_cost(lang)
    all_files[f"{lang}_cost_freedom_chart"] = chart_path

print("\nGenerated charts:")
for chart_name, chart_path in all_files.items():
    print(f"- {chart_name}: {chart_path}")
