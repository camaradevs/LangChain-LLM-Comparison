# LLM Benchmark Chart Generator

A Python tool for generating visualization charts for LLM benchmarking data in multiple languages.

## Features

- Generates comparative charts for LLM benchmark data
- Supports multiple chart types:
  - Freedom score vs. benchmark performance
  - Cost vs. benchmark performance 
  - Power (average benchmark) vs. freedom
- Multilingual support:
  - English
  - Spanish
  - Portuguese
- High-resolution PNG outputs (2400x1800px)
- Parallel processing for efficient chart generation

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Generate all charts in all supported languages:

```bash
python src/main.py
```

### Advanced Options

```bash
python src/main.py --data-file src/data/data.json --output-dir images --languages en es pt --parallel
```

### Command-line Arguments

- `--data-file`: Path to LLM benchmark data file (default: `src/data/data.json`)
- `--benchmarks-file`: Path to benchmarks description file (default: `src/data/benchmarks.json`)
- `--models-file`: Path to LLM models file (default: `src/data/llms.json`)
- `--output-dir`: Directory to save generated charts (default: `images`)
- `--languages`: Languages to generate charts in (default: `en es pt`)
- `--benchmarks`: Specific benchmarks to generate charts for (default: all)
- `--log-level`: Logging level (default: `INFO`)
- `--parallel`: Use parallel processing for chart generation
- `--workers`: Number of worker processes for parallel processing (default: number of CPU cores)

## Data Format

The tool expects JSON data files with the following structure:

### LLM Benchmark Data

```json
[
  {
    "model": "Model Name",
    "cost": 0.01500,
    "mmlu": 88.0,
    "hellaswag": 95.4,
    "freedom": 30.0,
    ...
  },
  ...
]
```

### Benchmark Descriptions

```json
{
  "text_model_benchmarks": [
    {
      "name": "MMLU",
      "full_name": "Massive Multitask Language Understanding",
      "description": "...",
      "category": "knowledge",
      "reference": "https://arxiv.org/abs/..."
    },
    ...
  ]
}
```

## Output

Charts are saved in the specified output directory, organized by language:

```
images/
├── en/
│   ├── freedom_vs_mmlu.png
│   ├── cost_vs_mmlu.png
│   ├── power_vs_freedom.png
│   └── ...
├── es/
│   └── ...
└── pt/
    └── ...
```