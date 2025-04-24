# Developer Guide

This document explains how to update the cost vs. performance charts for the LLM comparison project.

## Quick Start

To update all charts with new data:

```bash
python3 -m venv venv
source venv/bin/activate
pip install matplotlib pandas
python3 src/generate.py
```

This command will regenerate all charts based on the current data and save them in the `/images` folder, overwriting existing images.

## Project Structure

```
/
├── images/                # Generated chart images
├── src/
│   ├── data/              # Data files
│   │   └── data.json      # Model data (cost and performance metrics)
│   └── generate.py        # Chart generation script
```

## Data Format

Edit the `src/data/data.json` file to update or add new models. The file contains an array of objects with the following structure:

```json
[
  {
    "model": "Model Name",
    "cost": 0.00000,       // USD per 1K tokens (average of input+output)
    "mmlu": 00.0,          // MMLU score (percentage)
    "hellaswag": 00.0,     // HellaSwag score (percentage)
    "humaneval": 00.0      // HumanEval score (percentage)
  }
]
```

## How to Update

1. **Update Data**: Edit the `src/data/data.json` file with new or updated model information.

2. **Generate Charts**: Run `python3 src/generate.py` to regenerate all charts.

3. **Verify Results**: Check the `/images` folder for the updated charts:
   - `cost_vs_mmlu.png`
   - `cost_vs_hellaswag.png`
   - `cost_vs_humaneval.png`

## Implementation Details

The chart generation script uses Matplotlib and:
- Reads model data from `src/data/data.json`
- Creates scatter plots with logarithmic x-axis for cost
- Labels each data point with the model name
- Saves high-resolution (300 DPI) charts to the `/images` folder

## Requirements

- Python 3.6+
- matplotlib
- pandas

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install matplotlib pandas
```