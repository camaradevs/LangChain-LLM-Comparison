import pandas as pd
import json
import numpy as np
import sys
sys.path.append('src')
from llm_chart_generator.data_loader import load_llm_data
from llm_chart_generator.chart_generator import generate_power_vs_freedom_chart

# Load data using the actual loader
df = load_llm_data('src/data/data.json')

print("DataFrame info:")
print(df.info())
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# Test the power score calculation
try:
    benchmarks = [col for col in df.columns if col not in ['model', 'cost', 'freedom']]
    print("\nBenchmarks:", benchmarks)
    
    # Test mean calculation
    df['power_score'] = df[benchmarks].mean(axis=1)
    print("\nPower scores calculated successfully")
    print("First 3 power scores:", df['power_score'].head(3).tolist())
    
    # Try to generate the chart
    output = generate_power_vs_freedom_chart(df, 'test_output', 'en')
    print(f"\nChart generated: {output}")
    
except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback
    traceback.print_exc()