import sys
sys.path.append('src')
from llm_chart_generator.data_loader import load_llm_data
from llm_chart_generator.chart_generator import generate_power_vs_freedom_chart

# Load data
df = load_llm_data('src/data/data.json')

# Generate a single chart
output = generate_power_vs_freedom_chart(df, 'test_output', 'en')
print(f"Chart generated: {output}")

# Check if file exists
import os
if os.path.exists(output):
    print(f"File exists: {output}")
    print(f"File size: {os.path.getsize(output)} bytes")
else:
    print(f"File does not exist: {output}")