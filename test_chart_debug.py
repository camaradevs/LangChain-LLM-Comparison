import sys
sys.path.append('src')
from llm_chart_generator.data_loader import load_llm_data
from llm_chart_generator.chart_generator import generate_power_vs_freedom_chart
import os
import traceback

# Load data
df = load_llm_data('src/data/data.json')

# Create output directory
os.makedirs('test_output/en', exist_ok=True)

try:
    # Generate a single chart
    output = generate_power_vs_freedom_chart(df, 'test_output', 'en')
    print(f"Chart generated: {output}")
    
    # Check if file exists
    if os.path.exists(output):
        print(f"File exists: {output}")
        print(f"File size: {os.path.getsize(output)} bytes")
    else:
        print(f"File does not exist: {output}")
        
except Exception as e:
    print(f"Error occurred: {str(e)}")
    traceback.print_exc()