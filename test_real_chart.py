import sys
sys.path.append('src')

from llm_chart_generator.data_loader import load_llm_data
import plotly.graph_objects as go
import plotly.io as pio
import os

# Load data
df = load_llm_data('src/data/data.json')

# Create a simple freedom vs mmlu figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['freedom'],
    y=df['mmlu'],
    mode='markers+text',
    text=df['model'],
    textposition='top center',
    marker=dict(size=10)
))

fig.update_layout(
    title="Freedom vs MMLU",
    xaxis_title="Freedom Score",
    yaxis_title="MMLU",
    width=1000,
    height=600
)

# Try to save using different methods
output_dir = 'test_output'
os.makedirs(output_dir, exist_ok=True)

# Method 1: fig.write_image
output1 = os.path.join(output_dir, 'method1_write_image.png')
print(f"Method 1: Saving to {output1}")
try:
    fig.write_image(output1, scale=3)
    if os.path.exists(output1):
        print(f"Success! Size: {os.path.getsize(output1)} bytes")
    else:
        print("File not created")
except Exception as e:
    print(f"Error: {e}")

# Method 2: pio.write_image  
output2 = os.path.join(output_dir, 'method2_pio_write.png')
print(f"\nMethod 2: Saving to {output2}")
try:
    pio.write_image(fig, output2, scale=3)
    if os.path.exists(output2):
        print(f"Success! Size: {os.path.getsize(output2)} bytes")
    else:
        print("File not created")
except Exception as e:
    print(f"Error: {e}")

# List all files
print("\nFiles created:")
for f in os.listdir(output_dir):
    print(f" - {f}: {os.path.getsize(os.path.join(output_dir, f))} bytes")