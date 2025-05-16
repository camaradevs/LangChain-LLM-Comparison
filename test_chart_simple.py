import sys
sys.path.append('src')

import plotly.graph_objects as go
import os

# Create a simple test figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='Test'))

# Create output directory
output_dir = 'test_images'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'simple_test.png')

print(f"Attempting to save to: {output_file}")

try:
    # Try plotly directly
    import plotly.io as pio
    pio.write_image(fig, output_file, scale=3)
    
    if os.path.exists(output_file):
        print(f"Success! File exists with size: {os.path.getsize(output_file)} bytes")
    else:
        print("File not created")
        
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

# List directory contents
print("\nDirectory contents:")
for root, dirs, files in os.walk(output_dir):
    for file in files:
        full_path = os.path.join(root, file)
        print(f" - {full_path}: {os.path.getsize(full_path)} bytes")