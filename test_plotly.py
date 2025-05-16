import plotly.graph_objects as go
import os

# Create a simple test figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='Test'))

# Try to save it
output_file = 'test_plot.png'
output_file = os.path.abspath(output_file)

print(f"Attempting to save to: {output_file}")

try:
    fig.write_image(output_file, scale=3)
    print(f"Success! File saved to: {output_file}")
    if os.path.exists(output_file):
        print(f"File exists: {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print(f"File does not exist: {output_file}")
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()