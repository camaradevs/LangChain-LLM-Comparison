import plotly.graph_objects as go
import plotly.io as pio
import os
import tempfile

# Debug kaleido
import kaleido
import plotly
print("Kaleido version:", kaleido.__version__)
print("Plotly version:", plotly.__version__)

# Check kaleido scope
print("\nKaleido Scope available:", hasattr(pio, 'kaleido'))
print("Kaleido Scope class:", type(pio.kaleido.scope))

# Create a simple test figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='Test'))

# Try different methods
# Method 1: Direct write_image
output_file = 'test_plot1.png'
print(f"\nMethod 1: Direct write_image to {output_file}")
try:
    fig.write_image(output_file)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")

# Method 2: Using full path
output_file = os.path.join(os.getcwd(), 'test_plot2.png')
print(f"\nMethod 2: Full path to {output_file}")
try:
    fig.write_image(output_file)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")

# Method 3: Using temp directory
with tempfile.TemporaryDirectory() as tmpdir:
    output_file = os.path.join(tmpdir, 'test_plot3.png')
    print(f"\nMethod 3: Temp directory {output_file}")
    try:
        fig.write_image(output_file)
        print("Success!")
        if os.path.exists(output_file):
            print(f"File exists in temp dir with size: {os.path.getsize(output_file)} bytes")
    except Exception as e:
        print(f"Error: {e}")

# Method 4: Using pio directly
output_file = 'test_plot4.png'
print(f"\nMethod 4: Using pio.write_image to {output_file}")
try:
    pio.write_image(fig, output_file)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")

# List all files created
print("\nFiles in current directory:")
for f in os.listdir('.'):
    if f.startswith('test_plot'):
        print(f" - {f}: {os.path.getsize(f)} bytes" if os.path.exists(f) else f" - {f}: does not exist")