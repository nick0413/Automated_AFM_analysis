import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Create data arrays
sin_array = np.sin(np.linspace(0, 4*np.pi, 100))
cos_array = np.cos(np.linspace(0, 2*np.pi, 100))

sin_x = np.linspace(0, 4*np.pi, 100)
cos_x = np.linspace(0, 2*np.pi, 100)

sin_df = pd.DataFrame({'x':sin_x, 'y': sin_array})
cos_df = pd.DataFrame({'x':cos_x, 'y': cos_array})

dfs = [sin_df, cos_df]

fig = go.Figure()

for df in dfs:
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines', name='sin(x)'))

# Create a Plotly figure

# Add sin(x) trace

# Add cos(x) trace

# Customize layout
fig.update_layout(
    title='Sin and Cos',
    xaxis_title='x',
    yaxis_title='y'
)

# Show the figure
fig.show()
fig.data = []