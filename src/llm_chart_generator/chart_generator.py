"""
Chart generation functions for LLM benchmark visualization.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from llm_chart_generator.translations import get_translation, get_benchmark_name

logger = logging.getLogger(__name__)

def generate_freedom_vs_benchmark_chart(
    data: dict, 
    output_dir: str, 
    language: str = 'en'
) -> str:
    """
    Generate a freedom vs benchmark performance chart.
    
    Args:
        data: Dictionary with chart data
        output_dir: Directory to save the chart
        language: Language for chart labels and text
        
    Returns:
        Path to the generated chart file
    """
    benchmark = data.get('benchmark', 'unknown')
    benchmark_display = get_benchmark_name(benchmark, language)
    
    # Advanced roadmap: bubble size, color by provider, custom legend, accessibility
    df = pd.DataFrame({
        'freedom': data['x'],
        'score': data['y'],
        'model': data['text'],
        'provider': data.get('provider', ['Other'] * len(data['x']))
    })


    # Bubble size scaling options (linear, sqrt, log)
    bubble_scale_mode = data.get('bubble_scale_mode', 'fixed')  # 'fixed', 'sqrt', 'log', 'linear'
    if bubble_scale_mode == 'sqrt':
        df['bubble_size'] = np.sqrt(data.get('bubble_metric', [1]*len(df))) * 40
    elif bubble_scale_mode == 'log':
        df['bubble_size'] = np.log1p(data.get('bubble_metric', [1]*len(df))) * 40
    elif bubble_scale_mode == 'linear':
        df['bubble_size'] = data.get('bubble_metric', [30]*len(df))
    else:
        df['bubble_size'] = 30  # fixed
    sizeref = 2.0 * df['bubble_size'].max() / (60**2)

    # Color palette by provider (color-blind safe, Okabe-Ito as option)
    provider_palette_default = {
        'OpenAI': '#10a37f',
        'Anthropic': '#b83280',
        'Google': '#4285F4',
        'Microsoft': '#00a4ef',
        'Meta': '#0668E1',
        'Cohere': '#2596be',
        'DeepSeek': '#6366f1',
        'Mistral': '#7c3aed',
        'HuggingFace': '#fbbf24',
        'Stability AI': '#e11d48',
        'Other': '#888888'
    }
    provider_palette_okabe_ito = {
        'OpenAI': '#E69F00',
        'Anthropic': '#56B4E9',
        'Google': '#009E73',
        'Microsoft': '#F0E442',
        'Meta': '#0072B2',
        'Cohere': '#D55E00',
        'DeepSeek': '#CC79A7',
        'Mistral': '#999999',
        'HuggingFace': '#E69F00',
        'Stability AI': '#56B4E9',
        'Other': '#888888'
    }
    palette_mode = data.get('palette_mode', 'default')  # 'default' or 'okabe_ito'
    provider_palette = provider_palette_okabe_ito if palette_mode == 'okabe_ito' else provider_palette_default
    df['color'] = df['provider'].map(lambda p: provider_palette.get(p, '#888888'))

    # Symbol by provider for accessibility
    provider_symbol_map = {
        'OpenAI': 'circle',
        'Anthropic': 'square',
        'Google': 'diamond',
        'Microsoft': 'cross',
        'Meta': 'triangle-up',
        'Cohere': 'star',
        'DeepSeek': 'triangle-down',
        'Mistral': 'x',
        'HuggingFace': 'hexagram',
        'Stability AI': 'pentagon',
        'Other': 'circle-open'
    }
    df['symbol'] = df['provider'].map(lambda p: provider_symbol_map.get(p, 'circle-open'))

    # Text position dictionary (allow custom, fallback to 'top center')
    text_positions = data.get('text_positions', {m: 'top center' for m in df['model']})

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.85, 0.15],
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]],
        horizontal_spacing=0.02
    )

    # Main bubble chart
    fig.add_trace(
        go.Scatter(
            x=df['freedom'],
            y=df['score'],
            mode='markers+text',
            marker=dict(
                size=df['bubble_size'],
                sizemode='area',
                sizeref=sizeref,
                color=df['color'],
                symbol=df['symbol'],
                line=dict(width=1, color='rgba(100,100,100,0.3)')
            ),
            text=df['model'],
            textposition=[text_positions.get(m, 'top center') for m in df['model']],
            textfont=dict(size=10, color='rgba(0,0,0,0.7)'),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                f'{get_translation("freedom_score_label", language)}: %{{x:.1f}}<br>' +
                f'{get_translation("benchmark_score_label", language, benchmark=benchmark_display)}: %{{y:.1f}}<br>' +
                'Provider: %{customdata}<br>' +
                'Bubble size: %{marker.size:.2f}<br>' +
                '<extra></extra>'
            ),
            customdata=df['provider'],
            name=''
        ),
        row=1, col=1
    )

    # Add provider legend (one trace per provider, invisible points)
    for provider in df['provider'].unique():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=15, color=provider_palette.get(provider, '#888888'), symbol=provider_symbol_map.get(provider, 'circle-open')),
                name=provider,
                showlegend=True
            ),
            row=1, col=1
        )

    # Bubble size legend (reference bubbles)
    legend_sizes = [10, 20, 30]
    legend_y = list(range(len(legend_sizes)))
    fig.add_trace(
        go.Scatter(
            x=[0] * len(legend_sizes),
            y=legend_y,
            mode='markers+text',
            marker=dict(
                size=legend_sizes,
                sizemode='area',
                sizeref=sizeref,
                color='rgba(100,100,100,0.5)',
                symbol='circle',
                line=dict(width=1, color='rgba(70,70,70,0.5)')
            ),
            text=[f"{s}" for s in legend_sizes],
            textposition="middle right",
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=2
    )

    # Layout and accessibility
    fig.update_layout(
        title=get_translation('freedom_vs_benchmark_title', language, benchmark=benchmark_display),
        xaxis=dict(
            title=get_translation('freedom_score_label', language),
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            title=get_translation('benchmark_score_label', language, benchmark=benchmark_display),
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        plot_bgcolor='rgba(245,245,245,0.8)',
        width=1000,
        height=600,
        margin=dict(l=80, r=80, t=100, b=80),
        legend=dict(
            title="Provider",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        )
    )
    # Hide axes for legend subplot
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    # Add legend title
    fig.add_annotation(
        x=0, y=1.15,
        xref="x2", yref="paper",
        text="Bubble size (arbitrary units)",
        showarrow=False,
        font=dict(size=12)
    )
    # Add accessibility note
    fig.add_annotation(
        x=0.95, y=0.05,
        xref="paper", yref="paper",
        text="Bubble size = fixed (for freedom)",
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.7)",
        borderpad=4
    )
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    lang_dir = os.path.join(output_dir, language)
    os.makedirs(lang_dir, exist_ok=True)
    
    # Generate filename
    output_file = os.path.join(lang_dir, f"freedom_vs_{benchmark}.png")
    output_file = os.path.abspath(output_file)
    
    # Save the figure
    pio.write_image(fig, output_file, scale=3)  # High resolution
    
    # Skip HTML generation
    # html_file = os.path.join(lang_dir, f"freedom_vs_{benchmark}.html")
    # fig.write_html(html_file)
    
    logger.info(f"Saved freedom vs {benchmark} chart to {output_file}")
    return output_file

def generate_cost_vs_benchmark_chart(
    data: dict, 
    output_dir: str, 
    language: str = 'en'
) -> str:
    """
    Generate a cost vs benchmark performance chart.
    
    Args:
        data: Dictionary with chart data
        output_dir: Directory to save the chart
        language: Language for chart labels and text
        
    Returns:
        Path to the generated chart file
    """
    benchmark = data.get('benchmark', 'unknown')
    benchmark_display = get_benchmark_name(benchmark, language)
    
    # Advanced roadmap: bubble size, color by provider, custom legend, accessibility
    # Prepare data
    df = pd.DataFrame({
        'cost': data['x'],
        'score': data['y'],
        'model': data['text'],
        'provider': data.get('provider', ['Other'] * len(data['x']))
    })

    # Bubble size: sqrt scaling for cost
    df['bubble_size'] = np.sqrt(df['cost']) * 40  # scale factor for visibility
    sizeref = 2.0 * df['bubble_size'].max() / (60**2)

    # Color palette by provider (color-blind safe)
    provider_palette = {
        'OpenAI': '#10a37f',
        'Anthropic': '#b83280',
        'Google': '#4285F4',
        'Microsoft': '#00a4ef',
        'Meta': '#0668E1',
        'Cohere': '#2596be',
        'DeepSeek': '#6366f1',
        'Mistral': '#7c3aed',
        'HuggingFace': '#fbbf24',
        'Stability AI': '#e11d48',
        'Other': '#888888'
    }
    df['color'] = df['provider'].map(lambda p: provider_palette.get(p, '#888888'))

    # Text position dictionary (optional, fallback to 'top center')
    text_positions = {m: 'top center' for m in df['model']}

    # Create subplots: main chart + bubble size legend
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.85, 0.15],
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]],
        horizontal_spacing=0.02
    )

    # Main bubble chart
    fig.add_trace(
        go.Scatter(
            x=df['cost'],
            y=df['score'],
            mode='markers+text',
            marker=dict(
                size=df['bubble_size'],
                sizemode='area',
                sizeref=sizeref,
                color=df['color'],
                line=dict(width=1, color='rgba(100,100,100,0.3)')
            ),
            text=df['model'],
            textposition=[text_positions.get(m, 'top center') for m in df['model']],
            textfont=dict(size=10, color='rgba(0,0,0,0.7)'),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                f'{get_translation("cost_label", language)}: $%{{x:.5f}}<br>' +
                f'{get_translation("benchmark_score_label", language, benchmark=benchmark_display)}: %{{y:.1f}}<br>' +
                'Provider: %{customdata}<br>' +
                'Cost Bubble: %{marker.size:.2f}<br>' +
                '<extra></extra>'
            ),
            customdata=df['provider'],
            name=''
        ),
        row=1, col=1
    )

    # Add provider legend (one trace per provider, invisible points)
    for provider in df['provider'].unique():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=15, color=provider_palette.get(provider, '#888888')),
                name=provider,
                showlegend=True
            ),
            row=1, col=1
        )

    # Bubble size legend (reference bubbles)
    legend_sizes = [0.1, 1.0, 5.0, 15.0]
    legend_sizes_scaled = np.sqrt(legend_sizes) * 40
    legend_y = list(range(len(legend_sizes)))
    fig.add_trace(
        go.Scatter(
            x=[0] * len(legend_sizes),
            y=legend_y,
            mode='markers+text',
            marker=dict(
                size=legend_sizes_scaled,
                sizemode='area',
                sizeref=sizeref,
                color='rgba(100,100,100,0.5)',
                line=dict(width=1, color='rgba(70,70,70,0.5)')
            ),
            text=[f"${s:.2f}" for s in legend_sizes],
            textposition="middle right",
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=2
    )

    # Layout and accessibility
    fig.update_layout(
        title=get_translation('cost_vs_benchmark_title', language, benchmark=benchmark_display),
        xaxis=dict(
            title=get_translation('cost_label', language),
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            type='log'
        ),
        yaxis=dict(
            title=get_translation('benchmark_score_label', language, benchmark=benchmark_display),
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        plot_bgcolor='rgba(245,245,245,0.8)',
        width=1000,
        height=600,
        margin=dict(l=80, r=80, t=100, b=80),
        legend=dict(
            title="Provider",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        )
    )
    # Hide axes for legend subplot
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    # Add legend title
    fig.add_annotation(
        x=0, y=1.15,
        xref="x2", yref="paper",
        text="Cost per 1k tokens",
        showarrow=False,
        font=dict(size=12)
    )
    # Add accessibility note
    fig.add_annotation(
        x=0.95, y=0.05,
        xref="paper", yref="paper",
        text="Bubble size = cost per 1k tokens",
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.7)",
        borderpad=4
    )
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    lang_dir = os.path.join(output_dir, language)
    os.makedirs(lang_dir, exist_ok=True)
    
    # Generate filename
    output_file = os.path.join(lang_dir, f"cost_vs_{benchmark}.png")
    output_file = os.path.abspath(output_file)
    
    # Save the figure
    pio.write_image(fig, output_file, scale=3)  # High resolution
    
    # Skip HTML generation
    # html_file = os.path.join(lang_dir, f"cost_vs_{benchmark}.html")
    # fig.write_html(html_file)
    
    logger.info(f"Saved cost vs {benchmark} chart to {output_file}")
    return output_file

def generate_power_vs_freedom_chart(
    df: pd.DataFrame,
    output_dir: str,
    language: str = 'en'
) -> str:
    """
    Generate a power vs freedom chart based on combined benchmark scores.
    
    Args:
        df: DataFrame with LLM benchmark data
        output_dir: Directory to save the chart
        language: Language for chart labels and text
        
    Returns:
        Path to the generated chart file
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()
    
    # Calculate power score (average of all benchmarks except freedom)
    benchmarks = [col for col in df.columns if col not in ['model', 'cost', 'freedom']]
    if not benchmarks:
        logger.error("No benchmark columns found for power score calculation")
        return ""
    
    df['power_score'] = df[benchmarks].mean(axis=1)
    
    # Advanced roadmap: bubble size, color by provider, custom legend, accessibility
    # Bubble size: sqrt scaling for cost if available, else fixed
    if 'cost' in df.columns:
        df['bubble_size'] = np.sqrt(df['cost']) * 40
    else:
        df['bubble_size'] = 30
    sizeref = 2.0 * df['bubble_size'].max() / (60**2)

    # Color palette by provider (color-blind safe)
    provider_palette = {
        'OpenAI': '#10a37f',
        'Anthropic': '#b83280',
        'Google': '#4285F4',
        'Microsoft': '#00a4ef',
        'Meta': '#0668E1',
        'Cohere': '#2596be',
        'DeepSeek': '#6366f1',
        'Mistral': '#7c3aed',
        'HuggingFace': '#fbbf24',
        'Stability AI': '#e11d48',
        'Other': '#888888'
    }
    if 'provider' in df.columns:
        df['color'] = df['provider'].map(lambda p: provider_palette.get(p, '#888888'))
    else:
        df['color'] = '#888888'

    # Text position dictionary (optional, fallback to 'top center')
    text_positions = {m: 'top center' for m in df['model']}

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.85, 0.15],
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]],
        horizontal_spacing=0.02
    )

    # Main bubble chart
    fig.add_trace(
        go.Scatter(
            x=df['power_score'],
            y=df['freedom'],
            mode='markers+text',
            marker=dict(
                size=df['bubble_size'],
                sizemode='area',
                sizeref=sizeref,
                color=df['color'],
                line=dict(width=1, color='rgba(100,100,100,0.3)')
            ),
            text=df['model'],
            textposition=[text_positions.get(m, 'top center') for m in df['model']],
            textfont=dict(size=10, color='rgba(0,0,0,0.7)'),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Power Score: %{x:.1f}<br>' +
                f'{get_translation("freedom_score_label", language)}: %{{y:.1f}}<br>' +
                'Provider: %{customdata}<br>' +
                '<extra></extra>'
            ),
            customdata=df['provider'] if 'provider' in df.columns else None,
            name=''
        ),
        row=1, col=1
    )

    # Add provider legend (one trace per provider, invisible points)
    if 'provider' in df.columns:
        for provider in df['provider'].unique():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=15, color=provider_palette.get(provider, '#888888')),
                    name=provider,
                    showlegend=True
                ),
                row=1, col=1
            )

    # Bubble size legend (reference bubbles)
    legend_sizes = [10, 20, 30]
    legend_y = list(range(len(legend_sizes)))
    fig.add_trace(
        go.Scatter(
            x=[0] * len(legend_sizes),
            y=legend_y,
            mode='markers+text',
            marker=dict(
                size=legend_sizes,
                sizemode='area',
                sizeref=sizeref,
                color='rgba(100,100,100,0.5)',
                line=dict(width=1, color='rgba(70,70,70,0.5)')
            ),
            text=[f"{s}" for s in legend_sizes],
            textposition="middle right",
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=2
    )

    # Layout and accessibility
    fig.update_layout(
        title=get_translation('power_vs_freedom_title', language),
        xaxis=dict(
            title='Power Score (Average Benchmark)',
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        yaxis=dict(
            title=get_translation('freedom_score_label', language),
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        plot_bgcolor='rgba(245,245,245,0.8)',
        width=1000,
        height=600,
        margin=dict(l=80, r=80, t=100, b=80),
        legend=dict(
            title="Provider",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        )
    )
    # Hide axes for legend subplot
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    # Add legend title
    fig.add_annotation(
        x=0, y=1.15,
        xref="x2", yref="paper",
        text="Bubble size (arbitrary units)",
        showarrow=False,
        font=dict(size=12)
    )
    # Add accessibility note
    fig.add_annotation(
        x=0.95, y=0.05,
        xref="paper", yref="paper",
        text="Bubble size = cost or fixed",
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.7)",
        borderpad=4
    )
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    lang_dir = os.path.join(output_dir, language)
    os.makedirs(lang_dir, exist_ok=True)
    
    # Generate filename
    output_file = os.path.join(lang_dir, "power_vs_freedom.png")
    output_file = os.path.abspath(output_file)
    
    # Save the figure
    pio.write_image(fig, output_file, scale=3)  # High resolution
    
    # Skip HTML generation
    # html_file = os.path.join(lang_dir, "power_vs_freedom.html")
    # fig.write_html(html_file)
    
    logger.info(f"Saved power vs freedom chart to {output_file}")
    return output_file