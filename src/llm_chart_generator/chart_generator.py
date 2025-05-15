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
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for data points
    fig.add_trace(
        go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='markers+text',
            marker=dict(
                size=15,
                color='rgba(8, 84, 158, 0.8)',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=data['text'],
            textposition='top center',
            hovertemplate=(
                '<b>%{text}</b><br>' +
                f'{get_translation("freedom_score_label", language)}: %{{x:.1f}}<br>' +
                f'{get_translation("benchmark_score_label", language, benchmark=benchmark_display)}: %{{y:.1f}}<br>' +
                '<extra></extra>'
            )
        )
    )
    
    # Update layout
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
        plot_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Add shaded region for high performance/high freedom area
    fig.add_shape(
        type="rect", 
        x0=50, y0=80, x1=100, y1=100,
        fillcolor="rgba(144, 238, 144, 0.3)", 
        line=dict(width=0), 
        layer="below"
    )
    
    # Add annotations for the optimal area
    fig.add_annotation(
        x=75, y=90,
        text=get_translation('high_value_region', language),
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-40
    )
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    lang_dir = os.path.join(output_dir, language)
    os.makedirs(lang_dir, exist_ok=True)
    
    # Generate filename
    output_file = os.path.join(lang_dir, f"freedom_vs_{benchmark}.png")
    
    # Save the figure
    fig.write_image(output_file, scale=3)  # High resolution
    
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
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for data points
    fig.add_trace(
        go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='markers+text',
            marker=dict(
                size=15,
                color='rgba(227, 119, 194, 0.8)',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=data['text'],
            textposition='top center',
            hovertemplate=(
                '<b>%{text}</b><br>' +
                f'{get_translation("cost_label", language)}: $%{{x:.5f}}<br>' +
                f'{get_translation("benchmark_score_label", language, benchmark=benchmark_display)}: %{{y:.1f}}<br>' +
                '<extra></extra>'
            )
        )
    )
    
    # Update layout
    fig.update_layout(
        title=get_translation('cost_vs_benchmark_title', language, benchmark=benchmark_display),
        xaxis=dict(
            title=get_translation('cost_label', language),
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            type='log'  # Log scale for cost
        ),
        yaxis=dict(
            title=get_translation('benchmark_score_label', language, benchmark=benchmark_display),
            gridcolor='lightgray',
            zerolinecolor='lightgray'
        ),
        plot_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    lang_dir = os.path.join(output_dir, language)
    os.makedirs(lang_dir, exist_ok=True)
    
    # Generate filename
    output_file = os.path.join(lang_dir, f"cost_vs_{benchmark}.png")
    
    # Save the figure
    fig.write_image(output_file, scale=3)  # High resolution
    
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
    # Calculate power score (average of all benchmarks except freedom)
    benchmarks = [col for col in df.columns if col not in ['model', 'cost', 'freedom']]
    if not benchmarks:
        logger.error("No benchmark columns found for power score calculation")
        return ""
    
    df['power_score'] = df[benchmarks].mean(axis=1)
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for data points
    fig.add_trace(
        go.Scatter(
            x=df['power_score'],
            y=df['freedom'],
            mode='markers+text',
            marker=dict(
                size=15,
                color='rgba(44, 160, 44, 0.8)',
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=df['model'],
            textposition='top center',
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Power Score: %{x:.1f}<br>' +
                f'{get_translation("freedom_score_label", language)}: %{{y:.1f}}<br>' +
                '<extra></extra>'
            )
        )
    )
    
    # Update layout
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
        plot_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Add shaded region for high power/high freedom area
    fig.add_shape(
        type="rect", 
        x0=85, y0=40, x1=100, y1=100,
        fillcolor="rgba(144, 238, 144, 0.3)", 
        line=dict(width=0), 
        layer="below"
    )
    
    # Add annotations for the optimal area
    fig.add_annotation(
        x=92.5, y=70,
        text=get_translation('high_value_region', language),
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-40
    )
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    lang_dir = os.path.join(output_dir, language)
    os.makedirs(lang_dir, exist_ok=True)
    
    # Generate filename
    output_file = os.path.join(lang_dir, "power_vs_freedom.png")
    
    # Save the figure
    fig.write_image(output_file, scale=3)  # High resolution
    
    # Skip HTML generation
    # html_file = os.path.join(lang_dir, "power_vs_freedom.html")
    # fig.write_html(html_file)
    
    logger.info(f"Saved power vs freedom chart to {output_file}")
    return output_file