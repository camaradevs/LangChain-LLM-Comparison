"""
Data loading and processing functions for LLM chart generation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> dict:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the loaded data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return {}

def load_llm_data(file_path: str) -> pd.DataFrame:
    """
    Load and process LLM benchmark data.
    
    Args:
        file_path: Path to the LLM benchmark data file
        
    Returns:
        DataFrame containing the processed data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        required_columns = ['model', 'cost']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns in {file_path}: {missing}")
            return pd.DataFrame()
        
        # Convert numeric columns to float (except model column)
        for col in df.columns:
            if col != 'model':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to numeric: {str(e)}")
        
        # Log any columns that contain non-numeric values
        for col in df.columns:
            if col != 'model':
                if df[col].isnull().any():
                    logger.warning(f"Column {col} contains non-numeric values that were converted to NaN")
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return pd.DataFrame()

def load_benchmarks(file_path: str) -> dict:
    """
    Load benchmark descriptions.
    
    Args:
        file_path: Path to the benchmarks description file
        
    Returns:
        Dictionary containing benchmark information
    """
    try:
        benchmarks_data = load_json_file(file_path)
        return benchmarks_data.get('text_model_benchmarks', [])
    except Exception as e:
        logger.error(f"Error loading benchmarks from {file_path}: {str(e)}")
        return []

def load_llm_models(file_path: str) -> list:
    """
    Load LLM model information.
    
    Args:
        file_path: Path to the LLM models file
        
    Returns:
        List of model information dictionaries
    """
    try:
        models_data = load_json_file(file_path)
        models_list = []
        
        for platform in models_data.get('platforms', []):
            platform_name = platform.get('name', '')
            for model in platform.get('models', []):
                model['platform'] = platform_name
                models_list.append(model)
                
        return models_list
    except Exception as e:
        logger.error(f"Error loading LLM models from {file_path}: {str(e)}")
        return []

def prepare_freedom_vs_benchmark_data(df: pd.DataFrame, benchmark: str) -> dict:
    """
    Prepare data for freedom vs benchmark chart.
    
    Args:
        df: DataFrame containing LLM benchmark data
        benchmark: Name of the benchmark to compare against freedom
        
    Returns:
        Dictionary with processed data for chart generation
    """
    if 'freedom' not in df.columns or benchmark not in df.columns:
        logger.error(f"Required columns missing for freedom vs benchmark chart")
        return {}
        
    result = {
        'x': df['freedom'].tolist(),
        'y': df[benchmark].tolist(),
        'text': df['model'].tolist(),
        'benchmark': benchmark
    }
    
    return result

def prepare_cost_vs_benchmark_data(df: pd.DataFrame, benchmark: str) -> dict:
    """
    Prepare data for cost vs benchmark chart.
    
    Args:
        df: DataFrame containing LLM benchmark data
        benchmark: Name of the benchmark to compare against cost
        
    Returns:
        Dictionary with processed data for chart generation
    """
    if 'cost' not in df.columns or benchmark not in df.columns:
        logger.error(f"Required columns missing for cost vs benchmark chart")
        return {}
        
    result = {
        'x': df['cost'].tolist(),
        'y': df[benchmark].tolist(),
        'text': df['model'].tolist(),
        'benchmark': benchmark
    }
    
    return result