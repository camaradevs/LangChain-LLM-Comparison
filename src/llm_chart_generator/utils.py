"""
Utility functions for the LLM chart generator.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional path to log file
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Add file handler
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def ensure_directory(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure
        
    Returns:
        The directory path
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def get_benchmark_columns(df) -> List[str]:
    """
    Get the list of benchmark columns from a DataFrame.
    
    Args:
        df: DataFrame with LLM benchmark data
        
    Returns:
        List of benchmark column names
    """
    # Exclude non-benchmark columns
    non_benchmarks = ['model', 'cost', 'platform']
    benchmarks = [col for col in df.columns if col not in non_benchmarks]
    return benchmarks