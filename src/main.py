#!/usr/bin/env python3
"""
LLM Chart Generator

This script generates visualization charts for LLM comparison data in multiple languages.
It creates freedom vs benchmark, cost vs benchmark, and power vs freedom charts.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any

import pandas as pd

from llm_chart_generator.data_loader import (
    load_llm_data, load_benchmarks, prepare_freedom_vs_benchmark_data,
    prepare_cost_vs_benchmark_data
)
from llm_chart_generator.chart_generator import (
    generate_freedom_vs_benchmark_chart, 
    generate_cost_vs_benchmark_chart,
    generate_power_vs_freedom_chart
)
from llm_chart_generator.utils import setup_logging, ensure_directory, get_benchmark_columns

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate LLM benchmark comparison charts')
    
    parser.add_argument('--data-file', type=str, 
                       default='src/data/data_updated_2025.json',
                       help='Path to LLM benchmark data file')
    
    parser.add_argument('--benchmarks-file', type=str, 
                       default='src/data/benchmarks.json',
                       help='Path to benchmarks description file')
    
    parser.add_argument('--models-file', type=str, 
                       default='src/data/llms.json',
                       help='Path to LLM models file')
    
    parser.add_argument('--output-dir', type=str, 
                       default='images',
                       help='Directory to save generated charts')
    
    parser.add_argument('--languages', type=str, nargs='+',
                       default=['en', 'es', 'pt'],
                       help='Languages to generate charts in')
    
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       help='Specific benchmarks to generate charts for (default: all)')
    
    parser.add_argument('--log-level', type=str,
                       default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing for chart generation')
    
    parser.add_argument('--workers', type=int,
                       default=os.cpu_count(),
                       help='Number of worker processes for parallel processing')
    
    return parser.parse_args()

def generate_charts(args):
    """
    Generate charts based on command line arguments.
    
    Args:
        args: Command line arguments
    """
    # Set up logging
    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting LLM benchmark chart generation")
    
    # Load data
    logger.info(f"Loading data from {args.data_file}")
    df = load_llm_data(args.data_file)
    
    if df.empty:
        logger.error("Failed to load LLM benchmark data")
        return 1
    
    # Load benchmarks info
    logger.info(f"Loading benchmark descriptions from {args.benchmarks_file}")
    benchmarks_info = load_benchmarks(args.benchmarks_file)
    
    # Get benchmark columns
    benchmark_columns = get_benchmark_columns(df)
    
    # Filter benchmarks if specified
    if args.benchmarks:
        benchmark_columns = [b for b in benchmark_columns if b in args.benchmarks]
        if not benchmark_columns:
            logger.error(f"None of the specified benchmarks were found in the data")
            return 1
    
    logger.info(f"Generating charts for benchmarks: {', '.join(benchmark_columns)}")
    logger.info(f"Generating charts in languages: {', '.join(args.languages)}")
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Track charts to generate
    chart_tasks = []
    
    # Add freedom vs benchmark charts
    for benchmark in benchmark_columns:
        if benchmark != 'freedom':  # Skip freedom vs freedom
            for language in args.languages:
                chart_data = prepare_freedom_vs_benchmark_data(df, benchmark)
                if chart_data:
                    chart_tasks.append({
                        'type': 'freedom_vs_benchmark',
                        'data': chart_data,
                        'output_dir': args.output_dir,
                        'language': language
                    })
    
    # Add cost vs benchmark charts
    for benchmark in benchmark_columns:
        for language in args.languages:
            chart_data = prepare_cost_vs_benchmark_data(df, benchmark)
            if chart_data:
                chart_tasks.append({
                    'type': 'cost_vs_benchmark',
                    'data': chart_data,
                    'output_dir': args.output_dir,
                    'language': language
                })
    
    # Add power vs freedom chart
    for language in args.languages:
        chart_tasks.append({
            'type': 'power_vs_freedom',
            'data': df,
            'output_dir': args.output_dir,
            'language': language
        })
    
    logger.info(f"Preparing to generate {len(chart_tasks)} charts")
    
    # Generate charts
    if args.parallel and len(chart_tasks) > 1:
        logger.info(f"Using parallel processing with {args.workers} workers")
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            
            for task in chart_tasks:
                if task['type'] == 'freedom_vs_benchmark':
                    future = executor.submit(
                        generate_freedom_vs_benchmark_chart,
                        task['data'],
                        task['output_dir'],
                        task['language']
                    )
                elif task['type'] == 'cost_vs_benchmark':
                    future = executor.submit(
                        generate_cost_vs_benchmark_chart,
                        task['data'],
                        task['output_dir'],
                        task['language']
                    )
                elif task['type'] == 'power_vs_freedom':
                    future = executor.submit(
                        generate_power_vs_freedom_chart,
                        task['data'],
                        task['output_dir'],
                        task['language']
                    )
                
                futures.append(future)
            
            # Wait for all futures to complete
            for future in futures:
                future.result()
    else:
        logger.info("Using sequential processing")
        
        for task in chart_tasks:
            try:
                if task['type'] == 'freedom_vs_benchmark':
                    generate_freedom_vs_benchmark_chart(
                        task['data'],
                        task['output_dir'],
                        task['language']
                    )
                elif task['type'] == 'cost_vs_benchmark':
                    generate_cost_vs_benchmark_chart(
                        task['data'],
                        task['output_dir'],
                        task['language']
                    )
                elif task['type'] == 'power_vs_freedom':
                    generate_power_vs_freedom_chart(
                        task['data'],
                        task['output_dir'],
                        task['language']
                    )
            except Exception as e:
                logger.error(f"Error generating chart: {str(e)}")
    
    logger.info("Chart generation complete")
    return 0

def main():
    """Main entry point for the script."""
    args = parse_args()
    return generate_charts(args)

if __name__ == "__main__":
    sys.exit(main())