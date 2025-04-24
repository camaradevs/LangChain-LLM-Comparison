#!/usr/bin/env python3
"""
LangChain LLM Comparison Chart Generator

This script runs all the image generation scripts in the project 
to create performance, cost, and freedom comparison charts for 
various LLM models in multiple languages (English, Portuguese, Spanish).
"""

import os
import sys
import argparse
import subprocess
import time
import importlib.util
from typing import List, Dict, Any

# Check and install required dependencies using a virtual environment
def check_and_install_dependencies():
    """Create a virtual environment and install required packages."""
    required_packages = ['matplotlib', 'pandas', 'numpy']
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_compare_venv")
    
    # Check if virtual environment exists
    venv_python = os.path.join(venv_dir, "bin", "python")
    if not os.path.exists(venv_python):
        print(f"Creating virtual environment at {venv_dir}...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
            print("Virtual environment created successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            print("\nPlease create a virtual environment manually and install the required packages:")
            print(f"  python -m venv {venv_dir}")
            print(f"  source {venv_dir}/bin/activate")
            for package in required_packages:
                print(f"  pip install {package}")
            sys.exit(1)
    
    # Install required packages in the virtual environment
    pip_path = os.path.join(venv_dir, "bin", "pip")
    try:
        print(f"Installing dependencies in virtual environment: {', '.join(required_packages)}")
        subprocess.check_call([pip_path, "install"] + required_packages)
        print("All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("\nPlease install the dependencies manually:")
        print(f"  source {venv_dir}/bin/activate")
        for package in required_packages:
            print(f"  pip install {package}")
        sys.exit(1)
        
    # Return the path to the virtual environment's Python
    return venv_python

# Constants
SUPPORTED_LANGUAGES = ["en", "pt", "es"]
DEFAULT_LANGUAGE = "en"
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate LLM comparison charts')
    parser.add_argument(
        '--lang', 
        choices=SUPPORTED_LANGUAGES + ['all'],
        default='all',
        help='Output language for terminal messages and charts (default: all - generates for all languages)'
    )
    return parser.parse_args()

def get_messages(lang: str) -> Dict[str, str]:
    """Get localized messages based on selected language."""
    messages = {
        "en": {
            "start": "Starting LLM Comparison Chart Generator",
            "end": "All charts generated successfully!",
            "error": "Error occurred:",
            "running": "Running",
            "completed": "Completed",
            "failed": "Failed",
            "generating_basic": "Generating basic benchmark charts...",
            "generating_freedom": "Generating freedom comparison charts...",
            "generating_freedom_vs": "Generating freedom vs benchmark charts...",
            "summary": "Generation Summary",
            "elapsed_time": "Total time elapsed",
            "seconds": "seconds",
            "charts_saved": "Charts saved to",
            "processing_languages": "Processing languages"
        },
        "pt": {
            "start": "Iniciando o Gerador de Gráficos de Comparação de LLM",
            "end": "Todos os gráficos foram gerados com sucesso!",
            "error": "Ocorreu um erro:",
            "running": "Executando",
            "completed": "Concluído",
            "failed": "Falhou",
            "generating_basic": "Gerando gráficos de benchmarks básicos...",
            "generating_freedom": "Gerando gráficos de comparação de liberdade...",
            "generating_freedom_vs": "Gerando gráficos de liberdade vs benchmark...",
            "summary": "Resumo da Geração",
            "elapsed_time": "Tempo total decorrido",
            "seconds": "segundos",
            "charts_saved": "Gráficos salvos em",
            "processing_languages": "Processando idiomas"
        },
        "es": {
            "start": "Iniciando el Generador de Gráficos de Comparación de LLM",
            "end": "¡Todos los gráficos se generaron con éxito!",
            "error": "Se produjo un error:",
            "running": "Ejecutando",
            "completed": "Completado",
            "failed": "Falló",
            "generating_basic": "Generando gráficos de referencia básicos...",
            "generating_freedom": "Generando gráficos de comparación de libertad...",
            "generating_freedom_vs": "Generando gráficos de libertad vs punto de referencia...",
            "summary": "Resumen de Generación",
            "elapsed_time": "Tiempo total transcurrido",
            "seconds": "segundos",
            "charts_saved": "Gráficos guardados en",
            "processing_languages": "Procesando idiomas"
        }
    }
    return messages.get(lang, messages["en"])

def run_script(script_path: str, lang: str, msgs: Dict[str, str], venv_python: str) -> bool:
    """Run a Python script with the given language parameter and handle errors."""
    try:
        print(f"\n{msgs['running']}: {os.path.basename(script_path)} ({lang})")
        
        # Run the script with the language parameter using the virtual environment's Python
        result = subprocess.run(
            [venv_python, script_path, lang],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print script output
        print(result.stdout)
        
        if result.stderr:
            print(f"Warning: {result.stderr}")
            
        print(f"{msgs['completed']}: {os.path.basename(script_path)}")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"{msgs['failed']}: {os.path.basename(script_path)}")
        print(f"{msgs['error']} {str(e)}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    
    except Exception as e:
        print(f"{msgs['failed']}: {os.path.basename(script_path)}")
        print(f"{msgs['error']} {str(e)}")
        return False

def main():
    """Main function to run all chart generation scripts."""
    # Check and install required dependencies
    venv_python = check_and_install_dependencies()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine which languages to process
    languages_to_process = SUPPORTED_LANGUAGES if args.lang == 'all' else [args.lang]
    
    # Use English for console messages by default
    msgs = get_messages("en")
    
    # Start timer
    start_time = time.time()
    
    # Print welcome message
    print(f"\n{'=' * 60}")
    print(f"{msgs['start']} - {msgs['processing_languages']}: {', '.join(languages_to_process)}")
    print(f"{'=' * 60}")
    
    # Define scripts to run
    scripts = [
        {"name": "generate.py", "message": msgs["generating_basic"]},
        {"name": "generate_freedom.py", "message": msgs["generating_freedom"]},
        {"name": "generate_freedom_vs.py", "message": msgs["generating_freedom_vs"]}
    ]
    
    # Track success/failure
    results = []
    
    # Process each language
    for lang in languages_to_process:
        # Get the localized language name
        language_names = {
            "en": "English", 
            "pt": "Portuguese", 
            "es": "Spanish"
        }
        lang_name = language_names.get(lang, lang.upper())
        
        print(f"\n{'=' * 60}")
        print(f"Generating charts for {lang_name} language")
        print(f"{'=' * 60}")
        
        # Run each script for the current language
        for script in scripts:
            print(f"\n{script['message']} ({lang})")
            script_path = os.path.join(SRC_DIR, script["name"])
            success = run_script(script_path, lang, msgs, venv_python)
            results.append({"name": f"{script['name']} ({lang})", "success": success})
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"{msgs['summary']}")
    print(f"{'=' * 60}")
    
    # Show results for each script
    for result in results:
        status = msgs["completed"] if result["success"] else msgs["failed"]
        print(f"{result['name']}: {status}")
    
    # Show overall statistics
    successful = sum(1 for r in results if r["success"])
    print(f"\n{successful}/{len(results)} {msgs['completed']}")
    print(f"{msgs['elapsed_time']}: {elapsed_time:.2f} {msgs['seconds']}")
    
    # Show where charts are saved
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    print(f"\n{msgs['charts_saved']}:")
    for lang in languages_to_process:
        print(f"{images_dir}/{lang}/")
    
    # Final success message if all scripts succeeded
    if all(r["success"] for r in results):
        print(f"\n{'=' * 60}")
        print(f"{msgs['end']}")
        print(f"{'=' * 60}")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())