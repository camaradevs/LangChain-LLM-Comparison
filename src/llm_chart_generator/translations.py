"""
Translations for chart labels and text in multiple languages.
"""

# Translations for chart elements
translations = {
    'en': {
        # Chart titles
        'freedom_vs_benchmark_title': '{benchmark} vs Freedom Score',
        'cost_vs_benchmark_title': '{benchmark} vs Cost',
        'power_vs_freedom_title': 'Power vs Freedom Balance',
        
        # Axis labels
        'freedom_score_label': 'Freedom Score',
        'benchmark_score_label': '{benchmark} Score',
        'cost_label': 'Cost per 1k Tokens (USD)',
        'performance_label': 'Performance Score',
        
        # Annotations
        'high_freedom_label': 'High Freedom',
        'low_freedom_label': 'Low Freedom',
        'high_benchmark_label': 'High {benchmark}',
        'low_benchmark_label': 'Low {benchmark}',
        'high_value_region': 'Optimal Region',
        
        # Other labels
        'models_legend': 'LLM Models',
    },
    'es': {
        # Chart titles
        'freedom_vs_benchmark_title': '{benchmark} vs Puntuación de Libertad',
        'cost_vs_benchmark_title': '{benchmark} vs Costo',
        'power_vs_freedom_title': 'Equilibrio entre Poder y Libertad',
        
        # Axis labels
        'freedom_score_label': 'Puntuación de Libertad',
        'benchmark_score_label': 'Puntuación de {benchmark}',
        'cost_label': 'Costo por 1k Tokens (USD)',
        'performance_label': 'Puntuación de Rendimiento',
        
        # Annotations
        'high_freedom_label': 'Alta Libertad',
        'low_freedom_label': 'Baja Libertad',
        'high_benchmark_label': 'Alto {benchmark}',
        'low_benchmark_label': 'Bajo {benchmark}',
        'high_value_region': 'Región Óptima',
        
        # Other labels
        'models_legend': 'Modelos LLM',
    },
    'pt': {
        # Chart titles
        'freedom_vs_benchmark_title': '{benchmark} vs Pontuação de Liberdade',
        'cost_vs_benchmark_title': '{benchmark} vs Custo',
        'power_vs_freedom_title': 'Equilíbrio entre Poder e Liberdade',
        
        # Axis labels
        'freedom_score_label': 'Pontuação de Liberdade',
        'benchmark_score_label': 'Pontuação de {benchmark}',
        'cost_label': 'Custo por 1k Tokens (USD)',
        'performance_label': 'Pontuação de Desempenho',
        
        # Annotations
        'high_freedom_label': 'Alta Liberdade',
        'low_freedom_label': 'Baixa Liberdade',
        'high_benchmark_label': 'Alto {benchmark}',
        'low_benchmark_label': 'Baixo {benchmark}',
        'high_value_region': 'Região Ótima',
        
        # Other labels
        'models_legend': 'Modelos LLM',
    }
}

# Mapping of benchmark names to their translated display names
benchmark_display_names = {
    'en': {
        'mmlu': 'MMLU',
        'hellaswag': 'HellaSwag',
        'humaneval': 'HumanEval',
        'gsm8k': 'GSM8K',
        'truthfulqa': 'TruthfulQA',
        'bbh': 'BBH',
        'arc': 'ARC',
        'winogrande': 'WinoGrande',
        'math': 'MATH',
        'piqa': 'PIQA',
        'siqa': 'SIQA',
        'drop': 'DROP',
        'glue': 'GLUE',
        'superglue': 'SuperGLUE',
        'boolq': 'BoolQ',
        'lambada': 'LAMBADA',
        'freedom': 'Freedom'
    },
    'es': {
        'mmlu': 'MMLU',
        'hellaswag': 'HellaSwag',
        'humaneval': 'HumanEval',
        'gsm8k': 'GSM8K',
        'truthfulqa': 'TruthfulQA',
        'bbh': 'BBH',
        'arc': 'ARC',
        'winogrande': 'WinoGrande',
        'math': 'MATH',
        'piqa': 'PIQA',
        'siqa': 'SIQA',
        'drop': 'DROP',
        'glue': 'GLUE',
        'superglue': 'SuperGLUE',
        'boolq': 'BoolQ',
        'lambada': 'LAMBADA',
        'freedom': 'Libertad'
    },
    'pt': {
        'mmlu': 'MMLU',
        'hellaswag': 'HellaSwag',
        'humaneval': 'HumanEval',
        'gsm8k': 'GSM8K',
        'truthfulqa': 'TruthfulQA',
        'bbh': 'BBH',
        'arc': 'ARC',
        'winogrande': 'WinoGrande',
        'math': 'MATH',
        'piqa': 'PIQA',
        'siqa': 'SIQA',
        'drop': 'DROP',
        'glue': 'GLUE',
        'superglue': 'SuperGLUE',
        'boolq': 'BoolQ',
        'lambada': 'LAMBADA',
        'freedom': 'Liberdade'
    }
}

def get_translation(key: str, language: str = 'en', **kwargs) -> str:
    """
    Get a translation for a specific key in the specified language.
    
    Args:
        key: Translation key
        language: Target language code ('en', 'es', or 'pt')
        **kwargs: Format parameters for string formatting
        
    Returns:
        Translated string, formatted with kwargs if applicable
    """
    # Default to English if language not supported
    if language not in translations:
        language = 'en'
    
    # Get translation or fallback to English
    translation = translations.get(language, {}).get(key)
    if not translation and language != 'en':
        translation = translations.get('en', {}).get(key, key)
    
    # Format the translation if kwargs are provided
    if translation and kwargs:
        try:
            return translation.format(**kwargs)
        except KeyError:
            return translation
    
    return translation or key

def get_benchmark_name(benchmark: str, language: str = 'en') -> str:
    """
    Get the display name for a benchmark in the specified language.
    
    Args:
        benchmark: Benchmark code
        language: Target language code ('en', 'es', or 'pt')
        
    Returns:
        Localized display name for the benchmark
    """
    # Default to English if language not supported
    if language not in benchmark_display_names:
        language = 'en'
    
    # Get benchmark name or fallback to English
    name = benchmark_display_names.get(language, {}).get(benchmark)
    if not name and language != 'en':
        name = benchmark_display_names.get('en', {}).get(benchmark, benchmark)
    
    return name or benchmark