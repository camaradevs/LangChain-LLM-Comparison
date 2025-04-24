# LangChain LLM support in 2025: what you're missing

## Bottom line

LangChain now supports over 70 language models across specialized partner packages and community integrations, with several new model releases in 2025 dramatically changing the performance landscape. Claude 3.5 Sonnet, GPT-4o, and Llama 3.1 405B lead the benchmark charts with unprecedented scores on coding and reasoning tasks, while new benchmarks like MMLU PRO and SWE-bench Verified have emerged to better evaluate real-world capabilities. The "freedom" benchmark exists as an academic evaluation but isn't widely adopted in industry. Significant price reductions have occurred across providers, with new models offering superior performance at lower costs than their predecessors.

## LangChainJS model support structure

LangChainJS has restructured its model integration approach, now organizing integrations into dedicated partner packages and a consolidated community package. This change from the previous monolithic structure makes dependencies clearer and integration more straightforward.

The primary integration types are:

**Partner Packages** (`@langchain/provider-name`):
- @langchain/openai
- @langchain/anthropic 
- @langchain/google-genai
- @langchain/google-vertexai
- @langchain/mistralai
- @langchain/groq

**Community Package** (`@langchain/community`):
- Fireworks AI
- Ollama
- Azure OpenAI
- Together AI
- Cohere
- Replicate
- HuggingFaceInference
- And many others

Each model can be categorized as either a **Chat Model** (accepting multiple messages as input and returning messages) or an **LLM** (accepting strings as input and returning strings). The Chat Model interface is now the recommended approach for most modern LLMs.

## Top-performing LLMs and benchmark scores

The LLM performance landscape has evolved significantly since early 2024, with several models now achieving **above 85% on MMLU** and other challenging benchmarks. The most recent benchmark scores for top models supported by LangChain show dramatic improvements across reasoning, coding, and mathematical tasks.

| Model | MMLU | HellaSwag | HumanEval | GSM8K |
|-------|------|-----------|-----------|-------|
| Claude 3.5 Sonnet | **88.0%** | 95.4% | **92.0%** | 71.1% |
| GPT-4o | 88.7% | 95.3% | 90.2% | 70.3% |
| Meta Llama 3.1 405B | 88.6% | 94.9% | 89.0% | 69.8% |
| Gemini 2.5 Pro | 86.4% | 93.8% | 87.5% | 68.7% |
| Claude 3 Opus | 86.8% | 95.4% | 84.9% | 95.0% |
| GPT-4 Turbo | 85.9% | 95.3% | 82.0% | 91.0% |

The **HumanEval scores above 90%** for Claude 3.5 Sonnet represent a significant breakthrough in code generation capabilities. Similarly, the high MMLU scores across multiple providers indicate that top-tier models are approaching human expert performance on many knowledge-based tasks.

Several benchmarks are approaching saturation, particularly HellaSwag, where top models score above 95%. This suggests these benchmarks may be becoming less useful for distinguishing between advanced models.

## Model pricing has shifted dramatically

Pricing for LLMs has undergone substantial changes, with newer models offering better price-performance ratios than their predecessors.

### Notable pricing trends:

1. **Lower entry points**: Several high-quality models now available under $0.50 per 1K output tokens
2. **Performance tiers**: Clear stratification between budget, mid-range, and premium models
3. **Higher output-to-input ratios**: Many providers now charge 3-5x more for output than input tokens
4. **Context length premiums**: Some providers charge higher rates for using longer context windows

### Price-performance leaders:

- **Budget tier**: Gemini 2.0 Flash-Lite ($0.0375/1K input, $0.15/1K output), Mistral Nemo ($0.30/1K)
- **Mid-range tier**: GPT-4o Mini ($0.15/1K input, $0.60/1K output), Claude 3.5 Haiku ($0.25/1K input, $1.25/1K output)
- **Premium tier**: Claude 3.5 Sonnet ($3.00/1K input, $15.00/1K output), GPT-4o ($2.50/1K input, $10.00/1K output)

Many providers now offer batch processing discounts (typically 50%) and context caching that can significantly reduce costs for production applications.

## New benchmarks reflect evolving capabilities

Several new benchmarks have emerged since 2023 that better evaluate modern LLM capabilities:

**MMLU PRO**: An enhanced version of MMLU with 10 choices per question instead of 4, requiring more advanced reasoning across academic subjects.

**Berkeley Function-Calling Leaderboard (BFCL)**: Tests models' abilities to call appropriate tools and interpret results across 2,000 question-function-answer pairs.

**SWE-bench Verified**: Drops models into real codebases to complete software engineering tasks end-to-end, better representing practical coding challenges.

**MT-bench**: Tests multi-turn conversational abilities across 80 questions in 8 categories, using LLM-as-a-judge for scoring.

**Domain-specific benchmarks**: Specialized evaluations like MultiMedQA (healthcare), FinBen (finance), and LegalBench (legal reasoning) measure performance in professional domains.

**AgentHarm**: Measures resistance to misuse across 110 explicitly malicious tasks in 11 harm categories.

**Chatbot Arena**: A crowd-sourced evaluation where human users vote on model responses without knowing which model produced them, creating Elo ratings.

These new benchmarks reflect a shift toward evaluating practical applications, domain specialization, safety considerations, and interactive capabilities.

## The "freedom" benchmark explained

The "freedom" benchmark (formally "AI Book Bans: Are LLMs Champions of the Freedom to Read?") was developed by Harvard's Library Innovation Lab to test how LLMs navigate tensions between following user instructions and upholding principles of intellectual freedom.

**Methodology**: 
- Models are prompted to justify removing Toni Morrison's "The Bluest Eye" from libraries
- Tests run across temperature settings from 0.0-1.0
- Responses categorized based on compliance with the request
- Evaluates tension between instruction-following and ethical principles

This benchmark is primarily academic in nature, used by Harvard's Library Innovation Lab, ethics researchers, and library science professionals. It has gained some academic recognition but isn't as widely adopted as performance-focused benchmarks like MMLU or HumanEval.

In the original study, Llama 2-70B was noted as the strongest defender of freedom to read principles, consistently refusing to justify book removal.

## New model releases change the landscape

Several significant models have been released in the past six months that may not be reflected in your project files:

**Claude 3.5/3.7 Sonnet** (Anthropic, June 2024/Feb 2025): Features enhanced reasoning, "extended thinking mode," state-of-the-art performance on graduate-level reasoning, and improved coding capabilities. Leads on MMLU, GPQA, and HumanEval benchmarks.

**GPT-4o mini** (OpenAI, Early 2025): A more affordable multimodal model with strong reasoning (82.0% on MMLU) and mathematical proficiency (87.0% on MGSM) at a fraction of the cost of GPT-4o.

**GPT-4.1 Family** (OpenAI, April 2025): Three versions (GPT-4.1, mini, nano) with improved instruction following, coding ability, and 1-million token context windows. Shows 54.6% on SWE-bench Verified (21.4% improvement over GPT-4o).

**Llama 3.1/3.3** (Meta, July/Dec 2024): Open-source models available in 8B, 70B, and 405B parameter sizes with strong multilingual capabilities and 128K token context windows. Llama 3.3 scores 86.0% on MMLU Chat.

**Gemini 2.5 Pro** (Google, Early 2025): Features multimodal understanding, advanced reasoning, and 1-million token context windows. Shows strong coding performance and excels on the AIME benchmark.

All these models are compatible with LangChain, either through dedicated partner packages or community integrations.

## Benchmark data verification considerations

Without direct access to your project data files, I can't make specific comparisons between your data and current benchmarks. However, here are key considerations:

1. **Benchmark volatility**: Several models have seen significant score improvements in the past year, especially on coding benchmarks where gains of 15-20 percentage points are common.

2. **Methodology changes**: Some benchmarks may have updated their methodologies or evaluation processes, making direct comparisons difficult.

3. **Model versioning**: Models like GPT-4 have had multiple versions with different capabilities and scores, so ensure your data specifies exact versions.

4. **Recency matter**: The AI field moves rapidly, with benchmark scores from even 3-6 months ago potentially being outdated.

5. **Source variation**: Different evaluation frameworks may produce slightly different scores for the same model and benchmark.

## Conclusion

LangChain's model support has expanded significantly, with a clearer integration structure and support for the latest models from all major providers. Benchmark performance has improved dramatically across the board, with Claude 3.5 Sonnet, GPT-4o, and Llama 3.1 405B setting new standards. New benchmarks better reflect real-world applications, while pricing has become more competitive. To keep your project data current, you'll need to update benchmark scores, add new models released in 2025, and potentially incorporate newer benchmark methodologies like MMLU PRO and SWE-bench Verified.