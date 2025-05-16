# 2025's AI titans: The definitive LLM benchmark showdown

In 2025's rapidly evolving AI landscape, large language models have continued their meteoric performance gains while expanding their capabilities across reasoning, multimodality, and ultra-long context processing. This comprehensive benchmark analysis captures the current state of leading LLMs as of May 2025, including pricing, capabilities, and performance across key benchmarks.

## The new performance ceiling: 90%+ on core benchmarks

The most advanced LLMs in 2025 have shattered previous performance records, with top models now exceeding 90% on critical benchmarks. DeepSeek R1 (90.8%), OpenAI o3 (92.9%), and Grok 3 (92.7%) lead on MMLU, while coding performance on HumanEval has reached unprecedented levels with Claude 3.5 Sonnet (93.7%) and OpenAI o4-mini (97.3%). Mathematical reasoning has seen particular advancement, with GSM8K scores exceeding 95% across multiple flagship models.

These improvements come alongside dramatic expansions in context windows, with Meta's Llama 4 Scout setting a new industry standard at 10 million tokens, while pricing has generally trended downward across equivalent model tiers, making advanced AI increasingly accessible.

## AWS Bedrock models ecosystem

AWS Bedrock continues to offer one of the most diverse model ecosystems, providing a single access point to both AWS-native models and third-party options with unified billing and security.

### AWS native models (Nova and Titan families)

| Model | Context Window | Input Price (per 1M tokens) | Output Price (per 1M tokens) | Key Capabilities | Notable Benchmarks |
|-------|----------------|----------------------------|------------------------------|-----------------|-------------------|
| Nova Pro | 300K | $0.50 | $2.40 | Advanced multimodal, Function calling | Strong CRAG, BFCL scores |
| Nova Lite | 300K | $0.15 | $0.60 | Multimodal (text, image, video) | Strong RAG performance |
| Nova Micro | 128K | $0.04 | $0.14 | Text-only, Fast inference | Good performance/cost ratio |
| Nova Premier | 1M | $2.50 | $12.50 | Complex reasoning, Model distillation | Not yet benchmarked |
| Titan Text Premier | 32K | $0.0008 | $0.0016 | High-quality text generation | Moderate MMLU, GSM8K |

### Third-party models on AWS Bedrock

| Provider | Top Models | Context Window | Input Price (per 1M tokens) | Output Price (per 1M tokens) | Notable Benchmarks |
|----------|-----------|----------------|----------------------------|------------------------------|-------------------|
| Anthropic | Claude 3.7 Sonnet | 200K | $0.003 | $0.015 | State-of-the-art coding |
| | Claude 3.5 Sonnet | 200K | $0.003 | $0.015 | MMLU: 88.7%, HumanEval: 92.0% |
| Meta | Llama 4 Scout | 10M | Varies by region | Varies by region | Advanced context processing |
| | Llama 3.1 405B | 128K | Varies by region | Varies by region | MMLU: 88.6%, Strong on HumanEval |
| Mistral AI | Mistral Large 2 | 128K | $0.008 | $0.024 | Strong MMLU performance |
| DeepSeek | DeepSeek-R1 | N/A | $0.00135 | $0.0054 | Top intelligence scores |
| Cohere | Command R+ | 128K | $0.0030 | $0.0150 | Advanced reasoning |

## OpenAI models: Benchmark performance

OpenAI's 2025 lineup features a two-track strategy: the GPT-4.1 series for mainstream applications and the "o" series for advanced reasoning tasks.

| Model | Release | Context Window | Input Price (1M tokens) | Output Price (1M tokens) | MMLU | GSM8K | HumanEval | TruthfulQA | ARC | WinoGrande |
|-------|---------|----------------|------------------------|--------------------------|------|-------|-----------|------------|-----|------------|
| o3 | Apr 2025 | 200K | $10.00 | $40.00 | 92.9% | N/A | 87.4% | N/A | N/A | N/A |
| o4-mini | Apr 2025 | 200K | $1.10 | $4.40 | 90.0% | N/A | 97.3% | N/A | N/A | N/A |
| GPT-4.1 | Apr 2025 | 1M | $2.00 | $8.00 | 90.2% | N/A | N/A | N/A | N/A | N/A |
| GPT-4.1 Mini | Apr 2025 | 1M | $0.40 | $1.60 | ~88% | N/A | N/A | N/A | N/A | N/A |
| GPT-4.1 Nano | Apr 2025 | 1M | $0.10 | $0.40 | 80.1% | N/A | N/A | N/A | N/A | N/A |
| GPT-4o | May 2024 | 128K | $5.00 | $15.00 | 88.7% | 97% | 90.2% | 59% | 96.3% | 87.5% |
| GPT-4o Mini | May 2024 | 128K | $0.50 | $1.50 | Mid-80s% | N/A | N/A | N/A | N/A | N/A |
| GPT-3.5 Turbo | Jan 2025 | 16K | $0.50 | $1.50 | 70% | 57.1% | N/A | 47% | 85.2% | 81.6% |
| O1 Reasoning | Sep 2024 | 200K | $15.00 | $60.00 | 84.1% | N/A | N/A | N/A | N/A | N/A |

**Key innovations**: GPT-4.1 family features million-token context windows, while the new o3 and o4-mini models establish new records for mathematical reasoning and code generation.

## Anthropic Claude models: Benchmark performance

Anthropic's Claude models have maintained their reputation for impressive performance and safe, helpful behavior with their 2025 lineup.

| Model | Release | Context Window | Input Price (1M tokens) | Output Price (1M tokens) | MMLU | GSM8K | HumanEval | TruthfulQA | BBH | MATH |
|-------|---------|----------------|------------------------|--------------------------|------|-------|-----------|------------|-----|------|
| Claude 3.7 Sonnet | Feb 2025 | 200K | $3.00 | $15.00 | N/A | N/A | N/A | N/A | N/A | N/A |
| Claude 3.5 Sonnet | Jun 2024 | 200K | $3.00 | $15.00 | 88.7% | 95.0% | 93.7% | N/A | N/A | N/A |
| Claude 3.5 Haiku | Nov 2024 | 200K | $0.80 | $4.00 | N/A | N/A | N/A | N/A | N/A | N/A |
| Claude 3 Opus | Mar 2024 | 200K | $15.00 | $75.00 | High 80s% | 95.0% | 84.9% | N/A | 86.8% | 60.1% |
| Claude 3 Sonnet | Mar 2024 | 200K | $3.00 | $15.00 | 81.5% | N/A | N/A | N/A | N/A | N/A |
| Claude 3 Haiku | Mar 2024 | 200K | $0.25 | $1.25 | N/A | N/A | N/A | N/A | N/A | N/A |

**Key innovations**: Claude 3.7 Sonnet introduces hybrid reasoning with standard and extended thinking modes, improving performance on complex tasks. Claude 3.5 Sonnet set new standards for coding tasks with 93.7% on HumanEval.

## Meta Llama models: Benchmark performance

Meta's Llama models have dramatically expanded their capabilities in 2025, culminating in the Llama 4 series with unprecedented context lengths.

| Model | Release | Context Window | Parameters | MMLU | GSM8K | HumanEval | MATH | MMLU-Pro | IFEval |
|-------|---------|----------------|------------|------|-------|-----------|------|----------|--------|
| Llama 4 Scout | Apr 2025 | 10M | 17B active/109B total | 79.6% | N/A | N/A | 50.3% | 58.2% | N/A |
| Llama 4 Maverick | Apr 2025 | 1M | 17B active/400B total | 85.5% | N/A | 77.6% | 61.2% | 62.9% | N/A |
| Llama 3.1 405B | Jul 2024 | 128K | 405B | 88.6% | 96.8% | 89.0% | 73.8% | 73.4% | 88.6% |
| Llama 3.3 70B | Dec 2024 | 128K | 70B | 86.0% | N/A | 88.4% | 77.0% | 68.9% | 92.1% |

**Key innovations**: Llama 4 Scout's 10 million token context window represents a massive leap forward in long-context processing, while the MoE architecture in both Llama 4 models allows for more efficient processing.

## DeepSeek models: Benchmark performance

DeepSeek has emerged as a formidable competitor in 2025 with their V3 and Reasoner models showing exceptional performance.

| Model | Release | Context Window | Architecture | MMLU | GSM8K | HumanEval | MATH | BBH | MMLU-Pro |
|-------|---------|----------------|--------------|------|-------|-----------|------|-----|----------|
| DeepSeek Reasoner (R1) | Jan 2025 | 64K | 671B total/37B active | 90.8% | N/A | 82.6% | N/A | N/A | N/A |
| DeepSeek V3 | Mar 2025 | 128K | 671B total/37B active | 88.5% | N/A | 84.76% | 71.02% | 83.40% | 75.9% |
| DeepSeek V2 | May 2024 | 128K | 236B total/21B active | 78.9% | 83.0% | 48.2% | 42.2% | 81.0% | N/A |
| DeepSeek Coder V2 | Jun 2024 | 128K | 236B total/21B active | 79.2% | 94.9% | 90.2% | 75.7% | 83.9% | N/A |

**Key innovations**: DeepSeek R1 exceeds 90% on MMLU, while DeepSeek Coder V2 shows exceptional performance on coding and math tasks despite much lower pricing than comparable models.

## Google Gemini models: Benchmark performance

Google's Gemini models have made significant strides in 2025, particularly with the release of Gemini 2.5 Pro.

| Model | Release | Context Window | Input Price (1M tokens) | Output Price (1M tokens) | MMLU | GPQA | LiveCodeBench | Other notable |
|-------|---------|----------------|------------------------|--------------------------|------|------|---------------|---------------|
| Gemini 2.5 Pro | May 2025 | 1M (2M soon) | $1.25-$2.50 | $10.00-$15.00 | 85.8% | 84.0% | 70.4% | AIME 2025: 86.7% |
| Gemini 1.5 Flash | May 2024 | 1M | $0.10 | $0.40 | N/A | N/A | N/A | Cost-effective |
| Gemini 1.5 Pro | Sep 2024 | 2M | $1.25+ | $5.00+ | 75.0% | N/A | N/A | Long context specialist |
| Gemini 1.0 Pro | Dec 2023 | 32K | $0.0125 | $0.0375 | 71.8% | N/A | N/A | Older model |

**Key innovations**: Gemini 2.5 Pro introduces dramatically improved performance while maintaining large context windows, and shows particularly strong results on specialized mathematics benchmarks.

## Notable new models (Late 2024 - May 2025)

Beyond the models requested in the original dataset, several significant new models have emerged:

| Model | Provider | Release | Key Features | Top Benchmarks |
|-------|----------|---------|--------------|----------------|
| Grok 3 | xAI | Feb 2025 | 1M context, "Think" mode | MMLU: 92.7%, AIME: 88.9% |
| Grok 3 Mini | xAI | Feb 2025 | Cost-effective version | Similar capabilities to Grok 3 |
| OpenAI o3 | OpenAI | Apr 2025 | Advanced reasoning | MMLU: 92.9%, MATH: 97.8% |
| Qwen2.5-Max | Alibaba | Jan 2025 | MoE architecture | MMLU: 87.9%, GSM8K: 94.5% |
| Qwen2.5-Math-72B | Alibaba | Early 2025 | Math specialist | MATH: 92.9% with tools |

These new models demonstrate the continued innovation in the field, with specialized reasoning capabilities, massive context windows, and increasingly competitive performance from multiple providers.

## The Freedom benchmark: Measuring intellectual censorship resistance

The "Freedom benchmark" (also known as "AI Book Bans: Freedom to Read") emerged in 2023 from the Harvard Library Innovation Lab as a specialized evaluation to assess how LLMs respond to censorship requests.

### Methodology

The benchmark uses a specific prompt asking models to justify removing Toni Morrison's "The Bluest Eye" from a library. Responses are categorized as:
- **Unequivocal Yes** (fully complying with ban request)
- **Yes with Nuance** (complying but with qualifications)
- **Unequivocally No** (refusing and defending freedom to read)

### 2023 original findings

- Approximately 75% of model responses complied with book ban requests
- Llama 2 70B stood out as the strongest defender of freedom to read
- No clear pattern regarding how temperature settings affected outcomes

### 2025 model trends

While comprehensive 2025 data specifically for this benchmark isn't publicly available, several trends can be inferred:

1. **Claude models** (Anthropic) likely show strong defense of intellectual freedom, given their Constitutional AI principles
2. **Llama models** (Meta) likely continue their strong stance against censorship
3. **Latest GPT and Gemini models** might show more nuanced responses balancing multiple considerations

Based on these providers' stated values and safety implementations, Claude 3.5/3.7 and Llama 3.1/4 models would likely perform best on freedom metrics, though formal evaluation is needed to confirm this.

## Key trends in 2025 LLM landscape

1. **Performance convergence at the top tier**: Leading models from different providers now score within a few percentage points of each other on major benchmarks, with top performers exceeding 90% on MMLU.

2. **Mixture-of-Experts dominance**: MoE architecture has become standard for high-performance models, allowing for larger effective parameter counts without proportional inference costs.

3. **Context window expansion**: 128K context is now standard, with several models reaching 1M tokens and Meta's Llama 4 Scout setting a new standard at 10M tokens.

4. **Cost reduction**: Newer models generally offer better performance at lower prices compared to their predecessors, with multiple price tiers available from most providers.

5. **Specialized reasoning modes**: Models increasingly feature distinct reasoning capabilities, with Claude 3.7's hybrid reasoning, OpenAI's "o" series, and Grok 3's "Think" mode all offering enhanced problem-solving capabilities.

This comprehensive benchmark analysis reveals a rapidly maturing LLM ecosystem with multiple viable options across different price points and specialized capabilities, allowing users to select models optimized for their specific use cases and budget constraints.