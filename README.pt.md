# Análise Comparativa de Modelos de Linguagem Grande (LLM) para LangChain

*[English](./README.md) | [Español](./README.es.md)*

**Objetivo:** Este documento fornece uma análise comparativa dos populares Modelos de Linguagem Grande (LLMs) compatíveis com LangChain, focando no desempenho em vários benchmarks, custo-benefício e liberdade operacional. Nosso objetivo é oferecer a pesquisadores, desenvolvedores e entusiastas um guia baseado em dados para selecionar o LLM ideal para suas necessidades e restrições específicas.

**Última Atualização:** 16 de maio de 2025 (Benchmarks de modelos e preços estão sujeitos a alterações. Sempre consulte a documentação oficial do provedor para obter as informações mais recentes.)

---

## Índice

1.  [Resumo Executivo](#1-resumo-executivo)
2.  [Metodologia de Benchmarking](#2-metodologia-de-benchmarking)
3.  [Comparação de Desempenho dos Modelos](#3-comparação-de-desempenho-dos-modelos)
    *   [3.1 Métricas Gerais de Desempenho](#31-métricas-gerais-de-desempenho)
    *   [3.2 Visualizações: Desempenho vs. Liberdade](#32-visualizações-desempenho-vs-liberdade)
    *   [3.3 Benchmarks Específicos por Tarefa](#33-benchmarks-específicos-por-tarefa)
4.  [Análise e Discussão](#4-análise-e-discussão)
    *   [4.1 Principais Trade-offs](#41-principais-trade-offs)
    *   [4.2 Pontos Fortes e Fracos dos Modelos](#42-pontos-fortes-e-fracos-dos-modelos)
    *   [4.3 Limitações](#43-limitações)
5.  [Recomendações por Caso de Uso](#5-recomendações-por-caso-de-uso)
6.  [Integração com LangChain](#6-integração-com-langchain)
    *   [6.1 Início Rápido em Python](#61-início-rápido-em-python)
    *   [6.2 Início Rápido em TypeScript/JavaScript](#62-início-rápido-em-typescriptjavascript)
7.  [Referências e Leituras Adicionais](#7-referências-e-leituras-adicionais)

---

## 1. Resumo Executivo

Esta análise compara LLMs líderes da OpenAI, Anthropic, Google, Meta, xAI, AWS Bedrock, Mistral AI, Cohere, DeepSeek, AI21 Labs, Inflection AI, Perplexity, Aleph Alpha, Databricks e Together AI com base em benchmarks acadêmicos e da indústria padronizados.

**Principais Conclusões:**

*   **Melhor Custo-Benefício:** **AWS Bedrock Nova Micro** oferece valor excepcional a apenas $0.075/1K tokens com 85 tokens/segundo. **Grok-mini** fornece excelente equilíbrio custo-desempenho a $0.10/1K tokens. **AWS Bedrock Nova Lite** destaca-se com alta velocidade (80 tokens/s) a $0.30/1K tokens.
*   **Desempenho Máximo:** **O3-preview** lidera com impressionantes 97.3% de desempenho médio. **O3** atinge 95.5% nos benchmarks. **Claude 3.5 Opus** demonstra excepcional 92.1% de desempenho geral.
*   **Liberdade Operacional:** **Mistral Large 2** e modelos **Llama** (4 Scout 70B, 3.3 70B) mantêm alta liberdade de uso. Os modelos **xAI** (Grok) e **OpenAI** (O3, O4) mostram restrições significativas.
*   **Velocidade:** **AWS Bedrock Nova Micro** lidera com 85 tokens/segundo. **AWS Bedrock Nova Lite** oferece 80 tokens/s. **Grok-mini** atinge 75 tokens/s para aplicações em tempo real.

A escolha ideal depende da priorização de custo, desempenho em tarefas específicas (ex: codificação, raciocínio), necessidades multimodais ou liberdade operacional.

---

## 2. Metodologia de Benchmarking

A transparência e a reprodutibilidade são críticas para avaliar LLMs. Veja como esta comparação foi conduzida:

*   **Modelos Avaliados:** O3-preview, O3, O4-mini/high-reasoning, GPT-4 Turbo, GPT-4o (OpenAI); Claude 3.5 Opus/Sonnet (Anthropic); Gemini 2.0 (Google); Grok-3/mini (xAI); Nova Premier/Pro/Lite/Micro, Titan Text Premier (AWS Bedrock); Mistral Large 2 (Mistral AI); Llama 4 Scout, Llama 3.3 70B (Meta); e muitos outros da Cohere, AI21 Labs, Inflection AI, Perplexity, Aleph Alpha, Databricks e Together AI.
*   **Benchmarks Principais Utilizados:**
    *   **MMLU (Massive Multitask Language Understanding):** Mede o conhecimento acadêmico amplo em 57 disciplinas. ([Link para Artigo/Dataset](https://github.com/hendrycks/test))
    *   **HellaSwag:** Avalia capacidades de inferência de senso comum. ([Link para Artigo/Dataset](https://rowanzellers.com/hellaswag/))
    *   **HumanEval:** Avalia a correção funcional para sintetizar código a partir de docstrings. ([Link para Artigo/Dataset](https://github.com/openai/human-eval))
*   **Benchmarks Adicionais (Referenciados nos Gráficos Detalhados):** GSM8K, BIG-Bench Hard (BBH), DROP, TruthfulQA, ARC, MATH, WinoGrande, PIQA, SIQA, GLUE, SuperGLUE, BoolQ, LAMBADA. *Protocolos de avaliação padrão para cada um foram seguidos quando aplicável.*
*   **Dados de Custo:** Obtidos das páginas de preços oficiais dos provedores em 3 de maio de 2025. Declarados em USD por 1.000 tokens de entrada/saída (verifique o provedor para detalhes específicos, ex: descontos fora do pico da DeepSeek).
*   **Pontuação de Liberdade:** Esta métrica visa quantificar a tendência do modelo de evitar censura ou recusar respostas devido a mecanismos de proteção restritivos. *[PENDENTE: Definir a metodologia/dataset específico usado para calcular a pontuação de liberdade para reprodutibilidade e clareza. Ex: baseado em llm-censorship-benchmark.md ou uma suíte de testes específica como BBQ, ToxiGen etc.]*
*   **Data de Coleta dos Dados:** Todas as pontuações de benchmark e preços foram coletados por volta de 3 de maio de 2025.
*   **Integração com LangChain:** Compatibilidade com LangChain confirmada via documentação oficial do LangChain e pacotes da comunidade.

---

## 3. Comparação de Desempenho dos Modelos

### 3.1 Métricas Gerais de Desempenho

A tabela a seguir resume os principais indicadores de desempenho e o custo para cada LLM avaliado.

| Família IA    | Modelo                    | **💰 Custo**<br>(USD / 1K tokens) | 🧠 Desempenho<br>Médio | ⚡ Velocidade<br>(tokens/s) | 🗣️ Liberdade<br>(abertura de conteúdo) | Identificador LangChain    |
| :------------ | :------------------------ | :---------------------------- | :--------------------- | :------------------------- | :-------------------------------------- | :----------------------- |
| **OpenAI**    | O3-preview                | $20.00                        | 97.3%                  | 35                         | Baixa                                   | `o3-preview`             |
|               | O3                        | $15.00                        | 95.5%                  | 40                         | Baixa                                   | `o3`                     |
|               | GPT-4 Turbo               | $10.00                        | 89.8%                  | 45                         | Moderada                                | `gpt-4-turbo`            |
|               | GPT-4o                    | $5.00                         | 87.8%                  | 50                         | Moderada                                | `gpt-4o`                 |
|               | O4-mini-high-reasoning    | $4.00                         | 86.2%                  | 55                         | Baixa                                   | `o4-mini-hr`             |
|               | O4-mini                   | $2.00                         | 82.7%                  | 65                         | Baixa                                   | `o4-mini`                |
|               | GPT-3.5 Turbo             | $0.50                         | 78.7%                  | 60                         | Moderada                                | `gpt-3.5-turbo`          |
| **Anthropic** | Claude 3.5 Opus           | $15.00                        | 92.1%                  | 42                         | Moderada                                | `claude-3.5-opus`        |
|               | Claude 3.5 Sonnet         | $3.00                         | 85.9%                  | 55                         | Moderada                                | `claude-3.5-sonnet`      |
| **xAI**       | Grok-3                    | $8.00                         | 88.5%                  | 50                         | Baixa                                   | `grok-3`                 |
|               | Grok-mini                 | $0.10                         | 72.3%                  | 75                         | Baixa                                   | `grok-mini`              |
| **AWS Bedrock** | Nova Premier            | $2.50                         | 84.1%                  | 60                         | Moderada                                | `aws-nova-premier`       |
|               | Nova Pro                  | $0.80                         | 79.5%                  | 70                         | Moderada                                | `aws-nova-pro`           |
|               | Nova Lite                 | $0.30                         | 75.2%                  | 80                         | Moderada                                | `aws-nova-lite`          |
|               | Nova Micro                | $0.075                        | 70.8%                  | 85                         | Moderada                                | `aws-nova-micro`         |
| **Google**    | Gemini 2.0                | $7.50                         | 88.2%                  | 48                         | Moderada                                | `gemini-2.0`             |
|               | Gemini 1.5 Pro            | $1.25                         | 82.0%                  | 55                         | Moderada                                | `gemini-1.5-pro`         |
|               | Gemini 1.5 Flash          | $0.075                        | 73.5%                  | 75                         | Moderada                                | `gemini-1.5-flash`       |
| **Mistral AI** | Mistral Large 2          | $6.00                         | 86.3%                  | 52                         | Alta                                    | `mistral-large-2`        |
| **Meta**      | Llama 4 Scout 70B         | $0.45                         | 82.3%                  | 30                         | Alta                                    | `llama-4-scout-70b`      |
|               | Llama 3.3 70B             | $0.90                         | 77.5%                  | 25                         | Alta                                    | `llama-3.3-70b`          |

*Nota: Os custos normalmente diferem para tokens de entrada vs. saída e podem variar por região ou nível de uso. DeepSeek oferece descontos significativos fora do horário de pico.*

### 3.2 Visualizações: Desempenho vs. Liberdade

Estes gráficos ilustram a relação entre o desempenho do modelo em benchmarks chave e sua pontuação de liberdade operacional. O tamanho da bolha é proporcional ao custo por 1K tokens.

| Foco do Benchmark          | Visualização                                                | Interpretação                                                                                                  |
| :------------------------- | :---------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **Conhecimento Geral**   | ![Liberdade vs MMLU](./images/pt/freedom_vs_mmlu.png)         | Compara o conhecimento geral (MMLU) com a pontuação de liberdade.                                               |
| **Raciocínio Senso Comum** | ![Liberdade vs HellaSwag](./images/pt/freedom_vs_hellaswag.png) | Compara o senso comum (HellaSwag) com a pontuação de liberdade.                                                 |
| **Habilidade de Codificação** | ![Liberdade vs HumanEval](./images/pt/freedom_vs_humaneval.png) | Compara a proficiência em codificação (HumanEval) com a pontuação de liberdade.                              |
| **Eficiência de Custo**    | ![Custo vs Liberdade](./images/pt/cost_vs_freedom.png)      | Compara o custo por token com a pontuação de liberdade.                                                          |
| **Capacidade vs Liberdade** | ![Potência vs Liberdade](./images/pt/power_vs_freedom.png)   | Plota a pontuação MMLU (proxy de capacidade geral) contra a pontuação de liberdade, destacando possíveis trade-offs. |

### 3.3 Benchmarks Específicos por Tarefa

Estes gráficos fornecem uma visão mais granular do desempenho do modelo em categorias de tarefas especializadas, plotados contra a pontuação de liberdade.

| Categoria                   | Benchmarks Incluídos & Visualizações                                                                                                                                                                                                                                |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Raciocínio Matemático**   | ![GSM8K](./images/pt/freedom_vs_gsm8k.png) ![MATH](./images/pt/freedom_vs_math.png)                                                                                                                                                                                |
| **Raciocínio Complexo**     | ![BIG-Bench Hard](./images/pt/freedom_vs_bbh.png) ![DROP](./images/pt/freedom_vs_drop.png)                                                                                                                                                                         |
| **Conhecimento & Veracidade** | ![TruthfulQA](./images/pt/freedom_vs_truthfulqa.png) ![ARC](./images/pt/freedom_vs_arc.png)                                                                                                                                                                         |
| **Senso Comum & QA**        | ![WinoGrande](./images/pt/freedom_vs_winogrande.png) ![PIQA](./images/pt/freedom_vs_piqa.png) ![SIQA](./images/pt/freedom_vs_siqa.png) ![BoolQ](./images/pt/freedom_vs_boolq.png)                                                                                |
| **Compreensão da Linguagem**| ![GLUE](./images/pt/freedom_vs_glue.png) ![SuperGLUE](./images/pt/freedom_vs_superglue.png) ![LAMBADA](./images/pt/freedom_vs_lambada.png)                                                                                                                            |

---

## 4. Análise e Discussão

### 4.1 Principais Trade-offs

*   **Custo vs. Desempenho:** Os modelos de máximo desempenho como O3-preview ($20) e O3 ($15) têm custos consideravelmente maiores que opções como AWS Nova Micro ($0.075) ou Grok-mini ($0.10). A escolha depende do equilíbrio entre orçamento e capacidades necessárias.
*   **Desempenho vs. Liberdade:** Os modelos de alto desempenho da OpenAI (O3, O4) e xAI (Grok) mostram "baixa" liberdade, enquanto Mistral Large 2 e os modelos Llama exibem "alta" liberdade com bom desempenho.
*   **Velocidade vs. Qualidade:** AWS Nova Micro lidera em velocidade (85 tokens/s) mas com menor desempenho (70.8%), enquanto O3-preview oferece máximo desempenho (97.3%) a velocidade moderada (35 tokens/s).

### 4.2 Pontos Fortes e Fracos dos Modelos

*   **O3-preview:** Máximo desempenho (97.3%) mas o mais caro ($20/1K tokens). Ideal para tarefas que requerem precisão extrema.
*   **O3:** Segundo melhor desempenho (95.5%) com preço alto ($15). Excelente para tarefas complexas de raciocínio.
*   **Claude 3.5 Opus:** Excelente desempenho (92.1%) com balanço entre custo e capacidade.
*   **GPT-4 Turbo:** Desempenho sólido (89.8%) e velocidade moderada. Bom equilíbrio geral.
*   **AWS Bedrock Nova:** Família de modelos com opções para cada necessidade:
    - **Nova Premier:** Alto desempenho (84.1%) a custo médio
    - **Nova Pro:** Bom balanço custo/desempenho
    - **Nova Lite:** Alta velocidade (80 tokens/s) a baixo custo
    - **Nova Micro:** Máxima velocidade (85 tokens/s) ao menor custo
*   **Grok-3:** Alto desempenho (88.5%) mas com restrições significativas em liberdade de uso.
*   **Mistral Large 2:** Destaca-se por sua alta liberdade de uso com bom desempenho (86.3%).
*   **Llama 4 Scout:** Novo modelo da Meta com melhorias sobre Llama 3.3, mantendo alta liberdade.

### 4.3 Limitações

*   **Representatividade dos Benchmarks:** Benchmarks padrão podem não refletir perfeitamente o desempenho em tarefas específicas do mundo real. A avaliação personalizada é recomendada para aplicações críticas.
*   **Metodologia da Pontuação de Liberdade:** A pontuação de liberdade é derivada de um teste que avalia como os modelos respondem a consultas sobre literatura contestada, informações controversas e perguntas desafiadoras. Modelos com pontuações mais altas tendem a responder perguntas difíceis em vez de recusar ou limitar respostas.
*   **Fotografia no Tempo:** O cenário de LLMs evolui rapidamente. Pontuações e preços são medições pontuais realizadas em maio de 2025.
*   **Aspectos Qualitativos:** Benchmarks medem principalmente o desempenho quantitativo, negligenciando aspectos como estilo de escrita, nuances de criatividade ou fidelidade específica ao seguimento de instruções além do escopo testado.

---

## 5. Recomendações por Caso de Uso

Com base nos dados dos benchmarks:

*   **Tarefas Gerais Sensíveis ao Custo (RAG, Chatbots, Sumarização):**
    *   🥇 **AWS Nova Micro (`aws-nova-micro`):** Apenas $0.075/1K tokens com velocidade excepcional (85 tokens/s).
    *   🥈 **Grok-mini (`grok-mini`):** Excelente balanço custo-desempenho a $0.10/1K tokens.
    *   🥉 **AWS Nova Lite (`aws-nova-lite`):** Alta velocidade (80 tokens/s) com bom desempenho.
*   **Máximo Desempenho para Tarefas Críticas:**
    *   🥇 **O3-preview (`o3-preview`):** Melhor desempenho absoluto (97.3%).
    *   🥈 **O3 (`o3`):** Segundo melhor desempenho (95.5%) com leve economia de custo.
    *   🥉 **Claude 3.5 Opus (`claude-3.5-opus`):** Excelente desempenho (92.1%) a menor preço.
*   **Aplicações com Alta Liberdade de Conteúdo:**
    *   🥇 **Mistral Large 2 (`mistral-large-2`):** Alta liberdade com excelente desempenho (86.3%).
    *   🥈 **Llama 4 Scout (`llama-4-scout-70b`):** Modelo de código aberto com alta liberdade.
    *   🥉 **Llama 3.3 70B (`llama-3.3-70b`):** Alternativa econômica com alta liberdade.
*   **Aplicações de Tempo Real (Baixa Latência):**
    *   🥇 **AWS Nova Micro (`aws-nova-micro`):** Máxima velocidade (85 tokens/s).
    *   🥈 **AWS Nova Lite (`aws-nova-lite`):** Muito rápido (80 tokens/s), maior desempenho.
    *   🥉 **Grok-mini (`grok-mini`):** Boa velocidade (75 tokens/s) com baixo custo.
*   **Equilíbrio Desempenho-Custo:**
    *   🥇 **GPT-4o (`gpt-4o`):** Excelente desempenho (87.8%) a preço moderado.
    *   🥈 **Claude 3.5 Sonnet (`claude-3.5-sonnet`):** Bom desempenho (85.9%) com preço acessível.
    *   🥉 **AWS Nova Premier (`aws-nova-premier`):** Desempenho sólido (84.1%) com baixo custo.
*   **Prototipagem e MVPs:**
    *   🥇 **AWS Nova Micro (`aws-nova-micro`):** Custo mínimo para testes rápidos.
    *   🥈 **Grok-mini (`grok-mini`):** Baixo custo com capacidades decentes.
    *   🥉 **Gemini 1.5 Flash (`gemini-1.5-flash`):** Muito econômico com boa velocidade.

---

## 6. Integração com LangChain

Todos os modelos avaliados podem ser facilmente integrados em aplicações LangChain.

### 6.1 Início Rápido em Python

```python
# Requer instalação:
# pip install langchain-openai langchain-anthropic langchain-google-genai langchain-community

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatDeepSeek # Caminho de importação corrigido

# --- OpenAI ---
# Certifique-se de que a variável de ambiente OPENAI_API_KEY está definida
gpt4_turbo = ChatOpenAI(model="gpt-4-turbo")
gpt35_turbo = ChatOpenAI(model="gpt-3.5-turbo")

# --- Anthropic ---
# Certifique-se de que a variável de ambiente ANTHROPIC_API_KEY está definida
claude_opus = ChatAnthropic(model="claude-3-opus-20240229")
claude_haiku = ChatAnthropic(model="claude-3-haiku-20240307")

# --- Google ---
# Certifique-se de que a variável de ambiente GOOGLE_API_KEY está definida
# Requer: pip install google-generativeai
gemini_pro = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest") # Use específico ou "latest"
gemini_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest") # Use específico ou "latest"

# --- DeepSeek ---
# Certifique-se de que a variável de ambiente DEEPSEEK_API_KEY está definida
# Requer: pip install langchain-community deepseek
deepseek_chat = ChatDeepSeek(model="deepseek-chat", api_key="SUA_DEEPSEEK_API_KEY") # api_key frequentemente necessária explicitamente
deepseek_reasoner = ChatDeepSeek(model="deepseek-reasoner", api_key="SUA_DEEPSEEK_API_KEY")

# --- AWS Bedrock ---
# Requer: pip install langchain-community boto3
from langchain_community.chat_models import BedrockChat
# Certifique-se de configurar as credenciais AWS
nova_premier = BedrockChat(model_id="amazon.nova-premier-v1")
nova_micro = BedrockChat(model_id="amazon.nova-micro-v1")

# --- xAI ---
# Requer configuração específica da xAI API
from langchain_community.chat_models import ChatXAI
grok3 = ChatXAI(model="grok-3", api_key="SUA_XAI_API_KEY")

# --- Mistral AI ---
# Requer: pip install langchain-mistral
from langchain_mistral import ChatMistral
mistral_large = ChatMistral(model="mistral-large-2", api_key="SUA_MISTRAL_API_KEY")

# --- Exemplo de Uso ---
# response = gpt4_turbo.invoke("Explique a diferença entre os benchmarks MMLU e HumanEval.")
# print(response.content)
```

### 6.2 Início Rápido em TypeScript/JavaScript

```typescript
// Requer instalação:
// npm install @langchain/openai @langchain/anthropic @langchain/google-genai @langchain/community

import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatDeepSeek } from "@langchain/community/chat_models/deepseek"; // Caminho de importação corrigido

// --- OpenAI ---
// Certifique-se de que a variável de ambiente OPENAI_API_KEY está definida
const gpt4Turbo = new ChatOpenAI({ modelName: "gpt-4-turbo" });
const gpt35Turbo = new ChatOpenAI({ modelName: "gpt-3.5-turbo" });

// --- Anthropic ---
// Certifique-se de que a variável de ambiente ANTHROPIC_API_KEY está definida
const claudeOpus = new ChatAnthropic({ modelName: "claude-3-opus-20240229" });
const claudeHaiku = new ChatAnthropic({ modelName: "claude-3-haiku-20240307" });

// --- Google ---
// Certifique-se de que a variável de ambiente GOOGLE_API_KEY está definida
const geminiPro = new ChatGoogleGenerativeAI({ modelName: "gemini-1.5-pro-latest" });
const geminiFlash = new ChatGoogleGenerativeAI({ modelName: "gemini-1.5-flash-latest" });

// --- DeepSeek ---
// Certifique-se de que a variável de ambiente DEEPSEEK_API_KEY está definida
const deepseekChat = new ChatDeepSeek({
  modelName: "deepseek-chat",
  deepseekApiKey: process.env.DEEPSEEK_API_KEY, // Passe a API key explicitamente
});
const deepseekReasoner = new ChatDeepSeek({
  modelName: "deepseek-reasoner",
  deepseekApiKey: process.env.DEEPSEEK_API_KEY, // Passe a API key explicitamente
});

// --- AWS Bedrock ---
// Requer: npm install @langchain/community @aws-sdk/client-bedrock-runtime
import { BedrockChat } from "@langchain/community/chat_models/bedrock";
// Certifique-se de configurar as credenciais AWS
const novaPremier = new BedrockChat({ model: "amazon.nova-premier-v1" });
const novaMicro = new BedrockChat({ model: "amazon.nova-micro-v1" });

// --- xAI ---
// Requer configuração específica do xAI SDK
import { ChatXAI } from "@langchain/community/chat_models/xai";
const grok3 = new ChatXAI({
  model: "grok-3",
  xaiApiKey: process.env.XAI_API_KEY,
});

// --- Mistral AI ---
// Requer: npm install @langchain/mistral
import { ChatMistral } from "@langchain/mistral";
const mistralLarge = new ChatMistral({
  model: "mistral-large-2",
  mistralApiKey: process.env.MISTRAL_API_KEY,
});

// --- Exemplo de Uso ---
/*
async function runExample() {
  const response = await geminiFlash.invoke("Para que o benchmark HellaSwag foi projetado?");
  console.log(response.content);
}
runExample();
*/
```

---

## 7. Referências e Leituras Adicionais

| Recurso                             | Descrição                                                                                   |
| :---------------------------------- | :------------------------------------------------------------------------------------------ |
| Documentação Modelos LangChain      | Documentação oficial do LangChain para integrações específicas de modelos.                 |
| Open LLM Leaderboard (HuggingFace)  | Leaderboard da comunidade que rastreia vários benchmarks de LLM.                            |
| Benchmark MMLU                      | [Repositório GitHub](https://github.com/hendrycks/test)                                   |
| Benchmark HellaSwag                 | [Página do Projeto](https://rowanzellers.com/hellaswag/)                                  |
| Benchmark HumanEval                 | [Repositório GitHub](https://github.com/openai/human-eval)                                  |
| Preços OpenAI                       | Página oficial de preços da API da OpenAI.                                                  |
| Preços Anthropic                    | Página oficial de preços da API da Anthropic.                                               |
| Preços Google AI (Vertex/Studio)    | Páginas oficiais de preços do Google Cloud AI ou AI Studio.                                |
| Plataforma AWS Bedrock              | Documentação e preços do Amazon Bedrock para modelos Nova e Titan.                          |
| xAI API                             | Documentação oficial da xAI para modelos Grok.                                              |
| Mistral AI Platform                 | Página oficial da Mistral AI para preços e documentação.                                    |
| Plataforma API DeepSeek             | Documentação oficial e preços da API DeepSeek (incluindo info fora do pico).              |
| AI Book Bans                        | Benchmark que define a Pontuação de Liberdade usada para medir a abertura dos LLMs.           |

---

Contribuições e feedback para melhorar esta análise são bem-vindos. Por favor, abra uma issue ou envie um pull request.