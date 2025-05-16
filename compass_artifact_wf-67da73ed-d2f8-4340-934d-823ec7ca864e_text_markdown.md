# Roadmap completo para visualizações de LLMs com Plotly

Este guia oferece um roteiro técnico abrangente para melhorar gráficos de comparação de LLMs usando Plotly, com foco em design eficaz, acessibilidade e implementação prática.

## A ciência por trás das visualizações eficazes de modelos

Para comparar LLMs visualmente, as técnicas multivariadas são essenciais - utilizando posição (eixos x/y), tamanho, cor e interatividade para comunicar várias dimensões de desempenho simultaneamente. O Plotly oferece todas as ferramentas necessárias para implementar estas técnicas sofisticadas com Python.

### 1. Mapeamento de tamanho das bolhas para custo por token

O dimensionamento correto das bolhas é crucial para representações precisas de custos. Três métodos de escalonamento são particularmente relevantes para dados de custo de LLMs:

**Escala linear** - ideal para modelos com faixas de preço semelhantes:
```python
# Mapeamento direto de valores brutos
fig = px.scatter(df, x='latencia', y='desempenho', size='custo_por_1k', size_max=60)
```

**Escala raiz quadrada** - melhor para percepção visual com variação moderada:
```python
import numpy as np
# Transformação raiz quadrada para melhor percepção visual
df['tamanho_escalonado'] = np.sqrt(df['custo_por_1k'])
fig = px.scatter(df, x='latencia', y='desempenho', size='tamanho_escalonado', size_max=60)
```

**Escala logarítmica** - recomendada para conjuntos de LLMs com diferenças extremas de preço:
```python
# Transformação logarítmica para distribuições altamente assimétricas
df['log_custo'] = np.log1p(df['custo_por_1k'])  # log1p = log(1+x) para lidar com zeros
fig = px.scatter(df, x='latencia', y='desempenho', size='log_custo', size_max=60)
```

Para controle fino do dimensionamento com Plotly Graph Objects, o parâmetro `sizeref` é crucial:

```python
# Cálculo do sizeref para dimensionamento ótimo
tamanhos = df['custo_por_1k']
tamanho_maximo_desejado = 40  # em pixels
sizeref = 2.0 * max(tamanhos) / (tamanho_maximo_desejado**2)

fig = go.Figure(data=[go.Scatter(
    x=df['latencia'],
    y=df['desempenho'],
    mode='markers',
    marker=dict(
        size=tamanhos,
        sizemode='area',  # 'area' para dimensionamento visualmente preciso
        sizeref=sizeref,
        sizemin=4  # tamanho mínimo para valores muito pequenos
    )
)])
```

### 2. Codificação de cores por provedor

As cores funcionam melhor quando são perceptivamente distintas e acessíveis. Aqui está uma paleta de cores otimizada para provedores de LLMs que mantém alguma conexão com identidades de marca:

```python
cores_provedores_llm = {
    'OpenAI': '#10a37f',       # Verde adaptado da marca
    'Anthropic': '#b83280',    # Roxo adaptado da marca
    'Google': '#4285F4',       # Azul da marca
    'Microsoft': '#00a4ef',    # Azul da marca
    'Meta': '#0668E1',         # Azul da marca
    'Cohere': '#2596be',       # Adaptado da marca
    'DeepSeek': '#6366f1',     # Roxo-azul
    'Mistral': '#7c3aed',      # Tom roxo
    'HuggingFace': '#fbbf24',  # Âmbar/dourado
    'Stability AI': '#e11d48'  # Tom vermelho
}

# Aplicando o mapa de cores em um gráfico de dispersão
fig = px.scatter(df, x='latencia', y='desempenho', 
                 color='provedor',
                 color_discrete_map=cores_provedores_llm,
                 size='custo_por_1k')

# Definindo a ordem das categorias (ex: por desempenho)
ordem_provedores = ['OpenAI', 'Anthropic', 'Google', 'Microsoft', 'Meta', 
                    'Cohere', 'DeepSeek', 'Mistral', 'HuggingFace', 'Stability AI']

fig = px.scatter(df, x='latencia', y='desempenho',
                 color='provedor',
                 color_discrete_map=cores_provedores_llm,
                 category_orders={'provedor': ordem_provedores},
                 size='custo_por_1k')
```

Para casos onde você prefere usar paletas pré-definidas que são seguras para daltonismo:

```python
# Paleta Okabe-Ito recomendada para daltonismo
fig = px.scatter(df, x='latencia', y='desempenho',
                 color='provedor',
                 color_discrete_sequence=px.colors.qualitative.Safe)

# Outras opções de paletas seguras para daltonismo
# px.colors.qualitative.Set2
# px.colors.qualitative.D3
# px.colors.qualitative.G10
```

### 3. Legendas eficazes para tamanho das bolhas

Como o Plotly não possui um método nativo para criar legendas de tamanho, devemos implementar uma abordagem personalizada:

```python
from plotly.subplots import make_subplots

# Criar figura com dois subplots lado a lado
fig = make_subplots(rows=1, cols=2, 
                   column_widths=[0.8, 0.2],
                   specs=[[{'type': 'scatter'}, {'type': 'scatter'}]])

# Calcular referência de tamanho
tamanhos = df['custo_por_1k']
sizeref = 2.0 * max(tamanhos) / (40**2)

# Adicionar gráfico de bolhas principal ao primeiro subplot
fig.add_trace(
    go.Scatter(
        x=df['latencia'],
        y=df['desempenho'],
        mode='markers+text',
        marker=dict(
            size=tamanhos,
            sizemode='area',
            sizeref=sizeref,
            color='rgba(66, 135, 245, 0.7)',
            line=dict(width=1, color='darkblue')
        ),
        text=df['modelo'],
        textposition="top center",
        name=''
    ),
    row=1, col=1
)

# Criar bolhas de referência para a legenda
# Selecionar alguns tamanhos representativos
tamanhos_legenda = [0.2, 1, 5, 10]
legenda_y = np.arange(len(tamanhos_legenda))

# Adicionar bolhas de referência ao segundo subplot
fig.add_trace(
    go.Scatter(
        x=[0] * len(tamanhos_legenda),
        y=legenda_y,
        mode='markers+text',
        marker=dict(
            size=tamanhos_legenda,
            sizemode='area',
            sizeref=sizeref,
            color='rgba(66, 135, 245, 0.7)',
            line=dict(width=1, color='darkblue')
        ),
        text=[f"R${s:.1f}" for s in tamanhos_legenda],  # Use R$ para Reais ou $ para dólares
        textposition="middle right",
        name='',
        showlegend=False
    ),
    row=1, col=2
)

# Atualizar layout
fig.update_layout(
    title='Desempenho vs Latência de LLMs',
    height=600, width=1000,
    showlegend=False
)

# Formatar o subplot da legenda de tamanho
fig.update_xaxes(visible=False, row=1, col=2)
fig.update_yaxes(visible=False, row=1, col=2)

# Adicionar um título para a legenda de tamanho
fig.add_annotation(
    x=0, y=1.1,
    xref="x2", yref="paper",
    text="Custo por 1k tokens",
    showarrow=False,
    font=dict(size=14)
)
```

Alternativamente, usar uma anotação simples para explicar o significado do tamanho:

```python
fig.add_annotation(
    x=0.95, y=0.05,
    xref="paper", yref="paper",
    text="Tamanho da bolha representa<br>custo por 1k tokens",
    showarrow=False,
    align="right",
    bgcolor="rgba(255, 255, 255, 0.8)",
    borderwidth=1,
    borderpad=4
)
```

### 4. Prevenção de sobreposição de texto

A sobreposição de texto é um problema comum em gráficos densos. Embora o Plotly não tenha um algoritmo automático de repulsão de rótulos, várias estratégias são eficazes:

**Posicionamento manual controlado:**
```python
fig.update_traces(textposition='top center')  # Opções: 'top left', 'top right', 'bottom left', etc.
```

**Posicionamento personalizado por modelo:**
```python
posicoes_dict = {
    'GPT-4': 'top center',
    'Claude': 'middle left',
    'Llama-2': 'bottom right',
    # ... mais posições
}

# Função para aplicar posições do dicionário
def obter_posicao(modelo):
    return posicoes_dict.get(modelo, 'top center')  # Posição padrão

fig.update_traces(textposition=[obter_posicao(modelo) for modelo in df['modelo']])
```

**Uso de tooltips em vez de rótulos permanentes:**
```python
fig = px.scatter(df, x="latencia", y="desempenho", 
                 hover_name="modelo", hover_data=["custo_por_1k", "tokens_contexto"])
```

**Templates de hover personalizados:**
```python
fig = go.Figure(go.Scatter(
    x=df['latencia'],
    y=df['desempenho'],
    hovertemplate=
    '<b>%{text}</b><br>'+
    'Desempenho: %{y:.2f}<br>'+
    'Latência: %{x} ms<br>'+
    'Custo/1k: R$%{marker.size:.2f}<br>',
    text=df['modelo'],
    mode='markers'
))
```

**Rotulagem seletiva para pontos importantes:**
```python
# Mostrar apenas rótulos para pontos que atendem a uma condição
df['rotulo_exibido'] = df.apply(
    lambda row: row['modelo'] if row['importancia'] > limite else '', 
    axis=1
)
fig = px.scatter(df, x="latencia", y="desempenho", text="rotulo_exibido")
```

### 5. Técnicas específicas do Plotly para comparações de LLMs

**Visualização completa com todas as técnicas combinadas:**

```python
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Dados de exemplo
df = pd.DataFrame({
    'modelo': ['GPT-3.5', 'GPT-4', 'Claude-2', 'Claude-3', 'Llama-2', 'Mistral', 'Gemini'],
    'desempenho': [0.75, 0.9, 0.8, 0.88, 0.7, 0.78, 0.85],
    'latencia': [200, 350, 280, 300, 150, 180, 320],
    'custo_por_1k': [0.5, 10, 8, 12, 0.2, 0.3, 5],
    'provedor': ['OpenAI', 'OpenAI', 'Anthropic', 'Anthropic', 'Meta', 'Mistral', 'Google']
})

# Mapeamento de cores por provedor (acessível para daltonismo)
cores_provedores = {
    'OpenAI': '#10a37f',      # Verde adaptado
    'Anthropic': '#b83280',   # Roxo adaptado
    'Meta': '#0668E1',        # Azul
    'Mistral': '#7c3aed',     # Tom roxo
    'Google': '#4285F4'       # Azul Google
}

# Criar figura com dois subplots (gráfico principal e legenda de tamanho)
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.85, 0.15],
    specs=[[{'type': 'scatter'}, {'type': 'scatter'}]],
    horizontal_spacing=0.02
)

# Transformação por raiz quadrada para melhor percepção visual de tamanho
tamanhos = np.sqrt(df['custo_por_1k']) * 20  # Multiplicar por um fator para melhor visibilidade
sizeref = 2.0 * max(tamanhos) / (50**2)

# Gráfico de bolhas principal
fig.add_trace(
    go.Scatter(
        x=df['latencia'],
        y=df['desempenho'],
        mode='markers+text',
        marker=dict(
            size=tamanhos,
            sizemode='area',  # Use 'area' para comparação visual precisa
            sizeref=sizeref,
            color=[cores_provedores[provedor] for provedor in df['provedor']],
            line=dict(width=1, color='darkgray')
        ),
        text=df['modelo'],
        textposition="top center",
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Desempenho: %{y:.2f}<br>" +
            "Latência: %{x} ms<br>" +
            "Provedor: %{customdata}<br>" +
            "Custo/1k: R$%{marker.size:.2f}<br>"
        ),
        customdata=df['provedor'],
        name=''
    ),
    row=1, col=1
)

# Criar legenda de tamanho (bolhas de referência)
tamanhos_legenda = [0.1, 1.0, 5.0, 10.0]  # Valores de custo representativos
posicoes_y_legenda = list(range(len(tamanhos_legenda)))
tamanhos_legenda_escalados = np.sqrt(tamanhos_legenda) * 20  # Aplicar mesma escala do gráfico principal

fig.add_trace(
    go.Scatter(
        x=[0] * len(tamanhos_legenda),
        y=posicoes_y_legenda,
        mode='markers+text',
        marker=dict(
            size=tamanhos_legenda_escalados,
            sizemode='area',
            sizeref=sizeref,
            color='rgba(50, 50, 50, 0.7)',
            line=dict(width=1, color='darkgray')
        ),
        text=[f"R${s:.1f}" for s in tamanhos_legenda],
        textposition="middle right",
        hoverinfo='skip',
        showlegend=False
    ),
    row=1, col=2
)

# Adicionar legendas de provedor usando traces separados para cada provedor
provedores_unicos = df['provedor'].unique()
for provedor in provedores_unicos:
    fig.add_trace(
        go.Scatter(
            x=[None],  # Valores vazios para não mostrar no gráfico
            y=[None],
            mode='markers',
            marker=dict(
                size=15,
                color=cores_provedores[provedor],
            ),
            name=provedor,
            showlegend=True
        ),
        row=1, col=1
    )

# Atualizar layout
fig.update_layout(
    title='Comparação de Modelos LLM por Desempenho, Latência e Custo',
    height=600,
    width=1000,
    xaxis=dict(
        title='Latência (ms)',
        gridcolor='white',
        gridwidth=2,
    ),
    yaxis=dict(
        title='Pontuação de Desempenho',
        gridcolor='white',
        gridwidth=2,
    ),
    plot_bgcolor='rgb(240, 240, 240)',
    legend=dict(
        title="Provedores",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Configurar o subplot da legenda de tamanho
fig.update_xaxes(visible=False, row=1, col=2)
fig.update_yaxes(visible=False, row=1, col=2)

# Adicionar título para a legenda de tamanho
fig.add_annotation(
    x=0, y=1.15,
    xref="x2", yref="paper",
    text="Custo por 1k tokens",
    showarrow=False,
    font=dict(size=12)
)

fig.show()
```

### 6. Boas práticas de visualização para gráficos LLM

1. **Limite de dimensões**: Restrinja as visualizações a 4-5 dimensões antes de considerar faceting
   - Posição (eixos x/y) → métricas principais (desempenho, latência)
   - Tamanho → custo por token
   - Cor → provedor ou família de modelos
   - Texto → nome do modelo

2. **Codificação consistente**: Use a mesma codificação visual para a mesma variável em todos os gráficos

3. **Escalonamento apropriado**:
   - Raiz quadrada para custos (melhora percepção visual)
   - Log para valores extremamente variados (ex: contagem de parâmetros)

4. **Informações ao passar o mouse**: Forneça contexto rico (modelo, provedor, métricas exatas)

5. **Elementos interativos**: Adicione funcionalidade de zoom, pan e filtro para exploração

6. **Lidar com dados densos**:
   ```python
   # Transparência para revelar padrões de densidade
   fig = px.scatter(df, x='feature1', y='feature2', opacity=0.7)
   
   # Jitter para evitar sobreposições perfeitas
   df['feature1_jitter'] = df['feature1'] + np.random.normal(0, 0.05, len(df))
   ```

### 7. Considerações de acessibilidade

**Deficiências de visão cromática**:
- Protanopia: sensibilidade reduzida à luz vermelha
- Deuteranopia: sensibilidade reduzida à luz verde 
- Tritanopia: sensibilidade reduzida à luz azul

**Diretrizes para uso de cores**:
1. **Use azul e laranja/amarelo** como cores contrastantes primárias (melhor que vermelho-verde)
2. **Evite combinações vermelho-verde** problemáticas para daltônicos
3. **Varie a luminância (brilho)** além da tonalidade

**Teste suas visualizações**:
- Simuladores de daltonismo: ColorOracle, Coblis, Sim Daltonism
- Verificadores de contraste: WebAIM Contrast Checker, Coolors

**Suplementação com elementos visuais adicionais**:
```python
# Adicionar formas diferentes além de cores
fig = px.scatter(df, x='latencia', y='desempenho',
                color='provedor', symbol='provedor')

# Adicionar rótulos diretamente no gráfico
fig.update_traces(textposition='top center', textfont_size=10)
```

## Implementação completa e otimizada

Abaixo está uma implementação completa que incorpora todas as recomendações acima em um único gráfico eficaz:

```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Criar dados fictícios para demonstração
def criar_dados_demo():
    provedores = ['OpenAI', 'Anthropic', 'Google', 'Meta', 'Mistral', 'Cohere']
    modelos = [
        'GPT-3.5', 'GPT-4', 'GPT-4o', 
        'Claude-2', 'Claude-3 Haiku', 'Claude-3 Opus',
        'Gemini-1.0', 'Gemini-1.5', 
        'Llama-2', 'Llama-3',
        'Mistral-7B', 'Mistral-8x7B',
        'Cohere-Command', 'Cohere-Command-R'
    ]
    
    # Mapear modelos para provedores
    mapeamento_provedor = {
        'GPT-3.5': 'OpenAI', 'GPT-4': 'OpenAI', 'GPT-4o': 'OpenAI',
        'Claude-2': 'Anthropic', 'Claude-3 Haiku': 'Anthropic', 'Claude-3 Opus': 'Anthropic',
        'Gemini-1.0': 'Google', 'Gemini-1.5': 'Google',
        'Llama-2': 'Meta', 'Llama-3': 'Meta',
        'Mistral-7B': 'Mistral', 'Mistral-8x7B': 'Mistral',
        'Cohere-Command': 'Cohere', 'Cohere-Command-R': 'Cohere'
    }
    
    # Desempenho inventado (0-1)
    desempenho = {
        'GPT-3.5': 0.78, 'GPT-4': 0.91, 'GPT-4o': 0.94,
        'Claude-2': 0.82, 'Claude-3 Haiku': 0.88, 'Claude-3 Opus': 0.93,
        'Gemini-1.0': 0.79, 'Gemini-1.5': 0.90,
        'Llama-2': 0.75, 'Llama-3': 0.87,
        'Mistral-7B': 0.73, 'Mistral-8x7B': 0.85,
        'Cohere-Command': 0.76, 'Cohere-Command-R': 0.84
    }
    
    # Latência fictícia (ms)
    latencia = {
        'GPT-3.5': 150, 'GPT-4': 450, 'GPT-4o': 280,
        'Claude-2': 320, 'Claude-3 Haiku': 220, 'Claude-3 Opus': 480,
        'Gemini-1.0': 280, 'Gemini-1.5': 380,
        'Llama-2': 190, 'Llama-3': 250,
        'Mistral-7B': 140, 'Mistral-8x7B': 280,
        'Cohere-Command': 210, 'Cohere-Command-R': 300
    }
    
    # Custo por 1k tokens (combinação input/output)
    custo = {
        'GPT-3.5': 0.5, 'GPT-4': 10.0, 'GPT-4o': 5.0,
        'Claude-2': 8.0, 'Claude-3 Haiku': 3.0, 'Claude-3 Opus': 15.0,
        'Gemini-1.0': 0.7, 'Gemini-1.5': 7.0,
        'Llama-2': 0.1, 'Llama-3': 0.3,
        'Mistral-7B': 0.2, 'Mistral-8x7B': 2.0,
        'Cohere-Command': 1.0, 'Cohere-Command-R': 4.0
    }
    
    df = pd.DataFrame({
        'modelo': modelos,
        'provedor': [mapeamento_provedor[m] for m in modelos],
        'desempenho': [desempenho[m] for m in modelos],
        'latencia': [latencia[m] for m in modelos],
        'custo_por_1k': [custo[m] for m in modelos]
    })
    
    # Adicionar transformações para visualização
    df['tamanho_escalonado'] = np.sqrt(df['custo_por_1k']) * 20
    
    return df

# Mapear cores por provedor (acessível para daltonismo)
cores_provedores = {
    'OpenAI': '#10a37f',      # Verde adaptado
    'Anthropic': '#b83280',   # Roxo adaptado
    'Google': '#4285F4',      # Azul Google
    'Meta': '#0668E1',        # Azul Meta
    'Mistral': '#7c3aed',     # Tom roxo
    'Cohere': '#2596be'       # Azul adaptado
}

# Posições de texto pré-definidas para evitar sobreposição
posicoes_texto = {
    'GPT-3.5': 'top center', 
    'GPT-4': 'top right', 
    'GPT-4o': 'middle right',
    'Claude-2': 'bottom right', 
    'Claude-3 Haiku': 'top left', 
    'Claude-3 Opus': 'bottom left',
    'Gemini-1.0': 'middle left', 
    'Gemini-1.5': 'top right',
    'Llama-2': 'bottom center', 
    'Llama-3': 'middle right',
    'Mistral-7B': 'top left', 
    'Mistral-8x7B': 'bottom right',
    'Cohere-Command': 'middle left', 
    'Cohere-Command-R': 'top center'
}

# Criar dados de exemplo
df = criar_dados_demo()

# Criar figura com dois subplots (gráfico principal e legenda de tamanho)
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.85, 0.15],
    specs=[[{'type': 'scatter'}, {'type': 'scatter'}]],
    horizontal_spacing=0.02
)

# Preparar dados para gráfico principal
sizeref = 2.0 * max(df['tamanho_escalonado']) / (60**2)

# Adicionar gráfico de bolhas principal
fig.add_trace(
    go.Scatter(
        x=df['latencia'],
        y=df['desempenho'],
        mode='markers+text',
        marker=dict(
            size=df['tamanho_escalonado'],
            sizemode='area',
            sizeref=sizeref,
            color=[cores_provedores[p] for p in df['provedor']],
            line=dict(width=1, color='rgba(100,100,100,0.3)')
        ),
        text=df['modelo'],
        textposition=[posicoes_texto.get(m, 'top center') for m in df['modelo']],
        textfont=dict(size=10, color='rgba(0,0,0,0.7)'),
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Desempenho: %{y:.2f}<br>" +
            "Latência: %{x} ms<br>" +
            "Provedor: %{customdata}<br>" +
            "Custo/1k tokens: R$%{marker.size:.2f}<br>" +
            "<extra></extra>"
        ),
        customdata=df['provedor'],
        name=''
    ),
    row=1, col=1
)

# Adicionar legendas de provedor usando traces separados
provedores_unicos = df['provedor'].unique()
for provedor in provedores_unicos:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=10,
                color=cores_provedores[provedor],
            ),
            name=provedor,
            showlegend=True
        ),
        row=1, col=1
    )

# Criar legenda de tamanho
tamanhos_legenda = [0.1, 1.0, 5.0, 15.0]
posicoes_y_legenda = list(range(len(tamanhos_legenda)))
tamanhos_legenda_escalados = np.sqrt(tamanhos_legenda) * 20

fig.add_trace(
    go.Scatter(
        x=[0] * len(tamanhos_legenda),
        y=posicoes_y_legenda,
        mode='markers+text',
        marker=dict(
            size=tamanhos_legenda_escalados,
            sizemode='area',
            sizeref=sizeref,
            color='rgba(100,100,100,0.5)',
            line=dict(width=1, color='rgba(70,70,70,0.5)')
        ),
        text=[f"R${s:.1f}" for s in tamanhos_legenda],
        textposition="middle right",
        hoverinfo='skip',
        showlegend=False
    ),
    row=1, col=2
)

# Atualizar layout
fig.update_layout(
    title='Comparação de Modelos LLM: Desempenho × Latência × Custo',
    height=600, width=1000,
    xaxis=dict(
        title='Latência (ms)',
        gridcolor='rgba(220,220,220,0.8)',
    ),
    yaxis=dict(
        title='Pontuação de Desempenho',
        gridcolor='rgba(220,220,220,0.8)',
        range=[0.65, 0.97],  # Ajustar para melhor visualização
    ),
    plot_bgcolor='rgba(245,245,245,0.8)',
    legend=dict(
        title="Provedores",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=10),
    )
)

# Configurar o subplot da legenda de tamanho
fig.update_xaxes(visible=False, row=1, col=2)
fig.update_yaxes(visible=False, row=1, col=2)

# Adicionar título para a legenda de tamanho
fig.add_annotation(
    x=0, y=1.15,
    xref="x2", yref="paper",
    text="Custo por 1k tokens",
    showarrow=False,
    font=dict(size=12)
)

# Adicionar nota explicativa
fig.add_annotation(
    x=0.95, y=0.05,
    xref="paper", yref="paper",
    text="Tamanho da bolha = Custo por 1k tokens",
    showarrow=False,
    align="right",
    bgcolor="rgba(255,255,255,0.7)",
    borderpad=4
)

fig.show()
```

## Observações finais

Você mencionou que gostaria de analisar um código Python existente que usa Plotly para gerar os gráficos, mas não incluiu nenhum código para análise. Se você compartilhar o código específico, posso fornecer recomendações mais direcionadas para sua implementação atual.

As técnicas apresentadas neste roadmap fornecem uma base sólida para criar visualizações de comparação de LLMs que são informativas, acessíveis e visualmente atraentes. Ao implementar estas práticas, seus gráficos serão mais eficazes na comunicação das diferenças complexas entre os vários modelos de linguagem disponíveis atualmente.