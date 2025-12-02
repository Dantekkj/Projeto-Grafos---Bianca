# Importação das bibliotecas necessárias
import numpy as np  # Para operações matemáticas e matrizes
import pandas as pd  # Para manipulação de dados em tabelas
import networkx as nx  # Para criação e análise de grafos
import matplotlib.pyplot as plt  # Para visualização de gráficos
import json  # Para exportar dados em formato JSON
from itertools import combinations  # Para criar combinações de elementos

def criar_dataset():
    """
    Cria o conjunto de dados com informações sobre pessoas e suas preferências.
    Retorna um DataFrame com: id, nome, idade, poder, gênero preferido e categoria.
    """

    # Dicionário com todos os dados das pessoas
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # Identificador único de cada pessoa
        'nome': [  # Nome de cada pessoa
            'Verônica',
            'Giovanna zangarelli',
            'Adelice Rocha Nogueira',
            'Bruna Luiza',
            'Isabela pina',
            'Emi',
            'Emi',
            'Ana Paula',
            'Lilian',
            'João Victor Zaparoli',
            'Israell evellin carneiro'
        ],
        'idade': [  # Faixa etária de cada pessoa
            '19 anos - 40 anos',
            '<=18 anos',
            '>61',
            '<=18 anos',
            '19 anos - 40 anos',
            '<=18 anos',
            '<=18 anos',
            '40 anos - 60 anos',
            '19 anos - 40 anos',
            '19 anos - 40 anos',
            '19 anos - 40 anos'
        ],
        'poder': [  # Superpoder escolhido por cada pessoa
            'Teletransporte',
            'Teletransporte',
            'Teletransporte',
            'Voar',
            'Teletransporte',
            'Teletransporte',
            'Teletransporte',
            'Ler mentes',
            'Ler mentes',
            'Invisibilidade',
            'Ler mentes'
        ],
        'genero_preferido': [  # Gênero de filme preferido de cada pessoa
            'Comédia',
            'Ficção Científica',
            'Drama',
            'Comédia',
            'Terror',
            'Comédia',
            'Comédia',
            'Drama',
            'Comédia',
            'Terror',
            'Comédia'
        ],
        'categoria': [  # O que mais chama atenção em um filme
            'Gênero',
            'Trailer',
            'Roteiro',
            'Trailer',
            'Ator',
            'Gênero',
            'Gênero',
            'Trailer',
            'Gênero',
            'Gênero',
            'Ator'
        ]
    }
    
    # Converte o dicionário em um DataFrame (tabela)
    df = pd.DataFrame(data)
    print("Dataset criado com sucesso!")
    print(f"Dimensões: {df.shape}")  # Mostra quantas linhas e colunas tem
    print("\nPrimeiras linhas:")
    print(df.head())  # Mostra as 5 primeiras linhas
    
    return df


def criar_matriz_incidencia(df):
    """
    Cria a matriz de incidência mostrando quantas pessoas escolheram cada combinação de poder e gênero.
    Linhas = Poderes, Colunas = Gêneros de filme.
    """
    # Pega todos os poderes únicos e ordena alfabeticamente
    poderes = sorted(df['poder'].unique())
    # Pega todos os gêneros únicos e ordena alfabeticamente
    generos = sorted(df['genero_preferido'].unique())
    
    # Cria uma matriz vazia com zeros (linhas = poderes, colunas = gêneros)
    matriz = np.zeros((len(poderes), len(generos)), dtype=int)
    
    # Para cada combinação de poder e gênero, conta quantas pessoas têm essa combinação
    for i, poder in enumerate(poderes):
        for j, genero in enumerate(generos):
            # Conta quantas pessoas têm esse poder E esse gênero preferido
            count = len(df[(df['poder'] == poder) & (df['genero_preferido'] == genero)])
            matriz[i, j] = count
    
    # Converte a matriz em DataFrame para facilitar visualização
    df_incidencia = pd.DataFrame(matriz, index=poderes, columns=generos)
    
    # Imprime a matriz na tela
    print("\n" + "="*80)
    print("MATRIZ DE INCIDÊNCIA (Poderes × Gêneros)")
    print("="*80)
    print(df_incidencia)
    
    return df_incidencia, poderes, generos

def criar_matriz_similaridade_pessoas(df):
    """
    Cria a matriz de similaridade entre PESSOAS.
    Baseado em: Poder, Gênero Preferido e Categoria (O que chama atenção)
    """
    # Pega a lista de nomes de todas as pessoas
    pessoas = df['nome'].tolist()
    n = len(pessoas)  # Número total de pessoas
    
    # Cria uma matriz vazia para guardar as similaridades
    matriz_sim = np.zeros((n, n))
    
    # Compara cada pessoa com todas as outras
    for i in range(n):
        for j in range(n):
            if i == j:  # Se for a mesma pessoa, similaridade = 1 (100%)
                matriz_sim[i, j] = 1.0
            else:
                # Pega os dados de cada pessoa
                p1 = df.iloc[i]
                p2 = df.iloc[j]
                
                # Conta quantos atributos são iguais
                matches = 0
                total_attributes = 3  # Poder, gênero e categoria
                
                if p1['poder'] == p2['poder']: matches += 1  # Mesmo poder?
                if p1['genero_preferido'] == p2['genero_preferido']: matches += 1  # Mesmo gênero?
                if p1['categoria'] == p2['categoria']: matches += 1  # Mesma categoria?
                
                # Calcula a similaridade (0 a 1)
                matriz_sim[i, j] = matches / total_attributes
    
    # Converte a matriz em DataFrame
    df_similaridade = pd.DataFrame(matriz_sim, index=pessoas, columns=pessoas)
    
    # Imprime a matriz na tela (arredondada para 2 casas decimais)
    print("\n" + "="*80)
    print("MATRIZ DE SIMILARIDADE (Pessoas × Pessoas)")
    print("="*80)
    print(df_similaridade.round(2))
    
    return df_similaridade


def criar_grafo_coocorrencia_geral(df):
    """
    Cria um grafo de coocorrência conectando:
    - Superpoderes
    - Gêneros de Filme
    - O que chama atenção (Categoria)
    
    SOMENTE categorias, poderes e gêneros - SEM pessoas!
    Se uma pessoa escolheu (Poder A, Gênero B, Categoria C), criamos arestas:
    A-B, A-C, B-C
    """
    # Cria um grafo vazio
    G = nx.Graph()
    
    # Adiciona nós (pontos) para cada poder
    for val in df['poder'].unique():
        G.add_node(f"P:{val}", tipo='poder', label=val)
    
    # Adiciona nós para cada gênero de filme
    for val in df['genero_preferido'].unique():
        G.add_node(f"G:{val}", tipo='genero', label=val)
    
    # Adiciona nós para cada categoria (o que chama atenção)
    for val in df['categoria'].unique():
        G.add_node(f"C:{val}", tipo='categoria', label=val)
        
    # Cria conexões (arestas) entre poderes, gêneros e categorias
    for idx, row in df.iterrows():
        # Pega o poder, gênero e categoria de cada pessoa
        p = f"P:{row['poder']}"
        g = f"G:{row['genero_preferido']}"
        c = f"C:{row['categoria']}"
        
        # Cria pares de conexões: poder-gênero, poder-categoria, gênero-categoria
        pairs = [(p, g), (p, c), (g, c)]
        
        # Para cada par, adiciona ou aumenta o peso da conexão
        for u, v in pairs:
            if G.has_edge(u, v):  # Se a conexão já existe
                G[u][v]['weight'] += 1  # Aumenta o peso (força da conexão)
            else:  # Se não existe
                G.add_edge(u, v, weight=1)  # Cria a conexão com peso 1
                
    print("\n" + "="*80)
    print("GRAFO DE COOCORRÊNCIA GERAL (Poderes, Gêneros, Categorias)")
    print("="*80)
    print(f"Nós: {G.number_of_nodes()}")
    print(f"Arestas: {G.number_of_edges()}")
    
    return G


def criar_grafo_incidencia_completo(df):
    """
    Cria um grafo completo mostrando pessoas, poderes, gêneros e categorias.
    Nós: Pessoas, Poderes, Gêneros e Categorias
    Arestas: Pessoas -> Poderes, Pessoas -> Gêneros e Pessoas -> Categorias
    """
    # Cria um grafo vazio
    G = nx.Graph()
    
    # Pega todas as opções únicas de poderes, gêneros e categorias
    poderes = sorted(df['poder'].unique())
    generos = sorted(df['genero_preferido'].unique())
    categorias = sorted(df['categoria'].unique())
    
    # Adiciona nós para cada poder
    for poder in poderes:
        G.add_node(f"P:{poder}", tipo='poder', label=poder)
    
    # Adiciona nós para cada gênero
    for genero in generos:
        G.add_node(f"G:{genero}", tipo='genero', label=genero)
    
    # Adiciona nós para cada categoria
    for categoria in categorias:
        G.add_node(f"C:{categoria}", tipo='categoria', label=categoria)
    
    # Para cada pessoa, adiciona ela no grafo e conecta com suas escolhas
    for idx, row in df.iterrows():
        pessoa_id = f"Pessoa:{row['nome']}"
        # Adiciona a pessoa como um nó
        G.add_node(pessoa_id, tipo='pessoa', label=row['nome'], idade=row['idade'])
        
        # Conecta a pessoa ao poder que ela escolheu
        G.add_edge(pessoa_id, f"P:{row['poder']}", weight=1)
        
        # Conecta a pessoa ao gênero que ela prefere
        G.add_edge(pessoa_id, f"G:{row['genero_preferido']}", weight=1)
        
        # Conecta a pessoa à categoria que ela escolheu
        G.add_edge(pessoa_id, f"C:{row['categoria']}", weight=1)
    
    print("\n" + "="*80)
    print("GRAFO DE INCIDÊNCIA COMPLETO (Pessoas, Poderes, Gêneros, Categorias)")
    print("="*80)
    print(f"Nós: {G.number_of_nodes()}")
    print(f"Arestas: {G.number_of_edges()}")
    print(f"Pessoas: {len(df)}")
    print(f"Poderes: {len(poderes)}")
    print(f"Gêneros: {len(generos)}")
    print(f"Categorias: {len(categorias)}")
    
    return G

def criar_grafo_similaridade_pessoas(df_similaridade, threshold=0.3):
    """
    Cria grafo de similaridade entre PESSOAS
    """
    # Cria um grafo vazio
    G = nx.Graph()
    # Pega a lista de todas as pessoas
    pessoas = df_similaridade.index.tolist()
    
    # Adiciona cada pessoa como um nó no grafo
    for p in pessoas:
        G.add_node(f"Pessoa:{p}", tipo='pessoa', label=p)
        
    # Conecta pessoas que têm similaridade acima do limite (threshold)
    for i, p1 in enumerate(pessoas):
        for j, p2 in enumerate(pessoas):
            if i < j:  # Evita duplicar conexões
                sim = df_similaridade.iloc[i, j]  # Pega o valor de similaridade
                if sim >= threshold:  # Se a similaridade for maior ou igual ao limite
                    # Cria uma conexão entre as duas pessoas
                    G.add_edge(f"Pessoa:{p1}", f"Pessoa:{p2}", weight=sim)
                    
    print("\n" + "="*80)
    print(f"GRAFO DE SIMILARIDADE DE PESSOAS (threshold={threshold})")
    print("="*80)
    print(f"Nós: {G.number_of_nodes()}")
    print(f"Arestas: {G.number_of_edges()}")
    
    return G


def calcular_metricas(G, nome_grafo):
    """
    Calcula métricas topológicas para um grafo:
    - Grau médio
    - Densidade
    - Coeficiente de clustering
    - Centralidade de grau
    - Centralidade de intermediação
    - Centralidade de proximidade
    - Componentes conectados
    """
    print("\n" + "="*80)
    print(f"MÉTRICAS TOPOLÓGICAS - {nome_grafo}")
    print("="*80)
    
    # Dicionário para guardar todas as métricas
    metricas = {}
    
    # Conta quantos nós (pontos) e arestas (conexões) o grafo tem
    metricas['num_nos'] = G.number_of_nodes()
    metricas['num_arestas'] = G.number_of_edges()
    print(f"Número de nós: {metricas['num_nos']}")
    print(f"Número de arestas: {metricas['num_arestas']}")
    
    # Calcula o grau médio (quantas conexões cada nó tem em média)
    graus = dict(G.degree())
    metricas['grau_medio'] = sum(graus.values()) / len(graus) if graus else 0
    print(f"Grau médio: {metricas['grau_medio']:.2f}")
    
    # Calcula a densidade (quão conectado o grafo está, de 0 a 1)
    metricas['densidade'] = nx.density(G)
    print(f"Densidade: {metricas['densidade']:.4f}")
    
    # Calcula o clustering (tendência de formar grupos/comunidades)
    metricas['clustering'] = nx.average_clustering(G)
    print(f"Coeficiente de clustering médio: {metricas['clustering']:.4f}")
    
    # Verifica se o grafo está totalmente conectado
    if nx.is_connected(G):
        metricas['num_componentes'] = 1
        print("Grafo conectado: Sim")
        
        # Calcula o diâmetro (maior distância entre dois nós)
        metricas['diametro'] = nx.diameter(G)
        print(f"Diâmetro: {metricas['diametro']}")
    else:
        # Se não está conectado, conta quantos grupos separados existem
        componentes = list(nx.connected_components(G))
        metricas['num_componentes'] = len(componentes)
        print(f"Grafo conectado: Não ({metricas['num_componentes']} componentes)")
        metricas['diametro'] = None
    
    # Centralidade de Grau: quais nós têm mais conexões
    print("\nCentralidade de Grau (Top 3):")
    degree_cent = nx.degree_centrality(G)
    metricas['centralidade_grau'] = degree_cent
    # Mostra os 3 nós mais conectados
    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:3]
    for no, valor in top_degree:
        print(f"  {no}: {valor:.4f}")
    
    # Centralidade de Intermediação: quais nós são pontes entre outros
    print("\nCentralidade de Intermediação (Top 3):")
    betweenness_cent = nx.betweenness_centrality(G)
    metricas['centralidade_intermediacao'] = betweenness_cent
    # Mostra os 3 nós que mais servem de ponte
    top_betweenness = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:3]
    for no, valor in top_betweenness:
        print(f"  {no}: {valor:.4f}")
    
    # Centralidade de Proximidade: quais nós estão mais próximos de todos os outros
    if nx.is_connected(G):
        print("\nCentralidade de Proximidade (Top 3):")
        closeness_cent = nx.closeness_centrality(G)
        metricas['centralidade_proximidade'] = closeness_cent
        # Mostra os 3 nós mais centrais
        top_closeness = sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:3]
        for no, valor in top_closeness:
            print(f"  {no}: {valor:.4f}")
    else:
        metricas['centralidade_proximidade'] = {}
    
    return metricas


def exportar_para_json(G, nome_arquivo):
    """
    Exporta o grafo para um arquivo JSON que pode ser usado na visualização web.
    """
    # Estrutura de dados para o JSON
    data = {
        'nodes': [],  # Lista de nós
        'links': []   # Lista de conexões
    }
    
    # Adiciona todos os nós do grafo
    for node in G.nodes(data=True):
        node_data = {
            'id': node[0],  # ID do nó
            **node[1]       # Outros atributos (tipo, label, etc)
        }
        data['nodes'].append(node_data)
    
    # Adiciona todas as conexões do grafo
    for edge in G.edges(data=True):
        weight = edge[2].get('weight', 1)  # Pega o peso da conexão

        # Converte para número se necessário
        if hasattr(weight, 'item'):
            weight = weight.item()
        link_data = {
            'source': edge[0],      # Nó de origem
            'target': edge[1],      # Nó de destino
            'weight': float(weight) # Peso da conexão
        }
        data['links'].append(link_data)
    
    # Salva o arquivo JSON
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nGrafo exportado para: {nome_arquivo}")

def exportar_metricas_json(metricas_dict, nome_arquivo):
    """
    Exporta as métricas calculadas para um arquivo JSON.
    """
    metricas_serializaveis = {}
    
    # Para cada grafo, organiza as métricas em um formato simples
    for grafo, metricas in metricas_dict.items():
        metricas_serializaveis[grafo] = {
            'num_nos': metricas['num_nos'],
            'num_arestas': metricas['num_arestas'],
            'grau_medio': round(metricas['grau_medio'], 2),
            'densidade': round(metricas['densidade'], 4),
            'clustering': round(metricas['clustering'], 4),
            'num_componentes': metricas['num_componentes'],
            'diametro': metricas['diametro'],
            # Top 5 nós com maior centralidade de grau
            'top_degree': sorted(
                metricas['centralidade_grau'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            # Top 5 nós com maior centralidade de intermediação
            'top_betweenness': sorted(
                metricas['centralidade_intermediacao'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    # Salva o arquivo JSON
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(metricas_serializaveis, f, ensure_ascii=False, indent=2)
    
    print(f"\nMétricas exportadas para: {nome_arquivo}")


def main():
    """
    Função principal que executa todo o pipeline de análise
    """
    print("="*80)
    print("ANÁLISE DE GRAFOS DE SUPER PODERES")
    print("="*80)
    
    # PASSO 1: Criar o conjunto de dados
    df = criar_dataset()
    
    # PASSO 2: Criar as matrizes de incidência e similaridade
    df_incidencia, poderes, generos = criar_matriz_incidencia(df)
    df_similaridade = criar_matriz_similaridade_pessoas(df)
    
    # PASSO 3: Criar os três tipos de grafos
    grafo_incidencia = criar_grafo_incidencia_completo(df)  # Pessoas conectadas a poderes, gêneros e categorias
    grafo_similaridade = criar_grafo_similaridade_pessoas(df_similaridade, threshold=0.3)  # Pessoas similares
    grafo_coocorrencia = criar_grafo_coocorrencia_geral(df)  # Poderes, gêneros e categorias que aparecem juntos
    
    # PASSO 4: Calcular métricas topológicas de cada grafo
    metricas_incidencia = calcular_metricas(grafo_incidencia, "GRAFO DE INCIDÊNCIA")
    metricas_similaridade = calcular_metricas(grafo_similaridade, "GRAFO DE SIMILARIDADE")
    metricas_coocorrencia = calcular_metricas(grafo_coocorrencia, "GRAFO DE COOCORRÊNCIA")
    
    # PASSO 5: Exportar os grafos para arquivos JSON (para visualização web)
    exportar_para_json(grafo_incidencia, 'grafo_incidencia.json')
    exportar_para_json(grafo_similaridade, 'grafo_similaridade.json')
    exportar_para_json(grafo_coocorrencia, 'grafo_coocorrencia.json')
    
    # Exportar as métricas para JSON
    metricas_dict = {
        'incidencia': metricas_incidencia,
        'similaridade': metricas_similaridade,
        'coocorrencia': metricas_coocorrencia
    }
    exportar_metricas_json(metricas_dict, 'metricas.json')
    
    # Salvar as matrizes e o dataset em arquivos CSV
    df_incidencia.to_csv('matriz_incidencia.csv')
    df_similaridade.to_csv('matriz_similaridade.csv')
    df.to_csv('dataset_superpoderes.csv', index=False)
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA!")
    print("="*80)
    print("\nArquivos gerados:")
    print("- dataset_superpoderes.csv")
    print("- matriz_incidencia.csv")
    print("- matriz_similaridade.csv")
    print("- matriz_coocorrencia.csv")
    print("- grafo_incidencia.json")
    print("- grafo_similaridade.json")
    print("- grafo_coocorrencia.json")
    print("- metricas.json")

# Executa o programa quando o arquivo é rodado diretamente
if __name__ == "__main__":
    main()
