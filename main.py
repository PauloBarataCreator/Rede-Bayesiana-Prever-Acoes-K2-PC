import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import yfinance as yf
from sklearn.cluster import AgglomerativeClustering

# Compatibilidade entre versões recentes e anteriores da pgmpy
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

try:
    from pgmpy.estimators import K2 as K2Score
except ImportError:
    from pgmpy.estimators import K2Score

from pgmpy.estimators import HillClimbSearch, PC, BayesianEstimator
from pgmpy.inference import VariableElimination

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. OBTENÇÃO DE DADOS (YAHOO! FINANCE OU SINTÉTICOS)
# ==============================================================================
def gerar_dados_sinteticos(n_amostras=4735, preco_inicial=20.0, volatilidade=0.02, seed=42):
    """Gera cotações diárias sintéticas caso os dados reais não sejam utilizados."""
    rng = np.random.default_rng(seed)
    abertura = np.zeros(n_amostras)
    fechamento = np.zeros(n_amostras)
    maior = np.zeros(n_amostras)
    menor = np.zeros(n_amostras)
    volume = np.zeros(n_amostras)

    preco_atual = preco_inicial
    for t in range(n_amostras):
        abertura[t] = preco_atual * (1 + rng.normal(0, volatilidade * 0.25))
        retorno = rng.normal(0.0002, volatilidade)
        fechamento[t] = max(1.0, abertura[t] * (1 + retorno))

        spread_sup = abs(rng.normal(0, volatilidade * 0.5))
        spread_inf = abs(rng.normal(0, volatilidade * 0.5))
        maior[t] = max(abertura[t], fechamento[t]) * (1 + spread_sup)
        menor[t] = max(0.5, min(abertura[t], fechamento[t]) * (1 - spread_inf))
        volume[t] = 1e6 * (1 + 15 * abs(retorno) + rng.exponential(0.5))
        preco_atual = fechamento[t]

    return pd.DataFrame({
        "Abertura": abertura,
        "Maior_Preco": maior,
        "Menor_Preco": menor,
        "Volume": volume,
        "Fechamento": fechamento
    })


def obter_dados_acao(ticker_yf, cfg_sintetico, usar_dados_reais=True):
    """
    Baixa cotações reais do Yahoo! Finance seguindo exatamente a preparação do artigo:
    - Período: 01/02/2000 até 27/11/2020
    - Treino: até 14/07/2014 | Teste: a partir de 15/07/2014 (3 anos / até 1095 amostras)
    - Remoção de Data, Fechamento Ajustado e dias sem pregão (valores nulos / volume 0)
    """
    if usar_dados_reais:
        try:
            df_raw = yf.download(
                ticker_yf,
                start="2000-02-01",
                end="2020-11-28",
                auto_adjust=False,
                progress=False
            )

            # Trata MultiIndex retornado por versões recentes do yfinance
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)

            # Remove dias sem pregão / registros incompletos
            df_raw = df_raw.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            df_raw = df_raw[df_raw["Volume"] > 0]

            # Remove 'Adj Close' (e posteriormente a Data/índice) mantendo os 5 atributos do artigo
            df_limpo = df_raw[["Open", "High", "Low", "Volume", "Close"]].copy()
            df_limpo.columns = ["Abertura", "Maior_Preco", "Menor_Preco", "Volume", "Fechamento"]

            # Separação temporal conforme o artigo
            df_treino = df_limpo.loc[:"2014-07-14"].reset_index(drop=True)
            df_teste = df_limpo.loc["2014-07-15":].iloc[:1095].reset_index(drop=True)

            if len(df_treino) > 500 and len(df_teste) > 100:
                print(f"  [Yahoo! Finance] {ticker_yf}: {len(df_treino)} treino | {len(df_teste)} teste")
                return df_treino, df_teste
            else:
                print(f"  [Aviso] Poucos dados reais para {ticker_yf}. Usando dados sintéticos.")
        except Exception as e:
            print(f"  [Aviso] Falha ao baixar {ticker_yf} ({e}). Usando dados sintéticos.")

    # Fallback ou modo sintético explícito
    n_treino, n_teste = 3640, 1095
    df_sint = gerar_dados_sinteticos(
        n_amostras=n_treino + n_teste,
        preco_inicial=cfg_sintetico["preco_inicial"],
        volatilidade=cfg_sintetico["volatilidade"],
        seed=cfg_sintetico["seed"]
    )
    print(f"  [Sintético] {ticker_yf}: {n_treino} treino | {n_teste} teste")
    return df_sint.iloc[:n_treino].reset_index(drop=True), df_sint.iloc[n_treino:].reset_index(drop=True)


# ==============================================================================
# 2. DISCRETIZAÇÃO HIERÁRQUICA
# ==============================================================================
def discretizar_hierarquico(df_treino, df_teste, n_intervalos):
    treino_disc = pd.DataFrame(index=df_treino.index)
    teste_disc = pd.DataFrame(index=df_teste.index)
    categorias = list(range(n_intervalos))

    for col in df_treino.columns:
        valores_treino = df_treino[col].values.reshape(-1, 1)
        clusterer = AgglomerativeClustering(n_clusters=n_intervalos, linkage="ward")
        labels = clusterer.fit_predict(valores_treino)

        medias_clusters = sorted([valores_treino[labels == i].mean() for i in range(n_intervalos)])
        bordas = [(medias_clusters[i] + medias_clusters[i + 1]) / 2.0 for i in range(n_intervalos - 1)]

        treino_disc[col] = pd.Categorical(np.digitize(df_treino[col].values, bordas), categories=categorias)
        teste_disc[col] = pd.Categorical(np.digitize(df_teste[col].values, bordas), categories=categorias)

    return treino_disc, teste_disc


# ==============================================================================
# 3. APRENDIZADO DE ESTRUTURA (K2 E PC) E PARÂMETROS
# ==============================================================================
def ajustar_parametros(estrutura, df_treino, n_intervalos):
    modelo = BayesianNetwork()
    modelo.add_nodes_from(df_treino.columns)
    modelo.add_edges_from(estrutura.edges())

    state_names = {col: list(range(n_intervalos)) for col in df_treino.columns}

    estimador = BayesianEstimator(modelo, df_treino, state_names=state_names)
    cpds = estimador.get_parameters(prior_type="K2", n_jobs=1)
    modelo.add_cpds(*cpds)

    return modelo


def treinar_rede_k2(df_treino, n_intervalos):
    ordem = ["Volume", "Abertura", "Menor_Preco", "Maior_Preco", "Fechamento"]
    black_list = [
        (ordem[j], ordem[i])
        for i in range(len(ordem))
        for j in range(i + 1, len(ordem))
    ]
    hc = HillClimbSearch(df_treino)

    try:
        from pgmpy.estimators import ExpertKnowledge
        conhecimento = ExpertKnowledge(forbidden_edges=black_list)
        estrutura = hc.estimate(
            scoring_method="k2",
            expert_knowledge=conhecimento,
            max_indegree=2,
            show_progress=False
        )
    except ImportError:
        estrutura = hc.estimate(
            scoring_method="k2",
            black_list=black_list,
            max_indegree=2,
            show_progress=False
        )

    return ajustar_parametros(estrutura, df_treino, n_intervalos)


def treinar_rede_pc(df_treino, n_intervalos):
    pc = PC(df_treino)
    estrutura = pc.estimate(
        variant="stable",
        ci_test="chi_square",
        significance_level=0.05,
        show_progress=False
    )
    if hasattr(estrutura, "to_dag"):
        estrutura = estrutura.to_dag()
    return ajustar_parametros(estrutura, df_treino, n_intervalos)


# ==============================================================================
# 4. INFERÊNCIA EXATA E AVALIAÇÃO
# ==============================================================================
def avaliar_modelo(modelo, df_teste, alvo="Fechamento"):
    inferencia = VariableElimination(modelo)
    colunas_evidencia = [c for c in df_teste.columns if c != alvo]

    acertos = 0
    confiancas = []

    for _, linha in df_teste.iterrows():
        evidencia = {col: int(linha[col]) for col in colunas_evidencia}
        resultado = inferencia.query(
            variables=[alvo],
            evidence=evidencia,
            show_progress=False
        )
        probs = resultado.values
        estado_previsto = int(np.argmax(probs))
        confianca_prevista = float(np.max(probs))

        if estado_previsto == int(linha[alvo]):
            acertos += 1
        confiancas.append(confianca_prevista)

    taxa_acerto = (acertos / len(df_teste)) * 100
    confianca_media = np.mean(confiancas) * 100
    return taxa_acerto, confianca_media


# ==============================================================================
# 5. VISUALIZAÇÃO GRÁFICA DOS DAGs COM NETWORKX
# ==============================================================================
def plotar_comparacao_dags(modelo_pc, modelo_k2, ticker, n_intervalos, salvar_arquivo=True):
    """Desenha lado a lado as estruturas de Rede Bayesiana induzidas por PC e K2."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Layout fixo em círculo/pentágono para facilitar a comparação visual direta
    pos = {
        "Volume": (0.0, 1.0),
        "Abertura": (-1.0, 0.2),
        "Menor_Preco": (-0.6, -0.9),
        "Maior_Preco": (0.6, -0.9),
        "Fechamento": (1.0, 0.2),
    }

    pares = [
        (axes[0], modelo_pc, f"Algoritmo PC — {ticker} ({n_intervalos} intervalos)", "#4C72B0"),
        (axes[1], modelo_k2, f"Algoritmo K2 — {ticker} ({n_intervalos} intervalos)", "#DD8452"),
    ]

    for ax, modelo, titulo, cor_no in pares:
        G = nx.DiGraph()
        G.add_nodes_from(modelo.nodes())
        G.add_edges_from(modelo.edges())

        # Destaca a variável alvo ('Fechamento') com cor diferenciada
        cores_nos = ["#55A868" if no == "Fechamento" else cor_no for no in G.nodes()]

        nx.draw(
            G,
            pos=pos,
            ax=ax,
            with_labels=True,
            node_color=cores_nos,
            node_size=3200,
            font_size=9,
            font_weight="bold",
            font_color="white",
            edge_color="#333333",
            width=1.8,
            arrows=True,
            arrowsize=20,
            connectionstyle="arc3,rad=0.12"
        )
        ax.set_title(titulo, fontsize=11, fontweight="bold", pad=12)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.3, 1.35)
        ax.axis("off")

    plt.suptitle(
        "Comparação Estrutural das Redes Bayesianas (Verde = Variável Alvo)",
        fontsize=12,
        y=0.98
    )
    plt.tight_layout()

    if salvar_arquivo:
        nome_arquivo = f"dag_{ticker}_{n_intervalos}_intervalos.png"
        plt.savefig(nome_arquivo, dpi=300, bbox_inches="tight")
        print(f"  [Gráfico salvo] {nome_arquivo}")

    plt.close(fig)


# ==============================================================================
# 6. EXECUÇÃO PRINCIPAL
# ==============================================================================
def executar_simulacao(usar_dados_reais=True, intervalo_plot_dag=20):
    acoes = {
        "PETR3": {"ticker_yf": "PETR3.SA", "preco_inicial": 22.0, "volatilidade": 0.015, "seed": 10},
        "VIVT3": {"ticker_yf": "VIVT3.SA", "preco_inicial": 45.0, "volatilidade": 0.022, "seed": 20},
        "EMBR3": {"ticker_yf": "EMBR3.SA", "preco_inicial": 18.0, "volatilidade": 0.030, "seed": 30},
    }

    intervalos_lista = [20, 30, 40, 50, 60]
    resultados = []

    for nome, cfg in acoes.items():
        print(f"\nProcessando ação {nome}...")
        df_treino, df_teste = obter_dados_acao(
            cfg["ticker_yf"],
            cfg,
            usar_dados_reais=usar_dados_reais
        )

        for k in intervalos_lista:
            treino_disc, teste_disc = discretizar_hierarquico(df_treino, df_teste, k)

            modelo_pc = treinar_rede_pc(treino_disc, k)
            acerto_pc, conf_pc = avaliar_modelo(modelo_pc, teste_disc)

            modelo_k2 = treinar_rede_k2(treino_disc, k)
            acerto_k2, conf_k2 = avaliar_modelo(modelo_k2, teste_disc)

            # Gera e salva a imagem comparativa dos DAGs no intervalo escolhido (ex.: 20)
            if k == intervalo_plot_dag:
                print(f"  Arestas PC ({k} int.): {list(modelo_pc.edges())}")
                print(f"  Arestas K2 ({k} int.): {list(modelo_k2.edges())}")
                plotar_comparacao_dags(modelo_pc, modelo_k2, nome, k)

            resultados.append({
                "Ação": nome,
                "Intervalos": k,
                "Acertos PC (%)": round(acerto_pc, 2),
                "Confiança PC (%)": round(conf_pc, 2),
                "Acertos K2 (%)": round(acerto_k2, 2),
                "Confiança K2 (%)": round(conf_k2, 2),
            })
            print(
                f"  Intervalos={k:2d} | PC: {acerto_pc:5.1f}% (conf {conf_pc:5.1f}%) "
                f"| K2: {acerto_k2:5.1f}% (conf {conf_k2:5.1f}%)"
            )

    df_res = pd.DataFrame(resultados)

    print("\n" + "=" * 78)
    print("TABELA COMPARATIVA (Reprodução da Tabela 1 do Artigo)")
    print("=" * 78)
    print(df_res.to_string(index=False))
    print("-" * 78)
    print(f"Média PC -> Acertos: {df_res['Acertos PC (%)'].mean():.2f}% | Confiança: {df_res['Confiança PC (%)'].mean():.2f}%")
    print(f"Média K2 -> Acertos: {df_res['Acertos K2 (%)'].mean():.2f}% | Confiança: {df_res['Confiança K2 (%)'].mean():.2f}%")

    # Salva a tabela em CSV para facilitar a importação no relatório
    df_res.to_csv("resultados_tabela1.csv", index=False)
    print("\n[Tabela salva] resultados_tabela1.csv")

    # Gráfico de desempenho (trade-off granularidade vs. acerto)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, nome in zip(axes, acoes.keys()):
        sub = df_res[df_res["Ação"] == nome]
        ax.plot(sub["Intervalos"], sub["Acertos PC (%)"], marker="o", linewidth=2, label="PC")
        ax.plot(sub["Intervalos"], sub["Acertos K2 (%)"], marker="s", linewidth=2, label="K2")
        ax.set_title(nome, fontweight="bold")
        ax.set_xlabel("Nº de Intervalos de Discretização")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    axes[0].set_ylabel("Taxa de Acertos (%)")
    plt.suptitle("Desempenho K2 vs. PC por Granularidade de Discretização", fontweight="bold")
    plt.tight_layout()
    plt.savefig("desempenho_k2_vs_pc.png", dpi=300, bbox_inches="tight")
    print("[Gráfico salvo] desempenho_k2_vs_pc.png")
    plt.show()


if __name__ == "__main__":
    # Altere para False caso queira forçar o uso dos dados sintéticos
    executar_simulacao(usar_dados_reais=True, intervalo_plot_dag=20)