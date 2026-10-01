import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARQUIVO = "tabela_como_a_prof_quer.csv"
ANO_REF = 2025                 
AMPLITUDE_65_MAIS = 10        

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


VARIAVEIS = {
    "Variável - População (Mil pessoas)": "populacao_mil",
    "Variável - Coeficiente de variação - População (%)": "cv_populacao",
    "Variável - Distribuição percentual da população por sexo segundo grupos de idade (%)": "pct_distribuicao",
    "Variável - Coeficiente de variação - Distribuição percentual da população por sexo segundo grupos de idade (%)": "cv_pct_distribuicao",
}
NUMERICAS = list(VARIAVEIS.values())

NIVEIS = [("Grande Região", 5), ("Unidade da Federação", 27),
          ("Município (capital)", 27), ("Região Metropolitana", 20)]

REGIAO_UF = {
    "Rondônia": "Norte", "Acre": "Norte", "Amazonas": "Norte", "Roraima": "Norte",
    "Pará": "Norte", "Amapá": "Norte", "Tocantins": "Norte",
    "Maranhão": "Nordeste", "Piauí": "Nordeste", "Ceará": "Nordeste",
    "Rio Grande do Norte": "Nordeste", "Paraíba": "Nordeste", "Pernambuco": "Nordeste",
    "Alagoas": "Nordeste", "Sergipe": "Nordeste", "Bahia": "Nordeste",
    "Minas Gerais": "Sudeste", "Espírito Santo": "Sudeste", "Rio de Janeiro": "Sudeste",
    "São Paulo": "Sudeste", "Paraná": "Sul", "Santa Catarina": "Sul",
    "Rio Grande do Sul": "Sul", "Mato Grosso do Sul": "Centro-Oeste",
    "Mato Grosso": "Centro-Oeste", "Goiás": "Centro-Oeste", "Distrito Federal": "Centro-Oeste",
}

raw = pd.read_csv(ARQUIVO, sep=";", header=None, names=range(15), dtype=str, encoding="utf-8-sig")
raw = raw.dropna(axis=1, how="all")

inicios = raw.index[raw[0].isin(VARIAVEIS.keys())]              
fontes = raw.index[raw[0].str.startswith("Fonte", na=False)]     

partes = []
for ini in inicios:
    variavel = VARIAVEIS[raw.loc[ini, 0]]
    anos = raw.loc[ini + 2, 2:].ffill()                         
    sexos = raw.loc[ini + 3, 2:]
    fim = fontes[fontes > ini][0]

    bloco = raw.loc[ini + 4: fim - 1].copy()
    bloco.columns = ["territorio", "grupo_idade"] + [f"{a}|{s}" for a, s in zip(anos, sexos)]
    bloco["bloco"] = (bloco["territorio"] != bloco["territorio"].shift()).cumsum()   

    longo = bloco.melt(id_vars=["bloco", "territorio", "grupo_idade"], var_name="chave", value_name="valor")
    longo["variavel"] = variavel
    partes.append(longo)
    print(f"{variavel}: {len(bloco)} linhas x {len(anos)} colunas")

df = pd.concat(partes, ignore_index=True)


print("\nSímbolos especiais do IBGE:", df["valor"].isin(["-", "X", "...", ".."]).sum())
df["valor"] = pd.to_numeric(df["valor"].replace("-", "0").str.replace(",", ".", regex=False), errors="coerce")

df[["ano", "sexo"]] = df["chave"].str.split("|", expand=True)
df["ano"] = df["ano"].astype(int)
df["sexo"] = df["sexo"].replace({"Homens": "Homem", "Mulheres": "Mulher"})

df = df.pivot_table(index=["bloco", "territorio", "grupo_idade", "ano", "sexo"],
                    columns="variavel", values="valor", aggfunc="first").reset_index()
df.columns.name = None



blocos = df.drop_duplicates("bloco")[["bloco", "territorio"]].sort_values("bloco").reset_index(drop=True)
blocos["nivel"] = np.repeat([n for n, _ in NIVEIS], [q for _, q in NIVEIS])
blocos["regiao"] = blocos["territorio"].map(REGIAO_UF).fillna(blocos["territorio"])
df = df.merge(blocos[["bloco", "nivel", "regiao"]], on="bloco").drop(columns="bloco")


AGREGADOS = ["5 a 13 anos", "14 a 17 anos", "60 anos ou mais"]
df = df[~df["grupo_idade"].isin(AGREGADOS)].copy()

limites = df["grupo_idade"].str.extract(r"(\d+)\D*(\d+)?").astype(float)
df["amplitude"] = (limites[1] - limites[0] + 1).fillna(AMPLITUDE_65_MAIS)
df["idade_num"] = limites[0] + df["amplitude"] / 2

df["pop_por_idade"] = df["populacao_mil"] / df["amplitude"]
df["pct_por_idade"] = df["pct_distribuicao"] / df["amplitude"]

ordem = df.drop_duplicates("grupo_idade").sort_values("idade_num")["grupo_idade"].tolist()
df["grupo_idade"] = pd.Categorical(df["grupo_idade"], categories=ordem, ordered=True)
df = df[["territorio", "nivel", "regiao", "grupo_idade", "idade_num", "amplitude", "sexo", "ano"]
        + NUMERICAS + ["pop_por_idade", "pct_por_idade"]]


print("\n===== DESCRIÇÃO DA BASE =====")
print(f"Instâncias: {df.shape[0]} | Atributos: {df.shape[1]}")
print("\nTipos dos atributos:")
print(df.dtypes.to_string())
print("\nValores ausentes:")
print(df.isna().sum().to_string())
print("\nFrequência das categorias:")
for c in ["nivel", "sexo", "ano"]:
    print(f"  {c}: {df[c].value_counts().sort_index().to_dict()}")

soma = df.groupby(["nivel", "territorio", "ano", "sexo"])["pct_distribuicao"].sum()
print(f"\nSoma do pct_distribuicao ao longo das idades (por território/ano/sexo): média {soma.mean():.1f}%")

uf = df[df["nivel"] == "Unidade da Federação"]
uf_ano = uf[uf["ano"] == ANO_REF]

print(f"\nEstatística descritiva (UFs, {len(uf)} linhas):")
print(uf[NUMERICAS + ["pop_por_idade"]].describe().round(2).T.to_string())
print("\nAssimetria:")
print(uf[NUMERICAS + ["pop_por_idade"]].skew().round(2).to_string())
print("\nMédia por grande região:")
print(uf.groupby("regiao")[["pop_por_idade", "cv_populacao"]].mean().round(1).to_string())
print("\nCoeficiente de variação médio por nível territorial (confiabilidade da estimativa):")
print(df.groupby("nivel")[["cv_populacao", "cv_pct_distribuicao"]].mean().round(1).to_string())


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(f"fig_{nome}.png", dpi=150)
    plt.show()


AZUL, LARANJA = "#4C72B0", "#DD8452"
COR_REGIAO = {"Norte": "#55A868", "Nordeste": "#DD8452", "Centro-Oeste": "#8172B3",
              "Sudeste": "#4C72B0", "Sul": "#C44E52"}


inicio = uf_ano["idade_num"] - uf_ano["amplitude"] / 2
bordas = sorted(inicio.unique()) + [inicio.max() + AMPLITUDE_65_MAIS]
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist([uf_ano.loc[uf_ano["sexo"] == s, "idade_num"] for s in ["Homem", "Mulher"]], bins=bordas,
        weights=[uf_ano.loc[uf_ano["sexo"] == s, "pop_por_idade"] for s in ["Homem", "Mulher"]],
        stacked=True, color=[AZUL, LARANJA], edgecolor="white", label=["Homem", "Mulher"])
ax.set_xticks(bordas)
ax.set(title=f"Distribuição da população por idade - Brasil (soma das UFs), {ANO_REF}",
       xlabel="Idade (anos) - largura de cada barra = faixa etária do IBGE",
       ylabel="Mil pessoas por ano de idade")
ax.text(0.99, 0.97, f"Faixa '65 anos ou mais' representada como 65-{65 + AMPLITUDE_65_MAIS - 1}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray")
ax.set_ylim(0, uf_ano.groupby("grupo_idade", observed=True)["pop_por_idade"].sum().max() * 1.15)
ax.legend(loc="upper left", ncols=2)
salvar(fig, "00_histograma_idade_populacao")



def boxplot(dados, grupo, valor, titulo, nome, log=False):
    cats = dados[grupo].drop_duplicates().sort_values().tolist()
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(cats)), 4.5))
    ax.boxplot([dados.loc[dados[grupo] == c, valor] for c in cats])
    ax.set_xticks(range(1, len(cats) + 1))
    girar = max(len(str(c)) for c in cats) > 8
    ax.set_xticklabels(cats, rotation=30 if girar else 0, ha="right" if girar else "center")
    if log:
        ax.set_yscale("log")
    ax.set(title=titulo, ylabel=valor + (" (escala log)" if log else ""))
    salvar(fig, nome)


boxplot(uf_ano, "grupo_idade", "pop_por_idade", f"População por ano de idade, por grupo de idade - {ANO_REF}",
        "02_boxplot_pop_idade", log=True)
boxplot(uf_ano, "regiao", "pop_por_idade", f"População por ano de idade, por grande região - {ANO_REF}",
        "03_boxplot_pop_regiao", log=True)
boxplot(uf_ano, "grupo_idade", "pct_por_idade", f"Distribuição percentual por ano de idade, por grupo - {ANO_REF}",
        "04_boxplot_pct_idade")
boxplot(df[df["ano"] == ANO_REF], "nivel", "cv_populacao", f"Coeficiente de variação por nível territorial - {ANO_REF}",
        "05_boxplot_cv_nivel")

q1, q3 = uf_ano["pop_por_idade"].quantile([0.25, 0.75])
outliers = uf_ano[uf_ano["pop_por_idade"] > q3 + 1.5 * (q3 - q1)]
print(f"\nOutliers de população (regra do IQR): {len(outliers)} -> {outliers['territorio'].value_counts().to_dict()}")


fig, ax = plt.subplots(figsize=(7, 4.5))
for sexo, cor in [("Homem", AZUL), ("Mulher", LARANJA)]:
    d = uf_ano[uf_ano["sexo"] == sexo]
    ax.scatter(d["idade_num"], d["pct_por_idade"], alpha=0.5, s=18, color=cor, label=sexo)
    media = d.groupby("idade_num")["pct_por_idade"].mean()
    ax.plot(media.index, media.values, color=cor, linewidth=2)
ax.set(title=f"Idade x Distribuição percentual por ano de idade - UFs, {ANO_REF}",
       xlabel="Idade (ponto médio da faixa)", ylabel="% da população do sexo, por ano de idade")
ax.legend()
salvar(fig, "06_dispersao_idade_pct")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(uf_ano["populacao_mil"], uf_ano["cv_populacao"], alpha=0.5, s=18, color=AZUL)
ax.set_xscale("log")
ax.set(title=f"População x Coeficiente de variação - UFs, {ANO_REF}",
       xlabel="População (mil pessoas, escala log)", ylabel="CV (%)")
salvar(fig, "07_dispersao_pop_cv")


corr = uf[NUMERICAS + ["pop_por_idade", "idade_num", "ano"]].assign(
    sexo_mulher=(uf["sexo"] == "Mulher").astype(int)).corr()
print("\nMatriz de correlação:")
print(corr.round(2).to_string())

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax)
ax.set_title("Matriz de correlação - UFs")
salvar(fig, "08_correlacao")


brasil = uf.groupby(["ano", "grupo_idade"], observed=True)["populacao_mil"].sum().unstack()
pct = brasil.div(brasil.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(7, 4.5))
for grupo, cor in [("0 a 4 anos", AZUL), ("65 anos ou mais", LARANJA)]:
    ax.plot(pct.index, pct[grupo], marker="o", color=cor, label=grupo)
ax.set_xticks(pct.index)
ax.set(title="Participação na população total - Brasil (soma das UFs)", xlabel="Ano", ylabel="%")
ax.legend()
salvar(fig, "09_evolucao_anos")
print("\n% da população por grupo de idade, por ano (Brasil):")
print(pct.round(1).to_string())

pir = uf_ano.pivot_table(index="grupo_idade", columns="sexo", values="pop_por_idade",
                         aggfunc="sum", observed=True)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(pir.index.astype(str), -pir["Homem"], color=AZUL, label="Homem")
ax.barh(pir.index.astype(str), pir["Mulher"], color=LARANJA, label="Mulher")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{abs(x):,.0f}"))
ax.set(title=f"Pirâmide etária - Brasil (soma das UFs), {ANO_REF}", xlabel="Mil pessoas por ano de idade")
ax.legend()
salvar(fig, "10_piramide_etaria")

print(f"\nMulheres para cada 100 homens ({ANO_REF}):")
print((pir["Mulher"] / pir["Homem"] * 100).round(1).to_string())

JOVENS = ["0 a 4 anos", "5 a 9 anos", "10 a 13 anos"]
faixas = uf_ano.groupby(["territorio", "regiao", "grupo_idade"], observed=True)["populacao_mil"].sum().unstack()
jovens, idosos = faixas[JOVENS].sum(axis=1), faixas["65 anos ou mais"]
envelhecimento = (idosos / jovens * 100).sort_values().reset_index(name="indice")
indice_brasil = idosos.sum() / jovens.sum() * 100

fig, ax = plt.subplots(figsize=(8, 7.5))
ax.barh(envelhecimento["territorio"], envelhecimento["indice"],
        color=envelhecimento["regiao"].map(COR_REGIAO))
ax.axvline(indice_brasil, color="gray", linestyle="--", linewidth=1)
ax.text(indice_brasil + 1, 0, f"Brasil: {indice_brasil:.0f}", color="gray", fontsize=8, va="center")
for y, v in enumerate(envelhecimento["indice"]):
    ax.text(v + 1, y, f"{v:.0f}", va="center", fontsize=8)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COR_REGIAO.values()],
          labels=list(COR_REGIAO), loc="center right")
ax.set(title=f"Índice de envelhecimento por UF - {ANO_REF}",
       xlabel="Pessoas com 65 anos ou mais para cada 100 crianças de 0 a 13 anos")
ax.margins(y=0.01)
salvar(fig, "11_indice_envelhecimento")
print(f"\nÍndice de envelhecimento ({ANO_REF}), Brasil = {indice_brasil:.1f}:")
print(envelhecimento.set_index("territorio")["indice"].round(1).to_string())


sexo_reg = uf_ano.pivot_table(index=["regiao", "grupo_idade"], columns="sexo", values="populacao_mil",
                              aggfunc="sum", observed=True)
razao_idade = (sexo_reg["Mulher"] / sexo_reg["Homem"] * 100).unstack("regiao")
total_reg = sexo_reg.groupby(level="regiao").sum()
razao_total = (total_reg["Mulher"] / total_reg["Homem"] * 100).sort_values(ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 2]})
ax1.bar(razao_total.index, razao_total.values, color=razao_total.index.map(COR_REGIAO))
for x, v in enumerate(razao_total.values):
    ax1.text(x, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
ax1.set_ylim(95, razao_total.max() + 2)
ax1.axhline(100, color="gray", linestyle="--", linewidth=1)
ax1.set(title="Total da população", ylabel="Mulheres para cada 100 homens")
ax1.tick_params(axis="x", rotation=20)
for regiao, cor in COR_REGIAO.items():
    ax2.plot(razao_idade.index.astype(str), razao_idade[regiao], marker="o", color=cor, label=regiao)
ax2.axhline(100, color="gray", linestyle="--", linewidth=1, label="Equilíbrio (100)")
ax2.set(title="Por grupo de idade", ylabel="Mulheres para cada 100 homens")
ax2.tick_params(axis="x", rotation=45)
ax2.legend()
fig.suptitle(f"Razão de sexo por grande região - {ANO_REF}")
salvar(fig, "12_razao_sexo_regiao")
print(f"\nMulheres para cada 100 homens por região ({ANO_REF}):")
print(razao_idade.round(1).to_string())


pop_reg = uf.groupby(["ano", "regiao"])["populacao_mil"].sum().unstack()
ano_ini, ano_fim = pop_reg.index.min(), pop_reg.index.max()
crescimento = ((pop_reg.loc[ano_fim] / pop_reg.loc[ano_ini] - 1) * 100).sort_values(ascending=False)
cresc_brasil = (pop_reg.loc[ano_fim].sum() / pop_reg.loc[ano_ini].sum() - 1) * 100

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(crescimento.index, crescimento.values, color=crescimento.index.map(COR_REGIAO))
for x, v in enumerate(crescimento.values):
    ax.text(x, v + 0.05, f"+{v:.1f}%", ha="center", fontsize=9)
ax.axhline(cresc_brasil, color="gray", linestyle="--", linewidth=1)
ax.text(len(crescimento) - 0.5, cresc_brasil + 0.05, f"Brasil: +{cresc_brasil:.1f}%",
        color="gray", fontsize=8, ha="right")
ax.set(title=f"Crescimento da população por grande região, {ano_ini}-{ano_fim}",
       ylabel=f"Variação da população {ano_ini}→{ano_fim} (%)")
ax.set_ylim(0, crescimento.max() * 1.15)
salvar(fig, "13_crescimento_regiao")
print(f"\nPopulação por região (mil pessoas) e crescimento {ano_ini}-{ano_fim}:")
print(pop_reg.T.round(0).assign(crescimento_pct=crescimento.round(2)).to_string())
