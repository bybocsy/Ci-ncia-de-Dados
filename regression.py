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


fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, col in zip(axes.flat, ["pop_por_idade", "cv_populacao", "pct_distribuicao", "cv_pct_distribuicao"]):
    ax.hist(uf_ano[col], bins=30, color=AZUL, edgecolor="white")
    ax.set(title=col, ylabel="Frequência")
fig.suptitle(f"Histogramas - UFs, {ANO_REF}")
salvar(fig, "01_histogramas")


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