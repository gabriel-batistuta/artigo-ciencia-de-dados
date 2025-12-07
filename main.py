#!/usr/bin/env python3
"""
analyze_chronic_absence_no_args.py  (versão corrigida)

Corrige erro "Pandas data cast to numpy dtype of object" ao ajustar Logit.
"""
import os
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)

# ------------------------- CONFIGURE AQUI -------------------------
PARAMS = {
    "csv_path": "data/student_monnitoring_data.csv",
    "output_dir": "outputs",
    "random_state": 42,
    "test_size": 0.2,
}
# -----------------------------------------------------------------

OUTPUT_DIR = PARAMS["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------- Helpers -------------------------

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Student ID','Date']).reset_index(drop=True)

    # attendance -> absent
    df['Attendance Status'] = df['Attendance Status'].astype(str)
    df['absent'] = (df['Attendance Status'].str.lower() == 'absent').astype(int)

    # rolling 7-window per student (contagem de faltas)
    df['absent_7sum'] = df.groupby('Student ID')['absent'].rolling(window=7, min_periods=1).sum().reset_index(level=0, drop=True)
    df['chronic_absence'] = (df['absent_7sum'] >= 3).astype(int)

    # lags
    df['absent_lag1'] = df.groupby('Student ID')['absent'].shift(1).fillna(0).astype(int)
    df['absent_lag2'] = df.groupby('Student ID')['absent'].shift(2).fillna(0).astype(int)

    # biometrics
    if 'Sleep Hours' in df.columns:
        df['sleep_hours'] = pd.to_numeric(df['Sleep Hours'], errors='coerce')
        df['mean_sleep'] = df.groupby('Student ID')['sleep_hours'].transform('mean')
        df['sleep_diff'] = df['sleep_hours'] - df['mean_sleep']
        df['sleep_3mean'] = df.groupby('Student ID')['sleep_hours'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    else:
        df['sleep_hours']=np.nan; df['mean_sleep']=np.nan; df['sleep_diff']=np.nan; df['sleep_3mean']=np.nan

    if 'Stress Level (GSR)' in df.columns:
        df['stress'] = pd.to_numeric(df['Stress Level (GSR)'], errors='coerce')
        df['stress_7mean'] = df.groupby('Student ID')['stress'].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    else:
        df['stress']=np.nan; df['stress_7mean']=np.nan

    df['mood'] = pd.to_numeric(df['Mood Score'], errors='coerce') if 'Mood Score' in df.columns else np.nan
    df['anxiety'] = pd.to_numeric(df['Anxiety Level'], errors='coerce') if 'Anxiety Level' in df.columns else np.nan

    # weekday and dummies (force numeric)
    df['weekday'] = df['Date'].dt.weekday
    wdummies = pd.get_dummies(df['weekday'].astype(int), prefix='wd', drop_first=True).astype(int)
    df = pd.concat([df, wdummies], axis=1)

    return df

def _drop_constant_columns(X_df):
    """Remove colunas com variância zero."""
    nunique = X_df.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X_df = X_df.drop(columns=const_cols)
    return X_df, const_cols

def safe_logit_with_cluster(df, y_col, X_cols, cluster_col='Student ID'):
    """
    Ajusta Logit (statsmodels) com robust cluster SE.
    Coerção de tipos e remoção de constantes implementadas.
    """
    result = {}
    # copy relevant cols
    cols = [y_col] + X_cols + [cluster_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        result['error'] = f"Colunas faltando no DataFrame: {missing}"
        return result

    df2 = df[cols].copy()

    # converter y e X para numéricos quando possível
    df2[y_col] = pd.to_numeric(df2[y_col], errors='coerce')
    for c in X_cols:
        df2[c] = pd.to_numeric(df2[c], errors='coerce')

    # drop rows com NaN em y ou X (cluster pode permanecer)
    df2 = df2.dropna(subset=[y_col] + X_cols).copy()
    if df2.shape[0] == 0:
        result['error'] = "Sem observações após conversão para numérico e dropna."
        return result

    y = df2[y_col].astype(int)

    X_df = df2[X_cols].copy()
    # remover constantes (zero variância)
    X_df_clean, const_cols = _drop_constant_columns(X_df)
    if const_cols:
        result['warning_drop_constant'] = const_cols

    if X_df_clean.shape[1] == 0:
        result['error'] = "Nenhum preditor válido após remoção de constantes."
        return result

    # adicionar constante (intercept)
    X = sm.add_constant(X_df_clean, has_constant='add')

    # garantir tipos numpy float
    X = X.astype(float)

    try:
        model = sm.Logit(y, X).fit(disp=False, method='newton')
        cov = cov_cluster(model, df2[cluster_col].values)
        se = np.sqrt(np.diag(cov))
        params = model.params

        table = []
        for i, name in enumerate(params.index):
            coef = float(params[name])
            se_i = float(se[i]) if i < len(se) else np.nan
            z = coef / se_i if (not np.isnan(se_i) and se_i>0) else np.nan
            from scipy.stats import norm
            pval = 2*(1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            OR = float(np.exp(coef))
            ci_lo = float(np.exp(coef - 1.96*se_i)) if not np.isnan(se_i) else np.nan
            ci_hi = float(np.exp(coef + 1.96*se_i)) if not np.isnan(se_i) else np.nan
            table.append({'param': name, 'coef': coef, 'se_cluster': se_i, 'z': z, 'p': pval, 'OR': OR, 'OR_lo': ci_lo, 'OR_hi': ci_hi})
        result['table'] = pd.DataFrame(table)
        result['model'] = model
        return result
    except Exception as e:
        result['error'] = str(e)
        return result

# ------------------------- Pipeline principal -------------------------

def run_pipeline(params):
    csv_path = params['csv_path']; out = params['output_dir']
    os.makedirs(out, exist_ok=True)

    print("Carregando CSV:", csv_path)
    df = load_csv(csv_path)
    print("Linhas originais:", len(df))

    df = preprocess(df)
    df.to_csv(os.path.join(out, "processed_data.csv"), index=False)
    print("Pré-processamento concluído. Linhas após preprocess:", len(df))

    TARGET = 'chronic_absence'
    FEATURES_LAGS = ['absent_lag1','absent_lag2']
    weekday_cols = [c for c in df.columns if c.startswith('wd_')]
    feat_for_model = [f for f in (FEATURES_LAGS + weekday_cols) if f in df.columns]

    print("Ajustando Logit (cluster-robust) para chronic_absence com lags e weekday...")
    res = safe_logit_with_cluster(df, TARGET, feat_for_model, cluster_col='Student ID')
    if 'error' in res:
        print("Erro no ajuste do modelo:", res['error'])
        with open(os.path.join(out, "model_chronic_logit_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Erro no ajuste do modelo: " + res.get('error','') + "\n")
            if 'warning_drop_constant' in res:
                f.write("Const cols removidas: " + str(res['warning_drop_constant']) + "\n")
    else:
        df_table = res['table']
        df_table.to_csv(os.path.join(out, "chronic_logit_table.csv"), index=False)
        with open(os.path.join(out, "model_chronic_logit_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Modelo Logit (chronic_absence ~ absent_lag1 + absent_lag2 + weekday dummies)\n\n")
            f.write(df_table.to_string(index=False))
        print("Modelo ajustado com sucesso. Tabela salva em outputs/chronic_logit_table.csv")

    # Treinar sklearn logreg para AUC (mesma lógica anterior)
    print("Treinando classificador LogReg (sklearn) para avaliar AUC...")
    df_model = df[[TARGET] + feat_for_model].copy()
    # converter numeric e dropar NA
    for c in feat_for_model:
        df_model[c] = pd.to_numeric(df_model[c], errors='coerce')
    df_model = df_model.dropna()
    if df_model[TARGET].nunique() < 2 or df_model.shape[0] < 50:
        auc = None
        print("Dados insuficientes/variabilidade insuficiente para treinar classificador.")
    else:
        X = df_model[feat_for_model].values
        y = df_model[TARGET].astype(int).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.get('test_size',0.2), random_state=params.get('random_state',42), stratify=y)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_prob)
        with open(os.path.join(out, "sklearn_auc.txt"), "w", encoding="utf-8") as f:
            f.write(f"AUC (LogisticRegression) para chronic_absence: {auc:.4f}\n")
        print(f"AUC (LogisticRegression) = {auc:.4f}")

    # Summário em PT-BR (similar ao anterior)
    summary_text_lines = []
    summary_text_lines.append("Resumo em Português (PT-BR) — análise automática\n")
    summary_text_lines.append("Hipóteses testadas:\n- Histórico de faltas (absent_lag1, absent_lag2) -> chronic_absence\n- Dia da semana (weekday) -> chronic_absence\n")
    if 'table' in res:
        df_table = res['table']
        for param in ['absent_lag1','absent_lag2']:
            if param in df_table['param'].values:
                row = df_table[df_table['param']==param].iloc[0]
                summary_text_lines.append(f"Parâmetro {param}: coef={row['coef']:.4f}, SE_cluster={row['se_cluster']:.4f}, z={row['z']:.3f}, p={row['p']:.4g}, OR={row['OR']:.3f} (95%CI {row['OR_lo']:.3f}–{row['OR_hi']:.3f})")
            else:
                summary_text_lines.append(f"Parâmetro {param}: não encontrado no resumo.")
        wd_rows = df_table[df_table['param'].isin(weekday_cols)]
        if not wd_rows.empty:
            summary_text_lines.append("\nEfeito por dia da semana (dummies wd_* vs baseline):")
            for _, r in wd_rows.iterrows():
                summary_text_lines.append(f"  {r['param']}: coef={r['coef']:.4f}, p={r['p']:.4g}, OR={r['OR']:.3f} (95%CI {r['OR_lo']:.3f}–{r['OR_hi']:.3f})")
    else:
        summary_text_lines.append("O modelo Logit não foi ajustado corretamente; ver model_chronic_logit_summary.txt")

    # Gráficos (Português)
    plt.figure(figsize=(7,4))
    rates = df.groupby('weekday')['chronic_absence'].mean()
    weekday_names = {0:'Segunda',1:'Terça',2:'Quarta',3:'Quinta',4:'Sexta',5:'Sábado',6:'Domingo'}
    x = [weekday_names.get(i,i) for i in rates.index]
    plt.bar(x, rates.values)
    plt.title("Taxa de ausência crônica por dia da semana")
    plt.xlabel("Dia da semana")
    plt.ylabel("Taxa de ausência crônica (≥3 faltas em 7 dias)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "taxa_chronic_por_dia_da_semana.png"))
    plt.close()

    if 'table' in res:
        sel = df_table[df_table['param'].isin(['absent_lag1','absent_lag2'])]
        if not sel.empty:
            plt.figure(figsize=(6,3))
            or_vals = sel['OR'].values; lo = sel['OR_lo'].values; hi = sel['OR_hi'].values; labels = sel['param'].values
            y_pos = np.arange(len(labels))
            plt.errorbar(or_vals, y_pos, xerr=[or_vals-lo, hi-or_vals], fmt='o')
            plt.yticks(y_pos, labels); plt.axvline(1.0, color='k', linestyle='--', linewidth=0.8)
            plt.xlabel('Odds Ratio (OR)'); plt.title('Odds Ratio para lags de falta → ausência crônica')
            plt.tight_layout(); plt.savefig(os.path.join(out, "or_lags_chronic.png")); plt.close()

    if 'auc' in locals() and auc is not None:
        fpr, tpr, thr = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6,5)); plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}'); plt.plot([0,1],[0,1], linestyle='--')
        plt.xlabel('Falso Positivo'); plt.ylabel('Verdadeiro Positivo'); plt.title('Curva ROC — Previsão de ausência crônica')
        plt.legend(loc='lower right'); plt.tight_layout(); plt.savefig(os.path.join(out, "roc_chronic.png")); plt.close()

    plt.figure(figsize=(7,4))
    s = df[['absent_lag1','absent_lag2']].sum()
    s.plot(kind='bar'); plt.title("Contagem de registros com absent_lag1 e absent_lag2 = 1"); plt.xlabel("Lag"); plt.ylabel("Contagem")
    plt.tight_layout(); plt.savefig(os.path.join(out, "contagem_lags.png")); plt.close()

    with open(os.path.join(out, "summary_ptbr.txt"), "w", encoding="utf-8") as f:
        for line in summary_text_lines:
            f.write(line + "\n")
        if 'auc' in locals() and auc is not None:
            f.write(f"\nAUC (classificador para chronic_absence com lags+weekday): {auc:.4f}\n")
        else:
            f.write("\nAUC não calculada.\n")

    print("Pronto. Outputs salvos em:", out)
    return out

if __name__ == "__main__":
    run_pipeline(PARAMS)
