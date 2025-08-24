import pkgutil, streamlit as st
installed = {m.name for m in pkgutil.iter_modules()}
st.write("🔎 Installed packages include plotly?", "plotly" in installed)

import os
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
from plotly import express as px

from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression


# ------------- Page config -------------
st.set_page_config(page_title="Mini Genetics EDA + LLM", page_icon="🧬", layout="wide")
st.title("🧬 Mini Genetics EDA + LLM Assistant")
st.caption("Lightweight EDA + tiny LLM Q&A + minimal ML baseline (dataset capped to ~1,000 rows).")


# ------------- Small LLM client (Hugging Face Inference API) -------------
class TinyLLM:
    """
    Uses a very small chat model on HF Inference API.
    Default: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (fast, free-tier friendly).
    Set environment variable HF_TOKEN with your HF API token.
    """
    def __init__(self, model: str = None, timeout: int = 45):
        self.model = model or os.environ.get("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.hf_token = os.environ.get("HF_TOKEN")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.hf_token)

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        if not self.available():
            # Offline fallback: basic string search in the context we gave it
            lines = [ln for ln in system_prompt.splitlines() if any(tok in ln.lower()
                     for tok in user_prompt.lower().split()[:5])]
            if not lines:
                lines = system_prompt.splitlines()[:15]
            return "LLM offline mode (set HF_TOKEN to enable). Context:\n\n" + "\n".join(lines[:40])

        headers = {"Authorization": f"Bearer {self.hf_token}"}
        # Simple prompt format; TinyLlama is chat-tuned but accepts raw prompts
        prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.2}}
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            json=payload,
            headers=headers,
            timeout=self.timeout
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
            return text.split("Assistant:", 1)[-1].strip()
        if isinstance(data, dict) and "error" in data:
            return f"[HF Inference Error] {data['error']}"
        return str(data)


# ------------- Data helpers -------------
def load_csv(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=";")

def cap_rows(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df.reset_index(drop=True)

def basic_eda(df: pd.DataFrame) -> Dict:
    eda = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isna().sum().to_dict(),
        "null_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "nunique": df.nunique(dropna=False).to_dict(),
        "desc_num": df.select_dtypes(include=np.number).describe().T,
        "desc_cat": df.select_dtypes(exclude=np.number).describe(include="all").T
    }
    return eda

def infer_task(y: pd.Series) -> str:
    # If numeric with many unique values => regression; else classification
    if y.dtype.kind in "ifu":
        return "classification" if y.nunique(dropna=True) <= 10 else "regression"
    return "classification"

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = list(X.select_dtypes(include=np.number).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    return pre


# ------------- Sidebar: data + LLM settings -------------
with st.sidebar:
    st.header("1) Upload CSV or use demo")
    up = st.file_uploader("CSV file", type=["csv"])
    use_demo = st.checkbox("Use built-in demo data", value=not bool(up))

    st.header("2) LLM (very small)")
    st.write("Provider: **Hugging Face Inference** (TinyLlama 1.1B by default).")
    st.write("Set `HF_TOKEN` in secrets or env. You can change model via `HF_MODEL`.")
    st.caption("Examples of tiny chat models: TinyLlama/TinyLlama-1.1B-Chat-v1.0, tiiuae/falcon-1b-instruct")

# ------------- Demo data (1000 rows by default) -------------
def demo_data(n=1000, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genes = ["BRCA1","BRCA2","TP53","APOE","CFTR","HBB"]
    impacts = ["synonymous","missense","nonsense","frameshift"]
    zyg = ["het","hom"]

    df = pd.DataFrame({
        "sample_id": [f"S{10000+i}" for i in range(n)],
        "gene": rng.choice(genes, size=n),
        "impact": rng.choice(impacts, size=n, p=[0.45,0.4,0.1,0.05]),
        "zygosity": rng.choice(zyg, size=n, p=[0.8,0.2]),
        "allele_frequency": np.clip(rng.normal(0.02, 0.015, size=n), 0, 0.3),
        "coverage": np.clip(rng.normal(120, 25, size=n), 20, 300).round(0),
        "age": np.clip(rng.normal(45, 16, size=n), 0, 95).round(0),
        "sex": rng.choice(["F","M"], size=n, p=[0.52,0.48])
    })
    df["label"] = np.where(
        (df["impact"].isin(["missense","nonsense","frameshift"])) &
        (df["allele_frequency"] < 0.01) &
        (df["coverage"] >= 60),
        "likely_pathogenic", "other"
    )

    # Inject a little missingness
    miss_idx = rng.choice(n, size=int(0.02*n), replace=False)
    df.loc[miss_idx, "allele_frequency"] = np.nan
    return df

# ------------- Load data -------------
if up is not None:
    df = load_csv(up)
elif use_demo:
    df = demo_data(1000)
else:
    st.info("Upload a CSV or use the demo data.")
    st.stop()

df = cap_rows(df, 1000)
st.success(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head(), use_container_width=True)


# ------------- EDA -------------
st.header("2) Quick EDA")

eda = basic_eda(df)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Missingness")
    miss_tbl = pd.DataFrame({
        "column": list(eda["null_counts"].keys()),
        "nulls": list(eda["null_counts"].values()),
        "null_%": list(eda["null_pct"].values())
    }).sort_values("null_%", ascending=False)
    st.dataframe(miss_tbl, use_container_width=True, height=240)
    fig_null = px.bar(miss_tbl, x="column", y="null_%", title="% missing by column")
    st.plotly_chart(fig_null, use_container_width=True)

with col2:
    st.subheader("Dtypes & Unique vals")
    dtype_tbl = pd.DataFrame({
        "column": list(eda["dtypes"].keys()),
        "dtype": list(eda["dtypes"].values()),
        "nunique": list(eda["nunique"].values())
    }).sort_values("nunique", ascending=False)
    st.dataframe(dtype_tbl, use_container_width=True, height=240)

st.subheader("Descriptive (numeric)")
num_desc = eda["desc_num"]
if not num_desc.empty:
    st.dataframe(num_desc, use_container_width=True, height=260)
else:
    st.write("No numeric columns.")

st.subheader("Distributions (quick)")
num_cols = df.select_dtypes(include=np.number).columns.tolist()[:6]
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()[:6]

if num_cols:
    with st.expander("Numeric histograms (top 6)"):
        for c in num_cols:
            st.plotly_chart(px.histogram(df, x=c, nbins=40, title=c), use_container_width=True)

if cat_cols:
    with st.expander("Categorical bars (top 6)"):
        for c in cat_cols:
            vc = df[c].astype(str).value_counts(dropna=False).head(20).reset_index()
            vc.columns = [c, "count"]
            st.plotly_chart(px.bar(vc, x=c, y="count", title=c), use_container_width=True)

if len(num_cols) >= 2:
    st.subheader("Small correlation heatmap")
    corr = df[num_cols].corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=False, aspect="auto", title="Correlation (numeric subset)"),
                    use_container_width=True)


# ------------- Minimal ML baseline -------------
st.header("3) Tiny ML baseline")

target = st.selectbox("Target column", options=df.columns, index=(list(df.columns).index("label") if "label" in df.columns else 0))
if target:
    y = df[target]
    X = df.drop(columns=[target])
    task = infer_task(y)
    st.write(f"Detected task: **{task}**")

    pre = make_preprocessor(X)
    model = LogisticRegression(max_iter=600) if task == "classification" else LinearRegression()
    pipe = Pipeline([("prep", pre), ("model", model)])

    strat = y if (task == "classification" and y.nunique() >= 2) else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    if st.button("Train baseline"):
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        if task == "classification":
            acc = accuracy_score(y_te, pred)
            f1m = f1_score(y_te, pred, average="macro")
            st.write(f"**Accuracy:** {acc:.3f} | **F1-macro:** {f1m:.3f}")
        else:
            mae = mean_absolute_error(y_te, pred)
            rmse = mean_squared_error(y_te, pred, squared=False)
            r2 = r2_score(y_te, pred)
            st.write(f"**MAE:** {mae:.3f} | **RMSE:** {rmse:.3f} | **R²:** {r2:.3f}")


# ------------- LLM Q&A (uses EDA + small summary) -------------
st.header("4) Ask the tiny LLM")

def eda_summary_text(eda: Dict, df: pd.DataFrame, max_lines=80) -> str:
    lines = []
    lines.append(f"Shape: {eda['shape'][0]} rows x {eda['shape'][1]} cols.")
    lines.append("Dtypes: " + ", ".join(f"{k}:{v}" for k, v in eda["dtypes"].items()))
    lines.append("Missing %: " + ", ".join(f"{k}:{v}%" for k, v in eda["null_pct"].items()))
    if not eda["desc_num"].empty:
        lines.append("Numeric summary (mean/std/min/median/max for first 5):")
        for c in list(eda["desc_num"].index)[:5]:
            d = eda["desc_num"].loc[c]
            lines.append(f"- {c}: mean={d['mean']:.4f}, std={d['std']:.4f}, min={d['min']:.4f}, 50%={d['50%']:.4f}, max={d['max']:.4f}")
    return "\n".join(lines[:max_lines])

llm_context = eda_summary_text(eda, df)
llm = TinyLLM()

sys_prompt = (
    "You are a very concise assistant for genetics-related tabular data. "
    "Answer conservatively, do not give medical advice. "
    "Use the provided EDA context when answering dataset questions.\n\n"
    f"EDA CONTEXT:\n{llm_context}\n"
)

user_q = st.text_input("Question (about the dataset or basic genetics):", placeholder="Which columns have most missing values? What is allele_frequency like?")
if st.button("Ask LLM"):
    with st.spinner("Querying tiny model..."):
        ans = llm.ask(sys_prompt, user_q or "Summarize the dataset at a glance.")
    st.markdown("**Assistant:**")
    st.write(ans)
