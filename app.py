Python
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI株スクリーニング",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📈 全銘柄 AIスクリーニング")
st.caption("毎日16:00自動更新 ｜ ファンダメンタル・テクニカル・成長性 総合評価")


@st.cache_data(ttl=1800)
def load_data():
    try:
        return pd.read_csv("latest_ranking.csv")
    except Exception:
        return pd.DataFrame()


df = load_data()

if df.empty:
    st.info(
        "💡 現在データを作成中、または初回バッチ未実行です。GitHub Actionsから手動実行してください。"
    )
else:
    with st.expander("🔍 条件で絞り込む・検索", expanded=False):
        market_filter = st.multiselect(
            "市場区分",
            options=df["market"].unique(),
            default=df["market"].unique(),
        )
        min_div = st.slider("最低配当利回り (%)", 0.0, 8.0, 1.5, step=0.5)
        max_per = st.slider("上限PER (倍)", 5.0, 100.0, 30.0, step=5.0)
        search_kw = st.text_input("銘柄名・コード検索", "")

    filtered_df = df[
        (df["market"].isin(market_filter))
        & (df["dividend_yield"] >= min_div)
        & (df["per"] <= max_per)
    ]

    if search_kw:
        filtered_df = filtered_df[
            filtered_df["name"].str.contains(search_kw, case=False, na=False)
            | filtered_df["ticker"].astype(str).str.contains(search_kw)
        ]

    st.subheader(f"🏆 本日の結果 (全{len(filtered_df)}件)")

    display_df = filtered_df[
        [
            "rank",
            "name",
            "ticker",
            "price",
            "composite_score",
            "dividend_yield",
            "per",
            "pbr",
            "sma25_dev",
        ]
    ].copy()

    display_df.columns = [
        "順位",
        "銘柄名",
        "コード",
        "株価",
        "総合スコア",
        "配当利回り(%)",
        "PER",
        "PBR",
        "25日乖離(%)",
    ]

    st.dataframe(
        display_df.style.background_gradient(
            cmap="YlGnBu", subset=["総合スコア"]
        ).format(
            {
                "株価": "¥{:,.0f}",
                "総合スコア": "{:.2f}",
                "配当利回り(%)": "{:.2f}%",
                "PER": "{:.1f}",
                "PBR": "{:.2f}",
                "25日乖離(%)": "{:+.1f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=600,
    )
