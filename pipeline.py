Python
import io
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.csv"


def get_tokyo_stock_list() -> pd.DataFrame:
    print("東証全銘柄一覧をJPXからダウンロード中...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }

    res = requests.get(JPX_LIST_URL, headers=headers)
    res.raise_for_status()

    df_jpx = pd.read_csv(io.BytesIO(res.content), encoding="cp932")

    df_jpx = df_jpx[
        df_jpx["市場・商品区分"].str.contains(
            "プライム|スタンダード|グロース", na=False
        )
    ].copy()

    df_jpx["ticker_code"] = (
        df_jpx["コード"].astype(str).str.zfill(4).str[:4]
    )
    df_jpx["yf_symbol"] = df_jpx["ticker_code"] + ".T"

    df_result = df_jpx[
        ["yf_symbol", "銘柄名", "市場・商品区分", "33業種区分"]
    ].rename(
        columns={"銘柄名": "name", "市場・商品区分": "market", "33業種区分": "sector"}
    )

    print(f"対象銘柄数: {len(df_result)} 銘柄を取得しました。")
    return df_result


def compute_zscore(series: pd.Series, invert: bool = False) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=series.index)
    z = (series - series.mean()) / std
    return -z if invert else z


def process_all_stocks(batch_size: int = 100, max_stocks: int = None):
    stock_list_df = get_tokyo_stock_list()

    if max_stocks:
        stock_list_df = stock_list_df.head(max_stocks)

    symbols = stock_list_df["yf_symbol"].tolist()
    name_map = dict(zip(stock_list_df["yf_symbol"], stock_list_df["name"]))
    market_map = dict(zip(stock_list_df["yf_symbol"], stock_list_df["market"]))

    records = []
    total = len(symbols)

    print(f"全 {total} 銘柄のスクリーニング計算を開始します...")

    for i in range(0, total, batch_size):
        chunk = symbols[i : i + batch_size]

        for symbol in chunk:
            try:
                t = yf.Ticker(symbol)
                info = t.info
                hist = t.history(period="3m")

                if hist.empty or len(hist) < 25:
                    continue

                close = hist["Close"]
                current_price = close.iloc[-1]
                sma25 = close.rolling(25).mean().iloc[-1]
                sma25_dev = (
                    ((current_price - sma25) / sma25) * 100
                    if sma25 > 0
                    else 0
                )

                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi14 = 100 - (100 / (1 + rs)).iloc[-1]

                per = info.get("forwardPE") or info.get("trailingPE") or np.nan
                pbr = info.get("priceToBook") or np.nan
                div_yield = (info.get("dividendYield") or 0) * 100
                roe = (info.get("returnOnEquity") or 0) * 100
                rev_growth = (info.get("revenueGrowth") or 0) * 100
                profit_growth = (info.get("earningsGrowth") or 0) * 100

                records.append(
                    {
                        "ticker": symbol.replace(".T", ""),
                        "name": name_map.get(symbol, info.get("shortName", symbol)),
                        "market": market_map.get(symbol, ""),
                        "price": current_price,
                        "per": per,
                        "pbr": pbr,
                        "dividend_yield": div_yield,
                        "sma25_dev": sma25_dev,
                        "rsi14": rsi14,
                        "roe": roe,
                        "rev_growth": rev_growth,
                        "profit_growth": profit_growth,
                    }
                )
            except Exception:
                continue

        time.sleep(1)

    df = pd.DataFrame(records).dropna(subset=["per", "pbr"])
    if df.empty:
        print("エラー: 有効な銘柄データがありませんでした。")
        return

    z_per = compute_zscore(df["per"], invert=True)
    z_pbr = compute_zscore(df["pbr"], invert=True)
    z_div = compute_zscore(df["dividend_yield"])
    score_fundamentals = (z_per + z_pbr + z_div) / 3

    z_dev = compute_zscore(df["sma25_dev"])
    z_rsi = compute_zscore(df["rsi14"])
    score_technical = (z_dev * 0.6) + (z_rsi * 0.4)

    z_roe = compute_zscore(df["roe"])
    z_rev = compute_zscore(df["rev_growth"])
    z_profit = compute_zscore(df["profit_growth"])
    score_growth = (z_rev * 0.4) + (z_profit * 0.4) + (z_roe * 0.2)

    df["composite_score"] = (
        (score_fundamentals * 0.35)
        + (score_technical * 0.30)
        + (score_growth * 0.35)
    )

    df["rank"] = (
        df["composite_score"].rank(ascending=False, method="min").astype(int)
    )
    df_sorted = df.sort_values("rank")

    df_sorted.to_csv("latest_ranking.csv", index=False)
    print("スクリーニング完了: 'latest_ranking.csv' に出力しました。")


if __name__ == "__main__":
    process_all_stocks()
