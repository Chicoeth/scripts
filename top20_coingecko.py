#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
top20_coingecko.py
==================
Identifica, dentro do top 200 por capitalização de mercado (CoinGecko), as 20
moedas com melhores retornos de 01/jul/2026 até hoje que estejam a no máximo
-50% de suas máximas históricas (ATH), e gera:

  output/top20_resultados.csv      tabela final (BTC na 1ª linha como referência)
  output/retornos_diarios.csv      série diária de retorno acumulado (p/ gráficos)
  output/tabela.png                tabela renderizada
  output/grafico_top5_retorno.png  linhas: destaque p/ 5 maiores retornos + BTC
  output/grafico_top5_ath.png      linhas: destaque p/ 5 mais próximas da ATH + BTC
  output/grafico_dispersao.png     dispersão: retorno × distância da ATH

Uso:
  python top20_coingecko.py                      # API pública (lento: ~6s/req)
  COINGECKO_API_KEY=CG-xxxx python top20_coingecko.py   # chave demo (mais rápido)
  python top20_coingecko.py --include-derivatives       # não excluir stables/wrapped
  python top20_coingecko.py --mock               # dados sintéticos (teste offline)

Requisitos: pip install requests pandas matplotlib
"""

import argparse
import datetime as dt
import math
import os
import sys
import time

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ---------------------------------------------------------------- configuração

START_DATE = dt.date(2026, 7, 1)          # início do período de retorno
TOP_N = 200                                # universo: top N por market cap
SELECT_N = 20                              # quantas moedas selecionar
MAX_ATH_DRAWDOWN = -50.0                   # corte: no máximo -50% da ATH

API_PUBLIC = "https://api.coingecko.com/api/v3"
API_PRO = "https://pro-api.coingecko.com/api/v3"

# Stablecoins e ativos "espelho" (wrapped/staked/pegged) excluídos por padrão —
# eles replicam outro ativo ou valem ~US$1, então poluem um ranking de retorno.
# Use --include-derivatives para mantê-los.
EXCLUDE_IDS = {
    # stablecoins / pegged
    "tether", "usd-coin", "dai", "usds", "ethena-usde", "first-digital-usd",
    "true-usd", "paypal-usd", "usdd", "frax", "gemini-dollar", "usdtb",
    "usual-usd", "ondo-us-dollar-yield", "paxos-standard", "tether-gold",
    "pax-gold", "falcon-finance", "resolv-usr", "global-dollar",
    "binance-bridged-usdt-bnb-smart-chain", "polygon-bridged-usdt-polygon",
    "arbitrum-bridged-usdt-arbitrum", "blackrock-usd-institutional-digital-liquidity-fund",
    # wrapped / staked / bridged
    "wrapped-bitcoin", "weth", "wrapped-steth", "staked-ether",
    "rocket-pool-eth", "coinbase-wrapped-btc", "coinbase-wrapped-staked-eth",
    "wrapped-eeth", "renzo-restaked-eth", "kelp-dao-restaked-eth",
    "mantle-staked-ether", "lombard-staked-btc", "tbtc", "solv-btc",
    "solv-protocol-solvbtc", "msol", "jito-staked-sol", "binance-staked-sol",
    "jupiter-staked-sol", "wrapped-beacon-eth", "wbnb", "binance-peg-weth",
    "bridged-tether", "l2-standard-bridged-weth-base", "kraken-wrapped-btc",
    "wrapped-hype", "liquid-staked-ethereum", "restaked-swell-ethereum",
    "wrapped-avax", "wrapped-tron", "staked-hype",
}
EXCLUDE_NAME_HINTS = ("wrapped", "staked", "restaked", "bridged", "usd", "stable")


# ------------------------------------------------------------------- API layer

def make_session(api_key: str | None):
    import requests
    s = requests.Session()
    s.headers.update({"Accept": "application/json",
                      "User-Agent": "top20-script/1.0"})
    base = API_PUBLIC
    if api_key:
        if api_key.startswith("CG-") and os.environ.get("COINGECKO_PRO"):
            base = API_PRO
            s.headers["x-cg-pro-api-key"] = api_key
        else:
            s.headers["x-cg-demo-api-key"] = api_key
    return s, base


def get_json(session, base, path, params, sleep_s, max_retries=6):
    """GET com backoff para 429/erros transitórios."""
    url = f"{base}{path}"
    delay = sleep_s
    for attempt in range(max_retries):
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 200:
            time.sleep(sleep_s)  # espaçamento entre chamadas (rate limit)
            return r.json()
        if r.status_code == 429:
            # Retry-After pode vir como 0; nunca re-tentar sem pausa.
            wait = max(float(r.headers.get("Retry-After") or 0), delay * 2)
            print(f"  [rate limit] aguardando {wait:.0f}s…", flush=True)
            time.sleep(wait)
            delay *= 2
            continue
        if r.status_code >= 500:
            time.sleep(delay)
            delay *= 2
            continue
        raise RuntimeError(f"HTTP {r.status_code} em {path}: {r.text[:200]}")
    raise RuntimeError(f"Falha após {max_retries} tentativas em {path}")


def fetch_top_markets(session, base, sleep_s, top_n):
    out = []
    per_page = 250
    pages = math.ceil(top_n / per_page)
    for page in range(1, pages + 1):
        out += get_json(session, base, "/coins/markets", {
            "vs_currency": "usd", "order": "market_cap_desc",
            "per_page": per_page, "page": page, "sparkline": "false",
        }, sleep_s)
    return out[:top_n]


def fetch_daily_series(session, base, sleep_s, coin_id, start_date):
    """Série de preços desde start_date, reamostrada para fechamento diário (UTC)."""
    days = (dt.date.today() - start_date).days + 2
    data = get_json(session, base, f"/coins/{coin_id}/market_chart", {
        "vs_currency": "usd", "days": days,
    }, sleep_s)
    prices = data.get("prices") or []
    if not prices:
        return None
    s = pd.Series(
        [p[1] for p in prices],
        index=pd.to_datetime([p[0] for p in prices], unit="ms", utc=True),
    )
    daily = s.resample("1D").last().dropna()
    daily.index = daily.index.date
    return daily


# ------------------------------------------------------------------- mock mode

def mock_universe(top_n, start_date, seed=42):
    """Dados sintéticos p/ testar o pipeline sem rede."""
    import random
    rnd = random.Random(seed)
    names = [("bitcoin", "BTC", "Bitcoin", -8.0)]
    for i in range(1, top_n):
        names.append((f"coin-{i}", f"C{i:03d}", f"Coin {i:03d}",
                      -rnd.uniform(2, 95)))
    markets, series = [], {}
    dates = pd.date_range(start_date, dt.date.today(), freq="D").date
    for rank, (cid, sym, name, athd) in enumerate(names, 1):
        drift = rnd.uniform(-0.004, 0.012)
        px, path = 100.0, []
        for _ in dates:
            px *= 1 + drift + rnd.gauss(0, 0.02)
            path.append(px)
        series[cid] = pd.Series(path, index=dates)
        markets.append({"id": cid, "symbol": sym.lower(), "name": name,
                        "market_cap_rank": rank, "current_price": px,
                        "ath_change_percentage": athd})
    return markets, series


# ------------------------------------------------------------------ pipeline

def is_derivative(m):
    if m["id"] in EXCLUDE_IDS:
        return True
    name = (m.get("name") or "").lower()
    return any(h in name for h in EXCLUDE_NAME_HINTS)


def build_dataset(args):
    start = dt.date.fromisoformat(args.start_date)
    if args.mock:
        markets, mock_series = mock_universe(args.top, start)
        session = base = None
    else:
        api_key = args.api_key or os.environ.get("COINGECKO_API_KEY")
        session, base = make_session(api_key)
        if not api_key:
            print("Sem chave de API: usando API pública com espaçamento de "
                  f"{args.sleep:.0f}s por chamada (defina COINGECKO_API_KEY "
                  "para acelerar — chave demo gratuita em coingecko.com/api).")
        markets = fetch_top_markets(session, base, args.sleep, args.top)
        mock_series = None
    print(f"Top {len(markets)} por market cap obtido.")

    # -- filtro ATH + exclusões
    eligible, excluded = [], []
    for m in markets:
        athd = m.get("ath_change_percentage")
        if athd is None or athd < MAX_ATH_DRAWDOWN:
            continue
        if m["id"] != "bitcoin" and not args.include_derivatives and is_derivative(m):
            excluded.append(f'{m["name"]} ({m["symbol"].upper()})')
            continue
        eligible.append(m)
    if excluded:
        print(f"Excluídos {len(excluded)} stables/wrapped/staked: "
              + ", ".join(excluded))
    print(f"{len(eligible)} moedas do top {args.top} estão a até "
          f"{-MAX_ATH_DRAWDOWN:.0f}% da ATH.")

    # -- garante BTC no conjunto (referência), mesmo se cair no filtro
    ids = {m["id"] for m in eligible}
    if "bitcoin" not in ids:
        btc = next((m for m in markets if m["id"] == "bitcoin"), None)
        if btc:
            eligible.append(btc)

    # -- séries diárias e retorno no período
    rows, series_by_id = [], {}
    for i, m in enumerate(eligible, 1):
        cid = m["id"]
        print(f"[{i}/{len(eligible)}] {m['name']}…", flush=True)
        try:
            daily = (mock_series[cid] if args.mock
                     else fetch_daily_series(session, base, args.sleep, cid, start))
        except Exception as e:
            print(f"  aviso: falhou ({e}); pulando.")
            continue
        if daily is None or len(daily) < 2:
            continue
        daily = daily[daily.index >= start]
        if len(daily) == 0 or daily.index[0] > start + dt.timedelta(days=2):
            print("  aviso: sem histórico desde 01/jul; pulando.")
            continue
        base_px = daily.iloc[0]
        if not base_px or base_px <= 0:
            continue
        ret = daily.iloc[-1] / base_px - 1
        series_by_id[cid] = (daily / base_px - 1) * 100
        rows.append({
            "id": cid, "nome": m["name"], "simbolo": m["symbol"].upper(),
            "rank_mcap": m.get("market_cap_rank"),
            "retorno_pct": ret * 100,
            "dist_ath_pct": m.get("ath_change_percentage"),
        })

    df = pd.DataFrame(rows)
    btc_row = df[df["id"] == "bitcoin"]
    alts = df[df["id"] != "bitcoin"].sort_values("retorno_pct", ascending=False)
    top = alts.head(args.select)
    final = pd.concat([btc_row, top], ignore_index=True)
    keep = set(final["id"])
    series = {cid: s for cid, s in series_by_id.items() if cid in keep}
    return final, series


# ------------------------------------------------------------------- paleta

PAL = {  # paleta validada (modo claro) — slots categóricos em ordem fixa
    "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
    "s4": "#eda100", "s5": "#e87ba4",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb",
    "context": "#c9c8c1", "wash": "#cde2fb",
}
SLOTS = [PAL["s1"], PAL["s2"], PAL["s3"], PAL["s4"], PAL["s5"]]


def style_axes(ax):
    ax.set_facecolor(PAL["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(PAL["axis"])
    ax.grid(axis="y", color=PAL["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=PAL["muted"], labelsize=9, length=0)


def fmt_pct(v, signed=True):
    s = f"{v:+.1f}%" if signed else f"{v:.1f}%"
    return s.replace(".", ",").replace("+", "+").replace("-", "−")


# ------------------------------------------------------------------- gráficos

def line_chart(final, series, highlight_ids, title, subtitle, path):
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    fig.patch.set_facecolor(PAL["surface"])
    style_axes(ax)

    id2sym = dict(zip(final["id"], final["simbolo"]))
    # contexto: demais moedas em cinza fino
    for cid, s in series.items():
        if cid in highlight_ids or cid == "bitcoin":
            continue
        ax.plot(s.index, s.values, color=PAL["context"], lw=1.0,
                alpha=0.85, zorder=1)
    # BTC: referência tracejada em tinta primária
    if "bitcoin" in series:
        s = series["bitcoin"]
        ax.plot(s.index, s.values, color=PAL["ink"], lw=2.0, ls=(0, (4, 2)),
                zorder=3, label="BTC (referência)")
    # destaques: slots categóricos em ordem fixa
    labels = []
    for slot, cid in enumerate(highlight_ids):
        s = series.get(cid)
        if s is None:
            continue
        c = SLOTS[slot]
        ax.plot(s.index, s.values, color=c, lw=2.2, zorder=4,
                label=id2sym.get(cid, cid))
        labels.append((s.index[-1], s.values[-1], id2sym.get(cid, cid), c))
    if "bitcoin" in series:
        s = series["bitcoin"]
        labels.append((s.index[-1], s.values[-1], "BTC", PAL["ink"]))

    # rótulos diretos na ponta direita, com anti-colisão simples
    labels.sort(key=lambda t: t[1])
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_gap, last_y = span * 0.04, None
    for x, y, txt, c in labels:
        y_lab = y if last_y is None else max(y, last_y + min_gap)
        last_y = y_lab
        ax.annotate(f" {txt} {fmt_pct(y)}", xy=(x, y_lab), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    fontweight="bold", color=c,
                    annotation_clip=False)
    ax.axhline(0, color=PAL["axis"], lw=1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold",
                 color=PAL["ink"], pad=18)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10,
            color=PAL["ink2"])
    ax.legend(loc="upper left", frameon=False, fontsize=9,
              labelcolor=PAL["ink2"])
    ax.margins(x=0.01)
    fig.subplots_adjust(right=0.86, left=0.07, top=0.87, bottom=0.08)
    fig.savefig(path, facecolor=PAL["surface"])
    plt.close(fig)
    print(f"OK  {path}")


def scatter_chart(final, path):
    df = final[final["id"] != "bitcoin"]
    btc = final[final["id"] == "bitcoin"].iloc[0] if (final["id"] == "bitcoin").any() else None
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=150)
    fig.patch.set_facecolor(PAL["surface"])
    style_axes(ax)
    ax.grid(axis="x", color=PAL["grid"], linewidth=0.8)

    med_x = df["retorno_pct"].median()
    med_y = df["dist_ath_pct"].median()
    # quadrante destacado: maior retorno E mais perto da ATH (acima das medianas)
    ax.axvspan(med_x, df["retorno_pct"].max() * 1.15, ymin=0, ymax=1,
               color="none")
    x_max = max(df["retorno_pct"].max(), (btc["retorno_pct"] if btc is not None else 0)) * 1.18 + 5
    x_min = min(0, df["retorno_pct"].min()) - 5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(MAX_ATH_DRAWDOWN * 1.04, 2)
    ax.fill_between([med_x, x_max], med_y, 2, color=PAL["wash"], alpha=0.45,
                    zorder=0)
    ax.text(x_max - (x_max - x_min) * 0.015, med_y + 1.2,
            "maior retorno + perto da ATH", ha="right", va="bottom",
            fontsize=9, color=PAL["s1"], fontweight="bold")
    ax.axvline(med_x, color=PAL["axis"], lw=1.0, ls=(0, (3, 3)))
    ax.axhline(med_y, color=PAL["axis"], lw=1.0, ls=(0, (3, 3)))

    ax.scatter(df["retorno_pct"], df["dist_ath_pct"], s=90,
               color=PAL["s1"], edgecolors=PAL["surface"], linewidths=2,
               zorder=4)
    for _, r in df.iterrows():
        ax.annotate(r["simbolo"], (r["retorno_pct"], r["dist_ath_pct"]),
                    xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=8, color=PAL["ink2"])
    if btc is not None:
        ax.scatter([btc["retorno_pct"]], [btc["dist_ath_pct"]], s=130,
                   marker="D", color=PAL["s2"],
                   edgecolors=PAL["surface"], linewidths=2, zorder=5)
        ax.annotate("BTC", (btc["retorno_pct"], btc["dist_ath_pct"]),
                    xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold", color=PAL["s2"])

    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel(f"Retorno desde {START_DATE.strftime('%d/%b')}",
                  fontsize=10, color=PAL["ink2"])
    ax.set_ylabel("Distância da máxima histórica (ATH)", fontsize=10,
                  color=PAL["ink2"])
    ax.set_title("Retorno × distância da ATH", loc="left", fontsize=14,
                 fontweight="bold", color=PAL["ink"], pad=18)
    ax.text(0, 1.02, "Top 20 retornos (jul→hoje) entre moedas a até −50% da ATH · "
            "linhas tracejadas = medianas", transform=ax.transAxes,
            fontsize=10, color=PAL["ink2"])
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.09)
    fig.savefig(path, facecolor=PAL["surface"])
    plt.close(fig)
    print(f"OK  {path}")


def table_png(final, path, today):
    df = final.copy()
    df["retorno"] = df["retorno_pct"].map(fmt_pct)
    df["dist_ath"] = df["dist_ath_pct"].map(lambda v: fmt_pct(v, signed=True))
    cols = ["nome", "simbolo", "retorno", "dist_ath"]
    header = ["Moeda", "Símbolo",
              f"Retorno 01/07–{today.strftime('%d/%m')}",
              "Dist. da ATH"]
    n = len(df)
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * (n + 2)), dpi=150)
    fig.patch.set_facecolor(PAL["surface"])
    ax.axis("off")
    tbl = ax.table(cellText=df[cols].values, colLabels=header,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PAL["grid"])
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#f0efec")
            cell.set_text_props(fontweight="bold", color=PAL["ink"])
        elif r == 1:  # BTC (referência)
            cell.set_facecolor(PAL["wash"])
            cell.set_text_props(color=PAL["ink"])
        else:
            cell.set_facecolor(PAL["surface"])
            cell.set_text_props(color=PAL["ink2"])
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
    ax.set_title(f"Top {len(df)-1} retornos (01/07/2026 → "
                 f"{today.strftime('%d/%m/%Y')}) · até −50% da ATH · "
                 "BTC como referência",
                 loc="left", fontsize=11, fontweight="bold", color=PAL["ink"])
    fig.savefig(path, facecolor=PAL["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"OK  {path}")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-key", default=None, help="chave CoinGecko (ou env COINGECKO_API_KEY)")
    ap.add_argument("--sleep", type=float, default=None,
                    help="segundos entre chamadas (padrão: 6 sem chave, 2.1 com)")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--select", type=int, default=SELECT_N)
    ap.add_argument("--start-date", default=START_DATE.isoformat())
    ap.add_argument("--include-derivatives", action="store_true",
                    help="não excluir stablecoins/wrapped/staked")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--mock", action="store_true", help="dados sintéticos (teste)")
    args = ap.parse_args()
    if args.sleep is None:
        has_key = bool(args.api_key or os.environ.get("COINGECKO_API_KEY"))
        args.sleep = 2.1 if has_key else 6.0
    if args.mock:
        args.sleep = 0

    os.makedirs(args.outdir, exist_ok=True)
    today = dt.date.today()

    final, series = build_dataset(args)
    if final.empty:
        sys.exit("Nenhuma moeda elegível — nada a fazer.")

    # ---- CSVs
    out_csv = os.path.join(args.outdir, "top20_resultados.csv")
    final_out = final[["nome", "simbolo", "rank_mcap", "retorno_pct", "dist_ath_pct"]].copy()
    final_out.columns = ["nome", "simbolo", "rank_market_cap",
                         "retorno_01jul_pct", "dist_ath_pct"]
    final_out.to_csv(out_csv, index=False, float_format="%.2f")
    print(f"OK  {out_csv}")

    id2sym = dict(zip(final["id"], final["simbolo"]))
    wide = pd.DataFrame({id2sym[cid]: s for cid, s in series.items()})
    wide.index.name = "data"
    daily_csv = os.path.join(args.outdir, "retornos_diarios.csv")
    wide.to_csv(daily_csv, float_format="%.3f")
    print(f"OK  {daily_csv}")

    # ---- tabela no terminal
    with pd.option_context("display.max_rows", None, "display.width", 120):
        show = final_out.copy()
        show["retorno_01jul_pct"] = show["retorno_01jul_pct"].round(1)
        show["dist_ath_pct"] = show["dist_ath_pct"].round(1)
        print("\n" + show.to_string(index=False) + "\n")

    # ---- gráficos
    alts = final[final["id"] != "bitcoin"]
    top5_ret = list(alts.sort_values("retorno_pct", ascending=False)["id"].head(5))
    top5_ath = list(alts.sort_values("dist_ath_pct", ascending=False)["id"].head(5))
    per = f"01/07/2026 → {today.strftime('%d/%m/%Y')}"
    table_png(final, os.path.join(args.outdir, "tabela.png"), today)
    line_chart(final, series, top5_ret,
               "Retorno acumulado — destaque: 5 maiores retornos",
               f"{per} · cinza = demais moedas do top 20 · tracejada = BTC",
               os.path.join(args.outdir, "grafico_top5_retorno.png"))
    line_chart(final, series, top5_ath,
               "Retorno acumulado — destaque: 5 mais próximas da ATH",
               f"{per} · cinza = demais moedas do top 20 · tracejada = BTC",
               os.path.join(args.outdir, "grafico_top5_ath.png"))
    scatter_chart(final, os.path.join(args.outdir, "grafico_dispersao.png"))
    print("\nConcluído. Arquivos em ./" + args.outdir)


if __name__ == "__main__":
    main()
