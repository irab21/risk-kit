"""
Find stocks and ETFs by company name (Yahoo Finance search), so people never need to know tickers.

search("reliance") -> [{"ticker": "RELIANCE.NS", "name": "Reliance Industries", "where": "NSE India",
                        "kind": "Stock"}, ...]
"""
import re

EXCHANGES = {"NSI": "NSE India", "BSE": "BSE India", "NMS": "Nasdaq", "NGM": "Nasdaq", "NCM": "Nasdaq",
             "NYQ": "NYSE", "ASE": "NYSE American", "PCX": "NYSE Arca", "BTS": "Cboe"}
KINDS = {"EQUITY": "Stock", "ETF": "ETF"}
PREFERENCE = ["NSI", "NMS", "NGM", "NCM", "NYQ", "PCX", "ASE", "BTS", "BSE"]  # NSE over BSE for Indian names

# Well-known names for one-click picks: display name -> ticker
POPULAR = {
    "India": {
        "Reliance Industries": "RELIANCE.NS", "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
        "Infosys": "INFY.NS", "Tata Consultancy Services": "TCS.NS", "Bharti Airtel": "BHARTIARTL.NS",
        "ITC": "ITC.NS", "Larsen & Toubro": "LT.NS", "State Bank of India": "SBIN.NS",
        "Hindustan Unilever": "HINDUNILVR.NS", "Bajaj Finance": "BAJFINANCE.NS", "Sun Pharmaceutical": "SUNPHARMA.NS",
        "Maruti Suzuki": "MARUTI.NS", "Asian Paints": "ASIANPAINT.NS", "Titan Company": "TITAN.NS",
        "Nifty 50 ETF (Nippon)": "NIFTYBEES.NS", "Gold ETF (Nippon)": "GOLDBEES.NS",
    },
    "US": {
        "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Amazon": "AMZN", "Alphabet (Google)": "GOOGL",
        "Meta Platforms": "META", "Berkshire Hathaway": "BRK-B", "JPMorgan Chase": "JPM", "Johnson & Johnson": "JNJ",
        "Coca-Cola": "KO", "S&P 500 ETF": "SPY", "Total bond market ETF": "BND", "Gold ETF": "GLD",
    },
}

_SUFFIX = re.compile(r"[,\s]+(limited|ltd\.?|inc\.?|incorporated|corporation|corp\.?|plc|co\.?|company limited)$",
                     re.IGNORECASE)


def clean_name(name):
    """'Reliance Industries Limited' -> 'Reliance Industries'; 'Apple Inc.' -> 'Apple'."""
    name = re.sub(r"\s+", " ", str(name or "")).strip()
    prev = None
    while prev != name:
        prev, name = name, _SUFFIX.sub("", name).strip()
    if name.isupper() and len(name) > 5:
        name = name.title()
    return name


def parse_quotes(quotes, limit=10):
    """Yahoo search quotes -> stocks/ETFs on major exchanges, one listing per company (NSE/US preferred)."""
    best = {}
    for q in quotes:
        kind, exch, sym = KINDS.get(q.get("quoteType")), q.get("exchange"), q.get("symbol")
        if not kind or exch not in EXCHANGES or not sym:
            continue
        name = clean_name(q.get("longname") or q.get("shortname") or sym)
        if kind == "Stock" and re.search(r"\b(etf|bees|index fund)\b", name, re.IGNORECASE):
            kind = "ETF"
        # one entry per company: Indian listings share the symbol on NSE (.NS) and BSE (.BO)
        key = ("IN", sym.split(".")[0]) if exch in ("NSI", "BSE") else ("US", name.lower())
        rank = PREFERENCE.index(exch)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, {"ticker": sym, "name": name, "where": EXCHANGES[exch], "kind": kind})
    return [v for _, v in best.values()][:limit]  # dicts keep Yahoo's relevance order


def search(query, limit=10, prefer="India"):
    query = (query or "").strip()
    if len(query) < 2:
        return []
    import yfinance as yf
    res = yf.Search(query, max_results=20, news_count=0, lists_count=0)
    found = parse_quotes(res.quotes, 20)
    home = ("NSE India", "BSE India") if prefer == "India" else ()
    return sorted(found, key=lambda x: x["where"] not in home)[:limit]  # stable: keeps relevance within groups


KNOWN = {t: n for group in POPULAR.values() for n, t in group.items()}
KNOWN.update({
    "QQQ": "Nasdaq 100 ETF", "AGG": "US bond market ETF", "VTI": "US total stock market ETF",
    "VXUS": "International stocks ETF", "TLT": "Long-term US Treasuries ETF", "IEF": "7–10 year US Treasuries ETF",
    "DBC": "Commodities ETF", "VT": "World stocks ETF", "^NSEI": "Nifty 50", "^GSPC": "S&P 500 index",
    "^BSESN": "Sensex",
})


def name_of(ticker):
    """Display name for a well-known ticker, else None."""
    return KNOWN.get(str(ticker).upper())


def lookup_name(ticker):
    """Company / fund name for a Yahoo ticker: the built-in list first, then Yahoo's search. None if unknown."""
    t = str(ticker).upper()
    if t in KNOWN:
        return KNOWN[t]
    import yfinance as yf
    for q in yf.Search(t, max_results=10, news_count=0, lists_count=0).quotes:
        if str(q.get("symbol", "")).upper() == t:
            return clean_name(q.get("longname") or q.get("shortname")) or None
    return None
