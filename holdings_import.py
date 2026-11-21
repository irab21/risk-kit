"""
Reads a holdings file exported from a broker (CSV or Excel) and turns it into portfolio weights.

Broker exports differ: some start with a few lines of account info, column names vary
("Symbol", "Instrument", "Ticker"; "Quantity", "Qty.", "Shares"; "Current Value", "Cur. val",
"Market Value"), and totals, cash and disclaimers are mixed in with the positions.
This module finds the header row, guesses which column is which, and cleans the rows.

Typical use:
    table = read_table(file_bytes, "positions.csv")
    cols = guess_columns(table)
    holdings, skipped = build_holdings(table, **cols)
"""
import csv
import io
import re

import numpy as np
import pandas as pd


def _norm(text):
    """Lowercase and keep only letters, digits and %: 'Cur. val' -> 'cur val'"""
    return re.sub(r"[^a-z0-9%]+", " ", str(text).lower()).strip()


CANDIDATES = {
    "ticker_col": ["symbol", "ticker", "tradingsymbol", "trading symbol", "instrument", "stock symbol",
                   "symbol cusip", "security symbol", "scrip", "code"],
    "name_col": ["product name", "scheme name", "scheme", "fund name", "fund", "security name", "investment name",
                 "scheme details", "name of the scheme"],
    "shares_col": ["quantity", "qty", "shares", "units", "quantity held", "no of shares", "shares held",
                   "position", "available qty", "quantity available", "balance units", "closing units",
                   "unit balance", "units held", "bal units", "closing balance units"],
    "value_col": ["current value", "market value", "cur val", "present value", "total value", "value",
                  "market val", "mkt val", "marketvalue", "current val", "valuation", "current market value"],
    "weight_col": ["weight", "weight %", "allocation", "allocation %", "% of account", "percent of account",
                   "% of portfolio", "portfolio %", "% of acct", "% of account holdings"],
}
CANDIDATES = {role: [_norm(c) for c in names] for role, names in CANDIDATES.items()}

TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=&^]{0,19}$")
MONEY_MARKET = re.compile(r"^[A-Z]{3}XX$")  # US money market funds: SPAXX, VMFXX, SWVXX, FDRXX...
NOT_TICKERS = {"CASH", "TOTAL", "ACCOUNT TOTAL", "PENDING ACTIVITY", "GRAND TOTAL", "SUBTOTAL", "NAN", "NONE"}

SUFFIXES = {
    "None (US and other Yahoo symbols)": "",
    "India – NSE (.NS)": ".NS",
    "India – BSE (.BO)": ".BO",
    "UK – London (.L)": ".L",
    "Canada – Toronto (.TO)": ".TO",
    "Australia – ASX (.AX)": ".AX",
    "Germany – Xetra (.DE)": ".DE",
    "Hong Kong (.HK)": ".HK",
    "Japan – Tokyo (.T)": ".T",
}


class PasswordRequired(ValueError):
    """The PDF is encrypted and needs a password (common for statements sent by email)."""


def read_table(data, filename, password=None):
    """
    Reads a CSV, Excel or PDF file (bytes) into a DataFrame of strings, using the first row that
    looks like a header (contains a ticker column name) and skipping any lines above it.
    For a PDF, `password` unlocks an encrypted statement.
    """
    name = filename.lower()
    if name.endswith(".pdf"):
        raw = _pdf_to_raw(data, password)
    elif name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(io.BytesIO(data), header=None, dtype=str)
    else:
        text = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        try:
            dialect = csv.Sniffer().sniff(text[:5000], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(io.StringIO(text), dialect))
        width = max((len(r) for r in rows), default=0)
        raw = pd.DataFrame([r + [""] * (width - len(r)) for r in rows])
    return _table_from_raw(raw)


HEADER_KEYS = set(CANDIDATES["ticker_col"]) | set(CANDIDATES["name_col"])


def _find_header_row(raw, max_rows=40):
    for i in range(min(len(raw), max_rows)):
        if {_norm(c) for c in raw.iloc[i]} & HEADER_KEYS:
            return i
    return None


def _has_digit(text):
    return any(ch.isdigit() for ch in str(text))


def _table_from_raw(raw):
    raw = raw.fillna("").astype(str).apply(lambda c: c.str.replace("\n", " ").str.strip())
    header_row = _find_header_row(raw)
    if header_row is None:
        raise ValueError("Couldn't find a header row with a ticker or fund name column "
                         "(for example 'Symbol', 'Ticker', 'Instrument' or 'Scheme Name').")

    original = list(raw.iloc[header_row])
    body_start = header_row + 1
    # Two-row headers: a group row ("Cost", "Portfolio Status") above the real labels
    if body_start < len(raw):
        nxt = list(raw.iloc[body_start])
        labels = [c for c in nxt if c]
        if len(labels) >= 2 and not any(_has_digit(c) for c in labels):
            original = [b or a for a, b in zip(original, nxt)]
            body_start += 1
    header = [c or f"Column {j+1}" for j, c in enumerate(original)]
    seen = {}
    for j, h in enumerate(header):  # make duplicate names unique
        if h in seen:
            seen[h] += 1
            header[j] = f"{h} ({seen[h]})"
        else:
            seen[h] = 0
    table = raw.iloc[body_start:].copy()
    table = table[~(table.values == list(raw.iloc[header_row])).all(axis=1)]  # header repeated on later pages
    # Columns without a header next to a text column are usually that text spilling over: join them
    for j in range(len(header) - 1, 0, -1):
        if not original[j] and original[j - 1]:
            table.iloc[:, j - 1] = (table.iloc[:, j - 1] + " " + table.iloc[:, j]).str.strip()
            table.iloc[:, j] = ""
    table.columns = header
    # Rows holding only the wrapped end of a name ("Reg - Gr") belong to the row above
    keep, last_data = [], None
    for idx, row in table.iterrows():
        filled = [c for c in table.columns if row[c]]
        if (len(filled) == 1 and last_data is not None and not _has_digit(row[filled[0]])
                and table.at[last_data, filled[0]] and not _has_digit(table.at[last_data, filled[0]])):
            table.at[last_data, filled[0]] += " " + row[filled[0]]
            continue
        if sum(_has_digit(row[c]) for c in filled) >= 2:
            last_data = idx
        elif len(filled) > 1 or not filled:
            last_data = None
        keep.append(idx)
    table = table.loc[keep]
    table = table[(table != "").any(axis=1)]  # drop blank rows
    table = table.loc[:, (table != "").any()]  # drop blank columns
    return table.reset_index(drop=True)


# -----------------------------------------------------------------------------
# PDF statements
# -----------------------------------------------------------------------------
MAX_PDF_PAGES = 40


def _pdf_to_raw(data, password=None):
    """Extracts the holdings table from a PDF: ruled tables first, then a layout-based reader."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise ValueError("Reading PDFs needs pdfplumber: pip install pdfplumber") from exc
    try:
        pdf = pdfplumber.open(io.BytesIO(data), password=password or "")
        pages = pdf.pages[:MAX_PDF_PAGES]
        _ = [p.chars for p in pages[:1]]  # forces decryption errors to surface here
    except Exception as exc:
        text = f"{type(exc).__name__} {exc!r}".lower()
        if "password" in text or "encrypt" in text:
            raise PasswordRequired("This PDF is password-protected." if not password
                                   else "That password didn't open the PDF.") from exc
        raise ValueError("Couldn't open this PDF. Is the file damaged?") from exc

    with pdf:
        if not any(p.chars for p in pages):
            raise ValueError("This PDF has no readable text (it may be a scanned image). "
                             "Download a CSV or Excel export from your broker instead.")
        raw = _pdf_ruled_tables(pages)
        if raw is None:
            raw = _pdf_layout_table(pages)
    if raw is None:
        raise ValueError("Couldn't find a holdings table with a ticker or fund name column "
                         "(such as 'Symbol' or 'Scheme Name') in this PDF.")
    return raw


def _pdf_ruled_tables(pages):
    """Tables drawn with lines. Keeps tables that have a ticker header, plus headerless continuations."""
    rows, width = [], None
    for page in pages:
        for table in page.extract_tables():
            table = [["" if c is None else str(c).replace("\n", " ").strip() for c in r] for r in table]
            has_header = _find_header_row(pd.DataFrame(table), max_rows=3) is not None
            if has_header:
                width = len(table[0])
                rows.extend(table)
            elif width is not None and table and len(table[0]) == width:
                rows.extend(table)  # the same table continued on the next page
            else:
                width = None
    if not rows:
        return None
    width = max(len(r) for r in rows)
    return pd.DataFrame([r + [""] * (width - len(r)) for r in rows])


# Words that appear in column headers. Lines just above or below a header line count as part of
# the header (wrapped labels like "Market / Value") only if every word is one of these.
HEADER_WORDS = set("""symbol ticker instrument scrip code security description name investment holding holdings
asset class type quantity qty shares units held position available price last close ltp current cur market mkt
value val total amount cost basis avg average book unrealized unrealised realized gain loss p l pl p&l return
change chg net day today s todays % of percent pct account acct portfolio weight allocation yield income estimated
est annual date acquired currency exchange isin cusip per share no number sector rate""".split())

_NUMBERISH = re.compile(r"^[(\-+]?[$€£₹¥]?\s?[\d,]*\.?\d+%?\)?$")


def _lines(words, tol=3):
    """Groups words into lines by vertical position"""
    lines = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(l, key=lambda w: w["x0"]) for l in lines]


def _is_number(text):
    return bool(_NUMBERISH.match(text.replace(" ", "")))


def _cells(line, gap):
    """Joins neighbouring words of one line into cells (a cell ends at a gap wider than `gap`;
    two numbers are never joined, since they belong to different columns)"""
    cells = []
    for w in line:
        if cells and w["x0"] - cells[-1]["x1"] <= gap and not (_is_number(w["text"]) and _is_number(cells[-1]["text"])):
            cells[-1]["text"] += " " + w["text"]
            cells[-1]["x1"] = w["x1"]
        else:
            cells.append({"text": w["text"], "x0": w["x0"], "x1": w["x1"]})
    return cells


def _pdf_layout_table(pages):
    """
    For statements without table lines: finds the header line (with 'Symbol', 'Ticker', ...),
    builds column labels from the header words (which may wrap over two or three lines),
    then assigns each value on the following lines to the column it sits under.
    """
    ticker_names = HEADER_KEYS
    out, labels = [], None
    for page in pages:
        words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
        if not words:
            continue
        lines = _lines(words)
        i = 0
        while i < len(lines):
            line = lines[i]
            if not any(_norm(w["text"]) in ticker_names for w in line):
                if labels is not None:
                    row = _layout_row(line, labels)
                    if row is not None:
                        out.append(row)
                i += 1
                continue
            # header found: gather header lines close above and below (wrapped labels)
            height = max(w["bottom"] - w["top"] for w in line)
            top, bottom = line[0]["top"], max(w["bottom"] for w in line)
            def is_header_line(l):
                if l is line:
                    return True
                if not top - 1.6 * height <= l[0]["top"] <= bottom + 1.2 * height:
                    return False
                tokens = [t for w in l for t in _norm(w["text"]).split()]
                return bool(tokens) and all(t in HEADER_WORDS for t in tokens)
            block = [w for l in lines if is_header_line(l) for w in l]
            groups = []
            # cells per header line, then merge cells that overlap horizontally across lines
            for hl in _lines(block):
                for c in _cells(hl, 0.6 * height):
                    match = next((g for g in groups if c["x0"] < g["x1"] + 1 and c["x1"] > g["x0"] - 1), None)
                    if match:
                        match["parts"].append(c["text"])
                        match["x0"], match["x1"] = min(match["x0"], c["x0"]), max(match["x1"], c["x1"])
                    else:
                        groups.append({"parts": [c["text"]], "x0": c["x0"], "x1": c["x1"]})
            groups.sort(key=lambda g: g["x0"])
            new_labels = [{"label": " ".join(g["parts"]), "x0": g["x0"], "x1": g["x1"]} for g in groups]
            if labels is None or [l["label"] for l in new_labels] != [l["label"] for l in labels]:
                labels = new_labels
                out.append([l["label"] for l in labels])
            else:
                labels = new_labels  # same columns on a new page: keep going, positions may shift
            block_bottom = max(w["bottom"] for w in block)
            i += 1
            while i < len(lines) and lines[i][0]["top"] <= block_bottom:
                i += 1
    if not out:
        return None
    width = max(len(r) for r in out)
    return pd.DataFrame([r + [""] * (width - len(r)) for r in out])


def _layout_row(line, labels):
    """Puts the cells of one text line under the header labels. Returns None for non-data lines."""
    height = max(w["bottom"] - w["top"] for w in line)
    cells = _cells(line, 0.9 * height)
    if not any(_is_number(c["text"]) for c in cells):
        return None  # wrapped descriptions, notes, section titles
    row = [""] * len(labels)
    for c in cells:
        overlaps = [max(0, min(c["x1"], l["x1"]) - max(c["x0"], l["x0"])) for l in labels]
        if max(overlaps) > 0:
            j = overlaps.index(max(overlaps))
        else:
            centre = (c["x0"] + c["x1"]) / 2
            j = min(range(len(labels)), key=lambda k: abs(centre - (labels[k]["x0"] + labels[k]["x1"]) / 2))
        row[j] = (row[j] + " " + c["text"]).strip()
    return row


def guess_columns(table):
    """Guesses the ticker, shares, value and weight columns. Missing roles map to None."""
    normalized = {col: _norm(col) for col in table.columns}
    guesses = {}
    for role, names in CANDIDATES.items():
        found = None
        for name in names:  # exact matches first, in order of preference
            found = next((c for c, n in normalized.items() if n == name and c not in guesses.values()), None)
            if found:
                break
        if found is None and role not in ("ticker_col", "name_col"):  # then "starts with", e.g. "Quantity (shares)"
            for name in names:
                if len(name) < 4:
                    continue
                found = next((c for c, n in normalized.items()
                              if n.startswith(name) and c not in guesses.values()), None)
                if found:
                    break
        guesses[role] = found
    return guesses


def guess_suffix(table):
    """Indian broker exports (e.g. Zerodha: Instrument, LTP, Cur. val) list NSE symbols without '.NS'"""
    cols = {_norm(c) for c in table.columns}
    return ".NS" if {"ltp", "cur val"} & cols or "instrument" in cols and "ltp" in cols else ""


def to_number(series):
    """Parses numbers written as '$1,234.50', '(12.3)', '45%', '₹ 10,000', '--'"""
    s = series.astype(str).str.strip()
    negative = s.str.startswith("(") & s.str.endswith(")") | s.str.startswith("-")
    cleaned = s.str.replace(r"[^0-9.]", "", regex=True)
    out = pd.to_numeric(cleaned.where(cleaned.str.count(r"\.") <= 1), errors="coerce")
    return out.where(~negative, -out)


def clean_ticker(text, suffix=""):
    """'  spaxx** ' -> 'SPAXX'; 'BRK/B' or 'BRK.B' -> 'BRK-B' (Yahoo's spelling); adds an exchange suffix if none"""
    t = str(text).strip().upper().rstrip("*").strip()
    t = re.sub(r"^(NSE|BSE|NASDAQ|NYSE|LSE)[:\s]+", "", t)
    if re.fullmatch(r"[A-Z]{1,5}[/.][A-Z]", t) and not suffix:
        t = re.sub(r"[/.]", "-", t)
    if suffix and "." not in t and t:
        t += suffix
    return t


def build_holdings(table, ticker_col=None, shares_col=None, value_col=None, weight_col=None,
                   prices=None, suffix="", name_col=None):
    """
    Turns an imported table into holdings.

    Weights come from, in order of preference: the value column; shares x `prices`
    (a dict or Series of latest prices by ticker); or the weight column.

    Returns (holdings, skipped):
    Without a ticker column, holdings are identified by `name_col` (mutual fund statements); their
    Ticker is left empty, to be filled by matching the names to scheme codes (see mutual_funds.py).

      holdings: DataFrame with Ticker, Name, Shares, Value, Weight % (one row per holding, duplicates summed)
      skipped: DataFrame with Row (the original text) and Reason
    """
    if ticker_col is None and name_col is None:
        raise ValueError("Choose which column holds the tickers (or the fund names).")
    by_name = ticker_col is None
    if value_col is None and shares_col is None and weight_col is None:
        raise ValueError("Choose a column for market value, shares or weight.")

    df = pd.DataFrame({"Raw": table[name_col if by_name else ticker_col].astype(str)})
    df["Name"] = (table[name_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
                  if name_col else "")
    df["Ticker"] = "" if by_name else df["Raw"].map(lambda t: clean_ticker(t, suffix))
    df["Shares"] = to_number(table[shares_col]) if shares_col else np.nan
    df["Value"] = to_number(table[value_col]) if value_col else np.nan
    df["Weight"] = to_number(table[weight_col]) if weight_col else np.nan

    if value_col is None and shares_col and prices is not None:
        price = pd.Series(prices, dtype=float)
        df["Value"] = df["Shares"] * df["Ticker"].map(price)

    measure = "Value" if df["Value"].notna().any() else "Weight"
    if measure == "Weight" and weight_col is None:
        raise ValueError("Couldn't get prices for the shares in this file. Add a market value column, "
                         "or check the exchange suffix.")

    skipped, keep = [], []
    for i, row in df.iterrows():
        base = row["Ticker"].split(".")[0] if suffix else row["Ticker"]
        raw = row["Raw"].strip()
        if by_name:
            if not raw or re.search(r"\btotal\b", raw, re.I):
                reason = "total or blank line"
            elif pd.isna(row[measure]):
                reason = f"no {measure.lower()}"
            elif row[measure] <= 0:
                reason = "zero or negative position"
            else:
                keep.append(i)
                continue
        elif not raw or base in NOT_TICKERS or raw.upper() in NOT_TICKERS:
            reason = "total or cash line"
        elif row["Raw"].strip().endswith("**") or MONEY_MARKET.match(row["Ticker"]):
            reason = "cash / money market fund (left out of the analysis)"
        elif not TICKER_PATTERN.match(row["Ticker"]):
            reason = "not a ticker symbol (cash, note or option?)"
        elif pd.isna(row[measure]):
            reason = "no price found" if measure == "Value" and shares_col and value_col is None else f"no {measure.lower()}"
        elif row[measure] < 0:
            reason = "short or negative position (not supported)"
        elif row[measure] == 0:
            reason = "zero position"
        else:
            keep.append(i)
            continue
        if row["Raw"].strip():
            skipped.append({"Row": row["Raw"].strip()[:40], "Reason": reason})

    if not keep:
        raise ValueError("No usable holdings found in the file. Check the column choices.")
    key = "Name" if by_name else "Ticker"
    agg = {"Shares": lambda x: x.sum(min_count=1), "Value": lambda x: x.sum(min_count=1),
           "Weight": lambda x: x.sum(min_count=1), "Name" if key == "Ticker" else "Ticker": "first"}
    kept = df.loc[keep].groupby(key, sort=False).agg(agg)
    kept["Weight %"] = 100 * kept[measure] / kept[measure].sum()
    holdings = kept.reset_index()[["Ticker", "Name", "Shares", "Value", "Weight %"]]
    holdings = holdings.sort_values("Weight %", ascending=False, ignore_index=True)
    return holdings, pd.DataFrame(skipped, columns=["Row", "Reason"])


TEMPLATE_CSV = "Ticker,Shares\nVTI,120\nVXUS,150\nBND,200\nGLD,15\n"
