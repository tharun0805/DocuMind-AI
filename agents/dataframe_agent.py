import re
import time
import pandas as pd
from loguru import logger
from langchain_core.prompts import PromptTemplate
from utils.llm_provider import get_shared_llm
from tools.dataframe_tool import load_dataframe, get_dataframe_info


# ── Generic helpers ────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, *keywords) -> str | None:
    """Find a column matching any keyword (exact then partial)."""
    for kw in keywords:
        kw = kw.lower().strip()
        for col in df.columns:
            if col.strip().lower() == kw:
                return col
        for col in df.columns:
            if kw in col.strip().lower():
                return col
    return None


def _extract_leading_number(val) -> float:
    """
    Extract leading number from any value.
    Works for any format: '3.5', '42', 'Score: 7', '2 Some text label'
    """
    try:
        s = str(val).strip()
        m = re.match(r'^([0-9]+(?:\.[0-9]+)?)', s)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def _detect_score_columns(df: pd.DataFrame, keyword: str = "") -> list[str]:
    """
    Dynamically detect scored/numeric columns from the actual DataFrame.
    If keyword given: find columns whose name contains that keyword AND looks
    like a scored column (contains 'question', 'q', 'item', or 'score').
    Otherwise: auto-detect by sampling values — any column whose values
    start with a digit followed by text is treated as a scored response column.
    Fully generic — works for any dataset regardless of domain.
    """
    if keyword:
        return [c for c in df.columns
                if keyword.lower() in c.lower() and
                any(w in c.lower() for w in ["question", "q", "item", "score"])]

    # Auto-detect: sample values starting with digit + whitespace + non-whitespace
    score_cols = []
    for col in df.columns:
        sample = df[col].dropna().head(10).astype(str)
        if sample.str.match(r'^\d+\s+\S').sum() >= 3:
            score_cols.append(col)
    return score_cols


def _compute_group_score(df: pd.DataFrame, keyword: str) -> pd.Series | None:
    """Compute row-wise total for any group of score columns matching a keyword."""
    cols = _detect_score_columns(df, keyword)
    if not cols:
        return None
    return df[cols].applymap(_extract_leading_number).sum(axis=1)


def _detect_id_column(df: pd.DataFrame) -> str | None:
    """
    Detect the primary identifier column dynamically from the actual column names.
    Tries common identity-like patterns — no hardcoded domain names.
    """
    # Exact/partial match on generic identity keywords
    identity_keywords = [
        "name", "id", "identifier", "label", "title",
        "subject", "participant", "user", "employee",
        "customer", "patient", "student", "person",
        "respondent", "entry", "record",
    ]
    return _find_col(df, *identity_keywords)


# ── Layer 1: Smart lookup (no LLM) ────────────────────────────────────────────

def _smart_lookup(df: pd.DataFrame, question: str) -> str | None:
    ql       = question.lower().strip()
    name_col = _detect_id_column(df)
    age_col  = _find_col(df, "age", "years", "dob", "age_years")

    # ── Column existence / listing ────────────────────────────────────────────
    if any(p in ql for p in ["what column", "list column", "show column",
                              "column names", "what fields", "headers",
                              "which column", "what are the column"]):
        cols = list(df.columns)
        return (f"The dataset has **{len(cols)} columns**:\n\n"
                + "\n".join(f"- {c}" for c in cols))

    # Column existence check: "is there a column named X"
    m = re.search(
        r'(?:is there|does|do you have|exists?)\s+(?:a\s+)?column\s+'
        r'(?:named|called|for)?\s*["\']?([^"\'?]+?)["\']?\s*(?:in|$|\?)',
        ql)
    if m:
        target  = m.group(1).strip()
        matches = [c for c in df.columns
                   if target.lower() in c.lower() or c.lower() in target.lower()]
        if matches:
            return (f"Yes, the column **'{matches[0]}'** exists.\n\n"
                    f"All columns: {', '.join(df.columns)}")
        score_cols = _detect_score_columns(df, target)
        if score_cols:
            return (f"There is no pre-existing '{target}' column, but I can compute it "
                    f"from the {len(score_cols)} related columns ({', '.join(score_cols[:3])}...).\n"
                    f"Ask me to generate a document with the computed total.")
        return (f"No column named '{target}' exists.\n\n"
                f"Available columns: {', '.join(df.columns)}")

    # ── Dataset shape ─────────────────────────────────────────────────────────
    if any(p in ql for p in ["how many row", "how many record", "how many entry",
                              "how many people", "total record", "dataset size",
                              "number of row", "count of record"]):
        return f"The dataset has **{len(df):,} records** across {len(df.columns)} columns."

    if any(p in ql for p in ["how many column", "number of column"]):
        return f"The dataset has **{len(df.columns)} columns**: {', '.join(df.columns)}"

    # ── First / last record ───────────────────────────────────────────────────
    for kw in ["first record", "first row", "first entry",
               "first person", "first participant"]:
        if kw in ql:
            if name_col:
                return f"The first entry is **{df[name_col].iloc[0]}**."
            return f"First record:\n{df.iloc[0].to_string()}"

    for kw in ["last record", "last row", "last entry", "last person"]:
        if kw in ql:
            if name_col:
                return f"The last entry is **{df[name_col].iloc[-1]}**."
            return f"Last record:\n{df.iloc[-1].to_string()}"

    # ── Oldest / youngest (if an age-like column exists) ─────────────────────
    if age_col:
        if any(w in ql for w in ["oldest", "eldest", "maximum age", "highest age"]):
            idx  = df[age_col].idxmax()
            name = df.loc[idx, name_col] if name_col else f"Row {idx}"
            return f"The oldest is **{name}** ({age_col}: {df.loc[idx, age_col]})."
        if any(w in ql for w in ["youngest", "minimum age", "lowest age"]):
            idx  = df[age_col].idxmin()
            name = df.loc[idx, name_col] if name_col else f"Row {idx}"
            return f"The youngest is **{name}** ({age_col}: {df.loc[idx, age_col]})."

    # ── Name-based lookup: "What is X's Y" ───────────────────────────────────
    if name_col:
        nm = re.search(
            r"(?:what\s+is|tell\s+me|show\s+me|get|find)?\s*"
            r"([A-Za-z][A-Za-z\s\.]{1,25}?)'s\s+([A-Za-z][A-Za-z\s]{1,40}?)(?:\?|$)",
            question, re.IGNORECASE)
        if nm:
            person, field = nm.group(1).strip(), nm.group(2).strip()
            _blocklist = {"first", "last", "the", "a", "this", "that",
                          "record", "entry"}
            if person.lower() not in _blocklist:
                mask = (df[name_col].astype(str).str.strip().str.lower()
                        == person.lower())
                if mask.sum() == 0:
                    mask = (df[name_col].astype(str).str.strip().str.lower()
                            .str.contains(person.lower(), regex=False))
                if mask.sum() > 0:
                    row = df[mask].iloc[0]

                    # Check if asking for a computed group score ("total X score")
                    total_m = re.search(r'total\s+(\w+)\s+score|(\w+)\s+total\s+score',
                                        field.lower())
                    if total_m:
                        keyword = (total_m.group(1) or total_m.group(2)).strip()
                        series  = _compute_group_score(df, keyword)
                        if series is not None:
                            score = series[mask].iloc[0]
                            cols  = _detect_score_columns(df, keyword)
                            return (f"**{person}'s total {keyword.upper()} score is "
                                    f"{int(score)}** (sum of {len(cols)} questions).")

                    # Generic field lookup
                    field_col = _find_col(df, field)
                    if field_col:
                        return f"**{person}'s {field_col}** is **{row[field_col]}**."
                    return f"Full record for {person}:\n{row.to_string()}"

    # ── Generic "total X score" for all records ───────────────────────────────
    total_m = re.search(
        r'total\s+(\w+)\s+score|(\w+)\s+(?:total|sum)\s+score|'
        r'compute\s+(\w+)\s+score|sum\s+of\s+(\w+)',
        ql)
    if total_m:
        keyword = next(g for g in total_m.groups() if g)
        series  = _compute_group_score(df, keyword)
        if series is not None and name_col:
            lines = [f"- **{df[name_col].iloc[i]}**: {int(series.iloc[i])}"
                     for i in range(min(len(df), 20))]
            return (f"**Total {keyword.upper()} Scores** "
                    f"(sum of {len(_detect_score_columns(df, keyword))} questions):\n\n"
                    + "\n".join(lines)
                    + (f"\n\n...and {len(df)-20} more" if len(df) > 20 else ""))

    # ── Count by category ─────────────────────────────────────────────────────
    cat_m = re.search(r'how many\s+(\w+)', ql)
    if cat_m:
        val = cat_m.group(1)
        for col in df.select_dtypes(include="object").columns:
            matches = df[col].astype(str).str.lower() == val.lower()
            if matches.sum() > 0:
                return f"There are **{matches.sum()}** records with {col} = '{val}'."

    # ── Average ───────────────────────────────────────────────────────────────
    if age_col and any(w in ql for w in ["average age", "mean age", "avg age"]):
        return f"The average age is **{df[age_col].mean():.1f}**."

    # ── List all entries ──────────────────────────────────────────────────────
    if any(p in ql for p in ["list all", "all names", "who are",
                              "all entries", "all records", "show all"]):
        if name_col:
            names = df[name_col].tolist()
            return ("All entries ({} total):\n\n".format(len(names))
                    + "\n".join(f"{i+1}. {n}" for i, n in enumerate(names[:50]))
                    + (f"\n...and {len(names)-50} more" if len(names) > 50 else ""))

    return None  # fall through to LLM


# ── Layer 2: LLM codegen ──────────────────────────────────────────────────────

_PROMPT = PromptTemplate(
    input_variables=["question", "df_info", "sample"],
    template="""You are a pandas expert. DataFrame variable = `df`.

Info:
{df_info}

Sample (2 rows):
{sample}

Question: {question}

Write ONE valid Python expression. Use only:
- list(df.columns)              ← NOT df.columns.to_string()
- df.shape
- df['col'].value_counts().to_string()
- df.loc[df['col']==val, 'other'].values[0]
- df['col'].mean()
- df.nlargest(n, 'col')[['col1','col2']].to_string()
- df.iloc[0].to_string()
- df.describe().to_string()
- df['col'].astype(str).str.strip().str.lower() == 'value'

NEVER use: .to_string() on Index/MultiIndex objects directly.
If result is DataFrame/Series: add .to_string() at end.

Expression only:""")


def run_dataframe_agent(question: str, file_path: str) -> str:
    logger.debug(f"DataFrame agent: {question[:60]}")

    try:
        df = load_dataframe(file_path)
    except Exception as e:
        return f"Could not load data file: {e}"

    # Layer 1: fast rule-based lookup
    result = _smart_lookup(df, question)
    if result:
        return result

    # Layer 2: LLM codegen
    df_info = get_dataframe_info(df)
    sample  = df.head(2).to_string()
    code    = ""

    for attempt in range(3):
        try:
            out  = (_PROMPT | get_shared_llm(temperature=0)).invoke({
                "question": question, "df_info": df_info, "sample": sample})
            code = (out.content.strip()
                    .replace("```python", "").replace("```", "").strip())
            lines = [l.strip() for l in code.split("\n")
                     if l.strip() and not l.strip().startswith("#")]
            if not lines:
                continue
            code = max(lines, key=len)
            code = code.replace("df.columns.to_string()", "list(df.columns)")

            # Guard: catch any domain-specific strings the LLM may have
            # hallucinated from training data before they run as code.
            from utils.validator import assert_no_domain_leakage
            assert_no_domain_leakage(code, context="LLM-generated DataFrame code")

            answer = eval(code, {"df": df, "pd": pd, "__builtins__": {}})  # noqa
            if answer is None:
                continue
            s = answer.to_string() if hasattr(answer, "to_string") else str(answer)
            if s.strip() not in {"", "nan", "None"}:
                return s

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                time.sleep(30 * (attempt + 1))
            elif "WinError 10054" in err or "ConnectionReset" in err:
                time.sleep(5)
            else:
                logger.error(f"Eval error: {e} | {code!r}")
                break

    return _safe_fallback(df, question)


# ── Layer 3: Safe fallback ────────────────────────────────────────────────────

def _safe_fallback(df: pd.DataFrame, question: str) -> str:
    ql = question.lower()
    try:
        if any(w in ql for w in ["column", "field", "header"]):
            return f"Columns ({len(df.columns)}): {', '.join(df.columns)}"
        if any(w in ql for w in ["how many", "count", "total", "number"]):
            return f"The dataset has **{len(df):,} records** across {len(df.columns)} columns."
        if any(w in ql for w in ["average", "mean", "avg"]):
            nums = df.select_dtypes(include="number")
            return nums.mean().round(2).to_string() if not nums.empty else "No numeric columns."
        if any(w in ql for w in ["first", "top", "head"]):
            name_col = _detect_id_column(df)
            if name_col:
                return f"First 5: {', '.join(str(v) for v in df[name_col].head(5).tolist())}"
        return (f"Dataset: **{len(df):,} records** × {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns[:10])}"
                + (f" (+{len(df.columns)-10} more)" if len(df.columns) > 10 else ""))
    except Exception:
        return "Could not process this query. Please try rephrasing."