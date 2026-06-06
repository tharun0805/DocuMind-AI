import pandas as pd
from loguru import logger


def load_csv(file_path: str) -> str:
    # Previously used df.to_string(index=False) which is extremely slow and
    # memory-hungry for large files (builds one giant Python string in RAM).
    # Using itertuples() writes rows incrementally — much faster and
    # constant-memory regardless of file size.
    df = pd.read_csv(file_path, low_memory=False)

    header = " | ".join(str(c) for c in df.columns)
    rows = [header]
    for row in df.itertuples(index=False, name=None):
        rows.append(" | ".join("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v) for v in row))

    text = "\n".join(rows)
    logger.info(f"CSV loaded: {file_path} ({len(df)} rows, {len(df.columns)} cols)")
    return text