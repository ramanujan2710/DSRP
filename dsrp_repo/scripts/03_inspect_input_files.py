import pandas as pd

ALIAS_PATH  = "/content/drive/MyDrive/niwiad/9606.protein.aliases.v12.0.txt"
STRING_PATH = "/content/drive/MyDrive/niwiad/9606.protein.physical.links.v12.0 2.txt"
CORUM_PATH  = "/content/drive/MyDrive/niwiad/corum_humanComplexes (1).txt"

def inspect_file(path, label, n_rows=5):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  Path: {path}")
    print(f"{'='*65}")

    for sep, sep_name in [("\t","tab"), (" ","space"), (",","comma")]:
        try:
            df = pd.read_csv(path, sep=sep, nrows=100,
                           comment="#", engine="python")
            if df.shape[1] > 1:
                print(f"  Separator  : {sep_name!r}")
                print(f"  Shape      : {df.shape[0]} rows × {df.shape[1]} cols (first 100 rows)")
                print(f"  Columns    : {list(df.columns)}")
                print(f"  Dtypes     :\n{df.dtypes.to_string()}")
                print(f"\n  First {n_rows} rows:")
                print(df.head(n_rows).to_string())
                print(f"\n  Null counts:\n{df.isnull().sum().to_string()}")
                for col in df.columns:
                    nu = df[col].nunique()
                    if nu <= 20:
                        print(f"\n  Unique values in '{col}' ({nu}): {df[col].unique().tolist()}")
                    else:
                        print(f"\n  Sample values in '{col}' ({nu} unique): {df[col].dropna().head(5).tolist()}")
                return df
        except Exception as e:
            print(f"  [{sep_name}] failed: {e}")
            continue

    print("  Could not parse as tabular — showing raw lines:")
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 10: break
            print(f"  L{i}: {line.rstrip()}")
    return None

df_alias  = inspect_file(ALIAS_PATH,  "STRING ALIAS FILE")
df_string = inspect_file(STRING_PATH, "STRING PHYSICAL LINKS FILE")
df_corum  = inspect_file(CORUM_PATH,  "CORUM COMPLEXES FILE")

print(f"\n{'='*65}")
print("  FULL FILE ROW COUNTS")
print(f"{'='*65}")
for path, label in [(ALIAS_PATH,"Alias"), (STRING_PATH,"STRING"), (CORUM_PATH,"CORUM")]:
    try:
        with open(path, "r", errors="replace") as f:
            n = sum(1 for _ in f)
        print(f"  {label}: {n:,} lines")
    except Exception as e:
        print(f"  {label}: error — {e}")
