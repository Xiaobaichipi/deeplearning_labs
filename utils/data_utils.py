import pandas as pd
import numpy as np
import io
import os


def load_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "gb18030", "latin-1"]
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, encoding=enc)
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
        df = pd.read_csv(filepath, encoding="latin-1")
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return df


def get_data_info(df):
    if df.empty or df.shape[1] == 0:
        return {
            "info": "Empty DataFrame",
            "describe": {},
            "dtypes": {},
            "null_counts": {},
            "null_pcts": {},
            "sample": [],
            "columns": [],
            "shape": [0, 0],
        }
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    desc = df.describe(include="all").to_dict()
    dtypes = {col: str(tp) for col, tp in df.dtypes.items()}
    null_counts = df.isnull().sum().to_dict()
    null_pcts = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    sample = df.head(50).to_dict(orient="records")
    columns = list(df.columns)
    shape = list(df.shape)
    return {
        "info": info_str,
        "describe": desc,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "null_pcts": null_pcts,
        "sample": sample,
        "columns": columns,
        "shape": shape,
    }


def clean_data(df, drop_duplicates=True, fill_na_method=None, fill_na_value=None,
               drop_columns=None, handle_outliers=False, outlier_method="iqr",
               outlier_factor=1.5):
    result = df.copy()
    report = []

    if drop_duplicates:
        before = len(result)
        result = result.drop_duplicates()
        after = len(result)
        report.append(f"Removed {before - after} duplicate rows")

    if drop_columns:
        cols_to_drop = [c for c in drop_columns if c in result.columns]
        if cols_to_drop:
            remaining = [c for c in result.columns if c not in cols_to_drop]
            if len(remaining) == 0:
                # Preserve at least one column
                keep = cols_to_drop.pop()
                report.append(f"Cannot drop all columns; kept '{keep}'")
            result = result.drop(columns=cols_to_drop)
            report.append(f"Dropped columns: {', '.join(cols_to_drop)}")

    if handle_outliers:
        num_cols = result.select_dtypes(include=[np.number]).columns
        removed_any = False
        for col in num_cols:
            if len(result) <= 3:
                report.append(f"Too few rows ({len(result)}) to detect outliers in '{col}'")
                continue
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - outlier_factor * iqr
            upper = q3 + outlier_factor * iqr
            mask = (result[col] >= lower) & (result[col] <= upper)
            n_removed = (~mask).sum()
            if n_removed > 0 and n_removed < len(result):
                result = result[mask]
                removed_any = True
                report.append(f"Removed {n_removed} outliers from '{col}'")
        if not removed_any:
            report.append("No outliers detected")

    return result, report


def fill_missing(df, strategy="auto", columns=None, fill_value=None):
    result = df.copy()
    report = []

    if columns is None:
        columns = result.columns.tolist()

    num_cols = result[columns].select_dtypes(include=[np.number]).columns
    cat_cols = result[columns].select_dtypes(include=["object", "category"]).columns

    for col in columns:
        if col not in result.columns:
            continue
        n_missing = result[col].isnull().sum()
        if n_missing == 0:
            continue

        if col in num_cols:
            if strategy == "auto" or strategy == "mean":
                val = result[col].mean()
                report.append(f"Filled {n_missing} missing values in '{col}' with mean ({val:.4f})")
            elif strategy == "median":
                val = result[col].median()
                report.append(f"Filled {n_missing} missing values in '{col}' with median ({val:.4f})")
            elif strategy == "zero":
                val = 0
                report.append(f"Filled {n_missing} missing values in '{col}' with 0")
            elif strategy == "constant" and fill_value is not None:
                val = fill_value
                report.append(f"Filled {n_missing} missing values in '{col}' with constant ({val})")
            elif strategy == "ffill":
                result[col] = result[col].ffill()
                report.append(f"Forward-filled {n_missing} missing values in '{col}'")
                continue
            elif strategy == "bfill":
                result[col] = result[col].bfill()
                report.append(f"Backward-filled {n_missing} missing values in '{col}'")
                continue
            else:
                val = result[col].mean()
                report.append(f"Filled {n_missing} missing values in '{col}' with mean ({val:.4f})")
            result[col] = result[col].fillna(val)

        elif col in cat_cols:
            if strategy == "mode" or strategy == "auto":
                val = result[col].mode().iloc[0] if not result[col].mode().empty else "Unknown"
                report.append(f"Filled {n_missing} missing values in '{col}' with mode ({val})")
            elif strategy == "constant" and fill_value is not None:
                val = fill_value
                report.append(f"Filled {n_missing} missing values in '{col}' with constant ({val})")
            else:
                val = result[col].mode().iloc[0] if not result[col].mode().empty else "Unknown"
                report.append(f"Filled {n_missing} missing values in '{col}' with mode ({val})")
            result[col] = result[col].fillna(val)

    return result, report


def split_data(df, target_col, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    y = df[target_col]
    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    X_num = X[num_cols].values.astype(np.float32)
    from sklearn.preprocessing import LabelEncoder
    X_cat_encoded = []
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        encoded = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        X_cat_encoded.append(encoded.reshape(-1, 1))

    if X_cat_encoded:
        X_processed = np.hstack([X_num] + X_cat_encoded).astype(np.float32)
    else:
        X_processed = X_num

    if y.dtype == "object" or str(y.dtype) == "category":
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y.values.astype(str))
        task_type = "classification"
    else:
        target_encoder = None
        y_encoded = y.values.astype(np.float32)
        task_type = "regression"

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=test_size, random_state=random_state
    )

    n_classes = len(np.unique(y_encoded)) if task_type == "classification" else 1

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(num_cols) + list(cat_cols),
        "target_name": target_col,
        "task_type": task_type,
        "n_classes": n_classes,
        "target_encoder": target_encoder,
        "input_dim": X_processed.shape[1],
    }


def normalize_data(X_train, X_test, method="minmax"):
    """Normalize features using training-set statistics.

    Parameters
    ----------
    X_train, X_test : np.ndarray
    method : 'minmax' or 'mean'

    Returns
    -------
    X_train_norm, X_test_norm, params
        params = {"method": ..., "min": ..., "max": ..., "mean": ..., "std": ...}
    """
    params = {"method": method}

    if method == "minmax":
        col_min = X_train.min(axis=0)
        col_max = X_train.max(axis=0)
        # Avoid division by zero for constant columns
        denom = col_max - col_min
        denom[denom == 0] = 1.0
        X_train_norm = (X_train - col_min) / denom
        X_test_norm = (X_test - col_min) / denom
        params["min"] = col_min.tolist()
        params["max"] = col_max.tolist()

    elif method == "mean":
        col_mean = X_train.mean(axis=0)
        col_std = X_train.std(axis=0)
        col_std[col_std == 0] = 1.0
        X_train_norm = (X_train - col_mean) / col_std
        X_test_norm = (X_test - col_mean) / col_std
        params["mean"] = col_mean.tolist()
        params["std"] = col_std.tolist()

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_train_norm, X_test_norm, params
