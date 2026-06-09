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


def _infer_granularity(dt_series):
    """Infer time granularity from the median gap between sorted timestamps."""
    if len(dt_series) < 2:
        return "hour"
    sorted_dt = dt_series.sort_values()
    gaps = sorted_dt.diff().dropna()
    if gaps.empty:
        return "hour"
    median_gap = gaps.median()
    if median_gap >= pd.Timedelta(days=28):
        return "month"
    elif median_gap >= pd.Timedelta(days=1):
        return "day"
    elif median_gap >= pd.Timedelta(hours=1):
        return "hour"
    else:
        return "minute"


def time_encoding(df, time_col, granularity="auto"):
    """Generate temporal features from a datetime column.

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    granularity : str
        One of "auto", "year", "month", "day", "hour", "minute".
        "auto" infers from the median timestamp gap.

    Returns
    -------
    (df_with_encoding, encoded_names, used_granularity)
    """
    if time_col not in df.columns:
        return df, [], "auto"
    try:
        dt = pd.to_datetime(df[time_col])
    except Exception:
        return df, [], "auto"

    if granularity == "auto":
        granularity = _infer_granularity(dt)

    result = df.copy()
    encoded_names = []

    feature_map = {
        "year": ["year"],
        "month": ["year", "month"],
        "day": ["year", "month", "day", "weekday"],
        "hour": ["year", "month", "day", "weekday", "hour"],
        "minute": ["year", "month", "day", "weekday", "hour", "minute"],
    }
    selected = feature_map.get(granularity, feature_map["hour"])

    if "year" in selected and "year" not in result.columns:
        result["year"] = ((dt.dt.year - 2000) / 100.0).astype(np.float32)
        encoded_names.append("year")
    if "month" in selected and "month" not in result.columns:
        result["month"] = dt.dt.month.astype(np.float32) / 12.0
        encoded_names.append("month")
    if "day" in selected and "day" not in result.columns:
        result["day"] = dt.dt.day.astype(np.float32) / 31.0
        encoded_names.append("day")
    if "weekday" in selected and "weekday" not in result.columns:
        result["weekday"] = dt.dt.weekday.astype(np.float32) / 7.0
        encoded_names.append("weekday")
    if "hour" in selected and "hour" not in result.columns:
        result["hour"] = dt.dt.hour.astype(np.float32) / 23.0
        encoded_names.append("hour")
    if "minute" in selected and "minute" not in result.columns:
        result["minute"] = dt.dt.minute.astype(np.float32) / 59.0
        encoded_names.append("minute")

    return result, encoded_names, granularity


def _create_sliding_windows(X, y, seq_len, pred_len):
    """Create sliding windows from 2D arrays.

    X: (n_samples, n_features)
    y: (n_samples,) or (n_samples,)

    Returns:
        X_windows: (n_windows, seq_len, n_features) — 3D
        y_windows: (n_windows, pred_len)
    """
    n = X.shape[0]
    n_features = X.shape[1]
    windows = []
    targets = []

    for i in range(n - seq_len - pred_len + 1):
        x_win = X[i:i + seq_len]               # (seq_len, n_features)
        y_win = y[i + seq_len:i + seq_len + pred_len]
        windows.append(x_win)
        targets.append(y_win)

    if not windows:
        raise ValueError(
            f"Not enough samples ({n}) for seq_len={seq_len} + pred_len={pred_len}. "
            f"Need at least {seq_len + pred_len} rows."
        )

    return np.array(windows, dtype=np.float32), np.array(targets, dtype=np.float32)


def split_data(df, target_col, test_size=0.2, random_state=42,
               time_series=False, time_col=None, seq_len=10, pred_len=1,
               label_len=0, time_granularity="auto"):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    if time_series and time_col and time_col in df.columns:
        # --- Time series path: chronological split + sliding window ---
        # Parse time column
        dt = pd.to_datetime(df[time_col])
        sorted_idx = np.argsort(dt.values)
        df_sorted = df.iloc[sorted_idx].reset_index(drop=True)

        # Generate time encoding features
        df_enc, enc_names, used_granularity = time_encoding(df_sorted, time_col, granularity=time_granularity)

        # Extract target BEFORE dropping columns (y is independent of time_col handling)
        y_raw = df_enc[target_col].values
        if y_raw.dtype == object or str(y_raw.dtype) == "category":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y_raw.astype(str)).astype(np.float32)
            task_type = "classification"
            n_classes = len(np.unique(y))
        else:
            target_encoder = None
            y = y_raw.astype(np.float32)
            task_type = "regression"
            n_classes = 1

        # Build feature matrix: drop target col and raw time col
        cols_to_drop = [target_col]
        if time_col in df_enc.columns and time_col != target_col:
            cols_to_drop.append(time_col)
        X = df_enc.drop(columns=cols_to_drop)

        # Encode categorical features
        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns

        X_num = X[num_cols].values.astype(np.float32)
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            encoded = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            X_num = np.hstack([X_num, encoded.reshape(-1, 1)])

        feature_names = list(num_cols) + list(cat_cols)

        # Chronological split (no shuffle)
        split_idx = int(len(X_num) * (1 - test_size))
        if split_idx < seq_len + pred_len:
            raise ValueError(
                f"Training set too small ({split_idx} rows) for seq_len={seq_len} + pred_len={pred_len}"
            )

        X_train_raw = X_num[:split_idx]
        y_train_raw = y[:split_idx]
        X_test_raw = X_num[split_idx:]
        y_test_raw = y[split_idx:]

        # Sliding window
        X_train, y_train = _create_sliding_windows(X_train_raw, y_train_raw, seq_len, pred_len)
        X_test, y_test = _create_sliding_windows(X_test_raw, y_test_raw, seq_len, pred_len)

        n_features = X_num.shape[1]
        input_dim = n_features

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
            "target_name": target_col,
            "task_type": task_type,
            "n_classes": n_classes,
            "target_encoder": target_encoder,
            "input_dim": input_dim,
            "is_time_series": True,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "label_len": label_len,
            "time_col": time_col,
            "time_encoding_features": enc_names,
            "time_granularity": used_granularity,
        }

    # --- General (non-time-series) path: existing behavior ---
    y = df[target_col]
    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    X_num = X[num_cols].values.astype(np.float32)
    encoders = {}
    X_cat_encoded = []
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
        "is_time_series": False,
    }


def normalize_data(X_train, X_test, method="minmax"):
    """Normalize features using training-set statistics.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Can be 2D (n_samples, n_features) or 3D (n_windows, seq_len, n_features).
    method : 'minmax' or 'mean'

    Returns
    -------
    X_train_norm, X_test_norm, params
        params = {"method": ..., "min": ..., "max": ..., "mean": ..., "std": ...}
    """
    params = {"method": method}

    # Reshape 3D → 2D for per-feature statistics
    was_3d = X_train.ndim == 3
    orig_shape = X_train.shape
    test_shape = X_test.shape
    if was_3d:
        X_train = X_train.reshape(-1, X_train.shape[-1])
        X_test = X_test.reshape(-1, X_test.shape[-1])

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

    # Reshape back to 3D if input was 3D
    if was_3d:
        X_train_norm = X_train_norm.reshape(orig_shape)
        X_test_norm = X_test_norm.reshape(test_shape)

    return X_train_norm, X_test_norm, params


def normalize_target(y_train, y_test, method="mean"):
    """Normalize target values for regression so MSE is scale-independent.

    For classification tasks, y is passed through unchanged.
    Returns (y_train, y_test, params) where params contains mean/std
    needed to invert the transformation later.
    """
    if method is None:
        return y_train, y_test, {"method": None}

    params = {"method": method}
    if method == "mean":
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        if y_std == 0:
            y_std = 1.0
        y_train_norm = (y_train - y_mean) / y_std
        y_test_norm = (y_test - y_mean) / y_std
        params["mean"] = y_mean
        params["std"] = y_std
    elif method == "minmax":
        y_min = float(np.min(y_train))
        y_max = float(np.max(y_train))
        denom = y_max - y_min
        if denom == 0:
            denom = 1.0
        y_train_norm = (y_train - y_min) / denom
        y_test_norm = (y_test - y_min) / denom
        params["min"] = y_min
        params["max"] = y_max
    else:
        return y_train, y_test, {"method": None}

    return y_train_norm, y_test_norm, params


def denormalize_target(y, y_scaler):
    """Inverse transform of normalize_target — restore original scale.

    Parameters
    ----------
    y : np.ndarray
        Normalized values.
    y_scaler : dict or None
        Scaler params produced by ``normalize_target``.

    Returns
    -------
    np.ndarray with original scale restored, or *y* unchanged when
    *y_scaler* is ``None``.
    """
    if y_scaler is None:
        return y
    method = y_scaler.get("method")
    if method == "mean":
        return y * y_scaler["std"] + y_scaler["mean"]
    elif method == "minmax":
        return y * (y_scaler["max"] - y_scaler["min"]) + y_scaler["min"]
    return y
