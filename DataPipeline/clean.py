import pandas as pd
from pandas.util import hash_pandas_object
from time import time
from Utils.gpu_check import _gpu_available
import numpy as np 
from typing import Literal

Backend = Literal["auto", "cpu", "gpu"]

def clean(
    df: pd.DataFrame,
    timestamp_col: str = "Date",
    drop_duplicate_rows: bool = True,
    drop_duplicate_cols: bool = True,
    drop_constant_columns: bool = True,
    drop_constant_rows: bool = True,
    replace_placeholders: bool = True, 
    placeholders = ("Null","null","NULL","NaN","nan","NAN","None","none","NONE"),  
    interpolate_missing: bool = False,
    convert_numeric: bool = True,
    sort_index: bool = True,
    verbose: bool = False,
    backend: Backend = "auto",
):

    started_as_pandas = isinstance(df, pd.DataFrame)
    using_gpu = False
    cudf = None

    if backend in ("gpu", "auto"):
        if _gpu_available():
            import cudf  # type: ignore
            using_gpu = True
        else:
            using_gpu = False

    if using_gpu and started_as_pandas:
        df = cudf.from_pandas(df)  # type: ignore

    start_total = time()
    df = df.copy()
    
    
    if verbose:
        shape = (int(df.shape[0]), int(df.shape[1]))
        print(f"[CLEAN] Starting. shape={shape}")


    timestamp = None
    if timestamp_col in df.columns:
        timestamp = df[timestamp_col].copy()
        if verbose:
            print(f"[CLEAN] Preserving timestamp column: '{timestamp_col}'")



    if drop_duplicate_rows:
        if verbose:
            print("[CLEAN] Dropping duplicate rows...")
        start_time = time()
        before = df.shape
        df = df.drop_duplicates()
        after = df.shape
        if verbose:
            print(f"[CLEAN] Original Row Count: {before[0]}, "
                  f"Row Count After Removing Duplicates: {after[0]}, "
                  f"Total Removed: {before[0] - after[0]} in {time() - start_time:.5f}s")



    if drop_duplicate_cols:
        if verbose:
            print("[CLEAN] Dropping duplicate columns...")
        start_time = time()
        before = df.shape

        if using_gpu:
            col_hash = {c: cudf.hash(df[c]).sum().item() for c in df.columns}  # type: ignore
        else:
            col_hash = {c: hash_pandas_object(df[c], index=False).sum() for c in df.columns}
        seen, keep = set(), []
        for c, h in col_hash.items():
            if h not in seen:
                seen.add(h)
                keep.append(c)
        df = df[keep]
        
        after = df.shape
        if verbose:
            print(f"[CLEAN] Original Column Count: {before[1]}, "
                  f"Column Count After Removing Duplicates: {after[1]}, "
                  f"Total Removed: {before[1] - after[1]} in {time() - start_time:.5f}s")



    if drop_constant_columns:
        if verbose:
            print("[CLEAN] Dropping constant columns...")
        start_time = time()
        before = df.shape
        
        num_cols = df.select_dtypes(include=['number'])
        if num_cols.shape[1] > 0:
            nunique = num_cols.nunique(dropna=True)
            constant_cols = nunique[nunique <= 1].index.tolist()
            if constant_cols:
                df = df.drop(columns=constant_cols)
        else:
            constant_cols = []

        after = df.shape
        if verbose:
            print(f"[CLEAN] Dropped {len(constant_cols)} constant columns (Columns: {before[1]} → {after[1]}) in {time() - start_time:.3f}s")



    if drop_constant_rows:
        if verbose: print('[CLEAN] Dropping constant rows...')
        start_time = time()
        before = df.shape

        num = df.select_dtypes(include=['number'])
        if num.shape[1] > 0:
            try:
                row_min = num.min(axis=1, skipna=True)
                row_max = num.max(axis=1, skipna=True)
                non_empty = num.count(axis=1) > 0
                const_mask = (row_min == row_max) & non_empty
                df = df.loc[~const_mask]
            except Exception:
                if verbose:
                    print("[CLEAN] Skipping constant-row drop on this backend (row-wise reductions unsupported).")

        after = df.shape
        if verbose:
            print(f"[CLEAN] Dropped {before[0] - after[0]} constant rows (Rows: {before[0]} → {after[0]}) in {time() - start_time:.3f}s")



    if replace_placeholders: 
        if verbose:
            print("[CLEAN] Checking for common placeholder values...")
        start_time = time()
        df = df.replace(to_replace=placeholders, value=np.nan)
        if verbose:
            nulls = df.isnull().sum().sum()
            print(f"[CLEAN] Total nulls after placeholder replacement: {nulls} in {time() - start_time:.5f}s")



    if sort_index:
        if verbose:
            print("[CLEAN] Sorting index...")
        start_time = time()
        df = df.sort_index()
        if verbose:
            print(f"[CLEAN] Index sorted in {time() - start_time:.3f}s") 
        

    
    if interpolate_missing:
        if verbose:
            print("[CLEAN] Interpolating missing values...")
        start_time = time()
        na_before = df.isna().sum().sum()
        try:
            na_before = int( na_before)
        except Exception:
            pass
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
        na_after = df.isna().sum().sum()
        try:
            na_after = int( na_after)
        except Exception:
            pass
        if verbose:
            print(f"[CLEAN] Missing values: {int(na_before)} → {int(na_after)} (Δ={int(na_before - na_after)}) in {time() - start_time:.3f}s")



    if timestamp_col in df.columns:
        df = df.drop(columns=[timestamp_col])



    if convert_numeric:
        if verbose:
            print("[CLEAN] Converting to numeric types...")
        start_time = time()
        obj_cols = df.select_dtypes(include=['object', 'string']).columns

        if len(obj_cols) > 0:
            if using_gpu:
                try:
                    import cudf  # type: ignore
                    for c in obj_cols:
                        try:
                            df[c] = cudf.to_numeric(df[c], errors='coerce') 
                        except Exception:
                            df[c] = df[c].astype('float64') 
                except Exception:
                    if verbose:
                        print("[CLEAN] Skipping numeric conversion on GPU (unsupported cuDF version).")
            else:
                for c in obj_cols:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

        if verbose:
            print(f"[CLEAN] Converted numeric types in {time() - start_time:.3f}s")

    if timestamp is not None:
        df[timestamp_col] = timestamp

    if started_as_pandas and using_gpu:
        df = df.to_pandas()

    if verbose:
        print(f"[CLEAN] Cleaning complete in {time() - start_total:.3f}s. Final shape: {df.shape}")

    return df
