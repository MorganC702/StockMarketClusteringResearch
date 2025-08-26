import pandas as pd
from pandas.util import hash_pandas_object
from time import time
from Utils.gpu_check import _gpu_available
from typing import Literal, Optional
import numpy as np 


def clean(
    df: pd.DataFrame,
    timestamp_col: str = "Date",
    symbol_col: str = "Symbol",
    drop_duplicate_rows: bool = True,
    drop_duplicate_cols: bool = True,
    drop_constant_columns: bool = True,
    drop_constant_rows: bool = True,
    replace_placeholders: bool = True, 
    placeholders = ("Null","null","NULL","NaN","nan","NAN","None","none","NONE"),  
    interpolate_missing: bool = False,
    convert_numeric: bool = True,
    sort_by: Optional[Literal["index", "timestamp"]]= "timestamp",
    verbose: bool = False,
    backend: Literal["auto", "cpu", "gpu"] = "auto",
):
    # Track the time taken to clean the data. 
    start_total = time()
    df = df.copy()
    
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


    if verbose:
        print("\n######################################################")
        print(f"Backend Is Utilizing {'GPU' if using_gpu else 'CPU'}")
        print("######################################################\n")
        
    
    if verbose:
        shape = (int(df.shape[0]), int(df.shape[1]))
        print(f"[---CLEAN---] Starting Shape={shape}")


    # Columns to protect from modification
    protect = {
        symbol_col, 
        timestamp_col
    }
    if verbose:
        print(f"[---CLEAN---] Preserving: {symbol_col} and {timestamp_col}\n")


    # Remove Duplicate Columns
    if drop_duplicate_cols:
        if verbose:
            print("[---CLEAN---] Step 1: Remove Dupicate Columns.")
        t0 = time()
        s1 = df.shape
        
        if using_gpu:
            # GPU-safe: hash on host (pandas) to avoid cuDF API differences
            col_hash = {c: hash_pandas_object(df[c].to_pandas(), index=False).sum()
                        for c in df.columns}
        else:
            col_hash = {c: hash_pandas_object(df[c], index=False).sum() 
                        for c in df.columns}

        seen, keep = set(), []
        for c, h in col_hash.items():
            if h not in seen:
                seen.add(h)
                keep.append(c)
        df = df[keep]
        
        s2 = df.shape
        if verbose:
            print(f"[---CLEAN---] Original Column Count: {s1[1]}, "
                f"Column Count After Removing Duplicates: {s2[1]}, "
                f"Total Removed: {s1[1] - s2[1]} in {time() - t0:.5f}s\n")



    # Remove Duplicate Rows
    if drop_duplicate_rows:
        if verbose:
            print("[---CLEAN---] Step 2: Remove Dupicate Rows.")
        t0 = time()
        s1 = df.shape
        df = df.drop_duplicates()
        s2 = df.shape
        if verbose:
            print(f"[---CLEAN---] Original Row Count: {s1[0]}, "
                  f"Row Count After Removing Duplicates: {s2[0]}, "
                  f"Total Removed: {s1[0] - s2[0]} in {time() - t0:.5f}s\n")
             


    # Drop Columns that contain the same value for each row.
    if drop_constant_columns:
        if verbose:
            print("[---CLEAN---] Step 3: Remove Constant Columns.")
        
        t0 = time()
        s1 = df.shape
        
        # Do not drop constant object or category colums (Symbol is expected to be constant)
        num_cols = df.select_dtypes(include=[np.number])
        if num_cols.shape[1] > 0:
            nunique = num_cols.nunique(dropna=True)
            nunique_pd = nunique.to_pandas() if using_gpu else nunique
            constant_cols = nunique_pd[nunique_pd <= 1].index.tolist()
            if constant_cols:
                df = df.drop(columns=constant_cols)
        else:
            constant_cols = []

        s2 = df.shape
        
        if verbose:
            print(f"[---CLEAN---] Original Column Count: {s1[1]}, "
                f"Column Count After Removing Constants: {s2[1]}, "
                f"Total Removed: {s1[1] - s2[1]} in {time() - t0:.5f}s\n")
            
            
            
    # Drop Rows that contain the same value for each column.
    if drop_constant_rows:
        if verbose:
            print('[---CLEAN---] Step 4: Remove Constant Rows.')
        t0 = time()
        s1 = df.shape

        if using_gpu:
            # Do the computation in pandas, then convert back to cuDF
            df_pd = df.to_pandas()
            num_pd = df_pd.select_dtypes(include=[np.number])
            if num_pd.shape[1] > 0:
                row_min = num_pd.min(axis=1, skipna=True)
                row_max = num_pd.max(axis=1, skipna=True)
                non_empty = num_pd.count(axis=1) > 0
                const_mask = (row_min == row_max) & non_empty
                df_pd = df_pd.loc[~const_mask]
            df = cudf.from_pandas(df_pd)  # back to GPU frame
        else:
            # Native pandas path
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] > 0:
                row_min = num.min(axis=1, skipna=True)
                row_max = num.max(axis=1, skipna=True)
                non_empty = num.count(axis=1) > 0
                const_mask = (row_min == row_max) & non_empty
                df = df.loc[~const_mask]

        s2 = df.shape
        
        if verbose:
            print(f"[---CLEAN---] Original Row Count: {s1[0]}, "
                f"Column Count After Removing Constants: {s2[0]}, "
                f"Total Removed: {s1[0] - s2[0]} in {time() - t0:.5f}s\n")
            
                 
                 
    # Replace any placeholder values with NaNs  
    if replace_placeholders:
        if verbose:
            print("[---CLEAN---] Step 5: Replacing Placeholder Values")
        
        t0 = time()

        if using_gpu:
            str_cols = [c for c in df.columns if str(df[c].dtype) in ( "object", "str", "string" )]
            for c in str_cols:
                if c in protect:
                    continue
                df[c] = df[c].mask(df[c].isin(placeholders), None)
        else:
            str_cols = df.select_dtypes(include=["object","string"]).columns
            for c in str_cols:
                if c in protect:
                    continue
                df[c] = df[c].replace(to_replace=placeholders, value=np.nan)

        nulls = df.isnull().sum().sum()
        try: nulls = int(nulls)
        except: pass
        if verbose:
            print(f"[---CLEAN---] Total Nulls After Placeholder Replacement: {nulls} in {time() - t0:.5f}s\n")


    # Sort the values in the dataframe by either index or time/date. 
    if sort_by:
        if verbose:
            print(f"[---CLEAN---] Step 6: Sorting by {sort_by.capitalize()}.")
        t0 = time()
        
        if sort_by == "index":
            df = df.sort_index()
        elif sort_by == "timestamp":
            
            if timestamp_col not in df.columns:
                raise ValueError(f"Column '{timestamp_col}' not found for timestamp sorting.")
            
            if timestamp_col in df.columns:
                if using_gpu:
                    df[timestamp_col] = df[timestamp_col].astype("datetime64[ns]")
                else:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

            df = df.sort_values(by=timestamp_col)
        else:
            raise ValueError(f"Invalid value for sort_by: {sort_by}. Must be 'index' or 'timestamp'.")
        
        if verbose:
            print(f"[---CLEAN---] Sorted by {sort_by} in {time() - t0:.3f}s\n")
        

    # Forward Fill and Backward fill to inpute Missing Values / NaNs 
    if interpolate_missing:
        if verbose:
            print("[---CLEAN---] Step 7: Interpolating missing and NaN values.")
        
        t0 = time()
        
        n1 = df.isna().sum().sum()
        try:
            n1 = int(n1)
        except Exception:
            pass
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
        n2 = df.isna().sum().sum()
        
        try:
            n2 = int( n2)
        except Exception:
            pass
        
        if verbose:
            print(f"[---CLEAN---] Initial Invalid values Count: {int(n1)}, "
                f"Invalid Values Count After Interpolating: {int(n2)}, "
                f"Total Interpolated: {int(n1 - n2)} in {time() - t0:.5f}s\n'm")
            

    # Convert dataset to numerical values (ignoring timestamp_col and symbol_col)
    if convert_numeric:
        if verbose:
            print("[---CLEAN---] Step 8: Converting Data to Numerical Values.")
        
        t0 = time()

        obj_cols = [c for c in df.select_dtypes(include=['object','string']).columns if c not in protect]
        if obj_cols:
            if using_gpu:
                try:
                    import cudf
                    for c in obj_cols:
                        try:
                            df[c] = cudf.to_numeric(df[c], errors='coerce')
                        except Exception:
                            try:
                                df[c] = df[c].astype('float64')
                            except Exception:
                                pass
                except Exception:
                    if verbose:
                        print("[---CLEAN---] Skipping GPU numeric conversion (cuDF not available).")
            else:
                for c in obj_cols:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

        if verbose:
            print(f"[---CLEAN---] Converted dataset to numeric types in {time() - t0:.3f}s\n")

    
    if started_as_pandas and using_gpu:
        df = df.to_pandas()

    if verbose:
        print(f"[---CLEAN---] Cleaning complete in {time() - start_total:.3f}s. Final shape: {df.shape}\n")

    return df
