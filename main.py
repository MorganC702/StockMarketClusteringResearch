from DataPipeline.clean import clean
from DataPipeline.engineer import engineer
import pandas as pd
import os


def main():

    backend = os.environ.get("PIPELINE_BACKEND", "auto")

    df = pd.read_parquet("./demo_data/Symbol=A/year=2020/month=2020-08.parquet")
    
    df = clean(
        df,
        timestamp_col="Date",
        symbol_col="Symbol",
        drop_duplicate_rows=True,
        drop_duplicate_cols=True,
        drop_constant_columns=True,
        drop_constant_rows=True,
        replace_placeholders=True, 
        placeholders=("Null","null","NULL","NaN","nan","NAN","None","none","NONE"),  
        interpolate_missing=False,
        convert_numeric=True,
        sort_index=True,
        verbose=True,
        backend=backend,   
    )
    
    assert df["Symbol"].notna().all(), "Symbol got coerced to NaN somewhere"
    assert df["Symbol"].dtype == "object" or str(df["Symbol"].dtype) == "category"
    assert "Date" in df.columns
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    assert df["Date"].is_monotonic_increasing, "Index not sorted by Date"

    df = engineer(
        df,
        time_block_feature="1H",
        domain_features=False,
        normalization=None,
        lagged_features=10,  
        pairwise_features=False,
        verbose=True,
    )
    
    print(df.head(25))  


if __name__ == "__main__":
    main()
