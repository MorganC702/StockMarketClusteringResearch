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
    
    df = engineer(
        df,
        time_block_feature="1H",
        domain_features=False,
        normalization=None,
        lagged_features=10,  
        pairwise_features=False,
        verbose=False,
    )
    
    print(df.head())  


if __name__ == "__main__":
    main()
