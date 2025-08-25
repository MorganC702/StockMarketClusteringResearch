from typing import Optional, Literal
import pandas as pd

def engineer(
    df: pd.DataFrame,
    time_block_feature: Optional[Literal['3m','5m','15m','1H','4H','1D']] = "1H",
    domain_features: bool = False,
    normalization: Optional[Literal['Volatility']] = None,
    lagged_features: Optional[int] = 10,  
    pairwise_features: bool = False,
    verbose: bool = False,
):
    
    if verbose:
        print("[ENGINEER] Engineering Features")
    
    return df
    