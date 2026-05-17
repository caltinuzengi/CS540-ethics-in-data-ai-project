"""
Privacy module — Edoardo Balzano
Privacy-aware preprocessing: sensitive feature handling and explanation-time privacy guards.
"""

import pandas as pd
import numpy as np


SENSITIVE_COLS = ["gender", "disability", "imd_band", "age_band", "highest_education", "region"]


def apply_feature_masking(df: pd.DataFrame, drop_sensitive: bool = True) -> pd.DataFrame:
    """
    Removes or generalizes sensitive demographic columns before model training.
    When drop_sensitive=True, columns are removed entirely (maximum privacy).
    When False, columns are binned/generalized (utility-privacy tradeoff).
    """
    df = df.copy()
    if drop_sensitive:
        cols_to_drop = [c for c in SENSITIVE_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)
    else:
        # Generalize imd_band into 3 buckets instead of 10
        if "imd_band" in df.columns:
            mapping = {
                "0-10%": "Low", "10-20%": "Low", "20-30%": "Low",
                "30-40%": "Medium", "40-50%": "Medium", "50-60%": "Medium",
                "60-70%": "High", "70-80%": "High", "80-90%": "High", "90-100%": "High",
            }
            df["imd_band_generalized"] = df["imd_band"].map(mapping).fillna("Unknown")
            df = df.drop(columns=["imd_band"])
    return df


def detect_proxy_features(
    df: pd.DataFrame, 
    sensitive_cols: list[str], 
    threshold: float = 0.5
) -> list[str]:
    """
    Identifies features that are highly correlated with sensitive columns.
    These 'proxy' features can leak private information if included in explanations.
    """
    # Use only numeric columns for correlation (or encoded versions)
    numeric_df = df.select_dtypes(include=[np.number])
    
    proxies = set()
    for s_col in sensitive_cols:
        # Check if sensitive col is in the numeric df (might be encoded)
        s_target = s_col if s_col in numeric_df.columns else f"{s_col}_enc"
        if s_target not in numeric_df.columns:
            continue
            
        correlations = numeric_df.corr()[s_target].abs()
        high_corr = correlations[correlations > threshold].index.tolist()
        
        for feature in high_corr:
            if feature != s_target and not feature.startswith(tuple(sensitive_cols)):
                proxies.add(feature)
                
    print(f"Detected {len(proxies)} proxy features with correlation > {threshold}: {list(proxies)}")
    return list(proxies)


def suppress_sensitive_from_explanation(
    shap_values: pd.Series,
    sensitive_cols: list[str],
    proxy_cols: list[str] = None
) -> pd.Series:
    """
    Removes sensitive and proxy columns from a SHAP Series.
    """
    proxy_cols = proxy_cols or []
    to_drop = []
    
    # Identify encoded sensitive columns
    for col in sensitive_cols:
        to_drop.append(col)
        to_drop.append(f"{col}_enc")
        
    to_drop.extend(proxy_cols)
    
    return shap_values.drop(
        labels=[c for c in to_drop if c in shap_values.index],
        errors="ignore",
    )


def enforce_k_anonymity(
    df: pd.DataFrame, 
    quasi_identifiers: list[str], 
    k: int = 5,
    action: str = "suppress"
) -> pd.DataFrame:
    """
    Ensures the DataFrame satisfies k-anonymity for the given quasi-identifiers.
    
    Args:
        df: Input DataFrame.
        quasi_identifiers: Columns that could be used for re-identification.
        k: Minimum group size.
        action: 'suppress' to remove rows in small groups, or 'mask' to replace them with NaN.
    
    Returns:
        A k-anonymous DataFrame.
    """
    df_clean = df.copy()
    present_qi = [q for q in quasi_identifiers if q in df_clean.columns]
    
    # Calculate group sizes
    group_counts = df_clean.groupby(present_qi).size().reset_index(name="_k_count")
    df_with_counts = df_clean.merge(group_counts, on=present_qi, how="left")
    
    # Identify rows that violate k-anonymity
    violators_mask = df_with_counts["_k_count"] < k
    
    if action == "suppress":
        # Remove the rows entirely
        df_result = df_clean[~violators_mask].copy()
    elif action == "mask":
        # Keep the rows but mask the quasi-identifiers
        df_clean.loc[violators_mask, present_qi] = np.nan
        df_result = df_clean
    else:
        raise ValueError("Action must be 'suppress' or 'mask'")
        
    print(f"k-anonymity ({k}) enforced. Action: {action}. Rows affected: {violators_mask.sum()}")
    return df_result


def generalize_categorical(
    df: pd.DataFrame, 
    column: str, 
    mapping: dict
) -> pd.DataFrame:
    """
    Generalizes a categorical column based on a provided hierarchy mapping.
    Helps satisfy k-anonymity by reducing granularity.
    """
    df = df.copy()
    if column in df.columns:
        df[column] = df[column].map(mapping).fillna(df[column])
    return df


def get_default_generalization_maps() -> dict:
    """Provides standard generalization hierarchies for OULAD sensitive features."""
    return {
        "imd_band": {
            "0-10%": "0-30%", "10-20%": "0-30%", "20-30%": "0-30%",
            "30-40%": "30-60%", "40-50%": "30-60%", "50-60%": "30-60%",
            "60-70%": "60-100%", "70-80%": "60-100%", "80-90%": "60-100%", "90-100%": "60-100%",
        },
        "age_band": {
            "0-35": "Under 55",
            "35-55": "Under 55",
            "55<=": "Over 55"
        },
        "highest_education": {
            "No Formal quals": "Pre-HE",
            "Lower Than A Level": "Pre-HE",
            "A Level or Equivalent": "HE Entry",
            "HE Qualification": "Post-Secondary",
            "Post Graduate Qualification": "Post-Secondary"
        }
    }
