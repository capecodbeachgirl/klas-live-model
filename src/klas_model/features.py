from __future__ import annotations

import pandas as pd


def add_intraday_features(obs: pd.DataFrame) -> pd.DataFrame:
    """Create KLAS-specific heating-curve features from time-ordered observations.

    Expected columns: timestamp, temp_f, dewpoint_f, wind_speed_kt, cloud_fraction,
    precip_in. Additional source columns are preserved.
    """
    out = obs.sort_values("timestamp").copy()
    out["temp_change_1obs_f"] = out["temp_f"].diff()
    out["temp_change_3obs_f"] = out["temp_f"].diff(3)
    if "dewpoint_f" in out:
        out["dewpoint_depression_f"] = out["temp_f"] - out["dewpoint_f"]
    if "precip_in" in out:
        out["rain_recent"] = out["precip_in"].fillna(0).rolling(3, min_periods=1).sum() > 0
    return out
