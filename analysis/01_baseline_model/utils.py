import os
from pathlib import Path

import numpy as np
import pandas as pd


def get_clean_dataset():
    input_dir = (
        Path(os.getenv("STORM_DATA_DIR")) / "analysis/01_baseline_model/input"
    )

    # Read in input data
    combined_input_data = pd.read_csv(input_dir / "combined_input_data.csv")

    # Set some values to 0
    # TODO: Check this equation
    def set_zeros(x):
        x_max = 25
        y_max = 50

        v_max = x[0]
        rainfall_max = x[1]
        damage = x[2]
        if pd.notnull(damage):
            value = damage
        elif v_max > x_max or rainfall_max > y_max:
            value = damage
        elif v_max < np.sqrt(
            (1 - (rainfall_max**2 / y_max**2)) * x_max**2
        ):
            value = 0
        # elif ((v_max < x_max)  and  (rainfall_max_6h < y_max) ):
        # elif (v_max < x_max ):
        # value = 0
        else:
            value = np.nan

        return value

    combined_input_data["DAM_perc_dmg"] = combined_input_data[
        ["HAZ_v_max", "HAZ_rainfall_Total", "DAM_perc_dmg"]
    ].apply(set_zeros, axis="columns")

    # TODO: I thought we want to keep NA damage values
    # Remove NA values
    combined_input_data = combined_input_data[
        combined_input_data["DAM_perc_dmg"].notnull()
    ]

    # Create cubed wind feature
    combined_input_data["HAZ_v_max_3"] = combined_input_data[
        "HAZ_v_max"
    ].apply(lambda x: x * x * x)

    # Drop Mun_Code since it's not a feature
    combined_input_data = combined_input_data.drop(columns="Mun_Code")

    return combined_input_data
