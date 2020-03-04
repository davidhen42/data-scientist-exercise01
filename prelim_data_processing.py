""" Applies a number of user defined functions to dataset.
    Before exporting dataset to CSV for analysis.
    To turn off a user defined function, simply comment it out."""

import pandas as pd
import numpy as np
from prelim_data_functions import educval_to_category, country_to_income_level, combine_gains_losses

#Load flattened data into dataframe. Replace ?'s with null values.
flattened_df = pd.read_csv('./datasets/flattened_data.csv')
flattened_df = flattened_df.replace('?', np.nan)

#If know marital status and sex for a record, 'relationship' column is redundant. Drop.
flattened_df = flattened_df.drop(columns='relationship')
prelim_prep_df = flattened_df

# Groups education variable into more meaningful categories to aid interpretability.
# See educval_to_category in prelim_data_functions.py for documentation.
prelim_prep_df = educval_to_category(prelim_prep_df)

# Should not be able to have any records with positive capital gains and losses.
# If it is the case that gains and losses are opposite sides of same variable, combine into one variable,
# equal to capital gains - capital losses. See combine_gains_losses in prelim_data_functions.py for documentation.
prelim_prep_df = combine_gains_losses(prelim_prep_df)

# Categorize countries of origin by income level to further model extensibility.
# See country_to_income_level in prelim_data_functions.py for documentation.
prelim_prep_df = country_to_income_level(prelim_prep_df)

# Export to CSV
export_file_name = "./prelim_prepped_data.csv"
prelim_prep_df.to_csv(export_file_name, index=False)
print(f"Success! {len(prelim_prep_df)} rows exported to {export_file_name}.")
