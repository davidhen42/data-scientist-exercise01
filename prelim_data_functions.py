import pandas as pd
import numpy as np


# Function to replace given values for education variable with broader, more meaningful categories.
def educval_to_category(unprepped_df):
    # Create dictionary of each education category and list of responses that fall into category.
    recordval_to_educ_cat = {}
    educ_cats = {'no_ms': ['Preschool','1st-4th'], 'no_hs':['5th-6th', '7th-8th'], 
                    'some_hs':['9th', '10th', '11th', '12th'], 'hs_grad':['HS-grad'], 
                    'assoc_prof':['Assoc-acdm', 'Assos-voc', 'Prof-school'], 
                    'some_college':['Some-college'], 'college_grad':['Bachelors'], 
                    'masters':['Masters'], 'doctorate':['Doctorate']}
    # Create dictionary of each record value response and its category. 
    for broad_category, record_values in educ_cats.items():
        for record_value in record_values: recordval_to_educ_cat[record_value] = broad_category
    # Replace each record value in the dataframe with its category using dictionary.
    prepped_df = unprepped_df.replace(recordval_to_educ_cat)
    print(f'Replaced record value for education level with more informative category value for {len(prepped_df)} rows.')
    return prepped_df



# Function to replace values for country of origin with countries' 1996 income categories (high, low, etc.)
    """ Categorizing countries by income level in 1996 makes model extensible to people 
        outside of the ~40 countries included in dataset. Utilize country per capita GDP levels 
        and income level cutoffs in World Bank dataset generated at https://data.worldbank.org/indicator/NY.GDP.PCAP.CD
        to categorize countries of origin by  
        1996 per capita GDP levels. """
def country_to_income_level(unprepped_df):
    # Load in world bank data. Replace missing values with value from nearest complete year, 2002.
    wbank_96 = pd.read_csv('wbank_96_gdp_per_capita.csv', header=0, usecols=['Country Name', '1996','2002'])
    wbank_96.loc[wbank_96['1996'].isnull(),'1996'] = wbank_96['2002']
    # Uncomment code below to check where country names in census and wbank_96 differ.
    """
    for value in set(unprepped_df['country']).sorted():
    if value not in set(wbank_96['Country Name']).sorted():
        print(value)
    print(set(wbank_96['Country Name']).sorted())
    """

    # Function to replace countries in unprepped df that do not match wbank_96 with corresponding wbank_96 names. 
    def standardize_countries(df_country):
        df_country = str(df_country)
        # Small number of differences, hard code unprepped df value to replacement value dictionary.
        replacement_dict = {"Columbia": "Colombia",
                        "Scotland": "United Kingdom",
                        "England": "United Kingdom",
                        "Holand-Netherlands":"Netherlands",
                        "Outlying-US(Guam-USVI-etc)":"Guam",
                        "Trinadad&Tobago":"Trinidad and Tobago",
                        "Hong":"Hong Kong SAR, China",
                        "Iran": "Iran, Islamic Rep.",
                        "Taiwan":"China",
                        "Yugoslavia":"Serbia",
                        "Laos":"Lao PDR"}
        # Replace all problem countries in unprepped_df with corresponding wbank_96 country value.
        if df_country in replacement_dict.keys(): df_country = replacement_dict[df_country]
        df_country = df_country.replace('-', ' ')
        # If country not among problem countries but not in wbank_96, replace country with null.
        if df_country != str(np.nan) and df_country not in set(wbank_96['Country Name']):
            if df_country == "South": df_country = str(np.nan)
            else:
                print(f'WARNING {df_country} not in World Bank dataset. Replacing with nan')
                df_country = str(np.nan)
        return df_country

    # Function to assign income level to each country in df. 
    def assign_income_levels(df_country):
        income_level = None
        # Get lower cutoffs for each income group from wbank96 dataset.
        income_cutoffs = {'Low': wbank_96[wbank_96['Country Name']=='Low income']['1996'].values[0],
                  'Lower Middle Income': wbank_96[wbank_96['Country Name']=='Lower middle income']['1996'].values[0],
                  'Middle Income': wbank_96[wbank_96['Country Name']=='Middle income']['1996'].values[0],
                  'Upper Middle Income': wbank_96[wbank_96['Country Name']=='Upper middle income']['1996'].values[0]}
        # Define non-immigrants as non immigrants, rather than by country of origin income.
        if df_country == "United States": income_level = "US"
        elif df_country == str(np.nan): income_level = np.nan
        else:
            # For immigrants, loop through income brackets and bracket upper cutoffs, starting with lowest bracket.
            for income_bracket, upper_cuttoff in income_cutoffs.items():
                #if df_country in set(wbank_96['Country Name']):
            # If country of origin gdp < bracket's upper cuttoff, country belongs in bracket. 
                if wbank_96[wbank_96['Country Name'] == df_country]['1996'].values[0] < income_cutoffs[income_bracket]:
                    income_level = income_bracket
                    break
                elif income_bracket == 'Upper Middle Income':
                    income_level = 'High Income'
                    break
                else:
                    continue
            else:
                print(f"WARNING: {df_country} not in world bank dataset.")
        return income_level
    
    # Apply both functions to appropriate column of unprepped dataframe.
    unprepped_df['country'] = unprepped_df['country'].apply(standardize_countries)
    unprepped_df['origin_c_income'] = None
    unprepped_df['origin_c_income'] = unprepped_df['country'].apply(assign_income_levels)
    prepped_df = unprepped_df.drop(columns=['country'])
    print(f'Replaced country of origin with income category of country of origin for {len(prepped_df)} rows.')
    return prepped_df

# Should not be able to have any records with positive capital gains and positive capital losses.
# Combine into one capital gains column that can contain negative losses, and drop the capital losses column.
def combine_gains_losses(unprepped_df):
    # Check that there are no records for which both capital gain and loss are > 0. Then proceed, or print error.
    if len(unprepped_df[(unprepped_df['capital_gain'] > 0) & (unprepped_df['capital_loss'] > 0)]) == 0:
        unprepped_df['capital_gain'] = unprepped_df['capital_gain'] - unprepped_df['capital_loss']
        prepped_df = unprepped_df.drop(columns=['capital_loss'])
        print(f'Replaced capital gains with capital gains - capital losses for {len(prepped_df)} rows.')
    else:
        print(f"WARNING: Rows found with nonzero values for both capital gain and loss. Gains and losses columns will not be combined or modified.")
        prepped_df = unprepped_df
    return prepped_df




        




