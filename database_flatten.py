import pandas as pd
import numpy as np
import sqlite3

#Establish database connection.
sql_connection = sqlite3.connect('exercise01.sqlite')
cursor = sql_connection.cursor()

#Write SQL query and execute. Joins used to flatten categorical/string fields. 
query_string = ("""SELECT records.id, records.age, 
                countries.name AS "country",
                education_levels.name AS "educ_level",
                marital_statuses.name AS "marital_status",
                occupations.name AS "occupation_name",
                races.name AS "race",
                relationships.name AS "relationship",
                sexes.name AS "sex",
                workclasses.name AS "workclass",
                capital_gain, capital_loss, hours_week, over_50k 
                FROM records LEFT JOIN countries ON country_id = countries.id  
                LEFT JOIN education_levels ON education_level_id = education_levels.id 
                LEFT JOIN marital_statuses ON marital_status_id = marital_statuses.id 
                LEFT JOIN occupations ON occupation_id = occupations.id 
                LEFT JOIN races ON race_id = races.id 
                LEFT JOIN relationships ON relationship_id = relationships.id 
                LEFT JOIN sexes ON sex_id = sexes.id 
                LEFT JOIN workclasses ON workclass_id = workclasses.id; """
               )
query_result = cursor.execute(query_string)

#Get column names. Save query result with column names to pandas dataframe.
table_cols = [column[0] for column in query_result.description]
full_data_df = pd.DataFrame.from_records(data = query_result.fetchall(), columns = table_cols).set_index('id')

#Export pandas dataframe to csv, print success message with number of rows. 
export_file_name = 'flattened_data.csv'
full_data_df.to_csv(export_file_name, index=False)
print(f"Success! {len(full_data_df)} rows exported to {export_file_name}.")
print(full_data_df.head())
