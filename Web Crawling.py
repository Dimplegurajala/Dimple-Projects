#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Step 1: WebScrapping using Chromedriver

import sqlite3
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

# Set up the WebDriver
driver = webdriver.Chrome()

# Navigate to the webpage containing the table
url = "https://tradingeconomics.com/matrix"
driver.get(url)

# Find the table using its ID
table = driver.find_element(By.ID, 'matrix')

# Initialize an empty list to store rows and column names
data = []
columns = []

# Extract column names from the table header
header_row = table.find_element(By.TAG_NAME, 'thead').find_element(By.TAG_NAME, 'tr')
header_cells = header_row.find_elements(By.TAG_NAME, 'th')
columns = [cell.text for cell in header_cells]

# Iterate through rows in the table
for row in table.find_elements(By.TAG_NAME, 'tr'):
    # Iterate through cells in each row
    cells = row.find_elements(By.TAG_NAME, 'td')
    
    # Extract text from each cell and append to the data list
    row_data = [cell.text for cell in cells]
    data.append(row_data)

# Create a Pandas DataFrame from the data and set column names
df = pd.DataFrame(data, columns=columns)

# Close the browser
driver.quit()


# In[15]:


#Step 2: Connecting to the Database

# Connect to SQLite database
conn = sqlite3.connect('project.db')

# Write the DataFrame to a SQLite table named 'your_table_name'
df.to_sql('matrix_table', conn, index=False, if_exists='replace')

# Commit changes and close the connection
conn.commit()
conn.close()

# Display the DataFrame
print(df)


# In[19]:


#Step 3:Downloading file as csv

import os
import sqlite3
import csv

# Connect to the SQLite database
conn = sqlite3.connect('project.db')

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Execute a simple SELECT query to fetch all rows from the table
cursor.execute("SELECT * FROM matrix_table")

# Fetch all rows as a list of tuples
rows = cursor.fetchall()

# Get the column names from the table
cursor.execute("PRAGMA table_info(matrix_table)")
columns = [column[1] for column in cursor.fetchall()]

# Close the cursor and connection
cursor.close()
conn.close()

# Specify the full path for the CSV file
csv_file_path = r'C:\Users\sawan\downloads\\matrix.csv'

# Ensure that the directory structure exists
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

# Write column names and rows to a CSV file
try:
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write the column names as the header
        csv_writer.writerow(columns)
        
        # Write the rows
        csv_writer.writerows(rows)

    print(f'Data has been saved to {csv_file_path}')
except Exception as e:
    print(f'Error: {e}')


# In[25]:


#Descriptive Analysis
import pandas as pd

# Replace 'matrix.csv' with the actual file path
file_path = r'C:\\Users\\sawan\\downloads\\matrix.csv'

# Read data from CSV file
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print("First Few Rows of the Dataset:")
print(data.head())

# Descriptive statistics
summary_stats = data.describe()
print("\nDescriptive Statistics:")
print(summary_stats)

# Display information about the dataset
print("\nDataset Information:")
print(data.info())


# In[26]:


#Regression Analysis
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = 'C:\\Users\\sawan\\downloads\\matrix.csv'
economic_data_df = pd.read_csv(file_path)

# Assuming 'GDP' is the dependent variable
dependent_variable = 'GDP'

# List of independent variables
independent_variables = ['GDP Growth', 'Interest Rate', 'Inflation Rate', 'Jobless Rate', 
                         'Gov. Budget', 'Debt/GDP', 'Current Account', 'Population']

# Selecting the relevant columns from the DataFrame
regression_data = economic_data_df[[dependent_variable] + independent_variables].dropna()

# Adding a constant term for the intercept
regression_data = sm.add_constant(regression_data)

# Performing the regression
model = sm.OLS(regression_data[dependent_variable], regression_data[independent_variables]).fit()

# Print the model summary
print(model.summary())

# Plotting the regression line with gridlines
sns.regplot(x=model.predict(regression_data[independent_variables]), y=regression_data[dependent_variable])
plt.xlabel('Predicted GDP (In Billions)')
plt.ylabel('Actual GDP (In Billions)')
plt.title('Regression Line')
plt.grid(True)  # Add gridlines
plt.show()

economic_data_df = pd.read_csv(file_path)

# Assuming 'GDP' is the dependent variable
dependent_variable = 'GDP'

# List of independent variables
independent_variables = ['GDP Growth', 'Interest Rate', 'Inflation Rate', 'Jobless Rate', 
                         'Gov. Budget', 'Debt/GDP', 'Current Account', 'Population']

# Selecting the relevant columns from the DataFrame
regression_data = economic_data_df[[dependent_variable] + independent_variables].dropna()

# Adding a constant term for the intercept
regression_data = sm.add_constant(regression_data)

# Performing the regression
model = sm.OLS(regression_data[dependent_variable], regression_data[independent_variables]).fit()

# Print the model summary
print(model.summary())

# Plotting the regression line with gridlines
sns.regplot(x=model.predict(regression_data[independent_variables]), y=regression_data[dependent_variable])
plt.xlabel('Predicted GDP (In Billions)')
plt.ylabel('Actual GDP (In Billions)')
plt.title('Regression Line')
plt.grid(True)  # Add gridlines
plt.show()


# In[27]:


#Visualization- 1.Boxplot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'C:\\Users\\sawan\\downloads\\matrix.csv'
df = pd.read_csv(file_path)

all_columns_df = df.copy()

# Specify the column you want to analyze
column_name = 'Inflation Rate'  
if column_name in all_columns_df.columns:
    # Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='variable', y='value', data=pd.melt(all_columns_df[[column_name]]))
    plt.title(f'Boxplot of {column_name}')
    plt.xlabel('Indicators')
    plt.ylabel(column_name)
    plt.show()


# In[28]:


# Visualization - 2. Bar chart

# Sort DataFrame by Gov. Budget in descending order and select top 10
df_top10 = df.sort_values(by='Gov. Budget', ascending=False).head(10)

# Sort DataFrame by Gov. Budget in ascending order and select bottom 10
df_least10 = df.sort_values(by='Gov. Budget', ascending=True).head(10)

# Plot: Gov. Budget Comparison for Top 10 Countries
plt.figure(figsize=(10, 6))
plt.bar(df_top10['Country'], df_top10['Gov. Budget'], color='skyblue')
plt.xlabel('Country')
plt.ylabel('Gov. Budget (% of GDP)')
plt.title('Top 10 Countries with High Gov Budget')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()

# Plot: Gov. Budget Comparison for Least 10 Countries
plt.figure(figsize=(10, 6))
plt.bar(df_least10['Country'], df_least10['Gov. Budget'], color='lightcoral')
plt.xlabel('Country')
plt.ylabel('Gov. Budget (% of GDP)')
plt.title('Bottom 10 Countries with Low Gov Budget')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()


# In[29]:


#Visualization - 

#3. Line Graph
# Visualization - Line Graph

# Sort DataFrame by GDP Growth Rate in descending order and select top 10
df_top10_growth = df.sort_values(by='GDP Growth', ascending=False).head(10)

# Sort DataFrame by GDP Growth Rate in ascending order and select bottom 10
df_least10_growth = df.sort_values(by='GDP Growth', ascending=True).head(10)

# Plot: GDP Growth Rate for Top 10 Countries
plt.figure(figsize=(10, 6))
plt.plot(df_top10_growth['Country'], df_top10_growth['GDP Growth'], marker='o', linestyle='-', color='green', label='Top 10')
plt.xlabel('Country')
plt.ylabel('GDP Growth Rate (%)')
plt.title('Top 10 Countries - GDP Growth Rate')
plt.legend()
plt.xticks(rotation=45, ha='right')  
plt.grid(True)
plt.show()


# In[30]:


# Plot: GDP Growth Rate for Least 10 Countries
plt.figure(figsize=(10, 6))
plt.plot(df_least10_growth['Country'], df_least10_growth['GDP Growth'], marker='o', linestyle='-', color='orange', label='Bottom 10')
plt.xlabel('Country')
plt.ylabel('GDP Growth Rate (%)')
plt.title('Bottom 10 Countries - GDP Growth Rate')
plt.legend()
plt.xticks(rotation=45, ha='right')  
plt.grid(True)
plt.show()


# In[31]:


#Sentiment Analysis on GDP

import pandas as pd

# Read data from CSV file
file_path = 'C:\\Users\\sawan\\downloads\\matrix.csv'
df = pd.read_csv(file_path)

# Define column names for thresholds
gdp_column = 'GDP'

# Define thresholds for GDP categories
developed_threshold = 1000
developing_threshold = 100

# Create a new column 'Development_Category' with default value 'Least Developed'
df['Development_Category'] = 'Least Developed'

# Categorize countries into 'Developed', 'Developing', and 'Least Developed'
df.loc[df[gdp_column] > developed_threshold, 'Development_Category'] = 'Developed'
df.loc[(df[gdp_column] <= developed_threshold) & (df[gdp_column] > developing_threshold),
                                                   'Development_Category'] = 'Developing'
df.loc[df[gdp_column] <= developing_threshold, 'Development_Category'] = 'Least Developed'

# Set the option to display all rows
pd.set_option('display.max_rows', None)

# Display the result
print(df[['Country', 'Development_Category']])

# Reset the option to its default value if needed
pd.reset_option('display.max_rows')


# In[32]:


import pandas as pd

# Read data from CSV file
file_path = r'C:\\Users\\sawan\\downloads\\matrix.csv'

try:
    # Read data from CSV file
    data = pd.read_csv(file_path)

    # Replace 'GDP' column with 'Population' column
    data['gdp'] = data['Population']

    # Define column names for thresholds
    population_column = 'gdp'

    # Define thresholds for Population categories
    highly_populated_threshold = 100
    least_populated_threshold = 10

    # Create a new column 'Population_Category' with default value 'Moderately Populated'
    data['Population_Category'] = 'Moderately Populated'

    # Categorize countries into 'Highly Populated', 'Least Populated', and 'Moderately Populated'
    data.loc[data[population_column] > highly_populated_threshold, 'Population_Category'] = 'Highly Populated'
    data.loc[data[population_column] <= least_populated_threshold, 'Population_Category'] = 'Least Populated'

    # Display all rows of the modified dataframe
    pd.set_option('display.max_rows', None)
    print("Modified Dataset:")
    print(data[['Country', 'Population_Category']])

except FileNotFoundError:
    print(f"File not found at path: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Reset the option to its default value
    pd.reset_option('display.max_rows')


# In[33]:


#Text Mining on Jobless Rate Column

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Read data from CSV file (replace 'matrix.csv' with the actual file path)
file_path = 'C:\\Users\\sawan\\downloads\\matrix.csv'
df = pd.read_csv(file_path)

# Handling missing values in the 'Population' column
df['Jobless Rate'] = df['Jobless Rate'].fillna(0)

# Create a dictionary with country names and corresponding Population values
country_population_dict = dict(zip(df['Country'], df['Jobless Rate']))

# Generate the Word Cloud with word frequencies based on Population values
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis',
                      max_words=100, collocations=False)

# Generate the Word Cloud with specified colors and sizes
wordcloud.generate_from_frequencies(country_population_dict)

# Plot the Word Cloud
plt.figure(figsize=(8, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

