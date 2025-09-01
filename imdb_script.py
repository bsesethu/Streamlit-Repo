import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

from functions_Assignment_2 import Data_Collection as DaC 
from functions_Assignment_2 import Data_Preperation as DaP
from functions_Assignment_2 import Data_Visualisation as DaV
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


df = pd.read_csv('imdb_top_1000.csv')

# From Assignment 2
# Check for mising values and document their percantage
    # Using functions written into functions_Assignment_2.py file
print('\n')
missing = DaC.check_missing(df) 

# Remove rows with missing critical data, from the following columns
df_new = DaC.remove_rows(df, 'Meta_score')
df_new = DaC.remove_rows(df_new, 'Gross')
print('\nAfter removal of null values')
missing_df_new = DaC.check_missing(df_new)


# Phase 2 Data preperation
# Drop duplicates
df_noDuplicates = DaC.dropDuplicates(df_new)
print('\nAfter dropped duplicates, Core info again:')
print(df_noDuplicates.info()) # There are no duplicates. Function would've kept the first occurance

# Testing the 'dropDuplicates' function
# df_G = pd.DataFrame({'Name': ['Cash', 'Money', 'Money', 'Ross'],
#                      'Sex': ['Yes', 'No', 'No', 'Yes'],
#                      'Number': [1, 5, 5, 8]
#                      })
# df_noDup = DaC.dropDuplicates(df_G)
# print(df_noDup) # The function works

# Convert runtime to numerical
newRuntime = []
cnt = 0
for row in df_noDuplicates['Runtime']:
    newRuntime.append(float(row.split(' ')[0])) # New list of numerical values only
    
# Reset indices so there are no skipped values
df_reset = df_noDuplicates
df_reset.reset_index(inplace= True) #NOTE Index values are now reset for all affilliated DFs being used from now on

df_noDuplicates['Runtime'] = pd.DataFrame({'Runtime': newRuntime}) # Make newRuntime list a DF column by overriding the original
print('\nCheck Runtime column values dtype:')
print(df_noDuplicates['Runtime'].info())

# For some reason I can't explain it's returning NaN values for the last 200 or so values of the 'Runtime' column, Let's fix this.
# Sorted: Reason for the issue is the index values of the df_duplicates DF, there are 750 values but indices go up to 997

# Extract Decade from Released_Year
df_res = DaP.newColumn_ReleasedDecade(df_noDuplicates)
print('\nFirst 5 entries in Released_Year and Released_Decade columns')
print(df_res[['Released_Year', 'Released_Decade']].head()) # It checks out

# Create a Lead_Actors column combining Star1, Star2, Star3, Star4.                          #NOTE IMPORTANT FUNCTION
df_res['Lead_Actors'] = df_res[['Star1', 'Star2', 'Star3', 'Star4']].agg(', '.join, axis= 1) # New column, all 4 cell values in one, using aggregate funtion where all values are joined along the columns axis
print('\nFirst 5 entries in new Lead_Actors column')
print(df_res['Lead_Actors'].head())


# Phase 3: Data Visualisation
# Histogram
# Convert 'IMDB_Rating' to a comparative rating with 'Meta_score', rating out of 100 not 10
df_res['IMDB_Rating'] = df_res['IMDB_Rating'] * 10
hist = DaV.doubleHistogram(df_res, 'IMDB_Rating', 'Meta_score', 24)

# Bar plot of Genre frequency
    # First need to find the top 10 genre frequency of occurance
df_Genre_top10 = DaP.genre_Frequencies(df_res) # Applying the function
print('\nTop 10 genres by Frequency')
print(df_Genre_top10)
# Generating the bar plot
bar = DaV.bar_plot(df_Genre_top10, 'Film_Genre', 'Frequency')

# Scatter plot of Gross vs. No_of_votes.
    # First convert Gross to number value
df_res['Gross'] = df_res['Gross'].str.replace(',', '') # Remove ','
df_res['Gross'] = pd.to_numeric(df_res['Gross'], errors= 'coerce') # 'coerce' to make errors encountered NaN
# print(df_res[df_res['Gross'] > 800e6]) # Just checking
scatter = DaV.scatterPlot(df_res, 'Gross', 'No_of_Votes')

# Box plot of IMDB_Rating by Certificate
df_res['Certificate'] = df_res['Certificate'].fillna('Unknown') # fil NaN values
# Return IMDB to a rating out of 10
df_res['IMDB_Rating'] = df_res['IMDB_Rating'] / 10 # Converting column values back to original state
box = DaV.boxplots(df_res, 'IMDB_Rating', 'Certificate')


# Phase 4: Applied Statistical Analysis
    # Compute mean, median, std for Gross, No_of_votes, IMDB_Rating.
print('\nComputed values for mean, median and standard deviation for the following columns:')
# print(df_res.info()) # Check dtypes
G_mean = round(np.mean(df_res['Gross']), 0)
G_median = round(np.median(df_res['Gross']), 0)
G_std = round(np.std(df_res['Gross']), 0)
print(f'Gross; mean: {G_mean}, median: {G_median}, std: {G_mean}') # G_std = G_mean for an exponential distributiion, of which this is.

# Histogram of Gross distribution
# hist_G = DaV.singleHistogram(df_res, 'Gross', 24, 'Gross', 'Frequency') # Checking the distribution, it's exponential hence std == mean
# Histogram of No_of_votes distribution
# hist_G = DaV.singleHistogram(df_res, 'No_of_Votes', 24, 'No_of_Votes', 'Frequency') # Checking the distribution, also exponential.

N_mean = round(np.mean(df_res['No_of_Votes']), 0)
N_median = round(np.median(df_res['No_of_Votes']), 0)
N_std = round(np.std(df_res['No_of_Votes']), 0)
print(f'No_of_Votes; mean: {N_mean}, median: {N_median}, std: {N_mean}') # Also exponential distribution

I_mean = round(np.mean(df_res['IMDB_Rating']), 1)
I_median = round(np.median(df_res['IMDB_Rating']), 1)
I_std = round(np.std(df_res['IMDB_Rating']), 1)
print(f'IMDB_Rating; mean: {I_mean}, median: {I_median}, std: {I_std}')

# Calculate Pearson correlation between Gross and No_of_votes
corr = round(df_res['Gross'].corr(df_res['No_of_Votes']), 5) 
print('\nCorrelation between Gross and No_of_Votes: ', corr)

# Use IQR to identify outliers in Gross
q1 = np.percentile(df_res['Gross'], 25)
q3 = np.percentile(df_res['Gross'], 75)
IQR = q3 - q1
bound_lower = q1 - (1.5 * IQR)
bound_upper = q3 + (1.5 * IQR)
outliers = []
for row in df_res['Gross']:
    if row < bound_lower:
        outliers.append(row)
    elif row > bound_upper:
        outliers.append(row)
print('\nBoundaries for outliers in the Gross column: ', bound_lower, bound_upper)


# Phase 5 Advanced Analysis
    # Director Analysis
df_directors = df_res.groupby('Director').agg(Sum_Gross= ('Gross', 'sum'), Average_Gross= ('Gross', 'mean'))
df_dir_sorted = df_directors.sort_values(by= 'Average_Gross', ascending= False)
print("\nComparing directors' Gross profits:")
print(df_dir_sorted.head(10))
print('Anthony Russo is the director with the highest average Gross')

#--------------------------------------
# Setting up the Streamlit WebApp
#--------------------------------------

# Narrow down the columns in the DF
df_st = df_res[['Series_Title', 'Released_Decade', 'Certificate', 'No_of_Votes', 'Runtime', 'Genre', 'IMDB_Rating', 'Director', 'Lead_Actors', 'Gross']]
print(df_st.info())
# App setup
st.set_page_config(page_title= 'IMDB Top 100 Movies and Series', layout= 'wide')
st.title('')
st.caption('')

# SIDEBAR FILTERS
#-----------------------
st.sidebar.header('🔎 Filters')
release_decade = st.sidebar.multiselect('Released_Decade', sorted(df_st['Released_Decade'].unique()), default= sorted(df_st['Released_Decade'].unique()))
certificate = st.sidebar.multiselect('Certificate', sorted(df_st['Certificate'].unique()), default= sorted(df_st['Certificate'].unique()))
# genre = st.sidebar.multiselect('Genre', sorted(df_st['Genre'].unique()), default= sorted(df['Genre'].unique()))
imdb = st.sidebar.slider('IMDB Rating', df_st['IMDB_Rating'].min(), df_st['IMDB_Rating'].max(), df_st['IMDB_Rating'].mean())
gross = st.sidebar.slider('Gross', df_st['Gross'].min().astype(float), df_st['Gross'].max().astype(float), df_st['Gross'].mean())

mask = ( # Not quite sure what this does
    df_st['Released_Decade'].isin(release_decade) &
    df_st['Certificate'].isin(certificate) &
    (df_st['IMDB_Rating'] >= imdb)
)
fdf = df_st.loc[mask].copy()
st.sidebar.success(f'Activate filters → Rows: {len(fdf)} / {len(df_st)}')
# KPIs
#-------------------------


# DATA TABLE + DOWNLOAD
#--------------------------
st.subheader('📄 Filtered Dataset')
st.dataframe(df_st.sort_values('Gross', ascending= False), use_container_width= True)

csv_buf = StringIO()
fdf.to_csv(csv_buf, index=False)
st.download_button("⬇️ Download filtered CSV", csv_buf.getvalue(), "students_filtered.csv", "text/csv")

st.divider()

# CHARTS
    # Plot 1
st.subheader(f'Box Plot of IMDB Rating by Certificate')
fig = px.box(fdf, x='IMDB_Rating', y='Certificate', title='Box Plot of IMDB Rating by Certificate')
if len(fdf):
    st.plotly_chart(fig)
else:
    st.info('No data after filters')

st.divider()

    # Plot 2
    # First convert Gross to number value
st.subheader(f'Scatter plot of Gross vs. No_of_votes')
if len(fdf):
    st.scatter_chart(fdf, x= 'Gross', y= 'No_of_Votes', color= '#ffaa00')
else:
    st.info('No data after filters')

st.divider()
    
    # Plot 3
# Bar plot of Genre frequency
    # First need to find the top 10 genre frequency of occurance. Already created, just use it
# if len(df_Genre_top10):
#     st.bar_chart(df_Genre_top10, x= 'Genre', y= 'Frequency')
# else:
#     st.info('No data after filters')
    

st.caption("Tip: Use the sidebar to filter. All charts use Streamlit’s built-in chart functions (no matplotlib).")
