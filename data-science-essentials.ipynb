# Project Background & Aims
The aim of the project is to draw meaningful conclusions from the dataset, by means of performing data profiling, cleansing and analysis in order to produce graphs and predictive models. The end goal of this project is to make accurate, data-focused predictions and insights from the raw data provided.

# Data Understanding
The dataset provided covers a comprehensive set of data, sorted neatly by columns containing important data that will be used to generate insights, with the main objective being to first profile the data thoroughly, identify any important issues and then preparing the data for more in-depth analysis.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
from pandas.plotting import andrews_curves
from itertools import combinations

royal_cruise=pd.read_csv("Assignment2_Dataset.csv", parse_dates=['Date'], dayfirst=False)

# Display the first 5 rows of dataframe
print('''First 5 rows of dataframe:''')
royal_cruise.head()

# Display the last 5 rows of dataframe
print('''Last 5 rows of dataframe:''')
royal_cruise.tail()

#Count the total number of rows in the dataset
num_rows = royal_cruise.shape[0] #Starts counting all rows starting from n = 0

# Print the number of rows counted
print(f"The total number of rows in the dataset is: {num_rows}")

# There are 9639 rows and 21 columns
print('''Shape of dataframe:''')
royal_cruise.shape

print('''Datatypes in dataframe:''')
royal_cruise.dtypes

#Finding out the number of categories and % for each individual category for categorical data columns
for column in royal_cruise.select_dtypes(include='object').columns:
    print(f"Column: {column}")
    
    # Counting each category 
    category_count = royal_cruise[column].value_counts()
    print("Count of each category:\n", category_count)
    
    # Percentage of each individual category
    category_percentage = royal_cruise[column].value_counts(normalize=True) * 100
    print("Percentage of each category:\n", category_percentage)
    print("\n" + "-"*50 + "\n")

#Finding out the number of categories and % for each individual category for numerical data columns
for column in royal_cruise.columns:
    print(f"Column: {column}")
    print("-" * 50)
    
    if royal_cruise[column].dtype == 'object':
        category_count = royal_cruise[column].value_counts()
        print("Count of each category:\n", category_count)
        
        category_percentage = royal_cruise[column].value_counts(normalize=True) * 100
        print("Percentage of each category:\n", category_percentage)
    else:
        column_stats = royal_cruise[column].describe()
        print("Basic statistics:\n", column_stats)
    
    print("\n" + "-"*50 + "\n")

# isnull() checks for missing values while sum() counts the totals for each column.
missing_values = royal_cruise.isnull().sum()
print(missing_values)

# There are 30 missing values for "Hour", 874 missing values for "Tickets_Sold", 4729 missing values for "Solar_Radiation" and 24 missing values for "Hit_Sales".

Convert all Tickets_Sold values to one data type first (numeric) as some values are inputed non-numerically.

Convert all instances of "Sold: n" to "n" in the Tickets_Sold column to make data more consistent.

Convert all data entries to round numbers to ensure consistency.

Impute the missing values for the "Hour" and "Tickets_Sold" columns with mean as both columns have normal distributions.

However, as both the "Solar_Radiation" and "Hit_Sales" columns are categorical in nature, impute the missing values in both columns with mode instead.

# Make all data entries in the 'Tickets_Sold' column have a consistent format.
royal_cruise['Tickets_Sold'] = royal_cruise['Tickets_Sold'].replace('Sold: ', ' ')

# Then, convert all entries in the 'Tickets_Sold' column into numeric data type.
royal_cruise['Tickets_Sold'] = pd.to_numeric(royal_cruise['Tickets_Sold'], errors='coerce')

# After changing the categories in 'Tickets_Sold', impute the missing values in both 'Hour' and 'Tickets_Sold' with mean.
royal_cruise['Hour'].fillna(royal_cruise['Hour'].mean(), inplace=True)
royal_cruise['Tickets_Sold'].fillna(royal_cruise['Tickets_Sold'].mean(), inplace=True)

# Lastly, impute the missing values in both 'Solar_Radiation' and 'Hit_Sales' with mode.
royal_cruise['Solar_Radiation'].fillna(royal_cruise['Solar_Radiation'].mode()[0], inplace=True)
royal_cruise['Hit_Sales'].fillna(royal_cruise['Hit_Sales'].mode()[0], inplace=True)

# Change all values to be consistent in order to ensure smoother data preparation.
royal_cruise['Hit_Sales'] = royal_cruise['Hit_Sales'].replace({'N': 'No', 
                                                               'NO': 'No', 
                                                               'Y': 'Yes', 
                                                               'y':'Yes',
                                                               'yes': 'Yes'})

# Change all instances of "Interrior" with "Interior"
royal_cruise['Cabin_Type'] = royal_cruise['Cabin_Type'].replace('Interrior', 'Interior')

# Round off all to whole numbers with the .round(0) function
royal_cruise['Tickets_Sold'] = royal_cruise['Tickets_Sold'].round(0)
royal_cruise['Tickets_Sold'].value_counts()

print('''
Duplicates:''')
print(royal_cruise.duplicated().sum()) #Finding duplicates in dataset

# Display the updated dataset 
royal_cruise_updated = royal_cruise.copy()
print("Updated cleaned dataframe:")
print(royal_cruise_updated.isnull().sum())
# Check duplicates
print("\nDuplicates:")
print(royal_cruise_updated.duplicated().sum())
# Check info
print('''
Updated cleaned dataframe data info:''')
print(royal_cruise_updated.info())
print('''Updated database info: ''')
print(royal_cruise_updated.head())

# The 'Hour' column should be binned according to the time of day.

Create a new column named 'Time_Period', and sort them into the following categories: Early Morning, Morning, Noon, Afternoon, Evening and Night.

By grouping them according to specific times of the day, we can understand which times of the day have the most/least activity, correlation with other factors and columns in the dataset, etc. in order to help plan out strategies more effectively.

# Bin the 'Hour' column
# Define bin labels and edges
# Bin labels = Early Morning, Morning, Noon, Afternoon, Night
bins = [0,7,12,15,18,23]
labels = ['Early Morning','Morning','Noon','Afternoon','Night']
royal_cruise_updated['Time_Period'] = pd.cut(royal_cruise_updated['Hour'],bins=bins,labels=labels,include_lowest=True)

# Show the new dataframe with binned column
royal_cruise_updated[['Hour','Time_Period']].value_counts()

# Insert your codes here to generate the result.

df = pd.DataFrame({
    'Cabin_Type': ['Balcony', 'Ocean View', 'Interior', 'Suite']
})

df_encoded = pd.get_dummies(df, columns=['Cabin_Type'])

print("One-hot encoded Cabin_Type columns:")
print(df_encoded)

df = pd.DataFrame({
    'Mode_Of_Booking': ['Travel Agent', 'Company Website', 'Phone Call', 'Mobile App', '3rd Party Website']
})

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['Mode_Of_Booking'])

print("One-hot encoded Mode_Of_Booking columns:")
print(df_encoded)

# Do Z-Score Transformation (Normalization) on weather related columns (variables).

Using a single index for weather related columns, such as 'Temperature', 'Humidity', 'Windspeed', 'Dewpoint_Temp', 'Rainfall' and 'Solar_Radiation' will be useful as they are all measured with different units, so spotting and analyzing any potential patterns without a single index will be difficult. Therefore, having a single index is beneficial for spotting correlations more accurately and effectively.

The 'Temperature', 'Humidity', 'Windspeed', 'Dewpoint_Temp', 'Rainfall' and 'Solar_Radiation' columns will be combined into one column called: 'Weather_Index'. This new column will be used to see if there are any correlations with other existing columns in the Pearson's Correlation Analysis matrix.

import pandas as pd
from scipy.stats import zscore

file_path = 'Assignment2_Dataset.csv'
royal_cruise_updated = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False)

# Make a new column called 'Weather_Index'
weather_index = ['Temperature', 'Humidity', 'Windspeed', 'Dewpoint_Temp', 'Rainfall', 'Solar_Radiation']

# Add to 'royal_cruise_updated'
for col in weather_index:
    royal_cruise_updated[col] = pd.to_numeric(royal_cruise_updated[col], errors='coerce')

# Do normalization
royal_cruise_updated[weather_index] = royal_cruise_updated[weather_index].apply(zscore)

# Turn them into a single score
royal_cruise_updated['Weather_Index'] = royal_cruise_updated[weather_index].mean(axis=1)

# Show the first few rows of the dataframe to verify that 'royal_cruise_updated' was normalized correctly.
print(royal_cruise_updated[['Weather_Index']].head())

file_path = 'Assignment2_Dataset.csv'
df = pd.read_csv(file_path)

print("Column names in DataFrame:")
print(df.columns)

df.dropna(subset=['Cabin_Type', 'Mode_of_Booking'], inplace=True) # drop NaN values in order to perform cleaning

numerical_columns = royal_cruise_updated[['Hour', 'Temperature', 
                                         'Humidity', 'Windspeed', 
                                         'Rainfall', 
                                         'Cruise_Duration','Weather_Index']]

correlation_matrix = numerical_columns.corr()
# Decide the size of the graph
plt.figure(figsize=(15,12))
# Create the heatmap and correlation matrix
sns.heatmap(correlation_matrix, annot=True, fmt=".3f", cmap="coolwarm", linewidths=2)

plt.xticks(rotation=0)  # Rotate x-axis labels to be upright
plt.yticks(rotation=0) 

# Label the correlation matrix
plt.title('Pearson Correlation Analysis (royal_cruise)')
plt.show()

# Load the dataframe
file_path = 'Assignment2_Dataset.csv'
df = pd.read_csv(file_path)

# Select columns for plotting
columns_to_plot = ['Tickets_Sold', 'Cruise_Duration']
df_plot = df[columns_to_plot].copy()  # Create a copy to avoid SettingWithCopyWarning

# Clean the 'Tickets_Sold' column
df_plot['Tickets_Sold'] = df_plot['Tickets_Sold'].str.extract('(\d+)').astype(float)

# Remove rows with missing values
df_plot = df_plot.dropna()

# Create a categorical variable for coloring
df_plot['Tickets_Sold_Category'] = pd.cut(df_plot['Tickets_Sold'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])

# Verify the data types
print("Data types after cleaning and adding categories:")
print(df_plot.dtypes)

# Plot Pairplot with color
sns.pairplot(df_plot, hue='Tickets_Sold_Category', diag_kind='kde', palette='viridis')
plt.suptitle('Pairplot of Tickets_Sold and Cruise_Duration with Color Categories', y=1.02)
plt.show()

# Load the dataset
df = royal_cruise_updated

# Group by 'Cabin_Type' and sum 'Tickets_Sold'
sales_by_cabin_type = df.groupby('Cabin_Type')['Tickets_Sold'].sum().reset_index()
    
# Round 'Tickets_Sold' to whole numbers
sales_by_cabin_type['Tickets_Sold'] = sales_by_cabin_type['Tickets_Sold'].round()

# Debugging check
print("Sales by Cabin Type DataFrame:")
print(sales_by_cabin_type)

# Plotting the pie chart
plt.figure(figsize=(10, 10))
plt.pie(sales_by_cabin_type['Tickets_Sold'], labels=sales_by_cabin_type['Cabin_Type'], autopct='%1.1f%%', colors=['#00BFFF', '#F3C300', '#3CB371', '#FF0000'])
plt.title('Distribution of Tickets Sold by Cabin Type')
plt.show()

# Load the dataset
df = royal_cruise_updated

# Convert 'Tickets_Sold' to numeric if necessary
df['Tickets_Sold'] = pd.to_numeric(df['Tickets_Sold'], errors='coerce')

# Set the plot style
sns.set(style="whitegrid")

# Plotting the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Hour', y='Tickets_Sold', palette='pastel')

# Customize the plot with titles and labels
plt.title('Tickets Sold by Hour of the Day', fontsize=14)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Tickets Sold', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Display the plot
plt.show()

# Load the dataset
df = royal_cruise_updated

# Convert 'Tickets_Sold' to numeric and 'Cruise_Duration' to categorical if necessary
df['Tickets_Sold'] = pd.to_numeric(df['Tickets_Sold'], errors='coerce')
df['Cruise_Duration'] = df['Cruise_Duration'].astype('category')

# Group by 'Cruise_Duration' and sum 'Tickets_Sold'
sales_by_cruise_duration = df.groupby('Cruise_Duration')['Tickets_Sold'].sum().reset_index()

# Convert 'Cruise_Duration' to numeric for plotting
sales_by_cruise_duration['Cruise_Duration_Num'] = sales_by_cruise_duration['Cruise_Duration'].cat.codes

# Set the plot style
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(12, 8))

# Bar plot with light red bars
sns.barplot(data=sales_by_cruise_duration, x='Cruise_Duration', y='Tickets_Sold', color='lightcoral', label='Total Tickets Sold')

# Line plot on top of the bar plot
plt.plot(sales_by_cruise_duration['Cruise_Duration_Num'], sales_by_cruise_duration['Tickets_Sold'], 
         color='blue', marker='o', linestyle='-', linewidth=2, markersize=8, label='Trend Line')

# Customize the plot with titles and labels
plt.title('Sales by Cruise Duration with Trend Line (3 hours to 14 hours long)', fontsize=14)
plt.xlabel('Cruise Duration', fontsize=24)
plt.ylabel('Total Tickets Sold', fontsize=24)
plt.xticks(ticks=sales_by_cruise_duration['Cruise_Duration_Num'], labels=sales_by_cruise_duration['Cruise_Duration'], rotation=45, fontsize=24)
plt.yticks(fontsize=24)

# Ensure y-axis labels are upright (default behavior)
plt.gca().tick_params(axis='y', labelrotation=0)

plt.legend()
plt.tight_layout()  # Adjust layout to avoid clipping

# Display the plot
plt.show()

df = royal_cruise_updated

# Group data by satisfaction score
grouped_data = df.groupby('Customer_Satisfaction_Score')['Tickets_Sold'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_data, x='Customer_Satisfaction_Score', y='Tickets_Sold', palette='pastel')
plt.title('Total Tickets Sold by Customer Satisfaction Score')
plt.xlabel('Customer Satisfaction Score')
plt.ylabel('Total Tickets Sold')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Load the dataset
df = royal_cruise_updated

# Calculate the distribution of Mode_of_Booking
booking_mode_counts = df['Mode_of_Booking'].value_counts()

# Define a colormap with enough colors for each mode
colormap = plt.get_cmap('viridis')
colors = colormap(range(len(booking_mode_counts)))

# Plotting the pie chart
plt.figure(figsize=(5,5))
plt.pie(booking_mode_counts, labels=booking_mode_counts.index, autopct='%1.1f%%', colors=['#FF6F61', '#00BFFF', '#F3C300', '#3CB371', '#FF69B4'])
plt.title('Booking Mode Analysis')
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy import stats

print(royal_cruise_updated.head())

df = royal_cruise_updated
# Columns to drop
columns_to_drop = ['S/N', 'Office', 'Customer_Satisfaction_Score', 'Latitude', 'Longitude', 'Date', 'Solar_Radiation']

# Columns already dropped: 'S/N', 'Office', 'Customer_Satisfaction_Score', 'Latitude', 'Longitude', 'Date', 'Solar_Radiation'
# Check the code snippet below

# Drop the columns
royal_cruise_updated = royal_cruise_updated.drop(columns_to_drop, axis=1)

# Verify the changes
print("First 5 rows of the updated royal_cruise_updated DataFrame:")
print(royal_cruise_updated.head())

print("\nSummary of the updated royal_cruise_updated DataFrame:")
print(royal_cruise_updated.info())

print("\nUpdated columns in the royal_cruise_updated DataFrame:")
print(royal_cruise_updated.columns)

print(royal_cruise_updated.head(10))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from itertools import combinations

# Sample placeholder DataFrame creation for testing
df = royal_cruise_updated

np.random.seed(2302494)

# Print column names to ensure required columns exist
print("Column names:", royal_cruise_updated.columns)

# Define the numerical factors
numerical_factors = ['Cruise_Duration', 'Weather_Index', 'Temperature', 'Humidity', 'Windspeed']

# Handle missing values
df_cleaned = royal_cruise_updated.dropna(subset=numerical_factors + ['Tickets_Sold'])

# Check if the DataFrame is cleaned correctly
print("Cleaned DataFrame shape:", df_cleaned.shape)

# Initialize an empty dictionary to store R^2 scores
r2_scores = {}

# Generate all possible combinations of numerical factors
all_combinations = []
for i in range(1, len(numerical_factors) + 1):
    all_combinations.extend(combinations(numerical_factors, i))

# Iterate over all possible combinations
for subset in all_combinations:
    X = df_cleaned[list(subset)]
    y = df_cleaned['Tickets_Sold']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    
    # Store the R^2 score rounded to 3 decimal places
    r2_scores[subset] = round(r2, 3)

# Print R^2 scores for all combinations
print("R^2 scores:")
for features, score in r2_scores.items():
    print(f'Features: {features}, R^2 Score: {score}')

# Plot the R^2 scores and highlight the highest R^2 scores
plt.figure(figsize=(14, 10))
feature_combinations = [', '.join(f) for f in r2_scores.keys()]
scores = list(r2_scores.values())

# Find the highest R^2 scores
max_r2 = max(scores)
max_indices = [i for i, score in enumerate(scores) if score == max_r2]

colors = ['skyblue' if i not in max_indices else 'coral' for i in range(len(scores))]
plt.barh(feature_combinations, scores, color=colors)
plt.xlabel('R^2 Score')
plt.ylabel('Feature Combinations')
plt.title('R^2 Score for Various Numerical Feature Combinations')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.show()

print(df.head())
print(df.describe())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming df is your DataFrame
df = royal_cruise_updated.copy()  # Make a copy of your DataFrame

# Exclude unwanted columns (modify if needed)
exclude_columns = ['Destination']  # Add any other columns you want to exclude
df = df.drop(columns=exclude_columns)

# Define feature columns and target column
feature_columns = ['Cruise_Duration', 'Hour', 'Temperature', 'Humidity', 'Windspeed', 'Dewpoint_Temp', 'Rainfall']
target_column = 'Tickets_Sold'

# Convert target column to binary (example threshold)
threshold = df[target_column].median()  # You can choose a different threshold
df[target_column] = (df[target_column] > threshold).astype(int)  # Binary classification

# Select features (X) and target (y)
X = df[feature_columns]
y = df[target_column]

# Ensure feature and target columns are numeric
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Drop rows with NaN values in both features and target
df = df.dropna(subset=feature_columns + [target_column])
X = df[feature_columns]
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2302494)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)  # Increased max_iter if convergence issues occur
model.fit(X_train, y_train)

# Predict using the model
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Assuming royal_cruise_updated is your DataFrame

# Define factors for clustering
factors = ['Cruise_Duration', 'Temperature', 'Tickets_Sold']
X = royal_cruise_updated[factors]

# Handle missing values by imputing
imputer = SimpleImputer(strategy='mean')  # Impute missing values with mean
X_imputed = imputer.fit_transform(X)

# Normalize factors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# The elbow method to find the optimal number of clusters
inertia = []
silhouette_avg = []
k_values = range(2, 11)  # Start from 2 because silhouette score is not defined for k=1

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    
    # Compute silhouette score
    cluster_labels = kmeans.labels_
    silhouette_avg.append(silhouette_score(X_scaled, cluster_labels))

# Plot elbow method results
plt.figure(figsize=(14, 6))

# Plot inertia
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), inertia + [None], marker='o')  # Add an extra value to match lengths
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_avg, marker='o', color='orange')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Optimal Number of Clusters')
plt.show()

# Apply K-means with the chosen number of clusters
optimal_k = 3  # Choose the optimal number of clusters based on the plots
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
royal_cruise_updated['Cluster'] = clusters

# Analyze cluster centers
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=factors)
print('Cluster Centers:')
print(cluster_centers)

# Visualize clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(royal_cruise_updated['Cruise_Duration'], royal_cruise_updated['Tickets_Sold'],
                      c=royal_cruise_updated['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Cruise Duration')
plt.ylabel('Tickets Sold')
plt.title('Clusters of Cruise Duration and Tickets Sold')
plt.colorbar(scatter, label='Cluster')
plt.show()
