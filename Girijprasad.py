import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data_path = "C:/Users/girij/Downloads/TSCO_ann (1).csv"
df = pd.read_csv(data_path)

# Convert columns to numeric where possible, forcing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Ensure the 'Date' column is in datetime format
df['price'] = pd.to_datetime(df['price'], errors='coerce')

# Drop rows with missing values in 'Date' or 'Close' columns
df = df.dropna(subset=['price', 'ann_return'])

# Print column names and first few rows to understand the dataset
print("Columns in the dataset:", df.columns)
print(df.head())

# Function to generate descriptive statistics
def calculate_statistics(df):
    # General statistics
    stats = df.describe()
    # Correlation matrix
    corr_matrix = df.corr()
    return stats, corr_matrix

# Function to plot a histogram
def plot_histogram(df, column):
    plt.figure(figsize=(8, 6))
    plt.hist(df[column].dropna(), bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Function to plot a line chart
def plot_line_chart(df, x_column, y_column):
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_column].dropna(), df[y_column].dropna(), marker='o', color='teal')
    plt.title(f'Line Chart of {y_column} over {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

# Function to plot a heatmap (correlation matrix) using matplotlib only
def plot_heatmap(df):
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    
    # Adding labels for heatmap
    plt.xticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

# Generate statistics
stats_summary, corr_matrix = calculate_statistics(df)
print("Statistical Summary:\n", stats_summary)
print("Correlation Matrix:\n", corr_matrix)

# Plot histogram for a selected column (choose an actual column name)
plot_histogram(df, column='ann_return')  # Replace 'Close' with actual column name from the dataset

# Plot line chart for selected x and y columns (choose actual columns)
plot_line_chart(df, x_column='price', y_column='ann_return')  # Replace 'Date' and 'Close' with actual column names

# Plot heatmap (correlation matrix)
plot_heatmap(df)
