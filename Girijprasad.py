import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_path = "C:/Users/girij/Downloads/TSCO_ann dataset.csv"
df = pd.read_csv(data_path)

df = df.apply(pd.to_numeric, errors='coerce')

df['price'] = pd.to_datetime(df['price'], errors='coerce')

df = df.dropna(subset=['price', 'ann_return'])

print("Columns in the dataset:", df.columns)
print(df.head())

def calculate_statistics(df):
    stats = df.describe()
    corr_matrix = df.corr()
    return stats, corr_matrix

def plot_histogram(df, column):
    plt.figure(figsize=(8, 6))
    plt.hist(df[column].dropna(), bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_line_chart(df, x_column, y_column):
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_column].dropna(), df[y_column].dropna(), marker='o', color='teal')
    plt.title(f'Line Chart of {y_column} over {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

def plot_heatmap(df):
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    
    plt.xticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

stats_summary, corr_matrix = calculate_statistics(df)
print("Statistical Summary:\n", stats_summary)
print("Correlation Matrix:\n", corr_matrix)

plot_histogram(df, column='ann_return')  # Replace 'Close' with actual column name from the dataset

plot_line_chart(df, x_column='price', y_column='ann_return')  # Replace 'Date' and 'Close' with actual column names

plot_heatmap(df)
