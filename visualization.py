# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_quartile_distribution(df, output_path='static/fa_hpscore_plot.png'):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Quartile', y='FAHP_Score', data=df)
    plt.title('FAHP Score Distribution by Quartile')
    plt.savefig(output_path)
    plt.close()
