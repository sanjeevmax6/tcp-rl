import pandas as pd
import matplotlib.pyplot as plt

# File paths
files = {
    'Listwise': 'listwise_training_metricslistwise_dqn_2500ep_dqn.csv',
    'Pointwise': 'pointwise_training_metricspointwise_dqn_2500ep_dqn.csv',
    'Pairwise': 'pairwise_training_metricspairwise_dqn_2500ep_dqn.csv'
}

def filter_monotonic_increasing(df):
    filtered_rows = []
    current_max = -float('inf')
    for _, row in df.iterrows():
        if row['APFD'] >= current_max:
            filtered_rows.append(row)
            current_max = row['APFD']
    return pd.DataFrame(filtered_rows)

# Create a plot
plt.figure(figsize=(12, 6))

# Process and plot each file
for label, filepath in files.items():
    # Read the CSV
    df = pd.read_csv(filepath)

    # Keep only relevant columns and filter up to episode 500
    df = df[['Episode', 'Build', 'APFD']]
    df = df[df['Episode'] <= 500]

    # Sort to get highest APFD per Build
    df_sorted = df.sort_values(by='APFD', ascending=False)
    df_deduped = df_sorted.drop_duplicates(subset='Build', keep='first')

    # Sort by Episode
    df_sorted_by_episode = df_deduped.sort_values(by='Episode')

    # Filter to keep only increasing APFD values
    df_monotonic = filter_monotonic_increasing(df_sorted_by_episode)

    # Plot
    plt.plot(df_monotonic['Episode'], df_monotonic['APFD'], label=label)

# Plot formatting
plt.title('Monotonically Increasing APFD (Episodes ≤ 500)')
plt.xlabel('Episode')
plt.ylabel('APFD')
plt.xticks(df_monotonic['Episode'])  # Show actual episode values as ticks
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
