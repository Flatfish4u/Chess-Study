import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure plotting style
sns.set(style="whitegrid")
logging.basicConfig(level=logging.INFO)

# Load CSV data

adhd_df = pd.read_csv('/Users/benjaminrosales/Desktop/Chess Study Materials & Data/CSV\'s from initial run /adhd_moves_df.csv')
general_df = pd.read_csv('/Users/benjaminrosales/Desktop/Chess Study Materials & Data/CSV\'s from initial run /general_moves_df.csv')

# Combine datasets
df = pd.concat([adhd_df, general_df], ignore_index=True)

# Ensure correct data types
df['EvalChange'] = pd.to_numeric(df['EvalChange'], errors='coerce')
df['TimeSpent'] = pd.to_numeric(df['TimeSpent'], errors='coerce')
df['WhiteElo'] = pd.to_numeric(df['WhiteElo'], errors='coerce')
df['BlackElo'] = pd.to_numeric(df['BlackElo'], errors='coerce')

# Drop rows with missing critical data
df = df.dropna(subset=['EvalChange', 'TimeSpent', 'WhiteElo', 'BlackElo'])

# Function to perform t-test or ANOVA
def perform_statistical_test(variable, df, test_results):
    # Separate the data by group
    group_data = {grp: vals.dropna() for grp, vals in df.groupby('Group')[variable]}
    groups = list(group_data.keys())
    
    if len(groups) == 2:
        # Check assumptions
        check_assumptions(variable, df)
        # Perform independent t-test
        stat, p_value = stats.ttest_ind(group_data[groups[0]], group_data[groups[1]], equal_var=False)
        test_type = 'Independent t-test'
    else:
        # Perform one-way ANOVA
        stat, p_value = stats.f_oneway(*(group_data[grp] for grp in groups))
        test_type = 'One-way ANOVA'
    
    result = {
        'Variable': variable,
        'Test': test_type,
        'Statistic': stat,
        'p-value': p_value
    }
    test_results.append(result)
    print(f"{test_type} for {variable}: Statistic={stat:.4f}, p-value={p_value:.4f}")

# Function to perform chi-square test
def perform_chi_squared_test(variable, df, test_results):
    contingency_table = pd.crosstab(df['Group'], df[variable])
    stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    result = {
        'Variable': variable,
        'Test': 'Chi-square test',
        'Statistic': stat,
        'p-value': p_value
    }
    test_results.append(result)
    print(f"Chi-square test for {variable}: Statistic={stat:.4f}, p-value={p_value:.4f}")

# Function to check statistical assumptions
from scipy.stats import shapiro, levene

def check_assumptions(variable, df):
    group_data = [df[df['Group'] == grp][variable].dropna() for grp in df['Group'].unique()]
    
    # Shapiro-Wilk Test for Normality
    for i, grp in enumerate(df['Group'].unique()):
        stat, p = shapiro(group_data[i])
        print(f'Shapiro-Wilk Test for {grp} {variable}: Statistic={stat:.4f}, p-value={p:.4f}')
    
    # Levene's Test for Homogeneity of Variances
    stat, p = levene(*group_data)
    print(f"Levene's Test for {variable}: Statistic={stat:.4f}, p-value={p:.4f}")

# Analysis Functions
def plot_performance_under_time_pressure(df, test_results):
    df_pressure = df[df["UnderTimePressure"]]
    if df_pressure.empty:
        logging.warning("No moves under time pressure to analyze.")
        return
    
    df_pressure["EvalChange"] = pd.to_numeric(df_pressure["EvalChange"], errors="coerce")
    df_pressure = df_pressure.dropna(subset=["EvalChange"])
    avg_eval_change = df_pressure.groupby("Group")["EvalChange"].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Group", y="EvalChange", data=avg_eval_change, palette="Set2")
    plt.title("Average Evaluation Change Under Time Pressure")
    plt.ylabel("Average Evaluation Change (centipawns)")
    plt.xlabel("Group")
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Group", y="EvalChange", data=df_pressure, palette="Set2", inner="quartile")
    plt.title("Distribution of Evaluation Change Under Time Pressure")
    plt.ylabel("Evaluation Change (centipawns)")
    plt.xlabel("Group")
    plt.show()
    
    perform_statistical_test("EvalChange", df_pressure, test_results)
    
    # Logistic Regression: Predicting accurate vs. inaccurate moves
    df_pressure['AccurateMove'] = df_pressure['ErrorCategory'].apply(lambda x: 1 if x == 'No Error' else 0)
    model = smf.logit('AccurateMove ~ C(Group)', data=df_pressure).fit(disp=False)
    print(model.summary())

def plot_accuracy_vs_time(df, test_results):
    df["TimeSpent"] = pd.to_numeric(df["TimeSpent"], errors="coerce")
    df["EvalChange"] = pd.to_numeric(df["EvalChange"], errors="coerce")
    df = df.dropna(subset=["TimeSpent", "EvalChange"])
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="TimeSpent", y="EvalChange", hue="Group", data=df, alpha=0.3, palette="Set1")
    sns.lineplot(x="TimeSpent", y="EvalChange", hue="Group", data=df, estimator="mean", palette="Set1")
    plt.title("Move Accuracy Relative to Move Time")
    plt.xlabel("Time Spent on Move (seconds)")
    plt.ylabel("Evaluation Change (centipawns)")
    plt.legend(title="Group")
    plt.show()
    
    perform_statistical_test("EvalChange", df, test_results)
    
    # Linear Regression: EvalChange vs TimeSpent and Group
    df['Group_encoded'] = df['Group'].map({'ADHD': 1, 'General': 0})
    df = df.dropna(subset=['EvalChange', 'TimeSpent', 'Group_encoded'])
    X = df[['TimeSpent', 'Group_encoded']]
    y = df['EvalChange']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

def plot_error_rate(df, test_results):
    error_counts = df.groupby(["Group", "ErrorCategory"]).size().reset_index(name="Count")
    total_counts = df.groupby("Group")["ErrorCategory"].count().reset_index(name="Total")
    error_counts = error_counts.merge(total_counts, on="Group")
    error_counts["Percentage"] = (error_counts["Count"] / error_counts["Total"]) * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(x="ErrorCategory", y="Percentage", hue="Group", data=error_counts, palette="Set3")
    plt.title("Error Rate Comparison")
    plt.ylabel("Percentage of Moves (%)")
    plt.xlabel("Error Category")
    plt.legend(title="Group")
    plt.xticks(rotation=45)
    plt.show()

    perform_chi_squared_test("ErrorCategory", df, test_results)

def plot_time_management(df, test_results):
    df["TimeSpent"] = pd.to_numeric(df["TimeSpent"], errors="coerce")
    df = df.dropna(subset=["TimeSpent"])

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="TimeSpent", hue="Group", common_norm=False, fill=True, alpha=0.5, palette="Set1")
    plt.title("Time Management Patterns")
    plt.xlabel("Time Spent on Move (seconds)")
    plt.ylabel("Density")
    plt.legend(title="Group")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Group", y="TimeSpent", data=df, palette="Set2")
    plt.title("Time Spent per Move by Group")
    plt.ylabel("Time Spent on Move (seconds)")
    plt.xlabel("Group")
    plt.show()

    perform_statistical_test("TimeSpent", df, test_results)

def stratify_by_elo(df, test_results):
    def categorize_elo(elo):
        if elo < 1400:
            return "<1400"
        elif elo < 2000:
            return "1400-1999"
        else:
            return "2000+"

    df["EloCategory"] = df.apply(
        lambda x: categorize_elo(x["WhiteElo"]) if x["Player"] == "White" else categorize_elo(x["BlackElo"]),
        axis=1,
    )
    df = df.dropna(subset=["EloCategory"])

    df["EvalChange"] = pd.to_numeric(df["EvalChange"], errors="coerce")
    df["TimeSpent"] = pd.to_numeric(df["TimeSpent"], errors="coerce")
    df = df.dropna(subset=["EvalChange", "TimeSpent"])

    g = sns.FacetGrid(df, col="EloCategory", hue="Group", col_wrap=3, height=4, sharey=False, palette="Set1")
    g.map_dataframe(sns.scatterplot, x="TimeSpent", y="EvalChange", alpha=0.3)
    g.add_legend(title="Group")
    g.set_axis_labels("Time Spent on Move (seconds)", "Evaluation Change (centipawns)")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Accuracy vs Time Spent by ELO Category")
    plt.show()

    plt.figure(figsize=(14, 8))
    sns.boxplot(x="EloCategory", y="EvalChange", hue="Group", data=df, palette="Set2")
    plt.title("Evaluation Change by ELO Category and Group")
    plt.ylabel("Evaluation Change (centipawns)")
    plt.xlabel("ELO Category")
    plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()

    for elo in df["EloCategory"].unique():
        subset = df[df["EloCategory"] == elo]
        perform_statistical_test("EvalChange", subset, test_results)

def perform_cluster_analysis(df):
    features = ['TimeSpent', 'EvalChange']
    df_cluster = df.dropna(subset=features)
    X = df_cluster[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # For simplicity, we'll use 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TimeSpent', y='EvalChange', hue='Cluster', data=df_cluster, palette='viridis')
    plt.title('Cluster Analysis of Time Spent vs Eval Change')
    plt.xlabel('Time Spent on Move (seconds)')
    plt.ylabel('Evaluation Change (centipawns)')
    plt.show()

# Initialize test results
test_results = []

# Run analysis functions
plot_performance_under_time_pressure(df, test_results)
plot_accuracy_vs_time(df, test_results)
plot_error_rate(df, test_results)
plot_time_management(df, test_results)
stratify_by_elo(df, test_results)
perform_cluster_analysis(df)

# Print test results
print("\nStatistical Test Results:")
for result in test_results:
    print(result)
