import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Student Performance Data Analysis
# Milestone 2: Data Quality and Exploratory Analysis

np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

# 1. Create a sample dataset
n = 120

df = pd.DataFrame({
    "Student_ID": range(1, n + 1),
    "Study_Hours": np.round(np.random.uniform(1, 8, n), 1),
    "Attendance": np.round(np.random.uniform(50, 100, n), 1),
    "Math_Marks": np.random.randint(30, 101, n),
    "Science_Marks": np.random.randint(30, 101, n),
    "English_Marks": np.random.randint(30, 101, n)
})

# Introduce issues to demonstrate data cleaning
df.loc[[3, 12, 25], "Math_Marks"] = np.nan
df.loc[[7, 19], "Attendance"] = np.nan
df.loc[10, "Science_Marks"] = 120  # Invalid mark
df.loc[20, "Attendance"] = 130     # Invalid attendance
df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

# Save raw data
df.to_csv("outputs/raw_student_data.csv", index=False)

print("DATA QUALITY REPORT")
print("-" * 40)
print("Original rows:", len(df))
print("Duplicate rows:", df.duplicated().sum())
print("\nMissing values:")
print(df.isnull().sum())

# 2. Remove duplicate records
df = df.drop_duplicates().copy()

# 3. Replace invalid values with missing values
mark_columns = ["Math_Marks", "Science_Marks", "English_Marks"]

for col in mark_columns:
    df.loc[~df[col].between(0, 100), col] = np.nan

df.loc[~df["Attendance"].between(0, 100), "Attendance"] = np.nan
df.loc[~df["Study_Hours"].between(0, 24), "Study_Hours"] = np.nan

# 4. Fill missing numeric values with column medians
numeric_columns = [
    "Study_Hours", "Attendance",
    "Math_Marks", "Science_Marks", "English_Marks"
]

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 5. Exploratory Data Analysis
df["Average_Marks"] = df[mark_columns].mean(axis=1)

df["Performance"] = np.where(
    df["Average_Marks"] >= 75, "High",
    np.where(df["Average_Marks"] >= 50, "Medium", "Low")
)

print("\nEXPLORATORY DATA ANALYSIS")
print("-" * 40)
print(df[numeric_columns + ["Average_Marks"]].describe().round(2))
print("\nPerformance groups:")
print(df["Performance"].value_counts())

print("\nCorrelation with average marks:")
print(
    df[["Study_Hours", "Attendance", "Average_Marks"]]
    .corr()["Average_Marks"]
    .round(2)
)

# 6. Save cleaned dataset
df.to_csv("outputs/cleaned_student_data.csv", index=False)

# 7. Create charts
sns.set_theme(style="whitegrid")

# Chart 1: Distribution of average marks
plt.figure(figsize=(8, 5))
sns.histplot(df["Average_Marks"], bins=12, kde=True, color="royalblue")
plt.title("Distribution of Student Average Marks")
plt.xlabel("Average Marks")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("outputs/marks_distribution.png")
plt.close()

# Chart 2: Study hours versus marks
plt.figure(figsize=(8, 5))
sns.regplot(
    data=df, x="Study_Hours", y="Average_Marks",
    scatter_kws={"alpha": 0.7}, line_kws={"color": "red"}
)
plt.title("Study Hours vs Average Marks")
plt.tight_layout()
plt.savefig("outputs/study_hours_vs_marks.png")
plt.close()

# Chart 3: Average marks by performance group
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Performance", hue="Performance",
              order=["Low", "Medium", "High"], legend=False)
plt.title("Number of Students by Performance Group")
plt.tight_layout()
plt.savefig("outputs/performance_groups.png")
plt.close()

# Chart 4: Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(
    df[["Study_Hours", "Attendance"] + mark_columns +
       ["Average_Marks"]].corr(),
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Student Data Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# 8. Generate a summary report
with open("outputs/project_summary.txt", "w", encoding="utf-8") as f:
    f.write("STUDENT PERFORMANCE ANALYSIS\n")
    f.write("=" * 35 + "\n")
    f.write(f"Students analyzed: {len(df)}\n")
    f.write(f"Average student marks: {df['Average_Marks'].mean():.2f}\n")
    f.write(f"Average attendance: {df['Attendance'].mean():.2f}%\n")
    f.write(f"Average study hours: {df['Study_Hours'].mean():.2f}\n\n")
    f.write("Performance group counts:\n")
    f.write(df["Performance"].value_counts().to_string())
    f.write("\n\nCorrelation with average marks:\n")
    f.write(
        df[["Study_Hours", "Attendance", "Average_Marks"]]
        .corr()["Average_Marks"].to_string()
    )
    f.write("\n\nNote: This is synthetic sample data.")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("Check the outputs folder for datasets, charts, and summary.")
