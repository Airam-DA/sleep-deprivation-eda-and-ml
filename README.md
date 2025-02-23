![brain_s](https://github.com/user-attachments/assets/ed24223a-ad43-45b9-ab0c-f027613b40cf)

# Sleep Deprivation & Cognitive Performance Analysis
This project analyzes the [Sleep Deprivation & Cognitive Performance dataset](https://www.kaggle.com/datasets/sacramentotechnology/sleep-deprivation-and-cognitive-performance/data) from Kaggle. The dataset explores the impact of sleep deprivation on cognitive performance and emotional regulation. However, due to poor predictive performance across all tested models, the dataset appears unsuitable for my prediction goals, requiring additional data.

## About the Dataset
This dataset is based on a 2024 study conducted in the Middle East with 60 participants from diverse backgrounds. It captures various factors, including:

- **Sleep-related variables:** Sleep hours, sleep quality, daytime sleepiness
- **Cognitive performance:** Stroop Task (reaction time), N-Back Test (working memory accuracy), Psychomotor Vigilance Task (PVT)
- **Emotional regulation:** Emotion regulation scores
- **Demographics & lifestyle:** Age, gender, BMI, caffeine intake, physical activity, stress levels

### Key Features:
- `sleep_hrs`: Sleep duration (hours)
- `sleep_quality`: Sleep quality score
- `daytime_sleepiness`: Daytime sleepiness level
- `stroop_time`: Reaction time on the Stroop task
- `alertness_time`: PVT reaction time
- `working_memory`: N-Back task accuracy (working memory measure)
- `emotion_regulation`: Emotional regulation score
- `age`: Participant’s age
- `gender`: Participant’s gender
- `BMI`: Body Mass Index
- `caffeine_intake`: Amount of caffeine consumed
- `stress_level`: Self-reported stress level
- `movement`: Physical activity level

## Data Preprocessing
- Removed unnecessary spaces and standardized column names
- Converted `gender` to binary encoding (`Female: 1`, `Male: 0`)
- Checked for missing values (none found)
- Checked for duplicates (none found)
- Reordered columns for consistency

```python
# Standardizing column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Renaming columns for clarity
column_mapping = {
    'participant_id': 'id',
    'sleep_hours': 'sleep_hrs',
    'sleep_quality_score': 'sleep_quality',
    'daytime_sleepiness': 'daytime_sleepiness',
    'stroop_task_reaction_time': 'stroop_time',
    'pvt_reaction_time': 'alertness_time',
    'n_back_accuracy': 'working_memory',
    'emotion_regulation_score': 'emotion_regulation',
    'physical_activity_level': 'movement'
}
df.rename(columns=column_mapping, inplace=True)
```

## Exploratory Data Analysis (EDA)
### **Summary statistics** to understand data distribution

### Distribution of Key Features

```python
plt.figure(figsize=(12, 12))
for i, col in enumerate(columns, 1):
    plt.subplot(4, 3, i)
    plt.hist(df[col], bins=10, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
```

### Boxplots for Feature Distributions
```python
plt.figure(figsize=(12, 8))
sns.boxplot(data=df.select_dtypes(include=[np.number]), orient="h", palette="coolwarm")
plt.title("Boxplot for Feature Distribution")
plt.show()
```
![boxplot](https://github.com/user-attachments/assets/374212ff-3620-4a82-afdd-f22c1d6f7b9d)

### Identified potential outliers using Z-scores:
```python
z_scores = zscore(df.select_dtypes(include=['float64', 'int64']))
df_z = pd.DataFrame(z_scores, columns=df.select_dtypes(include=['float64', 'int64']).columns)
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_z, orient="h", palette='coolwarm')
plt.title("Z-Scores for Outliers Detection")
plt.show()
```
![outliers](https://github.com/user-attachments/assets/876776cf-c39b-49c0-8633-df81b436f47c)

### Pairplot Analysis

```python
sns.pairplot(df[['sleep_hrs', 'sleep_quality', 'daytime_sleepiness', 'stroop_time', 'alertness_time', 'working_memory']])
plt.show()
```

### Correlation Matrix
```python
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix))
plt.figure(figsize=(15,8), dpi=80)
sns.heatmap(corr_matrix, annot=True, mask=mask, fmt=".2f", cmap="coolwarm")
plt.show()
```
![corr_matrix](https://github.com/user-attachments/assets/54ae8109-9da0-4da3-81d1-2f451d01ac2b)

## Findings
- Weak correlations between sleep variables and cognitive performance measures.
- No strong predictors for `emotion_regulation`, `memory` or `reaction_times`.
- Very poor model performance across all models tried indicates that additional data is necessary.
- ML models explored:
    - KNN Regressor
    - Linear Regression
    - Random Forest Regressor
    - Gradient Boosting (XGBoost)

## Conclusion
Despite initial hopes, this dataset does not provide strong predictive power for my intended goals. 

