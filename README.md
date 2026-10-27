# 🎢 Exploratory Data Analysis (EDA) with Pandas: Roller Coaster Edition

A beginner-friendly, step-by-step walkthrough of **Exploratory Data Analysis** using **Python, Pandas, Matplotlib and Seaborn** on a real-world roller coaster dataset (1,087 coasters × 56 columns).

> 📖 **This README is written to teach.** You can read it top to bottom and understand every EDA concept, every function used, and every finding, without opening the notebook. The notebook is there when you want to run the code yourself.

---

## 📑 Table of Contents

1. [What is EDA and why do it?](#-what-is-eda-and-why-do-it)
2. [The dataset](#-the-dataset)
3. [The 5-step EDA workflow](#-the-5-step-eda-workflow)
4. [Step 0: Setup](#step-0--setup)
5. [Step 1: Data Understanding](#step-1--data-understanding)
6. [Step 2: Data Preparation](#step-2--data-preparation)
7. [Step 3: Feature Understanding (Univariate)](#step-3--feature-understanding-univariate-analysis)
8. [Step 4: Feature Relationships (Bivariate and Multivariate)](#step-4--feature-relationships)
9. [Step 5: Ask a Question](#step-5--ask-a-question-about-the-data)
10. [Cheat sheet](#-eda-cheat-sheet)
11. [Lessons and pitfalls](#-lessons--common-pitfalls)
12. [Run it yourself](#-run-it-yourself)
13. [Credits](#-credits)

---

## 🔍 What is EDA and why do it?

**Exploratory Data Analysis (EDA)** is the process of *looking at your data before modelling it*. You summarize it, clean it, plot it and question it, so you know what you actually have.

EDA answers questions like:

- How big is the data? What does each column mean, and what type is it?
- Which values are **missing**, **duplicated** or **suspicious**?
- How is each variable **distributed** (typical values, spread, extremes)?
- Which variables are **related** to each other?
- What interesting **questions** can this data answer?

Skipping EDA is how people end up training models on broken data. Good EDA saves hours of debugging later.

---

## 📦 The dataset

| Property | Value |
|----------|-------|
| File | `coaster_db.csv` (Roller Coaster Database, Kaggle) |
| Raw size | **1,087 rows × 56 columns** |
| After cleaning | **990 rows × 13 columns** |
| Each row | One roller coaster |

### Columns kept for analysis

| Column (after rename) | Meaning | Type |
|---|---|---|
| `Coaster_Name` | Name of the coaster | text |
| `Location` | Park or place | text |
| `Status` | Operating / Closed / Removed … | category |
| `Manufacturer` | Company that built it | text |
| `Year_Introduced` | Year it was introduced | integer |
| `latitude`, `longitude` | Geographic position | float |
| `Type_Main` | **Steel**, **Wood** or **Other** | category |
| `Opening_Date` | Opening date | datetime |
| `Speed_mph` | Top speed in miles per hour | float |
| `Height_ft` | Height in feet | float |
| `Inversions` | Number of upside-down elements | integer |
| `Gforce` | Maximum G-force | float |

The raw file contains a mix of messy text columns (like `"600 ft (180 m)"`) **and** already-cleaned numeric versions (like `height_ft`, `speed_mph`, `Gforce_clean`). The analysis uses the cleaned ones.

---

## 🗺️ The 5-step EDA workflow

```
┌────────────────┐   ┌────────────────┐   ┌──────────────────────┐
│ 1. Understand  │ → │ 2. Prepare     │ → │ 3. Single features   │
│ shape, dtypes, │   │ select, rename,│   │ (univariate):        │
│ describe       │   │ missing, dupes │   │ hist, KDE, counts    │
└────────────────┘   └────────────────┘   └──────────┬───────────┘
                                                     ↓
                       ┌─────────────────────────────┴───────────┐
                       │ 4. Relationships (bivariate/multivariate):│
                       │ scatter, pairplot, correlation heatmap   │
                       └─────────────────────────────┬───────────┘
                                                     ↓
                                     ┌───────────────────────────┐
                                     │ 5. Ask a question and     │
                                     │ answer it with groupby    │
                                     └───────────────────────────┘
```

---

## Step 0 · Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

plt.style.use('ggplot')          # nicer default plot style
pd.set_option('max_columns', 200) # show up to 200 columns when printing a DataFrame

df = pd.read_csv('coaster_db.csv')
```

| Library | Role |
|---|---|
| **pandas** | Load, clean and summarize tabular data (`DataFrame`) |
| **numpy** | Numerical operations under the hood |
| **matplotlib** | Base plotting library |
| **seaborn** | Statistical plots built on matplotlib (scatter with colour, pairplot, heatmap) |

---

## Step 1 · Data Understanding

**Goal:** get a first feel for the data before changing anything.

| Command | What it tells you | Result here |
|---|---|---|
| `df.shape` | `(rows, columns)` | **(1087, 56)** |
| `df.head()` / `df.tail()` | First / last rows: a quick visual sanity check | Rows like *Switchback Railway* (1884), *Flip Flap Railway* (1895) |
| `df.columns` | List of all column names | 56 columns, with duplicates in spirit (e.g. `Speed` *and* `speed_mph`) |
| `df.dtypes` | Data type of every column | Almost everything is `object` (text); only a few are numeric |
| `df.describe()` | Count, mean, std, min, quartiles, max for numeric columns | See below |

### Reading `describe()`

```
            speed_mph   height_ft   Inversions_clean   Gforce_clean
count          937        171            1087              362
mean          48.6       102.0           1.33             3.82
min            5.0        13.1           0                0.8
25%           37.3        51.8           0                3.4
50% (median)  49.7        91.2           0                4.0
75%           58.0       131.2           2                4.5
max          149.1       377.3          14               12.0
```

What a careful analyst notices:

- **`count` differs per column**: 1,087 rows exist, but `height_ft` has only 171 values and `Gforce_clean` only 362. A lot of data is **missing**.
- **Mean ≈ median for speed** (48.6 vs 49.7) suggests a fairly balanced middle, while the **max of 149.1 mph** is far from the typical ~50 mph, so there are fast extremes.
- **Inversions: the median is 0**. More than half of the coasters have no inversions at all.
- Raw `height_value` has a **max of 3,937** while cleaned `height_ft` maxes at 377.3. That gap is a red flag: the raw column mixes units. This is exactly the kind of issue `describe()` exposes.

---

## Step 2 · Data Preparation

**Goal:** make the data tidy and trustworthy for analysis.

### 2.1 Drop irrelevant columns

56 columns is too many. Keep only what answers your questions:

```python
df = df[['coaster_name', 'Location', 'Status', 'Manufacturer',
         'year_introduced', 'latitude', 'longitude', 'Type_Main',
         'opening_date_clean', 'speed_mph', 'height_ft',
         'Inversions_clean', 'Gforce_clean']].copy()
```

> 💡 `.copy()` creates an independent DataFrame and avoids pandas' *SettingWithCopyWarning*. (`df.drop([...], axis=1)` is the alternative when you want to remove a few columns instead of selecting a few.)

### 2.2 Fix data types

Dates stored as text can't be sorted or subtracted properly. Convert them:

```python
df['opening_date_clean'] = pd.to_datetime(df['opening_date_clean'])
```

### 2.3 Rename columns

Clear, consistent names make code readable:

```python
df = df.rename(columns={'coaster_name': 'Coaster_Name',
                        'year_introduced': 'Year_Introduced',
                        'opening_date_clean': 'Opening_Date',
                        'speed_mph': 'Speed_mph',
                        'height_ft': 'Height_ft',
                        'Inversions_clean': 'Inversions',
                        'Gforce_clean': 'Gforce'})
```

### 2.4 Count missing values

```python
df.isna().sum()
```

| Column | Missing |
|---|---|
| Height_ft | **916** (≈ 84%) |
| Gforce | **725** |
| Longitude / Latitude | 275 each |
| Opening_Date | 250 |
| Status | 213 |
| Speed_mph | 150 |
| Manufacturer | 59 |
| Coaster_Name, Location, Year_Introduced, Type_Main, Inversions | 0 |

**Why it matters:** any statistic or plot built on a column inherits its gaps. `Height_ft` is missing for most rows, so conclusions about height rest on a small sample.

### 2.5 Find and remove duplicates

**Check 1: fully identical rows**

```python
df.loc[df.duplicated()]   # → empty: no exact duplicates
```

**Check 2: same coaster name**

```python
df.loc[df.duplicated(subset=['Coaster_Name'])]
```

This returns rows, but are they real duplicates? Inspect one:

```python
df.query('Coaster_Name == "Crystal Beach Cyclone"')
```

Two rows appear with the same name, park, speed and opening date but a different `Year_Introduced` (1926 vs 1927). That is a **true duplicate with a data-entry difference**. However, a name alone is *not* a safe key, because different parks can have coasters with the same name.

**Fix: define "duplicate" using a combination of columns:**

```python
df = df.loc[~df.duplicated(subset=['Coaster_Name', 'Location', 'Opening_Date'])] \
       .reset_index(drop=True).copy()
```

- `~` means "NOT": keep rows that are *not* duplicates
- `subset=[...]` defines what counts as the same record
- `reset_index(drop=True)` renumbers rows 0…n−1

After this step the dataset has **990 rows** (728 Steel + 191 Wood + 71 Other).

---

## Step 3 · Feature Understanding (Univariate Analysis)

**Univariate analysis** studies **one variable at a time**: its typical values, spread, shape and outliers.

### 3.1 Categorical variables → counts and bar charts

```python
df['Type_Main'].value_counts()
```

| Type | Count | Share |
|---|---|---|
| Steel | 728 | ~74% |
| Wood | 191 | ~19% |
| Other | 71 | ~7% |

Steel dominates the dataset.

```python
df['Year_Introduced'].value_counts().head(10).plot(
    kind='bar', title='Top 10 Years Coasters Introduced')
```

Looking at the top-10 years, **1999 (46), 2000 (45), 1998 (30), 2001 (29)** and **2002 (28)** lead, so late-1990s to early-2000s was a boom period for new coasters. The dataset covers **101 distinct years**, from 1884 to 2022.

### 3.2 Numeric variables → histogram and KDE

```python
df['Speed_mph'].plot(kind='hist', bins=20, title='Coaster Speed (mph)')
df['Speed_mph'].plot(kind='kde',  title='Coaster Speed (mph)')
```

| Plot | What it shows | Watch out for |
|---|---|---|
| **Histogram** | Splits values into **bins** and counts how many fall in each; bar height = frequency | The number of bins changes the look: too few hides detail, too many looks noisy |
| **KDE** (Kernel Density Estimate) | A **smooth curve** estimating the distribution; area under it = 1 | Can show density beyond the real data range; it's an estimate, not raw data |

**How to read a distribution:**

- **Centre**: where is the peak? (Speed peaks around 50 mph.)
- **Spread**: how wide is it? (Std ≈ 16.7 mph.)
- **Shape**: symmetric, skewed, or multiple peaks?
- **Outliers**: isolated values far from the rest (a few coasters go past 100 mph).

### 3.3 Boxplot (concept from the outline)

A boxplot compresses a distribution into five numbers and is the fastest way to spot outliers:

```python
sns.boxplot(x=df['Speed_mph'])   # extra snippet, not run in the original notebook
```

```
   ├──────┤█████│█████├────────┤   ● ●  ●
  whisker  Q1  median  Q3   whisker     outliers
           └─ IQR = Q3 − Q1 ─┘
```

- Box = middle 50% of data (Q1 to Q3), line inside = **median**
- Whiskers = typically up to 1.5 × IQR from the box
- Dots beyond whiskers = **outliers**

---

## Step 4 · Feature Relationships

**Bivariate analysis** looks at **two variables together**; **multivariate** looks at three or more. This is where patterns appear.

### 4.1 Scatter plot: two numeric variables

```python
df.plot(kind='scatter', x='Speed_mph', y='Height_ft',
        title='Coaster Speed vs. Height')
```

Each dot is one coaster. The cloud trends **upward**: taller coasters tend to be faster (physics: more height → more potential energy → higher speed).

### 4.2 Add a third variable with colour (`hue`)

```python
sns.scatterplot(x='Speed_mph', y='Height_ft', hue='Year_Introduced', data=df)
```

Colour encodes the **year**, so you can see whether newer coasters sit in a different region of the plot than older ones.

### 4.3 Pairplot: every pair at once

```python
sns.pairplot(df,
             vars=['Year_Introduced', 'Speed_mph', 'Height_ft',
                   'Inversions', 'Gforce'],
             hue='Type_Main')
```

A **grid** of plots (5 × 5 = 25 panels here):

- **Off-diagonal**: scatter plots of each pair of variables
- **Diagonal**: the distribution of each variable
- **Colour**: Steel vs Wood vs Other, so you can compare the groups

It's a quick overview, though it can get slow on big datasets.

### 4.4 Correlation: put a number on the relationship

```python
df_corr = df[['Year_Introduced', 'Speed_mph', 'Height_ft',
              'Inversions', 'Gforce']].dropna().corr()
sns.heatmap(df_corr, annot=True)
```

**Pearson correlation (r)** measures **linear** association between two variables:

| r value | Meaning |
|---|---|
| **+1** | Perfect positive: both rise together |
| **0** | No linear relationship |
| **−1** | Perfect negative: one rises as the other falls |

Rule of thumb: |r| < 0.3 weak · 0.3–0.7 moderate · > 0.7 strong.

**Results:**

| | Year | Speed | Height | Inversions | Gforce |
|---|---|---|---|---|---|
| **Year_Introduced** | 1.00 | 0.17 | 0.14 | −0.21 | 0.16 |
| **Speed_mph** | 0.17 | 1.00 | **0.73** | −0.03 | **0.61** |
| **Height_ft** | 0.14 | **0.73** | 1.00 | −0.08 | 0.47 |
| **Inversions** | −0.21 | −0.03 | −0.08 | 1.00 | 0.28 |
| **Gforce** | 0.16 | **0.61** | 0.47 | 0.28 | 1.00 |

**Interpretation:**

- 🔥 **Speed ↔ Height (0.73)**: strong positive. Taller coasters are faster.
- 🔥 **Speed ↔ G-force (0.61)**: moderate-to-strong. Faster rides pull more G's.
- **Height ↔ G-force (0.47)**: moderate.
- **Speed ↔ Inversions (−0.03)**: essentially none. Going upside down says nothing about speed.
- **Year ↔ Inversions (−0.21)**: weak negative. Newer coasters have slightly *fewer* inversions in this data.

> ⚠️ **Two important caveats**
> 1. **Correlation ≠ causation.** A high r says two things move together, not that one causes the other.
> 2. `.dropna()` removes every row with *any* missing value across these five columns. Since `Height_ft` is mostly missing, the correlations are computed on a **much smaller subset** than 990 rows.

### 4.5 Reading a heatmap

A heatmap colours each cell by value. `annot=True` prints the number in each cell. Look for the brightest and darkest cells off the diagonal (the diagonal is always 1.0 because a variable correlates perfectly with itself).

---

## Step 5 · Ask a Question About the Data

The best EDA is **question-driven**. The notebook asks:

> **Which locations have the fastest roller coasters (minimum of 10 coasters)?**

```python
ax = df.query('Location != "Other"') \
       .groupby('Location')['Speed_mph'] \
       .agg(['mean', 'count']) \
       .query('count >= 10') \
       .sort_values('mean')['mean'] \
       .plot(kind='barh', figsize=(12, 5),
             title='Average Coaster Speed by Location')
```

Read the chain line by line. Each step transforms the data:

| Line | What happens |
|---|---|
| `query('Location != "Other"')` | Remove the vague "Other" bucket, which isn't a real place |
| `groupby('Location')['Speed_mph']` | Group coasters by location and look at their speeds |
| `.agg(['mean', 'count'])` | For each location, compute average speed **and** how many coasters |
| `.query('count >= 10')` | Keep only locations with **at least 10** coasters, so averages are reliable |
| `.sort_values('mean')['mean']` | Sort by average speed, keep the mean column |
| `.plot(kind='barh', ...)` | Horizontal bar chart: location vs average speed |

**Why the `count >= 10` filter?** A park with a single record-breaking coaster would otherwise "win" with an average based on one data point. Minimum sample sizes make comparisons fair.

This **filter → group → aggregate → filter → sort → plot** pattern is one of the most useful in all of pandas.

---

## 🧾 EDA Cheat Sheet

| Concept | Code | Use it to… |
|---|---|---|
| Size of data | `df.shape` | Know rows × columns |
| Peek at data | `df.head()`, `df.tail()` | Eyeball real values |
| Column types | `df.dtypes` | Find text-that-should-be-number/date |
| Summary stats | `df.describe()` | See range, centre, spread, hints of outliers |
| Missing values | `df.isna().sum()` | Find gaps per column |
| Duplicates | `df.duplicated(subset=[...])` | Detect repeated records |
| Remove duplicates | `df.loc[~df.duplicated(...)]` | Keep one copy |
| Convert to date | `pd.to_datetime(col)` | Enable date operations |
| Rename | `df.rename(columns={...})` | Clean, consistent names |
| Category counts | `col.value_counts()` | Frequency of each category |
| Histogram | `kind='hist'` | Distribution shape via bins |
| KDE | `kind='kde'` | Smooth distribution estimate |
| Boxplot | `sns.boxplot()` | Median, IQR, outliers |
| Scatter | `kind='scatter'` / `sns.scatterplot` | Relationship between two numbers |
| Pairplot | `sns.pairplot` | All pairwise relationships |
| Correlation | `df.corr()` | Strength of linear relationship |
| Heatmap | `sns.heatmap(corr, annot=True)` | Visualize a correlation matrix |
| Group stats | `groupby().agg()` | Compare categories |
| Filter rows | `df.query('...')` | Readable conditions |

---

## 🎓 Lessons & Common Pitfalls

1. **Look before you leap.** `shape`, `dtypes` and `describe` take seconds and prevent hours of confusion.
2. **Missing data is information.** Know *how much* is missing before trusting any number.
3. **Define "duplicate" carefully.** Name alone isn't unique; combine columns.
4. **Check units and extremes.** A max of 3,937 next to 377 means mixed units, not a super-coaster.
5. **Match the plot to the data type.** Counts → bar chart, one number → histogram/KDE, two numbers → scatter, many numbers → pairplot/heatmap.
6. **Correlation is not causation**, and it only captures *linear* patterns.
7. **Beware small samples.** Averages from 2 items mislead; enforce minimum counts.
8. **Always end with a question.** EDA is about discovering insights, not just drawing charts.

---

## ▶️ Run it yourself

### Requirements

- Python 3.8+
- pandas · numpy · matplotlib · seaborn · jupyter

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook
```

### Get the data

Download the **Roller Coaster Database** (`coaster_db.csv`) from Kaggle and update the path in the notebook:

```python
df = pd.read_csv('coaster_db.csv')   # original path: ../input/rollercoaster-database/coaster_db.csv
```

### Version notes

- `pd.set_option('max_columns', 200)` can raise an error in newer pandas. Use `pd.set_option('display.max_columns', 200)` instead.
- The original path `../input/...` is a Kaggle notebook path; change it for local runs.

---

## 🚀 Where to go next

- Add a **boxplot** for each numeric column and flag outliers with the IQR rule
- **Impute or drop** missing values (e.g. `Height_ft`) and see how correlations change
- Compare **Steel vs Wood** with `groupby('Type_Main')`
- Plot coaster **locations on a map** using `latitude` / `longitude`
- Track how **average speed changes by decade**
- Move on to **feature engineering** and a first **prediction model** (e.g. predict speed from height)

---

## 🙌 Credits

- Tutorial and original notebook: **Medallion Data Science** (YouTube), *Introduction to Exploratory Data Analysis Using Pandas*
- Dataset: Roller Coaster Database (Kaggle)
- Practised and documented by **Thiraviaraj R**

---

⭐ If this README helped you understand EDA, consider starring the repo!
