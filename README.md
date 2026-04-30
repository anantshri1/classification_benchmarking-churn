# Telco Customer Churn — Classification Benchmarking

A supervised binary classification project on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), benchmarking five classifiers with a focus on deliberate feature engineering to surface churn-relevant signals from the raw tabular data.

## The Challenge
Customer churn — the rate at which subscribers cancel their service — is a direct revenue loss. The goal here is to predict, from a customer's account and service profile, whether they will churn. Because failing to identify a churner (false negative) is typically more costly than a false alarm (false positive), model evaluation emphasises recall on the churned class alongside AUC-ROC.

**Source:** [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Size:** 7,043 customers × 21 raw features  
**Target:** `Churn` — whether the customer left within the last month

### Raw Features

| Feature | Type | Description |
|---|---|---|
| `customerID` | ID | Unique customer identifier — dropped before modelling |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary (0/1) | Whether the customer is 65 or older |
| `Partner` | Categorical | Whether the customer has a partner |
| `Dependents` | Categorical | Whether the customer has dependents |
| `tenure` | Numeric | Number of months the customer has been with the company |
| `PhoneService` | Categorical | Whether the customer has a phone line |
| `MultipleLines` | Categorical | Whether the customer has multiple lines (or no phone service) |
| `InternetService` | Categorical | DSL / Fibre optic / No |
| `OnlineSecurity` | Categorical | Whether the customer has online security add-on |
| `OnlineBackup` | Categorical | Whether the customer has online backup add-on |
| `DeviceProtection` | Categorical | Whether the customer has device protection add-on |
| `TechSupport` | Categorical | Whether the customer has tech support add-on |
| `StreamingTV` | Categorical | Whether the customer streams TV |
| `StreamingMovies` | Categorical | Whether the customer streams movies |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categorical | Whether billing is paperless |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | Numeric | Current monthly charge in USD |
| `TotalCharges` | Numeric | Total amount charged to date (stored as string in raw data — requires casting) |
| `Churn` | Target | Yes / No — converted to 1 / 0 |

**Data quality notes:**
- `TotalCharges` is read as `object` dtype and must be coerced to numeric; 11 rows with empty strings become `NaN` and are dropped (< 0.2% of data).
- No duplicate rows.
- No other missing values in the raw file.

## Feature Engineering

The raw dataset contains serviceable features but several important signals are implicit or require combination. Twelve new features are constructed in three conceptual groups.

### 1. Service Breadth Features

#### `TotalServices`

```python
service_cols = [
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]
df['TotalServices'] = (
    df[service_cols]
    .apply(lambda col: col.map({'Yes': 1, 'No': 0,
                                 'No internet service': 0,
                                 'No phone service': 0}))
    .sum(axis=1)
)
```

Counts how many of the nine available services a customer has subscribed to, treating "No internet service" and "No phone service" as equivalent to "No" (i.e., not subscribed). The resulting integer ranges from 0 to 9.

**Why it matters:** Service breadth is a proxy for switching cost. A customer using six services faces more friction in cancelling than one using only a base phone line. Research on churn consistently shows that customers with fewer services are more likely to leave.

#### `ChargePerService`

```python
df['ChargePerService'] = df['MonthlyCharges'] / (df['TotalServices'] + 1)
```

The monthly charge divided by the number of services plus one. The `+ 1` prevents division by zero for customers with `TotalServices = 0`, and also acts as a mild Laplace smoothing — without it, a customer with one service would look identical (in ratio terms) to a customer with one service at twice the price.

**Why it matters:** This normalises cost by usage. A high `ChargePerService` means the customer is paying a lot relative to what they're getting, which may indicate overpricing or underutilisation — both potential churn drivers. A low value suggests good value for money, which is associated with retention.

---

### 2. Tenure-Based Features

#### `ChargesTenureRatio`

```python
df['ChargesTenureRatio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
```

Monthly charges divided by tenure in months (again `+ 1` to handle new customers with `tenure = 0`).

**Why it matters:** For a new customer, this ratio is high — they're paying full price with no accumulated loyalty. For a long-tenure customer, the ratio is low. Empirically, recent customers churn at higher rates; this feature lets models capture that effect on a continuous scale rather than relying solely on raw tenure.

#### `TotalChargesExpected` and `ChargesGap`

```python
df['TotalChargesExpected'] = df['MonthlyCharges'] * df['tenure']
df['ChargesGap'] = df['TotalChargesExpected'] - df['TotalCharges']
```

`TotalChargesExpected` is what a customer *would* have paid if their monthly rate had always been the same as today's. `ChargesGap` is the difference between that expectation and what they actually paid.

**Why it matters:** A positive gap (expected > actual) means the customer's monthly rate has increased over time — they were paying less before. This could reflect a promotional rate that has expired, a plan upgrade, or price creep. Customers who have seen their bills rise may be more likely to shop around. A negative gap means they've been paying more historically — possibly on a premium plan they've since downgraded.

#### `IsNewCustomer`

```python
df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
```

Binary flag: 1 if the customer has been with the company six months or less.

**Why it matters:** The first few months are a high-risk period. Customers who signed up under a promotion and find the regular price disappointing, or who have not yet integrated the service into their routine, are most likely to churn early. This discretises the high-risk region of the tenure distribution.

#### `IsLoyalCustomer`

```python
df['IsLoyalCustomer'] = (df['tenure'] >= 48).astype(int)
```

Binary flag: 1 if the customer has been with the company four years or more.

**Why it matters:** The complement of `IsNewCustomer`. Long-tenure customers have demonstrated strong retention behaviour; they are unlikely to churn absent a triggering event. Flagging them lets a model explicitly discount churn probability for this cohort.

#### `TenureBucket` → `TenureScore`

```python
df['TenureBucket'] = pd.cut(
    df['tenure'],
    bins=[0, 6, 12, 24, 48, 72],
    labels=['0-6 mo', '6-12 mo', '12-24 mo', '24-48 mo', '48-72 mo']
)
tenure_map = {'0-6 mo': 0, '6-12 mo': 1, '12-24 mo': 2, '24-48 mo': 3, '48-72 mo': 4}
df['TenureScore'] = df['TenureBucket'].map(tenure_map)
```

Tenure is binned into five intervals and then mapped to an ordinal integer (0–4). `TenureBucket` is subsequently dropped; only the numeric `TenureScore` is retained for modelling.

**Why it matters:** Raw `tenure` is a continuous variable, but churn risk does not decrease linearly with tenure — the relationship is non-linear, with the steepest drop in churn risk occurring in the first one to two years. Binning captures these non-linear thresholds explicitly and in a form that is interpretable. The ordinal encoding preserves the monotonic ordering (longer tenure → higher score → lower expected churn) without imposing a specific functional shape.

---

### 3. Risk Profile Features

#### `IsMonthToMonth`

```python
df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
```

Binary flag for month-to-month contract holders.

**Why it matters:** Contract type is the single strongest predictor of churn in this dataset. Month-to-month customers have no lock-in; they can cancel without penalty at any time. By isolating this as a binary flag, models can interact it easily with other risk features.

#### `IsElectronicCheck`

```python
df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
```

Binary flag for customers paying by electronic check.

**Why it matters:** Electronic check customers churn at a substantially higher rate than those on automatic payments (bank transfer or credit card). One interpretation is that automatic payment customers have higher inertia — cancelling requires an active decision to stop the payment as well. Electronic check customers face no such friction. This payment method is also correlated with lower income brackets and month-to-month contracts.

#### `NoOnlineSecurity` and `NoTechSupport`

```python
df['NoOnlineSecurity'] = (df['OnlineSecurity'] == 'No').astype(int)
df['NoTechSupport']    = (df['TechSupport'] == 'No').astype(int)
```

Binary flags for customers who do not have online security or tech support add-ons.

**Why it matters:** These add-ons are associated with retention; customers who opt into them are paying for services that increase switching cost and signal investment in the relationship. Customers without them have less tying them to the provider, and when things go wrong (a security incident, a technical fault) they have less support to fall back on — increasing frustration and churn propensity.

#### `HighRiskProfile`

```python
df['HighRiskProfile'] = (
    (df['IsMonthToMonth'] == 1) &
    (df['NoOnlineSecurity'] == 1) &
    (df['IsElectronicCheck'] == 1)
).astype(int)
```

Binary flag: 1 if the customer satisfies all three conditions simultaneously — month-to-month contract, no online security, and electronic check payment.

**Why it matters:** This is a conjunction feature that identifies the intersection of three individually high-risk signals. Each condition alone is associated with elevated churn; their co-occurrence defines a customer archetype (low commitment, low add-on engagement, friction-free cancellation) that is strongly over-represented in churned customers. A single composite flag is more stable as a model input than relying on the model to discover the three-way interaction itself — particularly important for linear models like Logistic Regression that cannot learn interactions without explicit construction.

---

### Encoding

All remaining categorical columns are label-encoded using `sklearn.LabelEncoder`. Note that a single encoder instance is reused across columns in a loop, which means the fitted state refers only to the last column encoded — this is fine here since no inverse-transform is performed, but be aware of it if extending the notebook.

## Modelling Pipeline

All models are trained on an 80/20 stratified train/test split (`random_state=0`), preserving the ~26% churn rate in both sets. Features are standardised with `StandardScaler` fitted on the training set only.

Class imbalance (roughly 3:1 retained-to-churned) is addressed via `class_weight='balanced'` for Logistic Regression, Random Forest, and SVM, and via explicit class weight calculation for the Neural Network.

| Model | Key Hyperparameters | Notes |
|---|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight='balanced'` | Baseline linear model; features scaled |
| Random Forest | `n_estimators=5000`, `class_weight='balanced'` | High estimator count for stable OOB variance |
| XGBoost | `n_estimators=400`, `max_depth=6`, `lr=0.01`, `subsample=0.8` | Gradient boosting; `eval_metric='logloss'` |
| SVM (Linear + RBF) | PCA preprocessing (`n_components=0.95`), 5-fold GridSearchCV over `C` (and `gamma` for RBF) | Dimensionality reduction before SVM for computational tractability |
| Neural Network | 64→32→1 Dense, BatchNorm + Dropout, Adam (`lr=0.001`), EarlyStopping on val AUC | Up to 200 epochs; best weights restored |

## Results

Model performance is evaluated via classification report (precision, recall, F1 per class) and AUC-ROC. Confusion matrices are plotted for each model. A combined ROC curve plot at the end of the notebook gives a direct AUC comparison across all five classifiers.

Given the class imbalance and the asymmetric cost of missing a churner, **recall on the churned class and AUC-ROC** are the primary metrics of interest, not overall accuracy.

| Model | AUC |
|---|---|
| Logistic Regression | 0.854 |
| Random Forest | 0.838 |
| Linear SVM + PCA | 0.836 |
| RBF SVM + PCA | 0.803 |
| XGBoost | 0.850 |
| Two-layer Neural Net | 0.851 |

## How to Run
1. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost kagglehub
```
A Kaggle account and API token are required for `kagglehub` to download the dataset. See the [Kaggle API docs](https://www.kaggle.com/docs/api) for setup.

2. Run `ClassificationAnalysis_Benchmarking_2.ipynb` top to bottom.

## Tech Stack

Python · pandas · scikit-learn · TensorFlow · XGBoost · matplotlib · seaborn
