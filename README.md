# **Credit Scoring Business Understanding**  

### **1. Basel II Accord & Interpretable Models**  
The **Basel II Accord** mandates rigorous risk measurement to ensure financial stability, requiring banks to maintain adequate capital reserves against potential losses. This regulatory framework emphasizes:  
- **Transparency**: Regulators and auditors must validate model logic to ensure compliance.  
- **Explainability**: Stakeholders (e.g., underwriters, customers) need clear reasoning for credit decisions to avoid discriminatory practices.  
- **Documentation**: Detailed model documentation is critical for audits and regulatory reviews.  

An **interpretable model** (e.g., Logistic Regression with Weight of Evidence (WoE)) aligns with these requirements by providing:  
- **Traceable decision paths** (e.g., coefficient signs and magnitudes).  
- **Easily auditable feature contributions** (unlike "black-box" models).  

*Example*: A bank using a complex ensemble model might struggle to justify why a loan was denied, risking regulatory penalties or reputational damage.  

---

### **2. Proxy Variables & Business Risks**  
Since direct **default labels** are often unavailable (e.g., due to short customer histories or incomplete data), proxy variables (e.g., 90-day delinquency, charge-offs) are used to approximate risk. However, this introduces risks:  
- **Misalignment**: Proxies may not perfectly correlate with true default behavior (e.g., a delinquent customer might repay eventually).  
- **Bias**: Over-reliance on proxies could exclude creditworthy applicants (e.g., thin-file customers).  
- **Regulatory Scrutiny**: Regulators may challenge the proxy’s validity, especially if it disproportionately impacts protected groups.  

*Mitigation*: Use **multiple proxies** (e.g., delinquency + utilization ratios) and validate their predictive power with robustness checks.  

---

### **3. Model Trade-offs: Simplicity vs. Performance**  

| **Aspect**               | **Simple Model (Logistic Regression + WoE)**       | **Complex Model (Gradient Boosting)**          |  
|--------------------------|----------------------------------------------------|-----------------------------------------------|  
| **Interpretability**      | High (clear feature importance, WoE bins)          | Low (hard to explain interactions)            |  
| **Regulatory Compliance**| Easier to document and justify                     | May require additional explainability tools (SHAP, LIME) |  
| **Predictive Power**      | Lower (linear assumptions)                         | Higher (captures non-linear patterns)         |  
| **Implementation Speed**  | Faster (fewer hyperparameters)                     | Slower (needs tuning, more compute resources) |  
| **Bias Detection**        | Straightforward (e.g., coefficient analysis)        | Requires post-hoc analysis                    |  

**Recommendation**: In regulated contexts, start with a **simple model** to establish baselines and compliance, then incrementally adopt complexity *only* if:  
- Performance gains justify added opacity.  
- Explainability tools (e.g., SHAP) are integrated to meet auditing needs.  

---

### **Key Takeaways**  
1. **Basel II prioritizes transparency** → Interpretable models reduce regulatory and reputational risks.  
2. **Proxy variables are imperfect** → Validate rigorously to avoid biased decisions.  
3. **Balance simplicity and performance** → Complexity must demonstrably improve outcomes without compromising compliance.  


# ** Task Two **  

## 🧪 Exploratory Data Analysis (EDA)

To build a reliable credit scoring model, I conducted a comprehensive exploratory data analysis (EDA) to understand the structure, distribution, and quality of the data provided by the eCommerce platform. Below is a summary of our key EDA tasks:

### 1. Central Tendency, Dispersion, and Shape

* Examined key numerical features such as `Amount` and `Value` to understand their **mean**, **median**, **standard deviation**, **IQR**, **skewness**, and **kurtosis**.
* Identified that both fields exhibit **positive skewness** and **long tails**, which is typical of financial transaction data.

### 2. Visualizing Distributions

* Created **enhanced histograms** and **boxplots** to analyze the distribution of `Amount` and `Value`, with mean and median annotations to highlight skewness.
* Applied **log transformation** on skewed variables for better visualization and modeling readiness.

### 3. Categorical Feature Analysis

* Analyzed the distribution of categorical variables such as `ChannelId`, `CountryCode`, `ProductCategory`, `CurrencyCode`, and `FraudResult`.
* Generated frequency bar plots to observe **class imbalance**, **high-cardinality categories**, and **potential grouping needs**.

### 4. Relationships Between Numerical Features

* Calculated the **correlation matrix** and plotted a heatmap to understand relationships between `Amount`, `Value`, and other numerical attributes.
* Used **pair plots** to inspect linear trends, and flagged highly correlated features for potential dimensionality reduction.
* Considered **Variance Inflation Factor (VIF)** to detect multicollinearity risks.

### 5. Missing Value Analysis and Imputation Strategy

* Identified columns with missing values and calculated the **missing count and percentage**.
* Selected **appropriate imputation strategies**:

  * Median imputation for numerical fields with low missingness
  * Mode or "Unknown" category filling for categorical fields
  * Created binary indicators for missing values where relevant
* Visualized missingness using a **heatmap** to identify potential data quality issues.

---

## ⚙️ Task 3: Feature Engineering

In this stage, we built a **robust, automated, and reproducible data processing pipeline** to transform raw transaction-level data into a model-ready format using `sklearn.pipeline.Pipeline` and `ColumnTransformer`. The goal was to prepare high-quality features that improve model accuracy, interpretability, and generalizability.

### ✅ 1. Aggregated Customer Behavior Features

We engineered aggregated features by grouping transactions by `CustomerId` to understand customer-level financial behavior:

* **Total Transaction Amount** – Sum of all transactions
* **Average Transaction Amount** – Mean per customer
* **Transaction Count** – Total number of transactions
* **Standard Deviation** – Variability in spending

### ✅ 2. Temporal Feature Extraction

From the `TransactionStartTime` field, we extracted key time-based features:

* **Transaction Hour**
* **Transaction Day**
* **Transaction Month**
* **Transaction Year**
* **Day of Week** (optional)

These help identify time-based patterns and behavioral signals.

### ✅ 3. Categorical Feature Encoding

We transformed categorical features into numerical format using:

* **One-Hot Encoding** for nominal categories (e.g., `ChannelId`, `ProductCategory`)
* **Label Encoding** for binary/ordinal fields (e.g., `FraudResult`)

This made them usable by machine learning models.

### ✅ 4. Missing Value Handling

We applied different strategies based on feature type:

* **Numerical**: Median imputation
* **Categorical**: Mode imputation or replacement with `"Unknown"`
* **Row/Column Removal**: Applied for targets or heavily missing fields

### ✅ 5. Scaling Numerical Features

To bring numerical variables to a comparable scale, we implemented:

* **Standardization** (mean = 0, std = 1) using `StandardScaler`
* **Normalization** (scaling to \[0, 1]) using `MinMaxScaler` (optional toggle)

These were applied via `Pipeline` to ensure consistency during training and inference.

### ✅ 6. End-to-End Preprocessing Pipeline

All steps were integrated using `Pipeline` and `ColumnTransformer`, enabling:

* Consistent preprocessing
* Modular and maintainable code
* Compatibility with model training and deployment workflows

---

## 🎯 Task 4: Proxy Target Variable Engineering

Since the dataset lacks a direct label for credit risk or default, we created a **proxy target variable** to identify high-risk customers programmatically. This enables supervised learning for credit scoring.

### Key Steps:

1. **Calculate RFM Metrics**
   For each customer (`CustomerId`), we computed the classic Recency, Frequency, and Monetary (RFM) features based on transaction history:

   * **Recency:** Days since last transaction relative to a snapshot date
   * **Frequency:** Number of transactions
   * **Monetary:** Total transaction amount

2. **Cluster Customers Using K-Means**
   We standardized RFM features and applied K-Means clustering (3 clusters, fixed random state) to segment customers into distinct behavioral groups.

3. **Identify High-Risk Cluster**
   By analyzing cluster statistics (mean Recency, Frequency, Monetary), we identified the cluster characterized by:

   * Highest Recency (longest inactivity)
   * Lowest Frequency and Monetary values
     This cluster was labeled as the **high-risk proxy group**.

4. **Assign Binary Target Label**
   We created a new binary column, `is_high_risk`, assigning:

   * `1` for customers in the high-risk cluster
   * `0` for all others

5. **Integrate Target Variable**
   The `is_high_risk` label was merged into the main customer-level dataset (RFM data), ready to be used as the target variable for model training.

