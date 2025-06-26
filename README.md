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


# ** Task One **  

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
