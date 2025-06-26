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
