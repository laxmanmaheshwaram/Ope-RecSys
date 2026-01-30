![OPE Research CI](https://github.com/laxmanmaheshwaram/Ope-RecSys/actions/workflows/tests.yml/badge.svg)
# Robust Off-Policy Evaluation for Contextual Bandits in Recommendation Systems

### **Abstract**

> This repository presents a systematic investigation into the reliability and stability of **Off-Policy Evaluation (OPE)** estimators within the context of bandit-based recommendation. While modern estimators like **Doubly Robust (DR)** aim to mitigate the variance of **Inverse Propensity Scoring (IPS)**, their performance is highly sensitive to environment dynamics and model calibration. We provide a rigorous benchmarking suite that analyzes estimator behavior under **varying policy divergence (KL-divergence)** and **reward model misspecification**. Our findings characterize the "breakdown points" of these estimators, demonstrating that while DR offers superior convergence in low-noise settings, it inherits significant bias when the reward model is poorly calibrated. This work serves as a research artifact for understanding the trade-offs between bias and variance in offline decision-making.

---

##  Background & Theoretical Framework

This project builds upon the foundational principles of **Counterfactual Risk Minimization (CRM)** and Offline Policy Evaluation. Our implementation and experimental design are informed by the following key research pillars:

* **The Doubly Robust (DR) Estimator:** As proposed by **Dudík et al. (2011)**, we utilize the DR estimator to bridge the gap between the high-variance **Inverse Propensity Score (IPS)** and the high-bias **Direct Method (DM)**.
* **The Common Support Assumption:** Our robustness study specifically tests the **Positivity Assumption**, which states that for any action  where , the logging policy must also have .
* **Benchmarking Standards:** Our experimental setup follows evaluation protocols established by the **Open Bandit Pipeline (Saito et al., 2020)**, focusing on the Mean Absolute Error (MAE) relative to the ground-truth reward of the target policy.

---

##  Experimental Analysis & Research Findings

### 1. Convergence Stability and Variance Reduction

Analysis of the OPE Convergence plot demonstrates the empirical validation of the **Doubly Robust (DR)** estimator's superiority in finite sample regimes.
![Alternative Text](results/ope_research_plot.png)
* **Bias-Variance Trade-off:** While the Direct Method (DM) shows lower variance, it suffers from asymptotic bias due to reward model misspecification.
* **DR Performance:** The DR estimator successfully "interpolates" between DM and IPS, achieving a lower Absolute Error than IPS while maintaining tighter confidence bounds (Standard Error shading) as  increases.

### 2. Robustness to Policy Divergence (Temperature Scaling)

We analyzed the impact of target policy "peakiness" (controlled via temperature ) on estimation error. As  decreases, the target policy  concentrates on actions rarely explored by the logging policy .

| Temperature () | IPS MAE | IPS Std. Dev | Research Insight |
| --- | --- | --- | --- |
| **2.0 (High Support)** | 0.046 | 0.027 | High overlap leads to stable, low-error estimates. |
| **0.5 (Mid Divergence)** | 0.111 | 0.067 | Error more than doubles as policy support vanishes. |
| **0.2 (Low Support)** | 0.113 | 0.081 | Extreme variance makes IPS unreliable for greedy policies. |

**Scientific Conclusion:** This confirms that the violation of **Common Support** leads to quadratic growth in variance, rendering standard IPS insufficient for evaluating exploitative policies.

### 3. Sensitivity Analysis: The Impact of Model Noise

Our sensitivity analysis reveals a critical research finding regarding the DR estimator's dependence on the reward model .
![Alternative Text](results/sensitivity_plot.png)
* **Observation:** In high-noise regimes (), the **IPS Baseline** outperformed the **Doubly Robust** estimate.
* **PhD-Grade Insight:** This suggests a "Curse of Misspecification." When reward model error exceeds the natural variance of importance weights, DR "inherits" this noise, causing it to underperform. This highlights the risk of using hybrid estimators in environments where the reward signal is difficult to model.

---

##  Project Structure & Reproducibility

### Structure

* `src/`: Core implementation of estimators and synthetic environments.
* `experiments/`: Standalone research scripts for robustness and sensitivity sweeps.
* `results/`: Output logs and research visualizations.

### Execution

To reproduce the research findings, run the following from the project root:

```bash
# Set up Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run Robustness Study (Policy Divergence)
python -m experiments.robustness_study

# Run Sensitivity Analysis (Reward Model Noise)
python -m experiments.sensitivity_analysis

```

---

## 📝 Limitations & Future Work

* **Static Support:** This study assumes a static action space. Future work should investigate **Infinite Action Spaces** where propensity scores vanish.
* **Off-Policy Selection:** We aim to extend this work to **Off-Policy Selection (OPS)**, determining not just the value of a policy, but the rank-order of multiple candidate policies under uncertainty.

---


##  Citation

```bibtex
@misc{laxmanmaheshwaram2026ope,
  author = {Laxman Maheshwaram},
  title = {Off-Policy Evaluation of Recommendation Policies Using Inverse Propensity Scoring},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/laxmanmaheshwaram/Ope-RecSys}}
}

```

---
##  How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/laxmanmaheshwaram/Ope-RecSys-phd.git
cd Ope-RecSys

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Execute the experiment:**
```bash
python main.py

```



---

##  References

* Dudík, M., Langford, J., & Li, L. (2011). *Doubly Robust Policy Evaluation and Learning*.
* Swaminathan, A., & Joachims, T. (2015). *Batch Learning from Logged Contextual Bandit Feedback*.

