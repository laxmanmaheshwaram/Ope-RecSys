
# Off-Policy Evaluation (OPE) for RecSys: A Comparative Study of IPS, DM, and DR

##  Research Overview

In recommendation systems (RecSys), deploying a new policy  (the target policy) to live users is often risky and expensive. This project investigates **Off-Policy Evaluation (OPE)**—a framework for estimating the performance of a new policy using only historical data collected by a different policy  (the logging policy).

### The Hypothesis

We test the **Bias-Variance Trade-off** in three foundational estimators:

1. **Inverse Propensity Scoring (IPS):** Is it truly unbiased under "common support" conditions?
2. **Direct Method (DM):** How much does model misspecification (reward modeling) bias the results?
3. **Doubly Robust (DR):** Does the combination of IPS and DM consistently yield the lowest Mean Squared Error (MSE)?

---

##  Methodology

The project implements a **Contextual Multi-Armed Bandit (CMAB)** simulator.

### Estimator Formulations

* **IPS (Unbiased but High Variance):**


* **Direct Method (Biased but Low Variance):**


* **Doubly Robust (Best of Both Worlds):**



---

##  Key Findings

Based on our synthetic experiments (see `results/plots`):

* **IPS** remains unbiased but becomes highly unstable when the target policy  is significantly different from the logging policy  (low propensity scores).
* **DM** is highly dependent on the quality of the `Ridge` regressor. If the latent reward function is non-linear, DM fails to converge to the ground truth.
* **DR** maintains robustness; even with a sub-optimal reward model, the importance sampling term corrects the bias.

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

