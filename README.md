![OPE Research CI](https://github.com/laxmanmaheshwaram/Ope-RecSys/actions/workflows/tests.yml/badge.svg)
# Off-Policy Evaluation of Recommendation Policies using Inverse Propensity Scoring

##  Research Overview

In production recommendation systems, evaluating a new policy  (target) using historical data from a logging policy  is critical for safe deployment. This project investigates the **Bias-Variance Trade-off** in Off-Policy Evaluation (OPE).

### Hypothesis

* **H1:** Inverse Propensity Scoring (IPS) provides an unbiased estimate but suffers from extreme variance when policies diverge.
* **H2:** The Direct Method (DM) provides low-variance estimates but introduces systematic bias due to reward model misspecification.
* **H3:** The Doubly Robust (DR) estimator achieves the lowest Mean Squared Error (MSE) by combining the strengths of both.

---

##  Project Structure

The repository is organized into a modular research package:

* `src/environment.py`: Synthetic Contextual Bandit simulator.
* `src/policies.py`: Implementation of stochastic logging () and greedy target () policies.
* `src/estimators.py`: Core logic for **IPS**, **DM**, and **DR** calculations.
* `src/utils.py`: Visualization engine with Standard Error shading.
* `main.py`: Main entry point for running multi-trial experiments.

---

##  Experimental Results

### Convergence Analysis

The following plots demonstrate how the estimators approach the **Ground Truth** (calculated via Monte Carlo simulation) as the number of logged samples  increases.

#### Single Trial Performance

* **Observation:** Notice the high-variance "spikes" in the IPS and DM lines around . This illustrates **propensity overfitting**, where a rare action with a high reward causes a temporary explosion in the estimate error.

#### Aggregated Performance (with Standard Error)
![Alternative Text](results/ope_research_plot.png)
* **Interpretation:** By running  trials, we visualize the **Standard Error (SE)** via the shaded regions.
* **Findings:** * **IPS (Blue):** Shows the widest confidence interval, confirming its high variance.
* **DM (Red):** Shows the narrowest interval but maintains a higher mean error, confirming systematic **bias**.
* **DR (Green):** Maintains the best balance, staying closer to the ground truth with significantly less volatility than IPS.



---

##  Conclusion

Our research confirms that for high-stakes recommendation tasks, the **Doubly Robust** estimator is the most reliable tool for OPE. It provides a "safety net": if the reward model is slightly inaccurate, the IPS component corrects it; if the importance weights are extreme, the reward model stabilizes the estimate.

---

##  Technical Appendix: Doubly Robust (DR) Formulation

The DR estimator leverages a reward model  as a control variate to reduce the variance of the IPS estimate.

### The Equation

### Key Properties

1. **Unbiasedness:** The estimator remains unbiased as long as *either* the propensity scores or the reward model is correctly specified.
2. **Variance Reduction:** By subtracting the predicted reward  from the actual reward , the importance weights only scale the **residual error**. This leads to a significantly more stable estimate than standard IPS when importance weights are large.

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

