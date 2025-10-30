# Code for efficient Epistemic Uncertainty Estimation in Probabilistic Ensembles  
*Authors: Lucas Berry & David Meger*  
*Affiliation: McGill University — Centre for Intelligent Machines*  
📧 Contact: [lucas.berry@mail.mcgill.ca](mailto:lucas.berry@mail.mcgill.ca)

---

## 📚 Papers Included

### 🔹 1. [Normalizing Flow Ensembles for Rich Aleatoric and Epistemic Uncertainty Modeling](https://arxiv.org/abs/2302.01312)  
**Key contributions:**  
- Memory-efficient NF ensembles using fixed dropout masks  
- Analytical/less-sampling method for aleatoric uncertainty via NF base distribution ensembles 
- Comprehensive benchmarks of epistemic & aleatoric uncertainty in regression/active-learning settings  

---

### 🔹 2. [Efficient Epistemic Uncertainty Estimation in Regression Ensemble Models Using Pairwise-Distance Estimators](https://arxiv.org/abs/2308.13498)  
**Key contributions:**  
- Closed-form estimates of epistemic uncertainty via pairwise-distance estimators  
- Significant speed-up vs. MC sampling, especially in higher dimensions  
- Strong active-learning results on high-dimensional regression tasks  

---

## 🚀 Getting Started  
**Python version:** 3.13+
```bash
pip install -r requirements.txt
cd nflows
pip install -e .
```
---

## Nflows Base
```
python main.py --base_distro --model nflows_ensemble --num_layers 1 --hids 50 --env Humanoid-v2 --ensemble_size 5 --points_2_add 10 --acquisition_function kl_exp
```
## Nflows Out
```
python main.py --model nflows_ensemble --num_layers 1 --hids 50 --env Humanoid-v2 --ensemble_size 5 --points_2_add 10 --acquisition_function kl_exp
```
---

| **Flag** | **Description** | **Possible Values** |
|-----------|-----------------|--------------------|
| `--model` | Type of ensemble model | `nflows_ensemble`, `nn_ensemble` |
| `--acquisition_function` | Active learning acquisition rule | `kl_exp`, `bhatt_exp`, `random`, `badge`, `bait`, `batchbald`, `sample_bald` |
| `--env` | Environment / dataset | `Pendulum-v0`, `Hopper-v2`, `Humanoid-v2`, `Ant-v2`, `bimodal`, `hetero` |


Note that setting `--acquisition_function` to `kl_exp` or `bhatt_exp` corresponds to using the KL or Bhattacharyya PairEpEsts, respectively.

