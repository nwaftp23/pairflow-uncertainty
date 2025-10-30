# 🧩 Datasets

The datasets used in our experiments were constructed to introduce **domain shift** between training, testing, and oracle data — a setup that closely mirrors conditions in **imitation learning** and **active learning** research.  

## Data Collection Protocol

1. **Training data:** collected using a *random policy* to ensure high variability and diverse, suboptimal samples.  
2. **Test data:** collected using an *expert policy* trained with **Soft Actor-Critic (SAC)**.  
3. **Oracle data:** a *mixed* dataset containing **10% expert** and **90% random** samples.

This design introduces a clear **distributional shift** between the random-policy training data and the expert-policy test data.  
Such a setup challenges the acquisition functions to effectively **identify and isolate expert-like samples** within the oracle dataset — a crucial ability for robust uncertainty estimation and data selection.
