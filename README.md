# Adversarial Attacks on Privacy
This repository stores the codebase for the paper 'Study of Adversarial Attacks on Privacy in Large Language Models'. 

## Abstract
The paper delves into the topic of safeguarding privacy in domain-specific language models using differential privacy techniques. Specifically, we focus on a clinical dataset since adversarial attacks on language models that are trained on healthcare data can exploit personally identifiable information (PII) like patient names, ages, addresses, and unique medical history. Moreover, we highlight optimization and the implementation of the differentially private stochastic gradient (DP-SGD) algorithm. DP-SGD involves adding random noise to the gradients, which prevents reverse engineering of the original data while allowing the model to converge. The report lays out our approach and methodology for building a transformer-based model for text summarization from scratch, implementing DP-SGD, and carrying out a comparative analysis of privacy leakage compared to the original model. Our results were not able to show a difference in the amount of PII outputted by the base SGD model as compared to the DP-SGD model. However, successfully implementing DP-SGD on sensitive clinical data would aid in the continuous development of healthcare domain-specific models that assist in clinical decision-making while upholding patient confidentiality.
