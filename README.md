# Adversarial Attacks on Privacy
This repository stores the codebase for the paper 'Study of Adversarial Attacks on Privacy in Large Language Models'. 

## Abstract
The paper delves into the topic of safeguarding privacy in domain-specific language models using differential privacy techniques. Specifically, we focus on a clinical dataset since adversarial attacks on language models that are trained on healthcare data can exploit personally identifiable information (PII) like patient names, ages, addresses, and unique medical history. Moreover, we highlight optimization and the implementation of the differentially private stochastic gradient (DP-SGD) algorithm. DP-SGD involves adding random noise to the gradients, which prevents reverse engineering of the original data while allowing the model to converge. The report lays out our approach and methodology for building a transformer-based model for text summarization from scratch, implementing DP-SGD, and carrying out a comparative analysis of privacy leakage compared to the original model. Our results were not able to show a difference in the amount of PII outputted by the base SGD model as compared to the DP-SGD model. However, successfully implementing DP-SGD on sensitive clinical data would aid in the continuous development of healthcare domain-specific models that assist in clinical decision-making while upholding patient confidentiality.

## Instructions
1) We provide `environment.yaml` which contains a list of libraries to set the environment for this project. You can create it using the command `conda env create -f environment.yaml`
2) You can download the dataset by running the `data_preprocess.py` file in the tasks folder, or you can download it directly from the Kaggle link (https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) and unzip it to the `data/raw/` folder. Running `data_preprocess.py` file will output processed data files ready for input to model training in the data/processed folder.
3) To train a model, you can run the `run_model.py` file while in the `tasks` folder using: `python run_model.py`
To run a specific model_type, execute any of the following:
- `python run_model.py dp-sgd`
- `python run_model.py base`
4) To interact with the model and visualize the model output, `model_playground.ipynb` is provided and contains functions to load the model checkpoint from the saved results for prompting. Note that due to GitHub file size restrictions, this repository does not contain any of our trained model checkpoints.

## Acknowledgements
We would like to thank Dr. Kira Zsolt and the teaching staff at Georgia Institute of Technology for their guidance and support.
