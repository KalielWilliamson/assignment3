## Instructions for running this code
1. Clone the repository
`git clone https://github.com/KalielWilliamson/assignment3.git`

2. Create a new conda environment with the required packages:
`conda env create -f environment.yml -n <env_name>`

3. Activate the environment:
`conda activate <env_name>`

4. Download the hepmass dataset from uci machine learning repository. Unzip it and place it in the artifacts directory.
link: https://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_train.csv.gz

5. Run the hyperparameter tuning script:
`python assignment3.py --dataset=<dataset_name>`
dataset_name can be either 'HEPMASS' or 'APS_SYSTEM_FAILURE'

6. Run the tuner_analysis notebook to analyze the results of the hyperparameter tuning script:
`jupyter notebook tuner_analysis.ipynb`
