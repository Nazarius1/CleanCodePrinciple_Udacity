# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project showcases the implementation of clean code principle.
Runtime: Python 3.9.7


## Files and data description
- bank_data (contains the file source to create test and training data)
- images/eda (contains images which provides eda information)
- images/results (contains images about model and performance results)
- logs (contains log files from both churn_library.py and churn_script_loggin_and_tests.py)
- models: RFC and LR (contains file models created from file churn_library.py)


## Running Files
1. Create venv. As I use conda, simply execute:
    conda create -name <venvname>

2. Before running the file, make sure the codes comply to python PEP8 style guide.
run the following code to remove unseen mistakes, such as empty spaces and unnecesssary lines:

    autopep8 --in-place --aggressive --aggressive churn_library.py

    autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py

3. Run the following command to get the feedback about your structure and its score:

    pylint churn_library.py

    pylint churn_script_logging_and_tests.py

4. Run the following command to produce the ML models and their analyses:

    python churn_library.py

5. Logs and test using the following command:

    python churn_script_logging_and_tests.py


6. Sequence steps should be:

 - import_data
 - perform_eda
 - encoding categorical columns
 - feature engineering
 - model creation
