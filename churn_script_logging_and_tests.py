'''
This module contains the functions needed to run tests for the project predict customer churn.
Author : Nazarius Hedi Suseno
Date: 10/08/2022
'''

from msilib.schema import Directory
import os
import logging
import pytest
import churn_library as cl


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cl.import_data("data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    logging.info("## Testing the EDA function ##")

# Check if the eda folder exist
    logging.info("Checking the existence of eda folder...")
    eda_path = 'images/eda'
    try:
        assert os.path.exists(eda_path)
        logging.info("SUCCESS: Folder exists...")
    except AssertionError as err:
        logging.error("ERROR: test_eda: folder doest not exists...")
        raise err

    logging.info("Importing data...")
    df_data = cl.import_data("data/bank_data.csv")

    logging.info("Performing EDA...")
    cl.perform_eda(df_data)
    logging.info("EDA done...")

    lst_files = os.listdir(eda_path)
    try:
        assert len(lst_files) == 5
        logging.info("number of files are correct...")
    except BaseException:
        logging.error("test_eda: either files not exists or less than 5...")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    cat_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    response = [cat + '_Churn' for cat in cat_lst]

    df_data = cl.import_data("data/bank_data.csv")
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    df_data = encoder_helper(df_data, cat_lst, response)

    try:
        assert len(response) == len(cat_lst)
        logging.info("SUCCESS: the category_list and the response list have the same length")
    except AssertionError as err:
        logging.error('ERROR: category_lst and response should have the same length!')
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    logging.info("## Testing the feature engineering function ##")

    cat_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    response = [cat + '_Churn' for cat in cat_lst]

    df_data = cl.import_data("data/bank_data.csv")
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    df_data = cl.encoder_helper(df_data, cat_lst, response)
    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
        df_data, 'Churn')

    total_size = df_data.shape[0]
    x_train_size = X_train.shape[0]
    x_test_size = X_test.shape[0]
    y_train_size = y_train.shape[0]
    y_test_size = y_test.shape[0]

    try:
        assert (x_train_size + x_test_size) == total_size
        logging.info('SUCCESS: number of features data equals the total size of the file source')
    except AssertionError as err:
        logging.error('ERROR: unequal data numbers between x_train+x_test data and the total data')
        raise err

    try:
        assert(y_train_size + y_test_size) == total_size
        logging.info('SUCCESS: number of target data equals the total size of the file source')
    except AssertionError as err:
        logging.error('ERROR: unequal data numbers between y_train + y_test_size data and the total data')
        raise err


def test_train_models():
    '''
    test train_models
    '''

    logging.info("## Testing the train model function ##")

    cat_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    response = [cat + '_Churn' for cat in cat_lst]

    df_data = cl.import_data("data/bank_data.csv")
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    df_data = cl.encoder_helper(df_data, cat_lst, response)
    x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
        df_data, 'Churn')

    rfc_path = 'models/rfc_model.pkl'
    lr_path = 'models/lrc_model.pkl'

    cl.train_models(x_train, x_test, y_train, y_test)

    from os.path import exists

    try:
        assert exists(rfc_path)
        logging.info('SUCCESS: model rfc.pkl exists')
    except BaseException:
        logging.error('ERROR: file rfc.pkl not found')

    try:
        assert exists(lr_path)
        logging.info('SUCCESSS: model lr.pkl exists')
    except BaseException:
        logging.error('ERROR: file lr.pkl not found')


if __name__ == "__main__":
    logging.basicConfig(
        filename='logs/churn_script_logging_and_tests.log',
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.info('Begin process logging and test process...')
    test_import()
    test_eda()
    test_train_models()
    test_perform_feature_engineering()
    logging.info('logging and test process finished')