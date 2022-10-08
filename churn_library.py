# library doc string
'''
This module contains the functions needed to run project predict customer churn
Author: Nazarius Hedi Suseno
Date: 10/08/2022
'''

# import libraries
import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
#os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        dataframe = pd.read_csv(pth)
        logging.info(
            'File {} has been succesfullly converted to csv'.format(pth))
        return dataframe
    except:
        logging.error(
            'Error occurred when trying to convert file {} to csv'.format(pth))


def perform_eda(data_frame):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    # cat_columns = [
    #     'Gender',
    #     'Education_Level',
    #     'Marital_Status',
    #     'Income_Category',
    #     'Card_Category'
    # ]

    # quant_columns = [
    #     'Customer_Age',
    #     'Dependent_count',
    #     'Months_on_book',
    #     'Total_Relationship_Count',
    #     'Months_Inactive_12_mon',
    #     'Contacts_Count_12_mon',
    #     'Credit_Limit',
    #     'Total_Revolving_Bal',
    #     'Avg_Open_To_Buy',
    #     'Total_Amt_Chng_Q4_Q1',
    #     'Total_Trans_Amt',
    #     'Total_Trans_Ct',
    #     'Total_Ct_Chng_Q4_Q1',
    #     'Avg_Utilization_Ratio'
    # ]

    # create EDA and save
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    plt.savefig('images/eda/churn_hist.png')

    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    plt.savefig('images/eda/Customer_Age_hist.png')

    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/Marital_Status_bar.png')

    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/eda/Total_Trans_Ct_hisplot.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/eda/corr_coeff.png')


def encoder_helper(data_frame, cat_list, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
    df: pandas dataframe
    cat_list: list of columns that contain categorical features
    response: string of response name [optional argument that
    could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    new_dict = {}
    for idx, col in enumerate(cat_list):
        new_dict[response[idx]] = data_frame.groupby(col).mean()['Churn']

    data_frame[response] = data_frame[cat_list]
    data_frame = data_frame.replace(new_dict)
    return data_frame


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that
              could be used for naming variables or index y column]
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df[response]

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # create random_forest_classifier report on test data
    rf_test_df = pd.DataFrame(classification_report(
        y_test, y_test_preds_rf, output_dict=True)).transpose()
    plt.figure()
    rf_test_df.plot()
    plt.savefig('images/results/rfc_test.png')

    # create random_forest_classifier report on training data
    rf_train_df = pd.DataFrame(classification_report(
        y_train, y_train_preds_rf, output_dict=True)).transpose()
    plt.figure()
    rf_train_df.plot()
    plt.savefig('images/results/rfc_train.png')

    # create logistic_regression_classifier report on test data
    lr_test_df = pd.DataFrame(classification_report(
        y_test, y_test_preds_lr, output_dict=True)).transpose()
    plt.figure()
    lr_test_df.plot()
    plt.savefig('images/results/lrc_test.png')

    # create logistic_regression_classifier report on train data
    lr_train_df = pd.DataFrame(classification_report(
        y_train, y_train_preds_lr, output_dict=True)).transpose()
    plt.figure()
    lr_train_df.plot()
    plt.savefig('images/results/lrc_test.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=45)
    plt.plot()
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    logging.info(
        'INFO: Begining the training of the random forest and linear regression')

    # Instanciating the Random Forest classifier
    rfc = RandomForestClassifier(random_state=42)
    # Instanciating the Logistic Regression classifier
    lrc = LogisticRegression(solver='lbfgs', max_iter=200)

    logging.info('INFO: Initialization of random forest parameters')

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Training the random forest model on the data
    logging.info('Fitting data into the random forest...')
    gscv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    gscv_rfc.fit(X_train, y_train)
    logging.info('Training random forest model finished')

    # Training the logistic regression model on the data
    logging.info('Fitting data into the linear regression model...')
    lrc.fit(X_train, y_train)
    logging.info('Training linear regression model finished')

    # Saving the best model of the random forest
    logging.info('Saving the random forest model...')
    joblib.dump(gscv_rfc.best_estimator_, 'models/rfc_model.pkl')
    logging.info('Random forest model has been saved')

    # Saving the logistic regression model
    logging.info('Saving the random forest model ...')
    joblib.dump(lrc, 'models/lrc_model.pkl')
    logging.info('Random forest model has been saved')

    # Creating the result plots
    logging.info('Creating ROC curves...')

    logging.info('INFO: Plotting the learner regression ROC curve.')
    # Plotting the ROC curve for the logistic regression
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))

    the_ax = plt.gca()
    logging.info('Plotting the Random forest ROC curve....')

    # Plotting the random forest ROC curve
    plot_roc_curve(gscv_rfc.best_estimator_,
                   X_test, y_test, ax=the_ax, alpha=0.8)
    lrc_plot.plot(ax=the_ax, alpha=0.8)
    plt.title('ROC curves of Random forest and Linear regression models')
    logging.info('Saving the figure ...')
    plt.savefig('images/results/roc_curves.png')
    logging.info('ROC curves are generated and saved')


if __name__ == "__main__":
    logging.basicConfig(
        filename='logs/churn_library.log',
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S")
    logging.info('Begin process churn_library...')
    DATA_PATH = "data/bank_data.csv"
    full_data = import_data(DATA_PATH)

    full_data['Churn'] = full_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    CAT_LIST = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    response = [cat + '_Churn' for cat in CAT_LIST]

    df_encode = encoder_helper(full_data, CAT_LIST, response)

    # perform EDA
    perform_eda(df_encode)

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        df_encode, 'Churn')
    # print(X_train.head)

    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

    RFC_PATH = 'models/rfc_model.pkl'
    LR_PATH = 'models/lrc_model.pkl'

    rfc_model = joblib.load(RFC_PATH)
    lr_model = joblib.load(LR_PATH)

    y_train_preds_lr = lr_model.predict(X_TRAIN)
    y_train_preds_rf = rfc_model.predict(X_TRAIN)
    y_test_preds_lr = lr_model.predict(X_TEST)
    y_test_preds_rf = rfc_model.predict(X_TEST)

    classification_report_image(Y_TRAIN,
                                Y_TEST,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf
                                )

    X_data = pd.concat([X_TRAIN, X_TEST])
    RFC_REPORT_PATH = 'images/results/feature_importance.png'
    feature_importance_plot(rfc_model, X_data, RFC_REPORT_PATH)
    logging.info('End process churn_library')