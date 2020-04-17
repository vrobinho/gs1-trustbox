#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Packages
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import glob
import os

# Parallelization
from multiprocessing import Pool


# predictive model

import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# MLflow specific
import mlflow.sklearn
import time

from google.cloud import storage

# Boosting the performance
from pandarallel import pandarallel

import click

# Util function
def read_xls(filename):
    return pd.read_excel(filename, index_col=None, header=list(range(1, 7)), dtype=str)

# Core function
@click.command()
@click.option('--n-jobs', default=2)
@click.option('--random-state', default=42)
@click.option('--bucket-name', default=None)
@click.option('--n-cpus', default=1, help='The number of cpus for the parallelisation')
def run(n_jobs, random_state, bucket_name, n_cpus): 
    start_time = time.time()

    pandarallel.initialize()

    with mlflow.start_run():
        # Read the data data
        # ---------------------------------------------------------------------------------------------------
        step_start_time = time.time()

        if bucket_name:
            path = bucket_name + '/data/recp_Complete_TB_download_in_Excel_1462/'
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs()
            for blob in blobs:
                print('Blobs: {}'.format(blob.name))
                destination_uri = '{}/{}'.format(os.getcwd(), blob.name)
                os_path_dir_name = "/".join(destination_uri.split("/")[:-1])
                if not os.path.exists(os_path_dir_name):
                    Path(os_path_dir_name).mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(destination_uri)
            path = os.getcwd() + '/data/recp_Complete_TB_download_in_Excel_1462/'
            all_files = glob.glob(path + "/*.xlsx")
        else:
            path = os.getcwd() + '/data/recp_Complete_TB_download_in_Excel_1462/'
            all_files = glob.glob(path + "/*.xlsx")
        all_files.sort()

        mlflow.log_metric('File Read Time', time.time() - step_start_time)
        step_start_time = time.time()

        excel_list = []

        mlflow.set_tag('data_type', 'files')
        for num, file in enumerate(all_files, start=0):
            mlflow.set_tag('data_file_' + str(num), file)

        with Pool(processes=n_cpus) as pool:
            all_excel_df = pool.map(read_xls, all_files)
            raw_start_df = pd.concat(all_excel_df, axis=0, ignore_index=True)

        raw_header_lines = pd.read_excel(
            os.getcwd() + '/data/recp_Complete_TB_download_in_Excel_1462/recp_Complete_TB_download_in_Excel_14620.xlsx'
            , header=None
            , sheet_name='Sheet0'
            , nrows=7)

        mlflow.log_metric('Concat Time', time.time() - step_start_time)
        step_start_time = time.time()

        # create headers
        raw_header_lines.replace('empty|HEADER', np.nan, regex=True, inplace=True)
        # n_clmn = range(0,raw_header_lines.shape[1])
        headers = list()
        for col in range(raw_header_lines.shape[1]):
            column_data = raw_header_lines.iloc[:, col]
            headers.append(column_data.str.cat(sep='.'))
        raw_start_df.columns = headers
        start_data = raw_start_df.iloc[8:]
        start_data.reset_index(drop=True, inplace=True)

        # select a subset of the data and simplify headers
        allergen_data = start_data[['productData.gtin',
                                    'basicProductInformationModule.productName',
                                    'basicProductInformationModule.productName/@languageCode',
                                    'productAllergenInformationModule.allergenRelatedInformation.allergen.allergenTypeCode',
                                    'productAllergenInformationModule.allergenRelatedInformation.allergen.levelOfContainmentCode',
                                    'foodAndBeverageIngredientInformationModule.ingredientStatement',
                                    'foodAndBeverageIngredientInformationModule.ingredientStatement/@languageCode']]
        allergen_data.loc[:, 'productData.gtin'].fillna(method='ffill', inplace=True)
        allergen_data.columns = ['gtin', 'productName', 'productName.langCode', 'allergenTypeCode',
                                'levelOfContainmentCode', 'ingredientStatement',
                                'ingredientStatement.langCode']
        allergen_data.loc[allergen_data['levelOfContainmentCode'] == 'FREE_FROM', 'allergenTypeCode'] = np.nan
        allergen_data = allergen_data[~((allergen_data['productName'].isna()) &
                                        (allergen_data['productName.langCode'].isna()) &
                                        (allergen_data['allergenTypeCode'].isna()) &
                                        (allergen_data['ingredientStatement'].isna()) &
                                        (allergen_data['ingredientStatement.langCode'].isna()))]  # remove rows with NaNs

        # create product name columns
        pivot_prodName = allergen_data.pivot(columns='productName.langCode', values='productName')
        pivot_prodName_subset = pivot_prodName[['de', 'en', 'fr', 'nl']]
        pivot_prodName_subset.columns = ['de.prodName', 'en.prodName', 'fr.prodName', 'nl.prodName']

        # create ingredient list columns
        pivot_ingrStat = allergen_data.pivot(columns='ingredientStatement.langCode', values='ingredientStatement')
        pivot_ingrStat_subset = pivot_ingrStat[['de', 'en', 'fr', 'nl']]
        pivot_ingrStat_subset.columns = ['de.ingrStat', 'en.ingrStat', 'fr.ingrStat', 'nl.ingrStat']

        # create allergen dummies
        allergens_dummies = allergen_data['allergenTypeCode'].str.get_dummies()

        mlflow.log_metric('Subset and headers Time', time.time() - step_start_time)
        step_start_time = time.time()

        # merge everything
        data_f = pd.concat([allergen_data['gtin'],
                            pivot_prodName_subset,
                            pivot_ingrStat_subset,
                            allergens_dummies], axis=1)

        mlflow.log_metric('Merge Time', time.time() - step_start_time)
        step_start_time = time.time()

        # fill nan
        data_f['en.prodName'] = data_f.groupby('gtin')['en.prodName'].parallel_apply(lambda x: x.ffill().bfill())
        data_f['nl.prodName'] = data_f.groupby('gtin')['nl.prodName'].parallel_apply(lambda x: x.ffill().bfill())
        data_f['fr.prodName'] = data_f.groupby('gtin')['fr.prodName'].parallel_apply(lambda x: x.ffill().bfill())
        data_f['de.prodName'] = data_f.groupby('gtin')['de.prodName'].parallel_apply(lambda x: x.ffill().bfill())

        data_f['en.ingrStat'] = data_f.groupby('gtin')['en.ingrStat'].parallel_apply(lambda x: x.ffill().bfill())
        data_f['nl.ingrStat'] = data_f.groupby('gtin')['nl.ingrStat'].parallel_apply(lambda x: x.ffill().bfill())
        data_f['fr.ingrStat'] = data_f.groupby('gtin')['fr.ingrStat'].parallel_apply(lambda x: x.ffill().bfill())
        data_f['de.ingrStat'] = data_f.groupby('gtin')['de.ingrStat'].parallel_apply(lambda x: x.ffill().bfill())

        mlflow.log_metric('Nan Time', time.time() - step_start_time)
        step_start_time = time.time()

        # data_f[['en.prodName', 'nl.prodName', 'fr.prodName', 'de.prodName']] = data_f[['en.prodName', 'nl.prodName',
        #     'fr.prodName', 'de.prodName']].fillna('Not Available')
        # data_f[['en.ingrStat', 'nl.ingrStat', 'fr.ingrStat', 'de.ingrStat']] = data_f[['en.ingrStat', 'nl.ingrStat',
        #     'fr.ingrStat', 'de.ingrStat']].fillna('Not Available')

        # aggregate per gtin
        aggregated_per_product = data_f.groupby('gtin')['en.prodName', 'fr.prodName', 'nl.prodName',
                                                        'de.ingrStat', 'en.ingrStat', 'fr.ingrStat', 'nl.ingrStat'].first()
        allergen_indicator_cols = data_f.columns[9:]
        aggregated_allergen_indicators = data_f.groupby('gtin')[allergen_indicator_cols].sum()
        aggregated_allergen_booleans = aggregated_allergen_indicators >= 1
        aggregated_data = pd.concat([aggregated_per_product, aggregated_allergen_booleans['AM']], axis=1)

        mlflow.log_metric('Agg Time', time.time() - step_start_time)
        step_start_time = time.time()

        # Predictive Model -------------------------------------------------------------

        ps = PorterStemmer()

        nltk.download('stopwords')
        stopwords = set(stopwords.words('english'))
        nltk.download('punkt')

        # remove products with no IngrStat
        data = aggregated_data[aggregated_data['en.ingrStat'].notnull()]

        # create tokens & list of ingredients
        listOfIngrLists = []
        for ingrList in data['en.ingrStat']:
            ingrList_wo = re.sub(r"[^a-zA-Z]+", ' ', ingrList).strip()
            lowerIngrList = ingrList_wo.lower()
            lowerIngrList = lowerIngrList.split(' ')
            tmplist = []
            for ingredient in lowerIngrList:
                if ingredient not in stopwords:
                    tmplist.append(ingredient)
            tmplist2 = []
            for ingredient in tmplist:
                tmplist2.append(ps.stem(ingredient))
            unique_ingr = set(tmplist2)
            listOfIngrLists.append(unique_ingr)
        data['en.ingrList'] = listOfIngrLists

        # dummies
        mlb = MultiLabelBinarizer()
        ingrDummies = pd.DataFrame(mlb.fit_transform(data['en.ingrList']), columns=mlb.classes_, index=data.index)

        # filter ingredients that appear only once
        freq_bool = ingrDummies.sum() > 1
        limited_ingrDummies = ingrDummies[freq_bool.index[freq_bool]]
        subset_data = data[['en.prodName', 'en.ingrStat', 'en.ingrList']]
        clean_df = pd.concat([subset_data, limited_ingrDummies], axis=1)

        # Create train and test set
        np.random.seed(14)
        X_train, X_test, y_train, y_test = train_test_split(limited_ingrDummies, data['AM'], test_size=0.2, random_state=42)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        mlflow.log_metric('Prep prediction Time', time.time() - step_start_time)
        step_start_time = time.time()

        mlflow.log_param('n_jobs', n_jobs)
        mlflow.log_param('random_state', random_state)

        step_start_time = time.time()
        # train RF
        clf = RandomForestClassifier(n_jobs=n_jobs, random_state=random_state)
        clf.fit(X_train, y_train)
        mlflow.log_metric('Train Time', time.time() - step_start_time)

        # test performance
        predictions = clf.predict(X_test)
        X_test['model_prediction'] = predictions
        mlflow.log_metric('Perf Time', time.time() - step_start_time)
        step_start_time = time.time()

        # AUC
        fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=predictions, pos_label=1)
        mlflow.log_metric('AUC Time', time.time() - step_start_time)
        step_start_time = time.time()
        mlflow.log_metric('AUC', metrics.auc(fpr, tpr))
        mlflow.sklearn.log_model(clf, 'model')
        mlflow.log_metric('Exec Time', time.time() - start_time)

if __name__ == '__main__':
    run()
