import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler

from src import utils
from src.config import *


processed_data_path = 'data/processed/'
target_col = 'SalePrice'


def preprocess(input_train_path, input_test_path, output_dir: str = processed_data_path) -> None:
    
    train = pd.read_csv(os.path.join('../data/raw/train.csv'), index_col='Id')
    test = pd.read_csv(os.path.join('../data/raw/test.csv'), index_col='Id')
    
    train_target = train[target_col]
    train_data = train.drop(columns=target_col)
    
    All_data = pd.concat([train_data, test], axis=0, sort=True)
    
    num_All_data = All_data.select_dtypes(include = ['int64', 'float64'])
    
    cat_col = ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']
    All_data.loc[:, cat_col] = All_data.loc[:, cat_col].astype('object')
    
    missing_columns = All_data.columns[All_data.isnull().any()].values
    missing_columns = len(All_data) - All_data.loc[:, np.sum(All_data.isnull())>0].count()

    to_impute_by_none = All_data.loc[:, NAN_HAS_SENSE]
    for i in to_impute_by_none.columns:
        All_data[i].fillna('None', inplace = True)
        

    to_impute_by_mode =  All_data.loc[:, CAT_COL]
    for i in to_impute_by_mode.columns:
        All_data[i].fillna(All_data[i].mode()[0], inplace = True)
        

    to_impute_by_median = All_data.loc[:, NUM_COL]
    for i in to_impute_by_median.columns:
        All_data[i].fillna(All_data[i].median(), inplace = True)


    le = LabelEncoder()

    df = All_data.reset_index().drop(columns=['Id','LotFrontage'], axis=1)
    df = df.apply(le.fit_transform) 

    df['LotFrontage'] = All_data['LotFrontage']
    df = df.set_index('LotFrontage').reset_index()
    

    All_data['LotFrontage'] = All_data.groupby(['BldgType'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    
    train_target = np.log1p(train_target)
    
    All_data_num = All_data.select_dtypes(include = ['int64', 'float64'])
    All_data_skewed = np.log1p(All_data_num[All_data_num.skew()[All_data_num.skew() > 0.5].index])

    All_data_normal = All_data_num[All_data_num.skew()[All_data_num.skew() < 0.5].index]

    All_data_num_all = pd.concat([All_data_skewed, All_data_normal], axis = 1)

    All_data_num.update(All_data_num_all)

    scaler = RobustScaler()
    All_data_num_scaled = scaler.fit_transform(All_data_num)
    All_data_num_scaled = pd.DataFrame(data = All_data_num_scaled, columns = All_data_num.columns, index = All_data_num.index)
    
    All_data_cat = All_data.select_dtypes(include = ['object']).astype('category')

    All_data_cat.LotShape.replace(to_replace = ['IR3', 'IR2', 'IR1', 'Reg'], value = [0, 1, 2, 3], inplace = True)
    All_data_cat.LandContour.replace(to_replace = ['Low', 'Bnk', 'HLS', 'Lvl'], value = [0, 1, 2, 3], inplace = True)
    All_data_cat.Utilities.replace(to_replace = ['NoSeWa', 'AllPub'], value = [0, 1], inplace = True)
    All_data_cat.LandSlope.replace(to_replace = ['Sev', 'Mod', 'Gtl'], value = [0, 1, 2], inplace = True)
    All_data_cat.ExterQual.replace(to_replace = ['Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)
    All_data_cat.ExterCond.replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
    All_data_cat.BsmtQual.replace(to_replace = ['None', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
    All_data_cat.BsmtCond.replace(to_replace = ['None', 'Po', 'Fa', 'TA', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)
    All_data_cat.BsmtExposure.replace(to_replace = ['None', 'No', 'Mn', 'Av', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)
    All_data_cat.BsmtFinType1.replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
    All_data_cat.BsmtFinType2.replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
    All_data_cat.HeatingQC.replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
    All_data_cat.Electrical.replace(to_replace = ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'], value = [0, 1, 2, 3, 4], inplace = True)
    All_data_cat.KitchenQual.replace(to_replace = ['Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)
    All_data_cat.Functional.replace(to_replace = ['Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
    All_data_cat.FireplaceQu.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
    All_data_cat.GarageFinish.replace(to_replace =  ['None', 'Unf', 'RFn', 'Fin'], value = [0, 1, 2, 3], inplace = True)
    All_data_cat.GarageQual.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
    All_data_cat.GarageCond.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
    All_data_cat.PavedDrive.replace(to_replace =  ['N', 'P', 'Y'], value = [0, 1, 2], inplace = True)
    All_data_cat.PoolQC.replace(to_replace =  ['None', 'Fa', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)
    All_data_cat.Fence.replace(to_replace =  ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'], value = [0, 1, 2, 3, 4], inplace = True)
    
    All_data_cat.loc[:, ['OverallQual', 'OverallCond']] = All_data_cat.loc[:, ['OverallQual', 'OverallCond']].astype('int64')
    All_data_label_encoded = All_data_cat.select_dtypes(include = ['int64'])
    
    All_data_one_hot = All_data_cat.select_dtypes(include=['category'])
    All_data_one_hot = pd.get_dummies(All_data_one_hot, drop_first=True)
    
    All_data_encoded = pd.concat([All_data_one_hot, All_data_label_encoded], axis=1)
    All_data_processed = pd.concat([All_data_num_scaled, All_data_encoded], axis=1)
    
    train_final = All_data_processed.iloc[:train.shape[0], :]
    test_final = All_data_processed.iloc[train.shape[0]:, :]
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
    utils.save_as_pickle(train_final, os.path.join(output_dir, 'train.pkl'))
    utils.save_as_pickle(test_final, os.path.join(output_dir, 'test.pkl'))
    utils.save_as_pickle(train_target, os.path.join(output_dir, 'train_target.pkl'))