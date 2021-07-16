import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data = pd.read_csv('./train.csv', sep=',')
# data.drop(['Id'], axis=1, inplace=True)
data = pd.read_csv('./test.csv', sep=',')

describe = data.describe()

unique_values = {}
for col in data.columns:
    unique_values[col] = pd.unique(data[col])

dummy_MSZoning = pd.get_dummies(data['MSZoning'])
data[dummy_MSZoning.columns] = dummy_MSZoning
corr = data.corr()
data.drop(['MSZoning'], inplace=True, axis=1)

dummy_MSSubClass = pd.get_dummies(data['MSSubClass'])
data[dummy_MSSubClass.columns] = dummy_MSSubClass
corr = data.corr()
data.drop(['MSSubClass'], inplace=True, axis=1)

data.drop(['LotFrontage'], axis=1, inplace=True)  # too many nan values

dummy_Street = pd.get_dummies(data['Street'])
data[dummy_Street.columns] = dummy_Street
corr = data.corr()
data.drop(dummy_Street.columns, inplace=True, axis=1)
data.loc[(data['Street'] == 'Grvl'), 'Street'] = 0
data.loc[(data['Street'] == 'Pave'), 'Street'] = 1
data['Street'] = data['Street'].astype(int)
describe = data.describe()  # Street 99% Pave
# plt.scatter(data['Street'], data['SalePrice'])
data.drop(['Street'], axis=1, inplace=True)

plt.clf()

data['Alley'].isnull().sum()
data.drop(['Alley'], axis=1, inplace=True)  # too many nan values

dummy_LotShape = pd.get_dummies(data['LotShape'])
data[dummy_LotShape.columns] = dummy_LotShape
corr = data.corr()
data.drop(['LotShape'], inplace=True, axis=1)

dummy_LandContour = pd.get_dummies(data['LandContour'])
data[dummy_LandContour.columns] = dummy_LotShape
corr = data.corr()
describe = data.describe()
data.drop(['LandContour'], inplace=True, axis=1)

dummy = pd.get_dummies(data['Utilities'])
data[dummy.columns] = dummy
corr = data.corr()
describe = data.describe()  # Utilities 99% AllPub

# plt.scatter(data['Utilities'], data['SalePrice'])
data.drop(['Utilities'], axis=1, inplace=True)  # one point NoSeWa
plt.clf()

dummy = pd.get_dummies(data['LotConfig'])
data[dummy.columns] = dummy
corr = data.corr()
describe = data.describe()
data.drop(['LotConfig'], inplace=True, axis=1)

data.loc[(data['LandSlope'] == 'Gtl'), 'LandSlope'] = 0
data.loc[(data['LandSlope'] == 'Mod'), 'LandSlope'] = 1
data.loc[(data['LandSlope'] == 'Sev'), 'LandSlope'] = 2
data['LandSlope'] = data['LandSlope'].astype(int)
dummy = pd.get_dummies(data['LandSlope'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['LandSlope'] + list(dummy.columns), axis=1, inplace=True)  # colinearity 5%

corr = data.corr()
describe = data.describe()
# data.drop(['LotConfig'], inplace=True, axis=1)

dummy = pd.get_dummies(data['Neighborhood'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['Neighborhood'], axis=1, inplace=True)

dummy = pd.get_dummies(data['Condition1'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['Condition1'], axis=1, inplace=True)

dummy = pd.get_dummies(data['Condition2'])
describe = dummy.describe()
columns = []
for col in dummy.columns:
    columns.append(col + '2')
data[columns] = dummy
corr = data.corr()
data.drop(['Condition2'], axis=1, inplace=True)

dummy = pd.get_dummies(data['BldgType'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['BldgType'], axis=1, inplace=True)

dummy = pd.get_dummies(data['HouseStyle'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['HouseStyle'], axis=1, inplace=True)

dummy = pd.get_dummies(data['RoofStyle'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['RoofStyle'], axis=1, inplace=True)

dummy = pd.get_dummies(data['RoofMatl'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['RoofMatl'], axis=1, inplace=True)

dummy = pd.get_dummies(data['Exterior1st'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['Exterior1st'], axis=1, inplace=True)

dummy = pd.get_dummies(data['Exterior2nd'])
describe = dummy.describe()
columns = []
for col in dummy.columns:
    columns.append(col + '2')
data[columns] = dummy
corr = data.corr()
data.drop(['Exterior2nd'], axis=1, inplace=True)

dummy = pd.get_dummies(data['MasVnrType'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['MasVnrType'], axis=1, inplace=True)

data.loc[(data['ExterQual'] == 'Ex'), 'ExterQual'] = 5
data.loc[(data['ExterQual'] == 'Gd'), 'ExterQual'] = 4
data.loc[(data['ExterQual'] == 'TA'), 'ExterQual'] = 3
data.loc[(data['ExterQual'] == 'Fa'), 'ExterQual'] = 2
data.loc[(data['ExterQual'] == 'Po'), 'ExterQual'] = 1
data['ExterQual'] = data['ExterQual'].astype(int)
corr = data.corr()

data.loc[(data['ExterCond'] == 'Ex'), 'ExterCond'] = 5
data.loc[(data['ExterCond'] == 'Gd'), 'ExterCond'] = 4
data.loc[(data['ExterCond'] == 'TA'), 'ExterCond'] = 3
data.loc[(data['ExterCond'] == 'Fa'), 'ExterCond'] = 2
data.loc[(data['ExterCond'] == 'Po'), 'ExterCond'] = 1
data['ExterCond'] = data['ExterCond'].astype(int)
corr = data.corr()

dummy = pd.get_dummies(data['Foundation'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['Foundation'], axis=1, inplace=True)

data.loc[(data['BsmtQual'] == 'Ex'), 'BsmtQual'] = 5
data.loc[(data['BsmtQual'] == 'Gd'), 'BsmtQual'] = 4
data.loc[(data['BsmtQual'] == 'TA'), 'BsmtQual'] = 3
data.loc[(data['BsmtQual'] == 'Fa'), 'BsmtQual'] = 2
data.loc[(data['BsmtQual'] == 'Po'), 'BsmtQual'] = 1
data.loc[(data['BsmtQual'] == 'NA'), 'BsmtQual'] = 0
data.fillna(0, inplace=True)
data['BsmtQual'] = data['BsmtQual'].astype(int)
corr = data.corr()

data.loc[(data['BsmtCond'] == 'Ex'), 'BsmtCond'] = 5
data.loc[(data['BsmtCond'] == 'Gd'), 'BsmtCond'] = 4
data.loc[(data['BsmtCond'] == 'TA'), 'BsmtCond'] = 3
data.loc[(data['BsmtCond'] == 'Fa'), 'BsmtCond'] = 2
data.loc[(data['BsmtCond'] == 'Po'), 'BsmtCond'] = 1
data.loc[(data['BsmtCond'] == 'NA'), 'BsmtCond'] = 0
data['BsmtCond'] = data['BsmtCond'].astype(int)
corr = data.corr()

data.loc[(data['BsmtExposure'] == 'Gd'), 'BsmtExposure'] = 3
data.loc[(data['BsmtExposure'] == 'Av'), 'BsmtExposure'] = 2
data.loc[(data['BsmtExposure'] == 'Mn'), 'BsmtExposure'] = 1
data.loc[(data['BsmtExposure'] == 'NA'), 'BsmtExposure'] = 0
data.loc[(data['BsmtExposure'] == 'No'), 'BsmtExposure'] = 0
data['BsmtExposure'] = data['BsmtExposure'].astype(int)
corr = data.corr()

data.loc[(data['BsmtFinType1'] == 'GLQ'), 'BsmtFinType1'] = 5
data.loc[(data['BsmtFinType1'] == 'ALQ'), 'BsmtFinType1'] = 4
data.loc[(data['BsmtFinType1'] == 'BLQ'), 'BsmtFinType1'] = 3
data.loc[(data['BsmtFinType1'] == 'Rec'), 'BsmtFinType1'] = 2
data.loc[(data['BsmtFinType1'] == 'LwQ'), 'BsmtFinType1'] = 1
data.loc[(data['BsmtFinType1'] == 'Unf'), 'BsmtFinType1'] = 0
data.loc[(data['BsmtFinType1'] == 'NA'), 'BsmtFinType1'] = 0
data['BsmtFinType1'] = data['BsmtFinType1'].astype(int)
corr = data.corr()

data.loc[(data['BsmtFinType2'] == 'GLQ'), 'BsmtFinType2'] = 5
data.loc[(data['BsmtFinType2'] == 'ALQ'), 'BsmtFinType2'] = 4
data.loc[(data['BsmtFinType2'] == 'BLQ'), 'BsmtFinType2'] = 3
data.loc[(data['BsmtFinType2'] == 'Rec'), 'BsmtFinType2'] = 2
data.loc[(data['BsmtFinType2'] == 'LwQ'), 'BsmtFinType2'] = 1
data.loc[(data['BsmtFinType2'] == 'Unf'), 'BsmtFinType2'] = 0
data.loc[(data['BsmtFinType2'] == 'NA'), 'BsmtFinType2'] = 0
data['BsmtFinType2'] = data['BsmtFinType2'].astype(int)
corr = data.corr()

dummy = pd.get_dummies(data['Heating'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['Heating'], axis=1, inplace=True)

data.loc[(data['HeatingQC'] == 'Ex'), 'HeatingQC'] = 5
data.loc[(data['HeatingQC'] == 'Gd'), 'HeatingQC'] = 4
data.loc[(data['HeatingQC'] == 'TA'), 'HeatingQC'] = 3
data.loc[(data['HeatingQC'] == 'Fa'), 'HeatingQC'] = 2
data.loc[(data['HeatingQC'] == 'Po'), 'HeatingQC'] = 1
data['HeatingQC'] = data['HeatingQC'].astype(int)
corr = data.corr()

data.loc[(data['CentralAir'] == 'N'), 'CentralAir'] = 0
data.loc[(data['CentralAir'] == 'Y'), 'CentralAir'] = 1
data['CentralAir'] = data['CentralAir'].astype(int)
corr = data.corr()

dummy = pd.get_dummies(data['Electrical'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['Electrical'], axis=1, inplace=True)

data.loc[(data['KitchenQual'] == 'Ex'), 'KitchenQual'] = 5
data.loc[(data['KitchenQual'] == 'Gd'), 'KitchenQual'] = 4
data.loc[(data['KitchenQual'] == 'TA'), 'KitchenQual'] = 3
data.loc[(data['KitchenQual'] == 'Fa'), 'KitchenQual'] = 2
data.loc[(data['KitchenQual'] == 'Po'), 'KitchenQual'] = 1
data['KitchenQual'] = data['KitchenQual'].astype(int)
corr = data.corr()

dummy = pd.get_dummies(data['Functional'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['Functional'], axis=1, inplace=True)

data.loc[(data['FireplaceQu'] == 'Ex'), 'FireplaceQu'] = 5
data.loc[(data['FireplaceQu'] == 'Gd'), 'FireplaceQu'] = 4
data.loc[(data['FireplaceQu'] == 'TA'), 'FireplaceQu'] = 3
data.loc[(data['FireplaceQu'] == 'Fa'), 'FireplaceQu'] = 2
data.loc[(data['FireplaceQu'] == 'Po'), 'FireplaceQu'] = 1
data.loc[(data['FireplaceQu'] == 'NA'), 'FireplaceQu'] = 0
data['FireplaceQu'] = data['FireplaceQu'].astype(int)
corr = data.corr()

dummy = pd.get_dummies(data['GarageType'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['GarageType'], axis=1, inplace=True)

data.loc[(data['GarageFinish'] == 'Fin'), 'GarageFinish'] = 3
data.loc[(data['GarageFinish'] == 'RFn'), 'GarageFinish'] = 2
data.loc[(data['GarageFinish'] == 'Unf'), 'GarageFinish'] = 1
data.loc[(data['GarageFinish'] == 'NA'), 'GarageFinish'] = 0
data['GarageFinish'] = data['GarageFinish'].astype(int)
corr = data.corr()

data.loc[(data['GarageQual'] == 'Ex'), 'GarageQual'] = 5
data.loc[(data['GarageQual'] == 'Gd'), 'GarageQual'] = 4
data.loc[(data['GarageQual'] == 'TA'), 'GarageQual'] = 3
data.loc[(data['GarageQual'] == 'Fa'), 'GarageQual'] = 2
data.loc[(data['GarageQual'] == 'Po'), 'GarageQual'] = 1
data.loc[(data['GarageQual'] == 'NA'), 'GarageQual'] = 0
data['GarageQual'] = data['GarageQual'].astype(int)
corr = data.corr()

data.loc[(data['GarageCond'] == 'Ex'), 'GarageCond'] = 5
data.loc[(data['GarageCond'] == 'Gd'), 'GarageCond'] = 4
data.loc[(data['GarageCond'] == 'TA'), 'GarageCond'] = 3
data.loc[(data['GarageCond'] == 'Fa'), 'GarageCond'] = 2
data.loc[(data['GarageCond'] == 'Po'), 'GarageCond'] = 1
data.loc[(data['GarageCond'] == 'NA'), 'GarageCond'] = 0
data['GarageCond'] = data['GarageCond'].astype(int)
corr = data.corr()

dummy = pd.get_dummies(data['PavedDrive'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['PavedDrive'], axis=1, inplace=True)

data.loc[(data['PoolQC'] == 'Ex'), 'PoolQC'] = 4
data.loc[(data['PoolQC'] == 'Gd'), 'PoolQC'] = 3
data.loc[(data['PoolQC'] == 'TA'), 'PoolQC'] = 2
data.loc[(data['PoolQC'] == 'Fa'), 'PoolQC'] = 1
data.loc[(data['PoolQC'] == 'NA'), 'PoolQC'] = 0
data['PoolQC'] = data['PoolQC'].astype(int)
corr = data.corr()

dummy = pd.get_dummies(data['Fence'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['Fence'], axis=1, inplace=True)

dummy = pd.get_dummies(data['MiscFeature'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['MiscFeature'], axis=1, inplace=True)

dummy = pd.get_dummies(data['SaleType'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['SaleType'], axis=1, inplace=True)

dummy = pd.get_dummies(data['SaleCondition'])
describe = dummy.describe()
data[dummy.columns] = dummy
corr = data.corr()
data.drop(['SaleCondition'], axis=1, inplace=True)
for col in data.columns:
    print(col)
    data[col] = data[col].astype(int)

corr = data.corr()

corr_saleprice = corr['SalePrice']
corr_saleprice = corr_saleprice.sort_values(ascending=False)

cols_with_good_corr = []
for col in corr_saleprice.index:
    if corr_saleprice[col] > 0.3:
        cols_with_good_corr.append(col)

data = data[cols_with_good_corr+['Id']]
