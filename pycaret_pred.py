from pycaret.regression import *
from data import data

cols = data[[40, 160, 0, 45, 80, 75, 85, 120, 50, 90, 70, 20, 60, 30, 180, 190]]
data[['s40', 's160', 's0', 's45', 's80', 's75', 's85', 's120', 's50', 's90', 's70', 's20', 's60', 's30', 's180',
      's190']] = cols
data.drop([40, 160, 0, 45, 80, 75, 85, 120, 50, 90, 70, 20, 60, 30, 180, 190], axis=1, inplace=True)
cols = data[[60]]
data[['s60']] = cols
data.drop([60], axis=1, inplace=True)
setup_data = setup(data=data, target='SalePrice')
best_model = compare_models()

test_df = pd.read_csv('./test.csv', sep=',')
test_df[['OthW', '2.5Fin', 'RRAe2', 'RRAn2', 'RRNn2', 'Metal', 'ClyTile', 'Other2', 'NoSeWa', 'Membran', 'ImStucc', 'Roll', 'Mix', 'Floor', 'TenC']] = 0
prediction_test = predict_model(best_model, test_df.drop(['Id'], axis=1))

prediction_to_save = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': prediction_test['Label']})
prediction_to_save.to_csv('./prediction3.csv', index=False)
tune_model = tune_model(best_model)