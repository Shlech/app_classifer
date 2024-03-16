# Импорт библиотек
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Читаем .tsv файлы с данными
test = pd.read_csv('test.tsv', sep='\t')
train = pd.read_csv('train.tsv', sep='\t')
valid = pd.read_csv('val.tsv', sep='\t')

# Удаляем колонку filename из train и valid
train.drop('filename', axis=1, inplace=True)
valid.drop('filename', axis=1, inplace=True)

# Преобразовываем в train, valid и test столбец libs в string, а затем в массив, состоящий из наименований библиотек
train['libs'] = train['libs'].astype('string')
train['libs'] = train['libs'].str.split(',')
valid['libs'] = valid['libs'].astype('string')
valid['libs'] = valid['libs'].str.split(',')
test['libs'] = test['libs'].astype('string')
test['libs'] = test['libs'].str.split(',')

# Преобразоваем столбец libs во множество столиков с наименованиями библиотек
mlb = MultiLabelBinarizer()
encoded_train = pd.DataFrame(mlb.fit_transform(train['libs']), columns=mlb.classes_, index=train.index)
result_train = pd.concat([encoded_train, train['is_virus']], axis=1)
train = result_train

mlb = MultiLabelBinarizer()
encoded_valid = pd.DataFrame(mlb.fit_transform(valid['libs']), columns=mlb.classes_, index=valid.index)
result_valid = pd.concat([encoded_valid, valid['is_virus']], axis=1)
valid = result_valid

mlb = MultiLabelBinarizer()
encoded_test = pd.DataFrame(mlb.fit_transform(test['libs']), columns=mlb.classes_, index=test.index)
test = encoded_test

train_v = train['is_virus']
valid_v = valid['is_virus']

# Находим общие колонки у train и valid
common_columns_tv = np.intersect1d(train.columns, valid.columns)
train = train[common_columns_tv]
valid = valid[common_columns_tv]

# Находим общие колонки у обновленного train и test, переопределяя переменную train, test и valid с новым подмножеством столбцов
common_columns_tt = np.intersect1d(train.columns, test.columns)
train = train[common_columns_tt]
test = test[common_columns_tt]
valid = valid[common_columns_tt]

# Выбираем колоки, у которых сумма больше 1, чтобы избавиться от малоинформативных столбцов
selected_columns = train.loc[:, train.sum() > 1]
selected_columns_headers = list(selected_columns.columns.values)
train_1 = train[selected_columns_headers]
valid_1 = valid[selected_columns_headers]
test_1 = test[selected_columns_headers]

# Объединяем наши итоговые признаки с тагретом
train_1 = pd.concat([train_1, train_v], axis=1)
valid_1 = pd.concat([valid_1, valid_v], axis=1)

# Убираем строки, состоящие только из нулей
train_1 = train_1.loc[(train_1 != 0).any(axis=1)]
valid_1 = valid_1.loc[(valid_1 != 0).any(axis=1)]

X_valid = valid_1.drop(['is_virus'], axis=1)
y_valid = valid_1['is_virus']
X_train = train_1.drop(['is_virus'], axis=1)
y_train = train_1['is_virus']
X_test = test_1

# Загружаем модель
filename = 'decision_tree_model.sav'
model = pickle.load(open(filename, 'rb'))

# Записываем предсказания модели в prediction.txt
prediction = model.predict(X_test)
predictions = pd.Series(prediction, name = 'prediction')
predictions.to_csv('prediction.txt', sep = ' ', index = False)