"""markdown
### Описание данных

**Целевая переменная**
- `selling_price`: цена продажи, числовая

**Признаки**
- `name` (string): модель автомобиля
- `year` (numeric, int): год выпуска с завода-изготовителя
- `km_driven` (numeric, int): пробег на дату продажи
- `fuel` (categorical: _Diesel_ или _Petrol_, или _CNG_, или _LPG_, или _electric_): тип топлива
- `seller_type` (categorical: _Individual_ или _Dealer_, или _Trustmark Dealer_): продавец
- `transmission` (categorical: _Manual_ или _Automatic_): тип трансмиссии
- `owner` (categorical: _First Owner_ или _Second Owner_, или _Third Owner_, или _Fourth & Above Owner_):
 какой по счёту хозяин?
- `mileage` (string, по смыслу числовой): пробег, требует предобработки
- `engine` (string, по смыслу числовой): рабочий объем двигателя, требует предобработки
- `max_power` (string, по смыслу числовой): пиковая мощность двигателя, требует предобработки
- `torque` (string, по смыслу числовой, а то и 2): крутящий момент, требует предобработки
- `seats` (numeric, float; по смыслу categorical, int)
"""

# Import libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import metrics

CUR_YEAR = 2023

# df.head(3)

# Data analyse
# Dataset size and information
# df.shape
# df.info()

# Numeric columns
# df.describe()

# Object columns
# df.describe(include='object')

# df_1 = df[:]
# df_1[['name', 'model']] = df['name'].str.split(' ', n=1, expand=True)
# df_1[['model', 'other']] = df_1['model'].str.split(' ', n=1, expand=True)
# df_1.head(3)
# df_1['seats'].fillna(0.0, inplace=True)
# df_1['seats'] = df_1['seats'].astype('object')
# df_1['mileage'].fillna("-1 unknown", inplace=True)
# df_1['engine'].fillna("-1 unknown", inplace=True)
# df_1['max_power'].fillna("-1 unknown", inplace=True)
# df_1['max_power'] = np.where(df_1['max_power'] == '0', '0 bhp', df_1['max_power'])
# df_1['torque'].fillna("-1 Nm -1 rpm", inplace=True)
# df_1['engine'].fillna("-1 unknown", inplace=True)
#
# df_1['max_power'].fillna("-1 unknown", inplace=True)
# df_1['max_power'] = np.where(df_1['max_power'] == '0', '0 bhp', df_1['max_power'])
#
# df_1['torque'].fillna("-1 Nm -1 rpm", inplace=True)

# df_2 = df_1[:]
#
# df_2['torque'] = np.where(df_2['torque'] == '380Nm(38.7kgm)@ 2500rpm', '380 Nm 2500 rpm', df_2['torque'])
# df_2['torque'] = np.where(df_2['torque'] == '250@ 1250-5000rpm', '250 Nm 5000 rpm	', df_2['torque'])
# df_2['torque'] = np.where(df_2['torque'] == '510@ 1600-2400', '510 Nm 2400 rpm	', df_2['torque'])
# df_2['torque'] = np.where(df_2['torque'] == '48@ 3,000+/-500(NM@ rpm)', '48 Nm 3000 rpm	', df_2['torque'])
# df_2['torque'] = np.where(df_2['torque'] == '210 / 1900', '210 Nm 1900 rpm	', df_2['torque'])
#
# df_2['torque'] = df_2['torque'].str.replace(r"[nN][Mm][@]", " Nm", regex=True)  # заменить nM@ NM@ Nm@ nm@ на Nm
# df_2['torque'] = df_2['torque'].str.replace(r"[nN][mM]", " Nm", regex=True)
# df_2['torque'] = df_2['torque'].str.replace(r",", "", regex=True)  # убрать ,
# df_2['torque'] = df_2['torque'].str.replace(r"at", "", regex=True)  # убрать at
# df_2['torque'] = df_2['torque'].str.replace(r"[0-9]+[-~]", "", regex=True)  # убрать первую половину значения rpm
# df_2['torque'] = df_2['torque'].str.replace("\+/-[0-9]+", " ", regex=True)  # убрать записи +/-500 и им подобные
# df_2['torque'] = df_2['torque'].str.replace(r'\(kgm[@ ]+rpm\)', " krpm ", regex=True)  # заменить скобки на krpm
#
# df_2['torque'] = df_2['torque'].str.replace(" /", " ", regex=True)  # убираем / из текста
# df_2['torque'] = df_2['torque'].str.replace(r"KGM", " kgm ", regex=True)  # приведем к одному виду
# df_2['torque'] = df_2['torque'].str.replace(r"RPM", "rpm", regex=True)  # приведем к одному виду
# df_2['torque'] = df_2['torque'].str.replace(r"kgm@ ", " kgm ", regex=True)  # уберем знак @ и добавим пробел
# df_2['torque'] = df_2['torque'].str.replace(r"@ ", " kgm ", regex=True)  # убираем знак @ из оставшихся записей
# df_2['torque'] = df_2['torque'].str.replace("krpm", " rpm", regex=False)  # приведем rpm к одному виду
# df_2['torque'] = df_2['torque'].str.replace("rpm", " rpm", regex=False)
# df_2['torque'] = df_2['torque'].str.replace("  ", " ", regex=False)  # уберем двойные пробелы
# df_2['torque'] = df_2['torque'].str.replace("  ", " ", regex=False)  # one more time

# df_3 = df_2[:]
# df_3[['mileage', 'mileage_scale']] = df_3['mileage'].str.split(' ', n=1, expand=True)
# df_3[['engine', 'engine_type']] = df_3['engine'].str.split(' ', n=1, expand=True)
# df_3[['max_power', 'power_scale']] = df_3['max_power'].str.split(' ', n=1, expand=True)
# df_3[['torque', 'torque_scale']] = df_3['torque'].str.split(' ', n=1, expand=True)
# df_3[['torque_scale', 'rpm_value']] = df_3['torque_scale'].str.split(' ', n=1, expand=True)

# df_3['rpm_value'].fillna("-1 rpm", inplace=True)
#
# df_3[['rpm_value', 'rpm_scale']] = df_3['rpm_value'].str.split(' ', n=1, expand=True)
# df_3['rpm_scale'].fillna("rpm", inplace=True)
# df_3['rpm_value'] = df_3['rpm_value'].astype('float')

# df_4 = df_3[['selling_price', 'name', 'model', 'other', 'fuel', 'transmission', 'year', 'seats', 'km_driven',
#              'seller_type', 'owner', 'mileage', 'mileage_scale', 'engine', 'engine_type', 'max_power', 'power_scale',
#              'torque', 'torque_scale', 'rpm_value', 'rpm_scale']]
# df_4['max_power'] = np.where(df_4['max_power'] == '', 0, df_4['max_power'])
# df_4['mileage'] = df_4['mileage'].astype('float')
# df_4['engine'] = df_4['engine'].astype('int')
# df_4['max_power'] = df_4['max_power'].astype('float')
# df_4['torque'] = df_4['torque'].astype('float')
#
# df_4['torque'] = np.where((df_4['torque_scale'] == 'kgm'), df_4['torque'] * 9.8, df_4['torque'])
#
# df_4['torque'] = np.where(df_4['torque'] == -1, df_4[df_4['torque'] != -1]['torque'].mean(), df_4['torque'])
#
# df_4['mileage'] = np.where(df_4['mileage'] == -1, df_4[df_4['mileage'] != -1]['mileage'].mean(), df_4['mileage'])
# df_4['engine'] = np.where(df_4['engine'] == -1, df_4[df_4['engine'] != -1]['engine'].mean(), df_4['engine'])
# df_4['max_power'] = np.where(df_4['max_power'] == -1, df_4[df_4['max_power'] != -1]['max_power'].mean(),
#                              df_4['max_power'])
# df_4['rpm_value'] = np.where(df_4['rpm_value'] == -1, df_4[df_4['rpm_value'] != -1]['rpm_value'].mean(),
#                              df_4['rpm_value'])

# for hue in categorial_features:
#   g = sns.PairGrid(df_4[['year', 'engine', 'max_power', 'selling_price', hue]], hue=hue, diag_sharey=False, height=4)
#   g.map_lower(sns.kdeplot, alpha=0.6, warn_singular=False)
#   g.map_upper(plt.scatter, alpha=0.3)
#   g.map_diag(sns.kdeplot, lw=3, alpha=0.6, warn_singular=False,
#              common_norm=False)  # каждая плотность по отдельности должна давать 1 при интегрировании
#   g.add_legend()
# plot data
# plot_data(df_4)


# Get data function
def open_data(path="data/mvp_main_datasets_cars.csv"):
    df_in = pd.read_csv(path)
    return df_in


# Preprocess data
def preprocess_data(df: pd.DataFrame, test=True):
    # split name to brand and model
    df[['name', 'model']] = df['name'].str.split(' ', n=1, expand=True)
    df[['model', 'other']] = df['model'].str.split(' ', n=1, expand=True)
    # fill nulls in data
    df['seats'].fillna(0.0, inplace=True)
    df['mileage'].fillna("-1 unknown", inplace=True)
    df['engine'].fillna("-1 unknown", inplace=True)
    df['max_power'].fillna("-1 unknown", inplace=True)
    df['torque'].fillna("-1 Nm -1 rpm", inplace=True)
    # df['engine'].fillna("-1 unknown", inplace=True)
    # df['torque'].fillna("-1 Nm -1 rpm", inplace=True)
    # df['max_power'].fillna("-1 unknown", inplace=True)
    # df['max_power'] = np.where(df['max_power'] == '0', '0 bhp', df['max_power'])
    # fix data in columns
    # max_power
    df['max_power'] = np.where(df['max_power'] == '0', '0 bhp', df['max_power'])
    # torque
    df['torque'] = np.where(df['torque'] == '380Nm(38.7kgm)@ 2500rpm', '380 Nm 2500 rpm', df['torque'])
    df['torque'] = np.where(df['torque'] == '250@ 1250-5000rpm', '250 Nm 5000 rpm	', df['torque'])
    df['torque'] = np.where(df['torque'] == '510@ 1600-2400', '510 Nm 2400 rpm	', df['torque'])
    df['torque'] = np.where(df['torque'] == '48@ 3,000+/-500(NM@ rpm)', '48 Nm 3000 rpm	', df['torque'])
    df['torque'] = np.where(df['torque'] == '210 / 1900', '210 Nm 1900 rpm	', df['torque'])
    # replace
    df['torque'] = df['torque'].str.replace(r"[nN][Mm][@]", " Nm", regex=True)  # заменить nM@ NM@ Nm@ nm@ на Nm
    df['torque'] = df['torque'].str.replace(r"[nN][mM]", " Nm", regex=True)
    df['torque'] = df['torque'].str.replace(r",", "", regex=True)  # убрать ,
    df['torque'] = df['torque'].str.replace(r"at", "", regex=True)  # убрать at
    df['torque'] = df['torque'].str.replace(r"[0-9]+[-~]", "", regex=True)  # убрать первую половину значения rpm
    df['torque'] = df['torque'].str.replace("\+/-[0-9]+", " ", regex=True)  # убрать записи +/-500 и им подобные
    df['torque'] = df['torque'].str.replace(r'\(kgm[@ ]+rpm\)', " krpm ", regex=True)  # заменить скобки на krpm
    df['torque'] = df['torque'].str.replace(" /", " ", regex=True)  # убираем / из текста
    df['torque'] = df['torque'].str.replace(r"KGM", " kgm ", regex=True)  # приведем к одному виду
    df['torque'] = df['torque'].str.replace(r"RPM", "rpm", regex=True)  # приведем к одному виду
    df['torque'] = df['torque'].str.replace(r"kgm@ ", " kgm ", regex=True)  # уберем знак @ и добавим пробел
    df['torque'] = df['torque'].str.replace(r"@ ", " kgm ", regex=True)  # убираем знак @ из оставшихся записей
    df['torque'] = df['torque'].str.replace("krpm", " rpm", regex=False)  # приведем rpm к одному виду
    df['torque'] = df['torque'].str.replace("rpm", " rpm", regex=False)
    df['torque'] = df['torque'].str.replace("  ", " ", regex=False)  # уберем двойные пробелы
    df['torque'] = df['torque'].str.replace("  ", " ", regex=False)  # one more time
    # split columns
    df[['mileage', 'mileage_scale']] = df['mileage'].str.split(' ', n=1, expand=True)
    df[['engine', 'engine_type']] = df['engine'].str.split(' ', n=1, expand=True)
    df[['max_power', 'power_scale']] = df['max_power'].str.split(' ', n=1, expand=True)
    df[['torque', 'torque_scale']] = df['torque'].str.split(' ', n=1, expand=True)
    df[['torque_scale', 'rpm_value']] = df['torque_scale'].str.split(' ', n=1, expand=True)
    # fill null in new columns
    df['rpm_value'].fillna("-1 rpm", inplace=True)
    # split
    df[['rpm_value', 'rpm_scale']] = df['rpm_value'].str.split(' ', n=1, expand=True)
    # fill null
    df['rpm_scale'].fillna("rpm", inplace=True)
    df['max_power'] = np.where(df['max_power'] == '', 0, df['max_power'])
    # df = df[['selling_price', 'name', 'model', 'other', 'fuel', 'transmission', 'year', 'seats', 'km_driven',
    #          'seller_type', 'owner', 'mileage', 'mileage_scale', 'engine', 'engine_type', 'max_power', 'power_scale',
    #          'torque', 'torque_scale', 'rpm_value', 'rpm_scale']]

    # change columns types
    df['mileage'] = df['mileage'].astype('float')
    df['engine'] = df['engine'].astype('int')
    df['max_power'] = df['max_power'].astype('float')
    df['torque'] = df['torque'].astype('float')
    df['rpm_value'] = df['rpm_value'].astype('float')
    df['seats'] = df['seats'].astype('object')
    # set torque to Nm
    df['torque'] = np.where((df['torque_scale'] == 'kgm'), df['torque'] * 9.8, df['torque'])
    # fill gaps with means
    df['torque'] = np.where(df['torque'] == -1, df[df['torque'] != -1]['torque'].mean(), df['torque'])
    df['mileage'] = np.where(df['mileage'] == -1, df[df['mileage'] != -1]['mileage'].mean(), df['mileage'])
    df['engine'] = np.where(df['engine'] == -1, df[df['engine'] != -1]['engine'].mean(), df['engine'])
    df['max_power'] = np.where(df['max_power'] == -1, df[df['max_power'] != -1]['max_power'].mean(),
                               df['max_power'])
    df['rpm_value'] = np.where(df['rpm_value'] == -1, df[df['rpm_value'] != -1]['rpm_value'].mean(),
                               df['rpm_value'])
    return df


# Open data
df_1 = open_data()

# Fix and rebuild data
df_4 = preprocess_data(df_1)


# plot some graf
def plot_data(df_in: pd.DataFrame):
    sns.histplot(df_in['selling_price'], kde=True)
    plt.show()

    sns.barplot(x='engine', y='selling_price', data = df_in, palette='summer', )
    plt.title('engine - selling_price')
    plt.xticks(rotation='vertical')
    plt.show()

    corr = df_in[['selling_price', 'year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'rpm_value']].corr()
    sns.heatmap(corr, cmap="crest")
    sns.heatmap(corr, annot=True)

    return df_in


# Split data
def split_data(df_in: pd.DataFrame, model_list, target_name):
    y = df_in[target_name]
    X = df_in[model_list]

    return X, y


def preprocess_data(df: pd.DataFrame, model_list, target_name, to_encode_list, test=True):
    df.dropna(inplace=True)

    if test:
        X_df, y_df = split_data(df, model_list, target_name)
    else:
        X_df = df

    for col in to_encode_list:
        dummy = pd.get_dummies(X_df[col], prefix=col)
        X_df = pd.concat([X_df, dummy], axis=1)
        X_df.drop(col, axis=1, inplace=True)

    if test:
        return X_df, y_df
    else:
        return X_df


# Create data set for model
# full list of columns
# ['selling_price', 'name', 'model', 'other', 'fuel', 'transmission', 'year', 'seats', 'km_driven',
#  'seller_type', 'owner', 'mileage', 'mileage_scale', 'engine', 'engine_type', 'max_power', 'power_scale',
#  'torque', 'torque_scale', 'rpm_value', 'rpm_scale']

# lists for model
target_feature = 'selling_price'  # target model
# lists of columns
# included in model
model_list = ['name', 'fuel', 'transmission', 'year', 'seats', 'km_driven',
              'seller_type', 'engine', 'engine_type', 'max_power', 'torque']
# not included
drop_list = ['selling_price', 'model', 'other', 'owner', 'mileage', 'mileage_scale',
             'power_scale', 'torque_scale', 'rpm_value', 'rpm_scale']
#
categorical = ['name', 'fuel', 'seats', 'seller_type']
real_features = ['year', 'transmission', 'max_power', 'engine', 'engine_type', 'torque', 'km_driven']

# split dataframe
X_data = df_4.drop(drop_list, axis=1)  # data for model
y_data = df_4[target_feature]  # target value

#
X_data['transmission'] = X_data['transmission'].map({'Automatic': 1, 'Manual': 0})
X_data['engine_type'] = X_data['engine_type'].map({'CC': 1, 'unknown': 0})


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.15, random_state=42)
X_train.shape, X_test.shape

# categorical = categorial_features
numeric_features = [col for col in X_train.columns if col not in categorical]

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first', handle_unknown="ignore", sparse_output=False), categorical),
    ('scaling', MinMaxScaler(), numeric_features)
])

X_train_transformed = column_transformer.fit_transform(X_train)
X_test_transformed = column_transformer.transform(X_test)

lst = list(column_transformer.transformers_[0][1].get_feature_names_out())
lst.extend(numeric_features)

X_train_transformed = pd.DataFrame(X_train_transformed, columns=lst)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=lst)

# X_train_transformed.head(3)

model = LinearRegression(fit_intercept=True)
model.fit(X_train_transformed, y_train)

model.score(X_test_transformed, y_test).round(3)

test_preds = model.predict(X_test_transformed)
train_preds = model.predict(X_train_transformed)

print(metrics.mean_squared_error(y_test, test_preds) ** 0.5)


def mean_absolute_percentage_error(y_true, y_pred):
    return 100 * (np.abs(y_true - y_pred) / y_true).mean()


print(mean_absolute_percentage_error(y_test, test_preds))

print(metrics.mean_squared_error(y_train, train_preds) ** 0.5,
      metrics.mean_absolute_error(y_train, train_preds),
      mean_absolute_percentage_error(y_train, train_preds))
