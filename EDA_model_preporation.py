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
from pickle import dump, load

CUR_YEAR = 2023
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL = True


# Get data function
def open_data(path="data/mvp_main_datasets_cars.csv"):
    df_in = pd.read_csv(path)
    return df_in


# plot some graf
def plot_data(df_in: pd.DataFrame):
    sns.histplot(df_in['selling_price'], kde=True)
    plt.show()

    sns.barplot(x='engine', y='selling_price', data=df_in, palette='summer', )
    plt.title('engine - selling_price')
    plt.xticks(rotation='vertical')
    plt.show()

    corr = df_in[['selling_price', 'year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'rpm_value']].corr()
    sns.heatmap(corr, cmap="crest")
    sns.heatmap(corr, annot=True)

    return df_in


# Split data
def split_data(df_in: pd.DataFrame, model_list, target_name):
    y_data_in = df_in[target_name]
    x_data_in = df_in[model_list]

    return x_data_in, y_data_in


# Preprocess data
def preprocess_data(df: pd.DataFrame, model_list, target_name, test=True):
    # split name to brand and model
    df[['name', 'model']] = df['name'].str.split(' ', n=1, expand=True)
    df[['model', 'other']] = df['model'].str.split(' ', n=1, expand=True)
    # fill nulls in data
    df['seats'].fillna(0.0, inplace=True)
    df['mileage'].fillna("-1 unknown", inplace=True)
    df['engine'].fillna("-1 unknown", inplace=True)
    df['max_power'].fillna("-1 unknown", inplace=True)
    df['torque'].fillna("-1 Nm -1 rpm", inplace=True)
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

    if test:
        x_df, y_df = split_data(df, model_list, target_name)
    else:
        x_df = df

    if test:
        return x_df, y_df
    else:
        return x_df


def transform_xy(x_data_in, y_data_in, categorical_columns):
    # Transform categorical columns to boolean
    x_data_in.loc[:, 'transmission'] = x_data_in['transmission'].map({'Automatic': 1, 'Manual': 0})
    x_data_in.loc[:, 'engine_type'] = x_data_in['engine_type'].map({'CC': 1, 'unknown': 0})
    # split dataset on test and train
    x_train, x_test, y_train, y_test = train_test_split(x_data_in, y_data_in, test_size=0.05, random_state=42)
    x_train.shape, x_test.shape

    # encode categorical columns that isn't boolean
    numeric_features = [col for col in x_data_in.columns if col not in categorical_columns]
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(drop='first', handle_unknown="ignore", sparse_output=False), categorical_columns),
        ('scaling', MinMaxScaler(), numeric_features)
    ])

    x_train_transformed = column_transformer.fit_transform(x_train)
    x_test_transformed = column_transformer.transform(x_test)

    lst = list(column_transformer.transformers_[0][1].get_feature_names_out())
    lst.extend(numeric_features)

    x_train_transformed_listed = pd.DataFrame(x_train_transformed, columns=lst)
    x_test_transformed_listed = pd.DataFrame(x_test_transformed, columns=lst)

    return x_train_transformed_listed, x_test_transformed_listed, y_train, y_test


def transform(x_data_in):
    # split dataset on test and train
    # x_train, x_test, y_train, y_test = train_test_split(x_data_in, y_data_in, test_size=0.15, random_state=42)
    # x_train.shape, x_test.shape
    # Transform categorical columns to boolean
    x_data_in.loc[:, 'transmission'] = x_data_in['transmission'].map({'Automatic': 1, 'Manual': 0})
    x_data_in.loc[:, 'engine_type'] = x_data_in['engine_type'].map({'CC': 1, 'unknown': 0})
    to_encode = categorical
    # encode categorical columns that isn't boolean
    numeric_features = [col for col in x_data_in.columns if col not in to_encode]
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(drop='first', handle_unknown="ignore", sparse_output=False), to_encode),
        ('scaling', MinMaxScaler(), numeric_features)
    ])

    x_train_transformed = column_transformer.fit_transform(x_data_in)
    #
    lst = list(column_transformer.transformers_[0][1].get_feature_names_out())
    lst.extend(numeric_features)
    #
    df = pd.DataFrame(x_train_transformed, columns=lst)

    return df


# fit model and save it in file
def fit_and_save_model(x_train_in, x_test_in, y_train_in, y_test_in, model_type, path="data/model_weights.mw"):
    if model_type == 'Linear':
        model = LinearRegression(fit_intercept=True)
        model.fit(x_train_in, y_train_in)

    else:
        print('Wrong solver name')

    # Main model metric R^2
    score = model.score(x_test_in, y_test_in).round(3)
    print(f"Model score is {score}")

    test_preds = model.predict(x_test_in)
    # train_preds = model.predict(X_train)

    # ********************    test how it works     ***************
    prediction = model.predict(x_test_in)[0]
    print(x_test_in.head(1))
    print("Selling_price", prediction)

    # ************************* end block ***************************

    mean_error = metrics.mean_squared_error(y_test_in, test_preds) ** 0.5

    print(f"Mean squared error regression loss: {mean_error}")
    print(f"Mean absolute percentage error: {mean_absolute_percentage_error(y_test_in, test_preds)}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")

    return mean_error


def load_model_and_predict(df, selling_price, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # recommendation = selling_price
    # low_price
    if (prediction - mean_error) < 0:
        low_price = selling_price / 2
    else:
        low_price = (prediction - mean_error)
    # hi_price
    if (prediction + mean_error) < low_price:
        hi_price = low_price
    else:
        hi_price = (prediction + mean_error)
    # recommendation
    if hi_price < selling_price:
        recommendation = "too high"
    elif low_price > selling_price:
        recommendation = "too low"
    else:
        recommendation = "good"

    return prediction, recommendation, hi_price, low_price


def mean_absolute_percentage_error(y_true, y_pred):
    return 100 * (np.abs(y_true - y_pred) / y_true).mean()


# Open data
df_1 = open_data()

# Settings
# lists for model
target_feature = 'selling_price'  # target model
# lists of columns
# included in model
to_model_list = ['name', 'fuel', 'transmission', 'year', 'seats', 'km_driven',
                 'seller_type', 'engine', 'engine_type', 'max_power', 'torque']
# not included
to_drop_list = ['selling_price', 'model', 'other', 'owner', 'mileage', 'mileage_scale',
                'power_scale', 'torque_scale', 'rpm_value', 'rpm_scale']
#
categorical = ['name', 'fuel', 'seats', 'seller_type']
real_features = ['year', 'transmission', 'max_power', 'engine', 'engine_type', 'torque', 'km_driven']

x_data, y_data = preprocess_data(df_1, to_model_list, target_feature,)

# Fix and rebuild data and then Create data set for model
X_train, X_test, y_train, y_test = transform_xy(x_data, y_data,  categorical)

# initialize and save model
mean_error = fit_and_save_model(X_train, X_test, y_train, y_test, model_type='Linear')

# prediction, prediction_probas = load_model_and_predict()
