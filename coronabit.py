import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
import csv
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


#us_dataset = pd.read_csv('us-data.csv')

# for row in us_dataset.iterrows():

#convert date to epoch
dict_data = []
with open( 'us-data.csv', 'r' ) as File:
    reader = csv.DictReader(File)
    for line in reader:
        tmp = line["date"].split("-")
        dict_data.append({'date': datetime.datetime(int(tmp[0]), int(tmp[1]), int(tmp[2]), 0, 0).strftime('%s')+'000', 'cases': line['cases'], 'deaths': line['deaths']})

# save to file
csv_columns = ['date','cases','deaths']
csv_file = "us-data-epoch.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
except IOError:
    print("I/O error")









#
#
# dataset.shape
#
#
# dataset.describe()
#
#
# dataset.isnull().any()
#
#
# dataset = dataset.fillna(method='ffill')
#
#
# # X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
# # y = dataset['quality'].values
#
# X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']]
# y = dataset['price']
#
#
# plt.figure(figsize=(15,10))
# plt.tight_layout()
# seabornInstance.distplot(dataset['quality'])
# plt.show()
#
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
#
#
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
#
#
# coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
# coeff_df
