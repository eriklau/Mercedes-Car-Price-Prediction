import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np

class DataCleaner:
    def __init__(self):
        source = "C:\\Users\\Larkin\\Desktop\\python\\prediction\\merc.csv"
        pd.options.display.max_rows = 10
        pd.options.display.float_format = "{:.2f}".format

        self.data = pd.read_csv(source)

    def _remove_unwanted_features(self):
        for i in self.data:
            if i == 'mpg' or i == 'price':
                continue
            else:
                del self.data[i]

    def encode_region(self):
        # reoptimize numpy array has too many items for onehot encoding
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        # print(self.data['region'])
        X = self.data.iloc[:, :].values
        print(X)
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
        X = np.array(ct.fit_transform(X))
        return X.size

    def _remove_extremes(self):
        for index, row in self.data.iterrows():
            if row['mpg'] > 80.0 or row['price'] > 80000.0:
                self.data = self.data.drop(index=index)
                self.data.reset_index(drop=True)

    def process_data(self):
        self._remove_unwanted_features()
        self._remove_extremes()
        return self.data

    def plot_graph(self):
        # Label the axes.
        plt.xlabel("Miles Per Gallon (MPG)")
        plt.ylabel("Price in Euros")
        random_examples = self.data.sample(n=100)
        plt.scatter(random_examples['mpg'],
                    random_examples['price'])
        plt.show()


a = DataCleaner()
a.process_data()
# print(a.data)
a.plot_graph()
# print(a.encode_region())