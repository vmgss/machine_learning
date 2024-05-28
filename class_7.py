import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import  LogisticRegression
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    data = pd.read_csv("medical_insurance.csv")
    print(data)
    x = data.drop(columns=['smoker'])
    y = data['smoker']
    encoder = LabelEncoder()
    encoder.fit(x['sex'])
    x['sex'] = encoder.transform(x['sex'])
    encoder.fit(x['region'])
    x['region'] = encoder.transform(x['region'])
    lr = LogisticRegression()
    lr.fit(x.values,y.values)
    print(lr.predict(x.iloc[0].values.reshape(1, -1)))
    print(x.values)
