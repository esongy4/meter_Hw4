import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessing:

    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = StandardScaler()

    def load_data(self):
        df = pd.read_csv(self.filepath, sep="\t", header=None)
        df.columns = [f"x{i}" for i in range(df.shape[1])]
        return df

    def split_data(self, df):
        X = df.drop(columns=["x36"])
        y = df["x36"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def scale_data(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled