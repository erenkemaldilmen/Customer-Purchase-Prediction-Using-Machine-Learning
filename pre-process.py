import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data():
    # Load the dataset from the specified file name
    data = pd.read_csv("customer_purchase_data.csv")

    # Print the first 5 rows of the dataset
    print("First 5 Rows of the Dataset:")
    print(data.head())

    # Print dataset information
    print("\nDataset Information:")
    print(data.info())

    # Check for missing values
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Check and remove duplicate records
    duplicate_count = data.duplicated().sum()
    print("\nNumber of Duplicate Records:", duplicate_count)
    data.drop_duplicates(inplace=True)

    # Encode categorical variables if 'Gender' column exists
    if 'Gender' in data.columns:
        label_encoder = LabelEncoder()
        data['Gender'] = label_encoder.fit_transform(data['Gender'])

    return data


# Example usage:
if __name__ == "__main__":
    preprocessed_data = load_and_preprocess_data()
    # You can now use 'preprocessed_data' for further analysis or model training
