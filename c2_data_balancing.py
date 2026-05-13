import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder

def balance_dataset(dataset: pd.DataFrame, class_name: str) -> pd.DataFrame:

    X = dataset.drop(class_name, axis=1)     # separates features and target class
    y = dataset[class_name]

    encoder = OrdinalEncoder()    # encodes categorical/discretised features into numeric codes
    X_encoded = encoder.fit_transform(X)

    smote = SMOTE(    # uses SMOTE to balance minority target classes
        random_state = 42,
        k_neighbors = 3 # maximum safe value for k_neighbor value
    )
    X_balanced_encoded, y_balanced = smote.fit_resample(X_encoded, y) # data is rebalanced here

    X_balanced_encoded = np.rint(X_balanced_encoded).astype(int) # SMOTE may create decimal values for features so entries are rounded back to the nearest valid category index

    for column_index, categories in enumerate(encoder.categories_): # makes sure rounded values stay within the valid range for each column
        max_valid_value = len(categories) - 1 
        X_balanced_encoded[:, column_index] = np.clip(
            X_balanced_encoded[:, column_index],
            0,
            max_valid_value
        )
    X_balanced = encoder.inverse_transform(X_balanced_encoded) # convert numeric category codes back to original category values

    balanced_dataset = pd.DataFrame(X_balanced, columns=X.columns) # rebuilds dataFrame

    balanced_dataset[class_name] = y_balanced # adds target class back

    balanced_dataset = balanced_dataset[dataset.columns] # keeps original column order

    balanced_dataset = balanced_dataset.astype(object) # matches original object-based format

    return balanced_dataset