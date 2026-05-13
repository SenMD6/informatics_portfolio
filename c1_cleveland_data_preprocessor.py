import copy

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer


def read_data(path: str) -> pd.DataFrame:
    dataset = None
    try:
        dataset = pd.read_csv(path, na_values='?')
    except FileNotFoundError as error:
        print(error)
        print(
            "The data you wanted to read was not at the location passed to the function. "
            "Please make sure to provide a correct path to file."
        )
    except TypeError as terror:
        print(terror)
        print("Please provide a proper path to file, the input is missing.")

    return dataset


def split_train_test(
        dataset: pd.DataFrame,
        class_name: str,
        test_size: float = 0.30,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Improved split:
    - reproducible because random_state is fixed;
    - stratified by sex + target so the train/test split keeps a more similar
      distribution of male/female and healthy/disease combinations.
    """

    stratify_column = dataset["sex"].astype(str) + "_" + dataset[class_name].astype(str)

    training_dataset, testing_dataset = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_column
    )

    return training_dataset.reset_index(drop=True), testing_dataset.reset_index(drop=True)


def impute_missing_values(
        training_data: pd.DataFrame,
        testing_data: pd.DataFrame,
        class_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Improved missing-value handling:
    - instead of dropping incomplete patients, we keep them;
    - numerical missing values are filled using the training median;
    - categorical missing values are filled using the training mode.

    This matters for fairness because row deletion can reduce representation,
    especially for smaller groups.
    """

    training_data = copy.deepcopy(training_data)
    testing_data = copy.deepcopy(testing_data)

    feature_columns = [column for column in training_data.columns if column != class_name]

    numeric_columns = [
        column for column in feature_columns
        if pd.api.types.is_numeric_dtype(training_data[column])
    ]

    categorical_columns = [
        column for column in feature_columns
        if column not in numeric_columns
    ]

    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy="median")

        training_data[numeric_columns] = numeric_imputer.fit_transform(
            training_data[numeric_columns]
        )

        testing_data[numeric_columns] = numeric_imputer.transform(
            testing_data[numeric_columns]
        )

    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy="most_frequent")

        training_data[categorical_columns] = categorical_imputer.fit_transform(
            training_data[categorical_columns]
        )

        testing_data[categorical_columns] = categorical_imputer.transform(
            testing_data[categorical_columns]
        )

    return training_data, testing_data


def discretize(
        vars_to_discretize: list[str],
        training_data: pd.DataFrame,
        testing_data: pd.DataFrame,
        class_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Improved discretisation:
    - uses k-means discretisation;
    - uses only 2 bins to avoid creating many small sparse categories;
    - fitted only on training data to avoid data leakage.

    Why 2 bins?
    The Cleveland dataset is small, so too many bins can make Naïve Bayes unstable.
    A smaller number of bins gives broader, more reliable categories.
    """

    training_data = copy.deepcopy(training_data)
    testing_data = copy.deepcopy(testing_data)

    discretiser = KBinsDiscretizer(
        n_bins=2,
        encode="ordinal",
        strategy="kmeans"
    )

    training_data[vars_to_discretize] = discretiser.fit_transform(
        training_data[vars_to_discretize]
    )

    testing_data[vars_to_discretize] = discretiser.transform(
        testing_data[vars_to_discretize]
    )

    return training_data, testing_data


def preprocess(data: pd.DataFrame, class_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Component 1 improved preprocessing pipeline.

    Pipeline:
    1. Copy original data.
    2. Split with fixed random seed and sex-target stratification.
    3. Impute missing values instead of deleting rows.
    4. Discretise continuous variables using 2-bin k-means discretisation.
    5. Convert to object type so the existing CategoricalNB classifier still works.
    """

    dataset = copy.deepcopy(data)

    training_dataset, testing_dataset = split_train_test(dataset, class_name)

    training_dataset, testing_dataset = impute_missing_values(
        training_dataset,
        testing_dataset,
        class_name
    )

    vars_to_discretize = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    training_dataset, testing_dataset = discretize(
        vars_to_discretize,
        training_dataset,
        testing_dataset,
        class_name
    )

    training_dataset = training_dataset.astype(object)
    testing_dataset = testing_dataset.astype(object)

    return training_dataset, testing_dataset