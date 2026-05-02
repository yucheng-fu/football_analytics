import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from src.feature_engineering.OpenFE.FeatureGenerator import FNode, Node
from src.feature_engineering.ColumnTransformer import ColumnTransformer


@pytest.fixture
def sample_data():
    """Provides a consistent training and test set for parity checks."""
    train = pd.DataFrame({"cat": ["A", "B", "A", "C"], "val": [10.0, 20.0, 10.0, 40.0]})
    test = pd.DataFrame(
        {
            "cat": ["A", "D"],
            "val": [15.0, 25.0],
        }
    )
    return train, test


@pytest.fixture
def sample_data_with_second_category():
    train = pd.DataFrame(
        {
            "cat": ["A", "B", "A", "C"],
            "cat2": ["X", "Y", "X", "Z"],
            "val": [10.0, 20.0, 10.0, 40.0],
        }
    )
    test = pd.DataFrame(
        {
            "cat": ["A", "A", "D"],
            "cat2": ["X", "Q", "X"],
            "val": [15.0, 25.0, 35.0],
        }
    )
    return train, test


def test_freq_transformation_maps_training_counts(sample_data):
    # Arrange
    train, test = sample_data
    freq_node = Node("freq", [FNode("cat")])
    nodes = [freq_node]
    auto_transformer = ColumnTransformer(cat_columns=["cat"])

    # Act
    auto_transformer.fit(train, feature_nodes=nodes)
    auto_result = auto_transformer.transform(test, feature_nodes=nodes)

    # Assert
    expected_result = test.copy()
    train_counts = train["cat"].value_counts()
    expected_result["freq(cat)"] = test["cat"].map(train_counts).astype(float)
    expected_result["cat"] = expected_result["cat"].astype("category")

    assert_frame_equal(auto_result, expected_result, check_dtype=False, atol=1e-8)


def test_groupby_then_mean_uses_training_aggregates(sample_data):
    # Arrange
    train, test = sample_data
    mean_node = Node("GroupByThenMean", [FNode("val"), FNode("cat")])
    nodes = [mean_node]
    auto_transformer = ColumnTransformer(cat_columns=["cat"])

    # Act
    auto_transformer.fit(train, feature_nodes=nodes)
    auto_result = auto_transformer.transform(test, feature_nodes=nodes)

    # Assert
    expected_result = test.copy()
    train_group_means = train.groupby("cat")["val"].mean()
    expected_result["GroupByThenMean(val,cat)"] = (
        test["cat"].map(train_group_means).astype(float)
    )
    expected_result["cat"] = expected_result["cat"].astype("category")

    assert_frame_equal(auto_result, expected_result, check_dtype=False, atol=1e-8)


def test_combine_transformation_assigns_ids_and_marks_unknown_pairs(
    sample_data_with_second_category,
):
    # Arrange
    train, test = sample_data_with_second_category
    combine_node = Node("Combine", [FNode("cat"), FNode("cat2")])
    nodes = [combine_node]
    auto_transformer = ColumnTransformer(cat_columns=["cat", "cat2"])

    # Act
    auto_transformer.fit(train, feature_nodes=nodes)
    auto_result = auto_transformer.transform(test, feature_nodes=nodes)

    # Assert
    expected_result = test.copy()
    train_pairs = train["cat"].astype(str) + "_" + train["cat2"].astype(str)
    _, uniques = train_pairs.factorize()
    pair_to_id = {pair: idx for idx, pair in enumerate(uniques)}

    test_pairs = test["cat"].astype(str) + "_" + test["cat2"].astype(str)
    expected_result["Combine(cat,cat2)"] = (
        test_pairs.map(pair_to_id).fillna(-1).astype(float)
    )
    expected_result["cat"] = expected_result["cat"].astype("category")
    expected_result["cat2"] = expected_result["cat2"].astype("category")

    assert_frame_equal(auto_result, expected_result, check_dtype=False, atol=1e-8)


def test_combine_then_freq_maps_pair_frequency_from_training(
    sample_data_with_second_category,
):
    # Arrange
    train, test = sample_data_with_second_category
    combine_freq_node = Node("CombineThenFreq", [FNode("cat"), FNode("cat2")])
    nodes = [combine_freq_node]
    auto_transformer = ColumnTransformer(cat_columns=["cat", "cat2"])

    # Act
    auto_transformer.fit(train, feature_nodes=nodes)
    auto_result = auto_transformer.transform(test, feature_nodes=nodes)

    # Assert
    expected_result = test.copy()
    train_pairs = train["cat"].astype(str) + "_" + train["cat2"].astype(str)
    pair_counts = train_pairs.value_counts()
    test_pairs = test["cat"].astype(str) + "_" + test["cat2"].astype(str)
    expected_result["CombineThenFreq(cat,cat2)"] = test_pairs.map(pair_counts).astype(
        float
    )
    expected_result["cat"] = expected_result["cat"].astype("category")
    expected_result["cat2"] = expected_result["cat2"].astype("category")

    assert_frame_equal(auto_result, expected_result, check_dtype=False, atol=1e-8)
