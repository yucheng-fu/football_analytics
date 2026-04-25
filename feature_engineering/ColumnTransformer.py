from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import List
from feature_engineering.OpenFE.FeatureGenerator import Node
from feature_engineering.OpenFE.openfe import tree_to_formula


class ColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        ohe_columns: list[str] | None = None,
        cat_columns: list[str] | None = None,
        use_ofe_features: bool = False,
    ):
        self.ohe_columns = ohe_columns or []
        self.cat_columns = cat_columns or []
        # handle_unknown='ignore' is crucial for consistent nested CV
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_nodes = []
        self.mappings_ = {}

    def fit(self, X: pd.DataFrame, y=None, feature_nodes: List[Node] | None = None):

        self.feature_nodes = feature_nodes or []
        self.mappings_ = {}
        for node in self.feature_nodes:

            formula = tree_to_formula(node)
            # 1. Frequency Encoding
            if node.name == "freq":
                col_data = node.children[0].calculate(X)
                self.mappings_[formula] = col_data.value_counts()

            # 2. GroupBy Aggregations (Mean, Std, Median, etc.)
            elif "GroupByThen" in node.name:
                val_col = node.children[0].calculate(X)
                key_col = node.children[1].calculate(X)
                agg_func = node.name.replace("GroupByThen", "").lower()
                self.mappings_[formula] = val_col.groupby(key_col).agg(agg_func)

            # 3. Categorical Combinations
            elif node.name == "Combine":
                d1 = node.children[0].calculate(X).astype(str)
                d2 = node.children[1].calculate(X).astype(str)
                temp = d1 + "_" + d2
                temp[d1.isna() | d2.isna()] = np.nan

                # Store Vocabulary: Pair String -> Integer ID
                _, uniques = temp.factorize()
                self.mappings_[formula] = {val: i for i, val in enumerate(uniques)}

            elif node.name == "CombineThenFreq":
                d1 = node.children[0].calculate(X).astype(str)
                d2 = node.children[1].calculate(X).astype(str)
                temp = d1 + "_" + d2
                temp[d1.isna() | d2.isna()] = np.nan
                self.mappings_[formula] = temp.value_counts()

            node.delete()

        if self.ohe_columns:
            # Fit only on the specified OHE columns
            self.encoder.fit(X[self.ohe_columns])
        return self

    def transform(
        self, X: pd.DataFrame, feature_nodes: List[Node] | None = None
    ) -> pd.DataFrame:
        df = X.copy()

        nodes = feature_nodes if feature_nodes is not None else self.feature_nodes
        for node in nodes:
            formula = tree_to_formula(node)

            if node.name == "freq":
                source_col = node.children[0].calculate(X)
                df[formula] = source_col.map(self.mappings_[formula]).astype(float)

            elif "GroupByThen" in node.name:
                key_col = node.children[1].calculate(X)
                df[formula] = key_col.map(self.mappings_[formula]).astype(float)

            elif node.name == "Combine":
                d1 = node.children[0].calculate(X).astype(str)
                d2 = node.children[1].calculate(X).astype(str)
                temp = d1 + "_" + d2
                temp[d1.isna() | d2.isna()] = np.nan
                # Fill unknown pairs with -1 to match training set factorize
                df[formula] = temp.map(self.mappings_[formula]).fillna(-1).astype(float)

            elif node.name == "CombineThenFreq":
                d1 = node.children[0].calculate(X).astype(str)
                d2 = node.children[1].calculate(X).astype(str)
                temp = d1 + "_" + d2
                temp[d1.isna() | d2.isna()] = np.nan
                df[formula] = temp.map(self.mappings_[formula]).astype(float)

            node.delete()

        # 1. Categorical handling (Native pandas categories)
        for col in self.cat_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # 2. One-Hot Encoding
        if self.ohe_columns:
            encoded_array = self.encoder.transform(df[self.ohe_columns])
            encoded_cols = self.encoder.get_feature_names_out(self.ohe_columns)

            # Create a temporary DF for encoded features
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoded_cols,
                index=df.index,  # Critical to keep indices aligned
            )

            # Combine and drop original OHE columns
            df = pd.concat([df, encoded_df], axis=1).drop(columns=self.ohe_columns)

        return df
