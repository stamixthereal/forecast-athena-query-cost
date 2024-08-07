# Copyright 2024 Stanislav Kazanov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
import pickle
import re
import warnings

import numpy as np
import optuna
import pandas as pd
import streamlit as st
import xgboost as xgb
from optuna.pruners import MedianPruner
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.utils.config import (
    DEFAULT_MODEL_FILE,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_POLY_FILE,
    DEFAULT_SCALER_FILE,
    IS_LOCAL_RUN,
)


def train_and_evaluate_model(
    query,
    transform_result,
    use_pretrained,
    in_memory_ml_attributes,
    save_ml_attributes_in_memory,
):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    warnings.filterwarnings("ignore")

    def extract_features(query):
        # Preprocess the query
        query_lower = query.lower()
        query_upper = query.upper()

        def count_occurrences(strings):
            if isinstance(strings, str):
                return query_lower.count(strings)
            return sum(map(query_lower.count, strings))

        def find_occurrences(pattern):
            return len(re.findall(pattern, query, re.IGNORECASE))

        features = {
            "query_length": len(query),
            "num_select_clauses": count_occurrences("select"),
            "num_joins": count_occurrences("join"),
            "num_subqueries": count_occurrences("select") - 1,
            "max_parentheses_depth": max([len(m) - 1 for m in re.findall(r"\([^()]*\)", query)] or [0]),
            "num_aggregation_functions": count_occurrences(["sum", "avg", "count", "max", "min"]),
            "limit_clause": int("LIMIT" in query_upper),
            "group_by_clause": int("GROUP BY" in query_upper),
            "having_clause": int("HAVING" in query_upper),
            "order_by_clause": int("ORDER BY" in query_upper),
            "distinct_query": int("distinct" in query_lower),
            "num_tables": count_occurrences("from"),
            "num_conditions": find_occurrences(r"\bWHERE\b"),
            "num_operators": sum(map(query.count, ["=", "<>", ">", "<", "BETWEEN", "LIKE", "IN"])),
            "distinct_functions": find_occurrences(r"\b\w+\s*\("),
            "date_functions": count_occurrences(["date", "day", "month", "year", "now"]),
            "string_functions": count_occurrences(["substring", "concat", "trim", "lower", "upper"]),
            "logical_operators": count_occurrences(["and", "or"]),
            "union_usage": count_occurrences("union"),
            "case_statements": count_occurrences("case when"),
            "nested_case_depth": max(
                [q.count("case when") for q in re.findall(r"case when(.*?)end", query, re.IGNORECASE)] or [0],
            ),
            "with_clause": int("WITH" in query_upper),
            "values_clause": int("VALUES" in query_upper),
            "on_clause": int("ON" in query_upper),
            "int_usage": count_occurrences("int"),
            "varchar_usage": count_occurrences("varchar"),
            "date_usage": count_occurrences("date"),
            "abs_function": count_occurrences("abs("),
            "ceil_function": count_occurrences("ceil("),
            "floor_function": count_occurrences("floor("),
            "cast_function": count_occurrences("cast("),
            "convert_function": count_occurrences("convert("),
            "plus_operator": query.count("+"),
            "minus_operator": query.count("-"),
            "multiply_operator": query.count("*"),
            "divide_operator": query.count("/"),
            "is_null_check": count_occurrences("is null"),
            "is_not_null_check": count_occurrences("is not null"),
            "alias_usage": find_occurrences(r"AS [\w_]+"),
            "temp_table_usage": int("TEMP" in query_upper) + int("TEMPORARY" in query_upper),
            "wildcard_usage": query.count("*"),
            "not_keyword_usage": count_occurrences("not"),
            "left_joins": count_occurrences("left join"),
            "right_joins": count_occurrences("right join"),
            "inner_joins": count_occurrences("inner join"),
            "full_joins": count_occurrences("full join"),
            "coalesce_usage": count_occurrences("coalesce("),
            "cte_usage": int("WITH" in query_upper and "AS" in query_upper),
            "having_with_groupby": int("HAVING" in query_upper and "GROUP BY" in query_upper),
            "between_operators": count_occurrences("between"),
            "not_conditions": count_occurrences("not"),
            "window_functions": count_occurrences(["row_number(", "rank(", "dense_rank(", "lead(", "lag("]),
            "subquery_in_from": find_occurrences(r"FROM\s*\("),
            "comment_usage": find_occurrences(r"--.*?$|\/*.*?\*\/"),
            "transaction_commands": count_occurrences(["commit", "rollback", "start transaction"]),
            "alter_command": int("ALTER" in query_upper),
            "drop_command": int("DROP" in query_upper),
            "math_functions": count_occurrences(["round(", "power(", "sqrt("]),
            "string_pattern_functions": count_occurrences(["patindex(", "charindex("]),
            "set_operations": count_occurrences(["union all", "intersect", "except"]),
            "exists_subqueries": count_occurrences(["exists(", "not exists("]),
            "cross_join": count_occurrences("cross join"),
            "time_functions": count_occurrences(["getdate(", "dateadd(", "datediff("]),
            "partitioned_table_queries": count_occurrences("partition"),
            "array_functions": count_occurrences("unnest("),
            "map_functions": count_occurrences(["element_at(", "map_keys(", "map_values("]),
            "pseudo_columns": count_occurrences(["$path", "$bucket"]),
            "serde_usage": count_occurrences("serde"),
            "table_properties": count_occurrences("external_location"),
            "file_format_usage": count_occurrences(["parquet", "orc", "json", "avro"]),
            "approximate_queries": count_occurrences(["approx_percentile(", "approx_distinct("]),
            "lateral_view": count_occurrences("lateral view"),
            "ctas_usage": count_occurrences("create table .* as select"),
        }

        if features["limit_clause"]:
            limit_match = re.search(r"LIMIT (\d+)", query, re.IGNORECASE)
            features["limit_number"] = int(limit_match.group(1)) if limit_match else 0
        else:
            features["limit_number"] = 0

        return list(features.values())

    def manual_interactions(query_features):
        join_table_interaction = query_features[2] * query_features[8]
        select_column_interaction = query_features[1] * query_features[9]
        return join_table_interaction, select_column_interaction

    if (
        use_pretrained
        and os.path.exists(DEFAULT_MODEL_FILE)
        and os.path.exists(DEFAULT_SCALER_FILE)
        and os.path.exists(DEFAULT_POLY_FILE)
    ):
        logging.info("Loading existing model and preprocessing objects...")
        model = xgb.XGBRegressor()
        model.load_model(DEFAULT_MODEL_FILE)
        with open(DEFAULT_SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)
        with open(DEFAULT_POLY_FILE, "rb") as f:
            poly = pickle.load(f)

        logging.info("Extracting features from the input query...")
        query_features = extract_features(query)
        query_interactions = manual_interactions(query_features)
        query_combined = np.hstack((query_features, query_interactions))
        query_poly = poly.transform([query_combined])
        query_scaled = scaler.transform(query_poly)
        return model.predict(query_scaled)[0]

    elif in_memory_ml_attributes and use_pretrained:
        logging.info("Loading existing model and preprocessing objects...")
        model = xgb.XGBRegressor()
        model.load_model(in_memory_ml_attributes["model"])
        scaler = in_memory_ml_attributes["scaler"]
        poly = in_memory_ml_attributes["poly_features"]
        logging.info("Extracting features from the input query...")
        query_features = extract_features(query)
        query_interactions = manual_interactions(query_features)
        query_combined = np.hstack((query_features, query_interactions))
        query_poly = poly.transform([query_combined])
        query_scaled = scaler.transform(query_poly)
        return model.predict(query_scaled)[0]
    else:
        logging.info("Loading the dataset...")
        data = pd.read_csv(DEFAULT_OUTPUT_FILE) if IS_LOCAL_RUN else transform_result.copy()

        logging.info(f"Loaded {len(data)} rows of data.")

        # Filter out null values in specified columns
        data = data[data["query"].notna() & data["cpu_time_ms"].notna() & data["peak_memory_bytes"].notna()]

        # Remove duplicate rows based on all columns
        data.drop_duplicates(inplace=True)

        # If you want to remove duplicates only based on the 'query' column, you would use:
        data.drop_duplicates(subset=["query"], inplace=True)

        # Convert 'query' column to uppercase
        data["query"] = data["query"].str.upper()

        X = data[["query", "cpu_time_ms"]].values
        y = data["peak_memory_bytes"].values

        logging.info("Splitting the dataset into train, validation, and test sets...")

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Feature Extraction
        logging.info("Extracting features from training data...")
        X_train_features = [extract_features(query[0]) for query in X_train]
        logging.info("Extracting features from validation data...")
        X_valid_features = [extract_features(query[0]) for query in X_valid]
        logging.info("Extracting features from test data...")
        X_test_features = [extract_features(query[0]) for query in X_test]

        # Manual Interaction
        X_train_interactions = [manual_interactions(features) for features in X_train_features]
        X_valid_interactions = [manual_interactions(features) for features in X_valid_features]
        X_test_interactions = [manual_interactions(features) for features in X_test_features]

        # Append interactions to features
        X_train_final = [
            features + list(interactions)
            for features, interactions in zip(X_train_features, X_train_interactions, strict=False)
        ]
        X_valid_final = [
            features + list(interactions)
            for features, interactions in zip(X_valid_features, X_valid_interactions, strict=False)
        ]
        X_test_final = [
            features + list(interactions)
            for features, interactions in zip(X_test_features, X_test_interactions, strict=False)
        ]

        # Convert to numpy arrays
        X_train_final = np.array(X_train_final)
        X_valid_final = np.array(X_valid_final)
        X_test_final = np.array(X_test_final)

        # Polynomial Features
        logging.info("Generating polynomial features...")
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_train_poly = poly.fit_transform(X_train_final)
        X_valid_poly = poly.transform(X_valid_final)
        X_test_poly = poly.transform(X_test_final)

        # Scaling
        logging.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_poly)
        X_valid_scaled = scaler.transform(X_valid_poly)
        X_test_scaled = scaler.transform(X_test_poly)

        def objective(trial):
            params = {
                "verbosity": 0,
                "n_estimators": 1000,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 0.5),
            }

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_valid_scaled, y_valid)],
                early_stopping_rounds=10,
                verbose=False,
            )
            preds = model.predict(X_valid_scaled)
            rmse = mean_squared_error(y_valid, preds, squared=False)
            return rmse

        if not use_pretrained:
            logging.info("Tuning hyperparameters with Optuna...")
            study = optuna.create_study(direction="minimize", pruner=MedianPruner())
            study.optimize(objective, n_trials=50)

            logging.info("Training the final model with best hyperparameters...")
            best_params = study.best_params
            model = xgb.XGBRegressor(**best_params)
            model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_valid_scaled, y_valid)],
                early_stopping_rounds=10,
                verbose=False,
            )

            if save_ml_attributes_in_memory:
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.poly_features = poly
            else:
                model.save_model(DEFAULT_MODEL_FILE)

                # Save the scaler and polynomial feature generator
                with open(DEFAULT_SCALER_FILE, "wb") as f:
                    pickle.dump(scaler, f)
                with open(DEFAULT_POLY_FILE, "wb") as f:
                    pickle.dump(poly, f)
            st.session_state.made_ml_training = True
        else:
            raise ValueError("Please use use_pretrained parameter...")

        logging.info("Evaluating the model on the test set...")
        y_pred = model.predict(X_test_scaled)

        def calculate_metrics(y_true, y_pred):
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            return mse, mae, r2

        mse, mae, r2 = calculate_metrics(y_test, y_pred)
        logging.info(f"Test MSE: {mse:.2f}")
        logging.info(f"Test MAE: {mae:.2f}")
        logging.info(f"Test R2: {r2:.2f}")

        logging.info("Extracting features from the provided query...")
        query_features = extract_features(query)
        query_interactions = manual_interactions(query_features)
        query_final = np.array(query_features + list(query_interactions)).reshape(1, -1)
        query_poly = poly.transform(query_final)
        query_scaled = scaler.transform(query_poly)
        query_pred = model.predict(query_scaled)[0]

        # Calculate the lower and upper bound for the prediction using MAE
        lower_bound = query_pred - mae
        upper_bound = query_pred + mae

        logging.info(f"Final prediction: {query_pred} bytes")
        logging.info(f"Prediction range: {lower_bound} bytes to {upper_bound} bytes")
        logging.info(f"Evaluation Metrics - MSE: {mse}, MAE: {mae}, R^2: {r2}")

        result = {}

        # Saving the model
        result["predicted_memory"] = query_pred
        result["lower_bound"] = lower_bound
        result["upper_bound"] = upper_bound
        result["mse"] = mse
        result["mae"] = mae
        result["r2"] = r2

        return result


def main(query, use_pretrained, transform_result, in_memory_ml_attributes, save_ml_attributes_in_memory):
    return train_and_evaluate_model(
        query=query,
        transform_result=transform_result,
        in_memory_ml_attributes=in_memory_ml_attributes,
        use_pretrained=use_pretrained,
        save_ml_attributes_in_memory=save_ml_attributes_in_memory,
    )


if __name__ == "__main__":
    main()
