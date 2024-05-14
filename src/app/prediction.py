import re
import warnings

import numpy as np
import logging
import optuna
import pandas as pd
import xgboost as xgb
from optuna.pruners import MedianPruner
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from src.utils.config import DEFAULT_OUTPUT_FILE, DEFAULT_MODEL_FILE


def train_and_evaluate_model():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    warnings.filterwarnings("ignore")

    # Constants
    LOWER_ALPHA = 0.05
    UPPER_ALPHA = 0.95

    # Load the sample dataset
    logging.info("Loading the dataset...")
    data = pd.read_csv(DEFAULT_OUTPUT_FILE)
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
                [q.count("case when") for q in re.findall(r"case when(.*?)end", query, re.IGNORECASE)] or [0]
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

    X_train = np.hstack((X_train_features, X_train_interactions))
    X_valid = np.hstack((X_valid_features, X_valid_interactions))
    X_test = np.hstack((X_test_features, X_test_interactions))

    # Polynomial Features
    poly = PolynomialFeatures(2)
    X_train_poly = poly.fit_transform(X_train)
    X_valid_poly = poly.transform(X_valid)
    X_test_poly = poly.transform(X_test)

    # Feature Scaling
    scaler = StandardScaler()
    logging.info("Scaling features...")
    X_train_poly = scaler.fit_transform(X_train_poly)
    X_valid_poly = scaler.transform(X_valid_poly)
    X_test_poly = scaler.transform(X_test_poly)

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")  # Use an appropriate strategy
    X_train_poly = imputer.fit_transform(X_train_poly)
    X_valid_poly = imputer.transform(X_valid_poly)
    X_test_poly = imputer.transform(X_test_poly)

    # Check for NaN values after imputation
    if np.isnan(X_train_poly).any() or np.isnan(X_valid_poly).any() or np.isnan(X_test_poly).any():
        raise ValueError("NaN values are still present in the data after imputation.")

    # PCA for Dimensionality Reduction
    pca = PCA(n_components=0.95)
    logging.info("Applying PCA for dimensionality reduction...")
    X_train_poly = pca.fit_transform(X_train_poly)
    X_valid_poly = pca.transform(X_valid_poly)
    X_test_poly = pca.transform(X_test_poly)

    # Clip Outliers for y_train
    logging.info("Clipping outliers for y_train...")
    quantile_transform = pd.Series(y_train).quantile([LOWER_ALPHA, UPPER_ALPHA])
    y_train_clipped = np.clip(y_train, quantile_transform[LOWER_ALPHA], quantile_transform[UPPER_ALPHA])

    # Clip Outliers for y_valid
    logging.info("Clipping outliers for y_valid...")
    quantile_transform_valid = pd.Series(y_valid).quantile([LOWER_ALPHA, UPPER_ALPHA])
    y_valid_clipped = np.clip(
        y_valid,
        quantile_transform_valid[LOWER_ALPHA],
        quantile_transform_valid[UPPER_ALPHA],
    )

    # Splitting training data to train and smaller validation set
    X_train_final, X_valid_final, y_train_final, y_valid_final = train_test_split(
        X_train_poly, y_train_clipped, test_size=0.1, random_state=42
    )

    logging.info("Starting hyperparameter optimization...")

    def get_trial_parameters(trial):
        """Get hyperparameters for the trial."""

        # Core parameters
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "objective": "reg:squarederror",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        }

        # Tree-specific parameters
        if params["booster"] == "gbtree" or params["booster"] == "dart":
            params.update(
                {
                    "max_depth": trial.suggest_int("max_depth", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
                    "gamma": trial.suggest_float("gamma", 0.0, 1),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1),
                    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 1.5),
                }
            )

        # Regularization
        params.update(
            {
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
            }
        )

        return params

    def train_xgb_model(params, x_train_data, y_train_data, x_valid_data, y_valid_data):
        """Train the XGBoost model."""

        model = xgb.XGBRegressor(**params)

        eval_set = [(x_train_data, y_train_data), (x_valid_data, y_valid_data)]
        model.fit(
            x_train_data,
            y_train_data,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False,
            eval_metric="mae",
        )

        return model

    def objective(trial):
        logging.debug("Starting a new trial...")

        params = get_trial_parameters(trial)

        model = train_xgb_model(params, X_train_poly, y_train_clipped, X_valid_poly, y_valid_clipped)

        y_pred = model.predict(X_test_poly)

        return mean_absolute_error(y_test, y_pred)

    pruner = MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=50)
    logging.info(f"Optimization completed with best parameters: {study.best_params}")

    logging.info("Training the best model...")
    best_params = study.best_params
    best_xgb_model = xgb.XGBRegressor(**best_params)
    best_xgb_model.fit(X_train_poly, y_train_clipped)

    # Feature Importance
    logging.info("Getting feature importances...")
    feature_importances = best_xgb_model.feature_importances_
    sorted_idx = feature_importances.argsort()[-10:][::-1]
    print(f"Top 10 important features:\n{sorted_idx}")

    # Cross-validation
    logging.info("Starting cross-validation...")
    scores = cross_val_score(
        best_xgb_model,
        X_train_poly,
        y_train_clipped,
        cv=5,
        scoring="neg_mean_squared_error",
    )
    rmse = np.sqrt(-scores.mean())
    print("Average RMSE from cross-validation:", rmse)

    # Evaluate on Test Data
    logging.info("Evaluating on test data...")
    xgb_predictions = best_xgb_model.predict(X_test_poly)

    # Prediction for a new query
    new_query = """SELECT * FROM "hdl-uat-catalog-db"."quote" WHERE cdfdate = date('2023-07-07');""".upper()
    logging.info(f"Predicting memory for new query: {new_query[:50]}...")
    new_features = extract_features(new_query)
    new_interactions = manual_interactions(new_features)
    new_features_extended = np.hstack([new_features, new_interactions])

    # Apply all transformations
    new_features_poly = poly.transform([new_features_extended])
    new_features_scaled = scaler.transform(new_features_poly)
    new_features_poly_pca = pca.transform(new_features_scaled)

    predicted_memory = best_xgb_model.predict(new_features_poly_pca)[0]

    BYTES_IN_ONE_GB = 1_073_741_824  # 2^30

    # Evaluation Metrics
    mse = mean_squared_error(y_test, xgb_predictions)
    mae = mean_absolute_error(y_test, xgb_predictions)
    r2 = r2_score(y_test, xgb_predictions)

    # Saving the model
    best_xgb_model.save_model(DEFAULT_MODEL_FILE)
    logging.info(f"Model saved to: {DEFAULT_MODEL_FILE}")

    # Calculate the lower and upper bound for the prediction using MAE
    lower_bound = predicted_memory - mae
    upper_bound = predicted_memory + mae

    logging.info(f"Final prediction: {predicted_memory} bytes")
    logging.info(f"Prediction range: {lower_bound} bytes to {upper_bound} bytes")
    logging.info(f"Evaluation Metrics - MSE: {mse}, MAE: {mae}, R^2: {r2}")

    print(f"Predicted memory for new query: {predicted_memory:.2f} bytes ({predicted_memory / BYTES_IN_ONE_GB:.2f} GB)")
    print(
        f"Prediction range: {lower_bound:.2f} bytes ({lower_bound / BYTES_IN_ONE_GB:.2f} GB) "
        f"to {upper_bound:.2f} bytes ({upper_bound / BYTES_IN_ONE_GB:.2f} GB)"
    )
    print(f"Mean squared error (MSE): {mse}")
    print(f"Mean absolute error (MAE): {mae}")
    print(f"R-squared (R^2): {r2}")


def main():
    train_and_evaluate_model()


if __name__ == "__main__":
    main()
