from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "memory_available_gb",
    "disk_usage",
    "disk_read_mb",
    "disk_write_mb",
    "network_sent_mb",
    "network_receive_mb",
    "ping_ms",
    "battery_percent",
    "running_processes",
    "hour",
    "day",
    "month",
    "dayofweek",
]
COLUMN_ALIASES = {
    "memory_availabale_gb": "memory_available_gb",
    "disk_right_md": "disk_write_mb",
    "network_set_mb": "network_sent_mb",
    "network_recive_mb": "network_receive_mb",
}
HEADER_ALIASES = {
    "mean_cpu_usage_rate": "cpu_usage", "e_time": "timestamp", "time": "timestamp",
    "cpu": "cpu_usage", "cpu_percent": "cpu_usage", "cpu_usage_percent": "cpu_usage",
    "cpu_util": "cpu_usage", "cpu_utilization": "cpu_usage", "cpu_utilization_percent": "cpu_usage",
    "memory": "memory_usage", "memory_percent": "memory_usage", "memory_usage_percent": "memory_usage",
    "mem_usage": "memory_usage", "mem_percent": "memory_usage", "memory_utilization": "memory_usage",
    "memory_available": "memory_available_gb", "available_memory_gb": "memory_available_gb",
    "mem_available_gb": "memory_available_gb", "available_memory": "memory_available_gb",
    "disk_percent": "disk_usage", "disk_usage_percent": "disk_usage", "disk_utilization": "disk_usage",
    "disk_read": "disk_read_mb", "read_mb": "disk_read_mb", "disk_reading_mb": "disk_read_mb",
    "disk_write": "disk_write_mb", "write_mb": "disk_write_mb", "disk_writing_mb": "disk_write_mb",
    "network_sent": "network_sent_mb", "network_sent_mb": "network_sent_mb", "bytes_sent_mb": "network_sent_mb",
    "network_received": "network_receive_mb", "network_received_mb": "network_receive_mb",
    "network_receive": "network_receive_mb", "bytes_received_mb": "network_receive_mb",
    "ping": "ping_ms", "latency": "ping_ms", "latency_ms": "ping_ms", "response_time_ms": "ping_ms",
    "battery": "battery_percent", "battery_pct": "battery_percent",
    "processes": "running_processes", "running_process_count": "running_processes",
    "day_of_week": "dayofweek", "weekday": "dayofweek", "date_time": "timestamp", "datetime": "timestamp",
    "cpuutilization": "cpu_usage", "cpuutilization_percent": "cpu_usage",
    "memoryutilization": "memory_usage", "memoryutilization_percent": "memory_usage",
    "memoryavailablegb": "memory_available_gb", "diskutilization": "disk_usage",
    "diskreadmb": "disk_read_mb", "diskwritemb": "disk_write_mb",
    "networksentmb": "network_sent_mb", "networkreceivedmb": "network_receive_mb",
    "memory_availabale_gb": "memory_available_gb", "disk_right_md": "disk_write_mb",
    "network_set_mb": "network_sent_mb", "network_recive_mb": "network_receive_mb",
}

st.set_page_config(
    page_title="Enterprise Server Failure Prediction",
    page_icon="🖥️",
    layout="wide",
)
st.title("Prediction Enterprise Server Failure ")
st.write("Estimate server failure risk from system performance metrics.")

model_path = Path(__file__).with_name("server_failure_model.pkl")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Model file not found: {model_path.name}. Run the training notebook first.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model ({type(exc).__name__}: {exc}). Install the versions in requirements.txt and retrain it.")
    st.stop()

model_features = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
canonical_model_features = [COLUMN_ALIASES.get(name, name) for name in model_features]
if canonical_model_features != FEATURE_COLUMNS:
    st.error(
        "The saved model uses an unsupported feature schema. Run data_model.ipynb from top to bottom to rebuild server_failure_model.pkl."
    )
    st.stop()

legacy_feature_names = {canonical: legacy for legacy, canonical in COLUMN_ALIASES.items()}

def model_input(features):
    """Order inputs for this model and preserve legacy model feature names."""
    ordered = features.loc[:, FEATURE_COLUMNS]
    if model_features != FEATURE_COLUMNS:
        ordered = ordered.rename(columns=legacy_feature_names)
    return ordered.loc[:, model_features]

st.sidebar.header("Server Metrics")
cpu_usage = st.sidebar.slider("CPU Usage (%)", 0.0, 100.0, 50.0)
memory_usage = st.sidebar.slider("Memory Usage (%)", 0.0, 100.0, 50.0)
memory_available_gb = st.sidebar.number_input("Memory Available (GB)", min_value=0.0, value=8.0)
disk_usage = st.sidebar.slider("Disk Usage (%)", 0.0, 100.0, 60.0)
disk_read_mb = st.sidebar.number_input("Disk Read (MB per sample)", min_value=0.0, value=50.0)
disk_write_mb = st.sidebar.number_input("Disk Write (MB per sample)", min_value=0.0, value=40.0)
network_sent_mb = st.sidebar.number_input("Network Sent (MB per sample)", min_value=0.0, value=10.0)
network_receive_mb = st.sidebar.number_input("Network Received (MB per sample)", min_value=0.0, value=10.0)
ping_ms = st.sidebar.number_input("Ping (ms)", min_value=0.0, value=20.0)
battery_percent = st.sidebar.slider("Battery (%)", 0.0, 100.0, 100.0)
running_processes = st.sidebar.number_input("Running Processes", min_value=0, value=100)
hour = st.sidebar.slider("Hour", 0, 23, 12)
day = st.sidebar.slider("Day", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 7)
dayofweek = st.sidebar.slider("Day of Week (Monday=0)", 0, 6, 1)

input_data = pd.DataFrame([{
    "cpu_usage": cpu_usage,
    "memory_usage": memory_usage,
    "memory_available_gb": memory_available_gb,
    "disk_usage": disk_usage,
    "disk_read_mb": disk_read_mb,
    "disk_write_mb": disk_write_mb,
    "network_sent_mb": network_sent_mb,
    "network_receive_mb": network_receive_mb,
    "ping_ms": ping_ms,
    "battery_percent": battery_percent,
    "running_processes": running_processes,
    "hour": hour,
    "day": day,
    "month": month,
    "dayofweek": dayofweek,
}], columns=FEATURE_COLUMNS)

if st.sidebar.button("Predict", type="primary"):
    prediction = int(model.predict(model_input(input_data))[0])
    probabilities = model.predict_proba(model_input(input_data))[0]
    classes = list(model.classes_)
    failure_probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    st.subheader("Prediction Result")
    if prediction == 0:
        st.success("Healthy Server")
    elif failure_probability < 0.70:
        st.warning("Warning: Monitor the Server")
    else:
        st.error("Critical: High Risk of Server Failure")
    st.metric("Failure Probability", f"{failure_probability:.2%}")
    st.progress(failure_probability)
    st.subheader("Current Server Metrics")
    st.dataframe(input_data, use_container_width=True)

    try:
        import matplotlib.pyplot as plt

        classifier = model.named_steps["classifier"]
        importance = classifier.feature_importances_
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.barh(FEATURE_COLUMNS, importance)
        axis.set_title("Feature Importance")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    except (AttributeError, KeyError, ImportError):
        st.info("Feature importance is unavailable for this model.")

st.markdown("---")
st.header("Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        import csv
        import io

        csv_text = uploaded_file.getvalue().decode("utf-8-sig")
        # Accept pasted collector exports where records are space-separated on one line.
        import re

        csv_text = csv_text.strip().strip("()")
        csv_text = csv_text.replace("\\_", "_")
        csv_text = re.sub(
            r"\s+(?=\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*,)",
            "\n",
            csv_text,
        )
        # Clipboard exports may wrap each physical line in outer quote marks.
        csv_lines = [line.strip().strip('"').strip() for line in csv_text.splitlines() if line.strip()]
        if (
            len(csv_lines) > 1
            and csv_lines[0].rstrip().lower().endswith("disk_right_")
            and csv_lines[1].lstrip().lower().startswith("md,")
        ):
            # Join the wrapped disk_right_md header in this collector export.
            csv_lines[0] = csv_lines[0].rstrip() + csv_lines[1].lstrip()
            del csv_lines[1]
        csv_text = "\n".join(csv_lines)
        records = csv.reader(
            io.StringIO(csv_text), delimiter=",", quoting=csv.QUOTE_NONE, skipinitialspace=True
        )
        rows = [row for row in records if row and any(value.strip() for value in row)]
        if not rows:
            raise pd.errors.EmptyDataError("The uploaded CSV is empty.")
        # If the entire header was quoted as one field, split its embedded commas.
        if len(rows[0]) == 1 and "," in rows[0][0]:
            rows[0] = next(
                csv.reader([rows[0][0].strip().strip('"')], delimiter=",")
            )
        # Also recover if the wrapped header was not joined at the text level.
        if (
            len(rows) > 1
            and rows[0]
            and rows[0][-1].strip().lower() == "disk_right_"
            and rows[1]
            and rows[1][0].strip().lower() == "md"
        ):
            rows[0][-1] += rows[1][0]
            rows[0].extend(rows[1][1:])
            del rows[1]
        headers = [value.strip().strip('"') for value in rows[0]]
        data_rows = []
        for line_number, row in enumerate(rows[1:], start=2):
            if not row or not any(value.strip() for value in row):
                continue
            # Repair collector rows with an empty leading field and a comma-split timestamp.
            if len(row) == len(headers) + 2 and not row[0].strip():
                date_part = row[1].strip().strip('"')
                time_part = row[2].strip().strip('"').strip()
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_part) and re.fullmatch(r"\d{2}:\d{2}:\d{2}", time_part):
                    row = [f"{date_part} {time_part}", *row[3:]]
            if len(row) != len(headers):
                st.error(
                    f"Malformed CSV row at line {line_number}: expected {len(headers)} fields, "
                    f"found {len(row)}. No rows were skipped. Parsed row: {row!r}"
                )
                st.stop()
            data_rows.append([value.strip().strip('"') for value in row])
        batch_data = pd.DataFrame(data_rows, columns=headers)
        # Collector exports may contain spaces at the end of each field.
        batch_data = batch_data.apply(
            lambda column: column.str.strip().str.strip('"') if column.dtype == "object" else column
        )
        # Normalize spaces, punctuation, camel case, and common exported names.
        import re

        normalized_columns = {}
        for column in batch_data.columns:
            name = str(column).lstrip("\ufeff").strip()
            name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
            name = name.lower().replace("%", "_percent")
            name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
            normalized_columns[column] = HEADER_ALIASES.get(name, name)
        batch_data = batch_data.rename(columns=normalized_columns)
        if "timestamp" in batch_data.columns:
            timestamp = pd.to_datetime(batch_data.pop("timestamp"), errors="coerce")
            batch_data["hour"] = timestamp.dt.hour
            batch_data["day"] = timestamp.dt.day
            batch_data["month"] = timestamp.dt.month
            batch_data["dayofweek"] = timestamp.dt.dayofweek
        batch_data = batch_data.drop(columns=["failure", "Prediction"], errors="ignore")
        available_features = [column for column in FEATURE_COLUMNS if column in batch_data.columns]
        if not available_features:
            detected = ", ".join(map(str, batch_data.columns)) or "(no columns found)"
            st.error(
                "CSV has no recognized model features. Detected columns: " + detected
                + ". Include at least one supported metric, such as cpu_usage or mean_cpu_usage_rate."
            )
        else:
            # The training pipeline has a median imputer, so omitted metrics can be
            # represented as missing values and filled using training data medians.
            raw_features = batch_data.reindex(columns=FEATURE_COLUMNS)
            features = raw_features.apply(pd.to_numeric, errors="coerce")
            ping_missing = pd.Series(False, index=raw_features.index)
            if "ping_ms" in raw_features.columns:
                ping_text = raw_features["ping_ms"].astype("string").str.strip()
                ping_missing = ping_text.str.lower().isin(
                    ["", "na", "n/a", "nan", "none", "null", "timeout", "timed out", "unavailable", "-"]
                )
                ping_numeric_text = ping_text.str.replace(r"\s*ms$", "", regex=True, case=False)
                features["ping_ms"] = pd.to_numeric(ping_numeric_text, errors="coerce")
            invalid_values = raw_features.notna() & features.isna()
            if "ping_ms" in invalid_values.columns:
                invalid_values.loc[ping_missing, "ping_ms"] = False
            if invalid_values.any().any():
                bad_columns = invalid_values.columns[invalid_values.any()].tolist()
                st.error("CSV contains non-numeric values in: " + ", ".join(bad_columns))
            else:
                batch_data["Prediction"] = model.predict(model_input(features))
                st.info(
                    "Metrics missing from this CSV were filled using the model's training medians: "
                    + ", ".join(column for column in FEATURE_COLUMNS if column not in available_features)
                    if len(available_features) < len(FEATURE_COLUMNS)
                    else "All model metrics were provided in the CSV."
                )
                st.subheader("Prediction Results")
                st.dataframe(batch_data, use_container_width=True)
                csv = batch_data.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results", csv, "prediction_results.csv", "text/csv")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")










