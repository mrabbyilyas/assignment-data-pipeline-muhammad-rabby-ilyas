"""Clean and transform the automobile dataset into a processed CSV."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "automobileEDA_dirty_training.csv"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "automobileEDA_processed.csv"
)

REQUIRED_COLUMNS = {
    "transaction_date",
    "make",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "stroke",
    "horsepower",
    "price",
    "horsepower-binned",
}

CATEGORICAL_COLUMNS = [
    "make",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "engine-type",
    "num-of-cylinders",
    "fuel-system",
    "horsepower-binned",
]

DATE_FORMATS = {
    r"\d{4}-\d{2}-\d{2}": "%Y-%m-%d",
    r"\d{2}/\d{2}/\d{4}": "%d/%m/%Y",
    r"\d{2}-\d{2}-\d{4}": "%m-%d-%Y",
    r"\d{2}-[A-Za-z]{3}-\d{4}": "%d-%b-%Y",
}


def extract(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV without modifying it."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Raw dataset is missing columns: {sorted(missing_columns)}")
    return data


def inspect(data: pd.DataFrame) -> None:
    """Print the initial dataset checks required by the assignment rubric."""
    print("\n=== INITIAL DATA INSPECTION ===")
    print("\nFirst five rows:")
    print(data.head().to_string(index=False))
    print(f"\nShape: {data.shape[0]:,} rows x {data.shape[1]:,} columns")
    print("\nColumns:")
    print(", ".join(data.columns))
    print("\nData types:")
    print(data.dtypes.to_string())
    print("\nMissing values by column:")
    print(data.isna().sum().to_string())
    print(f"\nExact duplicate copies: {int(data.duplicated().sum()):,}")
    print("\nRelevant categorical values:")
    for column in CATEGORICAL_COLUMNS:
        values = sorted(data[column].dropna().astype(str).unique().tolist())
        print(f"- {column} ({len(values)}): {values}")


def parse_transaction_dates(values: pd.Series) -> pd.Series:
    """Parse the four known date formats without ambiguous day/month guessing."""
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    for pattern, date_format in DATE_FORMATS.items():
        mask = text.str.fullmatch(pattern, na=False)
        parsed.loc[mask] = pd.to_datetime(
            text.loc[mask], format=date_format, errors="coerce"
        )

    invalid = text.notna() & parsed.isna()
    if invalid.any():
        examples = sorted(text.loc[invalid].unique().tolist())[:5]
        raise ValueError(f"Unsupported transaction date values: {examples}")

    # Both missing dates are internal gaps in a daily sequence.
    return parsed.interpolate().bfill().ffill()


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, standardize values, and impute missing data."""
    data = raw.drop_duplicates().copy()

    for column in CATEGORICAL_COLUMNS:
        data[column] = data[column].astype("string").str.strip().str.lower()

    data["drive-wheels"] = data["drive-wheels"].replace({"awd": "4wd"})
    data["transaction_date"] = parse_transaction_dates(
        data["transaction_date"]
    ).dt.strftime("%Y-%m-%d")

    for column in ["make", "num-of-doors"]:
        data[column] = data[column].fillna(data[column].mode().iat[0])

    for column in ["stroke", "horsepower", "price"]:
        data[column] = data[column].fillna(data[column].median())

    horsepower_bins = pd.cut(
        data["horsepower"],
        bins=[-float("inf"), 101, 155, float("inf")],
        labels=["low", "medium", "high"],
    )
    data["horsepower-binned"] = data["horsepower-binned"].fillna(
        horsepower_bins.astype("string")
    )

    return data


def report_cleaning(raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    """Print the before-and-after cleaning results."""
    deduplicated_raw = raw.drop_duplicates().reset_index(drop=True)
    changed_columns = [
        column
        for column in cleaned.columns
        if not (
            deduplicated_raw[column]
            .astype("string")
            .fillna("<MISSING>")
            .equals(cleaned[column].astype("string").fillna("<MISSING>"))
        )
    ]

    print("\n=== CLEANING SUMMARY ===")
    print(f"Rows: {len(raw):,} before -> {len(cleaned):,} after")
    print(
        f"Missing values: {int(raw.isna().sum().sum()):,} before -> "
        f"{int(cleaned.isna().sum().sum()):,} after"
    )
    print(f"Exact duplicate copies removed: {int(raw.duplicated().sum()):,}")
    print(f"Columns cleaned: {', '.join(changed_columns)}")


def transform(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Min-Max scale price and one-hot encode body style."""
    data = cleaned.copy()

    minimum, maximum = data["price"].agg(["min", "max"])
    if minimum == maximum:
        raise ValueError("Cannot Min-Max scale price because all values are equal")

    price_position = data.columns.get_loc("price") + 1
    data.insert(
        price_position,
        "price_minmax",
        (data["price"] - minimum) / (maximum - minimum),
    )

    encoded_body_style = pd.get_dummies(
        data.pop("body-style"), prefix="body_style", dtype="int64"
    )
    return pd.concat([data, encoded_body_style], axis=1)


def report_transformation(cleaned: pd.DataFrame, processed: pd.DataFrame) -> None:
    """Print concrete examples of the required transformations."""
    first_index = cleaned.index[0]
    body_style = cleaned.loc[first_index, "body-style"]
    encoded_column = f"body_style_{body_style}"

    print("\n=== TRANSFORMATION SUMMARY ===")
    print(
        f"Min-Max scaling: price {cleaned.loc[first_index, 'price']:,.2f} -> "
        f"price_minmax {processed.loc[first_index, 'price_minmax']:.6f}"
    )
    print(
        f"One-hot encoding: body-style '{body_style}' -> "
        f"{encoded_column}={processed.loc[first_index, encoded_column]}"
    )
    print(
        "New body-style columns: "
        + ", ".join(
            column for column in processed.columns if column.startswith("body_style_")
        )
    )


def validate(raw: pd.DataFrame, processed: pd.DataFrame) -> None:
    """Fail before writing if the processed dataset breaks core guarantees."""
    expected_rows = len(raw.drop_duplicates())
    if len(processed) != expected_rows:
        raise ValueError(f"Expected {expected_rows} processed rows, got {len(processed)}")
    if processed.isna().any().any():
        missing = processed.isna().sum().loc[lambda counts: counts.gt(0)].to_dict()
        raise ValueError(f"Processed data still contains missing values: {missing}")
    if processed.duplicated().any():
        raise ValueError("Processed data still contains duplicate rows")
    if not processed["transaction_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("Transaction dates are not consistently formatted as YYYY-MM-DD")
    if not processed["price_minmax"].between(0, 1).all():
        raise ValueError("price_minmax contains values outside the range [0, 1]")

    encoded_columns = [
        column for column in processed.columns if column.startswith("body_style_")
    ]
    if not encoded_columns or not processed[encoded_columns].sum(axis=1).eq(1).all():
        raise ValueError("Body-style one-hot encoding is invalid")


def load(processed: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    """Write the validated dataframe to the processed-data directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(path, index=False)


def main() -> None:
    raw = extract()
    inspect(raw)

    cleaned = clean(raw)
    report_cleaning(raw, cleaned)

    processed = transform(cleaned)
    report_transformation(cleaned, processed)
    validate(raw, processed)
    load(processed)

    print("\n=== LOAD SUMMARY ===")
    print("ETL pipeline completed successfully.")
    print(f"Rows: {len(raw):,} raw -> {len(processed):,} processed")
    print(f"Columns: {raw.shape[1]} raw -> {processed.shape[1]} processed")
    print(f"Exact duplicate copies removed: {int(raw.duplicated().sum()):,}")
    print(f"Missing values remaining: {int(processed.isna().sum().sum()):,}")
    print(f"Output: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
