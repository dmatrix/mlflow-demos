import random
from datetime import date, timedelta


def load_fema_disaster_data(spark, catalog: str, schema: str, table_name: str) -> str:
    """
    Generate 200 fabricated FEMA disaster records and write them to a Unity Catalog
    managed table.

    Returns the fully qualified table name: catalog.schema.table_name
    """
    random.seed(42)

    states = [
        "California", "Texas", "Florida", "Louisiana", "North Carolina",
        "New York", "Oklahoma", "Mississippi", "Alabama", "Georgia",
        "Tennessee", "Missouri", "Arkansas", "South Carolina", "Virginia",
        "Colorado", "Oregon", "Washington", "Iowa", "Nebraska",
    ]
    disaster_types = ["Hurricane", "Wildfire", "Flood", "Tornado", "Earthquake"]

    records = []
    for i in range(1, 201):
        year = random.choice(range(2020, 2026))
        d_type = random.choice(disaster_types)
        state = random.choice(states)
        severity = random.randint(2, 5)
        affected_pop = random.randint(500, 500_000)
        aid = round(random.uniform(100_000, 50_000_000), 2)
        start_day = date(year, 1, 1)
        decl_date = start_day + timedelta(days=random.randint(0, 364))

        records.append({
            "disaster_id": f"DR-{year}-{i:04d}",
            "year": year,
            "state": state,
            "disaster_type": d_type,
            "severity": severity,
            "affected_population": affected_pop,
            "federal_aid_amount": aid,
            "declaration_date": decl_date.isoformat(),
        })

    fq_table = f"{catalog}.{schema}.{table_name}"

    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

    import pandas as pd
    from pyspark.sql import functions as F

    df = spark.createDataFrame(pd.DataFrame(records))
    df = df.withColumn("declaration_date", F.col("declaration_date").cast("date"))
    df.write.mode("overwrite").saveAsTable(fq_table)

    spark.sql(
        f"""COMMENT ON TABLE {fq_table} IS
        'Fabricated FEMA disaster records (2020-2025) for 20 US states. """
        f"""Includes disaster type, severity (2-5), affected population, and federal aid amounts.'"""
    )

    return fq_table
