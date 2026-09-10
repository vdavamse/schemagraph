from schemagraph.bench.gold_sql import gold_columns

SCHEMA = {
    "fin.cybersyn.timeseries": {"id_rssd", "variable", "value", "date"},
    "fin.cybersyn.entities": {"id_rssd", "name", "is_active"},
    "bigquery-public-data.ga4.events_*": {"event_name", "user_pseudo_id"},
}


def resolve(raw: str) -> str | None:
    raw = raw.lower()
    if raw in SCHEMA:
        return raw
    if raw.startswith("bigquery-public-data.ga4.events_"):
        return "bigquery-public-data.ga4.events_*"
    return None


def test_gold_columns_resolve_aliases_ctes_and_computed_names():
    sql = '''
    WITH "F" AS (
      SELECT "ID_RSSD"::NUMBER AS "ID_NUM", "VARIABLE", MAX("VALUE") AS "TotalAssets"
      FROM "FIN"."CYBERSYN"."TIMESERIES" WHERE "DATE" = '2022-12-31' GROUP BY 1, 2
    )
    SELECT "E"."NAME", "F"."TotalAssets"
    FROM "FIN"."CYBERSYN"."ENTITIES" AS "E" JOIN "F" ON "E"."ID_RSSD" = "F"."ID_NUM"
    WHERE "E"."IS_ACTIVE" = TRUE
    '''
    g = gold_columns(sql, "snowflake", SCHEMA, resolve)
    assert g.parsed and g.tables == {"fin.cybersyn.timeseries", "fin.cybersyn.entities"}
    assert g.columns == {
        ("fin.cybersyn.timeseries", "id_rssd"),
        ("fin.cybersyn.timeseries", "variable"),
        ("fin.cybersyn.timeseries", "value"),
        ("fin.cybersyn.timeseries", "date"),
        ("fin.cybersyn.entities", "name"),
        ("fin.cybersyn.entities", "id_rssd"),
        ("fin.cybersyn.entities", "is_active"),
    }
    assert g.unresolved == {"id_num", "totalassets"}  # CTE aliases are not base columns


def test_gold_columns_bigquery_shard_and_unparsable():
    sql = "SELECT event_name, COUNT(DISTINCT user_pseudo_id) FROM `bigquery-public-data.ga4.events_20210101` GROUP BY 1"
    g = gold_columns(sql, "bigquery", SCHEMA, resolve)
    assert g.columns == {("bigquery-public-data.ga4.events_*", "event_name"), ("bigquery-public-data.ga4.events_*", "user_pseudo_id")}
    assert not gold_columns("SELECT FROM WHERE (", "snowflake", SCHEMA, resolve).parsed
