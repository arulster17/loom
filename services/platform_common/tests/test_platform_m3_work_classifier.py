"""The "work" classifier behind B3's "no work query ran before the role check" tests
(docs/m3-review-fixes-spec.md, "Billing", B3: "no work query runs on any pool connection
until the check has passed"; the role check "reads only the system catalogs ... names no
table of the schema").

`platform_m3_review_proxy.work_classifier` decides, for each statement a service sent,
whether it was work against the schema. Those tests can only fail for the right reason
if a catalog read that merely mentions a schema name (in a literal compared or listed, a
comment, an alias, another relation's column) is not work, and can only pass for the
right reason if every way of reaching a schema object (any case, quoted,
`public.`-qualified, in a CTE, after a comment, a second statement, a literal read as an
object name: `'orgs'::regclass`, `CAST('orgs' AS regclass)`, `regclass 'orgs'`, or a
function's direct argument such as `to_regclass('orgs')`) and every write, lock, LISTEN
or NOTIFY is work. This file pins both sides, without a database (the services' own role
check among the not-work cases), and checks that the schema's object names come from
the real migrated schema."""

from __future__ import annotations

import psycopg
import pytest
from platform_m3_review_proxy import schema_object_names, work_classifier

from loom_platform.settings import ROLE_CHECK_SQL

NAMES = [
    "orgs",
    "users",
    "sessions",
    "api_keys",
    "usage_events",
    "usage_events_2026_11",
    "loom_org_usage_total",
    "loom_ensure_usage_partitions",
    "Odd Name",
]

NOT_WORK = {
    "session_user": "SELECT session_user::text, current_user::text",
    "begin": "BEGIN",
    "commit": "COMMIT",
    "rollback": "ROLLBACK",
    "set": "SET search_path TO public",
    "show": "SHOW server_version",
    "select_1": "SELECT 1",
    "names_in_literals": (
        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relname IN ('orgs', 'users', 'api_keys')"
    ),
    "literal_with_verbs": "SELECT 'DELETE FROM orgs; INSERT INTO users'",
    "escape_string": r"SELECT E'it\'s orgs' AS x",
    "dollar_quoted": "SELECT $$UPDATE orgs SET x = 1$$, $tag$ FROM users $tag$",
    "line_comment": "SELECT 1 -- FROM orgs",
    "block_comment": "/* SELECT * FROM orgs; DELETE FROM users */ SELECT 1",
    "output_alias": "SELECT count(*) AS sessions FROM pg_stat_activity",
    "qualified_column": "SELECT d.sessions FROM pg_stat_database d",
    "other_schema": "SELECT * FROM pg_catalog.pg_tables WHERE schemaname = 'public'",
    "catalog_functions": (
        "SELECT pg_has_role(session_user, r.oid, 'MEMBER') FROM pg_roles r "
        "WHERE r.rolsuper AND r.rolname IN (session_user, current_user)"
    ),
    "unquoted_other_case_of_quoted_name": "SELECT * FROM pg_class WHERE odd = 1",
    "role_check_shape": (
        "SELECT session_user::text, current_user::text, "
        "  ARRAY(SELECT r.rolname::text FROM pg_roles r "
        "   WHERE r.rolsuper AND r.rolname IN (session_user, current_user) ORDER BY 1), "
        "  ARRAY(SELECT c.relname::text FROM pg_class c "
        "   JOIN pg_namespace n ON n.oid = c.relnamespace "
        "   WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') "
        "   AND (pg_has_role(session_user, c.relowner, 'MEMBER') "
        "     OR pg_has_role(current_user, c.relowner, 'MEMBER')) ORDER BY 1)"
    ),
    "role_check_itself": ROLE_CHECK_SQL,  # the services' own check, as they send it
    # A schema name in a literal that is not read as an object name.
    "values_list": "SELECT x FROM (VALUES ('orgs'), ('users')) v(x)",
    "any_array": "SELECT 1 FROM pg_class WHERE relname = ANY (ARRAY['orgs', 'users'])",
    "in_list_after_call": "SELECT 1 FROM pg_class WHERE lower(relname) IN ('orgs', 'users')",
    "compared_to_call": "SELECT 1 FROM pg_class WHERE lower(relname) = 'orgs'",
    "concatenated_argument": "SELECT length('x' || 'orgs')",
    "cast_to_text": "SELECT CAST('orgs' AS text)",
    "typed_literal_text": "SELECT text 'orgs'",
    "catalog_name_as_argument": "SELECT to_regclass('pg_catalog.pg_class')",
    "other_schema_as_argument": "SELECT has_table_privilege(current_user, 'other.orgs', 'SELECT')",
}

WORK = {
    "table": "SELECT * FROM orgs",
    "table_upper_case": "select * from ORGS",
    "quoted": 'SELECT * FROM "orgs"',
    "quoted_odd_name": 'SELECT * FROM "Odd Name"',
    "public_qualified": "SELECT * FROM public.orgs",
    "public_qualified_quoted": 'SELECT * FROM "public"."users"',
    "partition": "SELECT * FROM usage_events_2026_11",
    "join": "SELECT 1 FROM users u JOIN sessions s ON s.user_id = u.id",
    "function": "SELECT loom_org_usage_total($1)",
    "function_qualified": "SELECT public.loom_ensure_usage_partitions($1, $2)",
    "after_comment": "/* hello */ SELECT id FROM api_keys WHERE key_hash = $1",
    "after_line_comment": "-- note\nSELECT 1 FROM usage_events",
    "regclass_literal": "SELECT 'orgs'::regclass",
    "regclass_qualified_literal": "SELECT 'public.usage_events'::regclass::oid",
    "regclass_catalog_qualified_type": "SELECT 'orgs'::pg_catalog.regclass",
    "cast_as_regclass": "SELECT CAST('orgs' AS regclass)",
    "cast_as_regclass_qualified": "SELECT cast('public.usage_events' AS pg_catalog.regclass)",
    "cast_as_regprocedure": "SELECT CAST('loom_org_usage_total(uuid)' AS regprocedure)",
    "typed_literal_regclass": "SELECT regclass 'orgs'",
    "typed_literal_quoted_name": "SELECT regclass '\"Odd Name\"'",
    "typed_literal_regproc": "SELECT regproc 'public.loom_ensure_usage_partitions'",
    "to_regclass_argument": "SELECT to_regclass('public.orgs')",
    "to_regclass_upper_case": "SELECT to_regclass('ORGS')",
    "privilege_check_argument": "SELECT has_table_privilege(current_user, 'orgs', 'INSERT')",
    "relation_size_argument": "SELECT pg_relation_size('orgs')",
    "catalog_qualified_call_argument": "SELECT pg_catalog.pg_total_relation_size('api_keys')",
    "function_privilege_argument": (
        "SELECT has_function_privilege('public.loom_org_usage_total(uuid)', 'EXECUTE')"
    ),
    "argument_after_nested_call": (
        "SELECT has_table_privilege(lower(current_user::text), 'usage_events', 'SELECT')"
    ),
    "insert_verb": "INSERT INTO t VALUES (1)",
    "update_after_comment": "/* x */ UPDATE t SET a = 1",
    "delete_in_cte": "WITH d AS (DELETE FROM t RETURNING 1) SELECT count(*) FROM d",
    "second_statement": "SELECT 1; TRUNCATE t",
    "for_update": "SELECT * FROM pg_class FOR UPDATE",
    "for_share": "SELECT * FROM pg_class FOR SHARE",
    "lock": "LOCK TABLE t IN ACCESS EXCLUSIVE MODE",
    "listen": "LISTEN loom_changes",
    "notify": "NOTIFY loom_changes, 'x'",
    "pg_notify": "SELECT pg_notify('loom_changes', 'x')",
    "advisory_lock": "SELECT pg_advisory_lock(42)",
    "advisory_xact_lock_qualified": "SELECT pg_catalog.pg_try_advisory_xact_lock(42)",
    "nextval": "SELECT nextval('some_seq')",
    "copy": "COPY t FROM STDIN",
    "do_block": "DO $$ BEGIN PERFORM 1; END $$",
    "call": "CALL some_procedure()",
    "create": "CREATE TEMP TABLE x (a int)",
}


@pytest.mark.parametrize("case", sorted(NOT_WORK))
def test_item_b3_catalog_reads_and_mentions_are_not_work(case):
    """A catalog read, transaction control, or a schema name only mentioned (literal,
    comment, alias, another relation's column or schema) is not work."""
    assert work_classifier(NAMES)(NOT_WORK[case]) is False, NOT_WORK[case]


@pytest.mark.parametrize("case", sorted(WORK))
def test_item_b3_schema_references_and_effects_are_work(case):
    """Every way of reaching a schema object, and every write, lock, LISTEN, NOTIFY or
    effectful built-in, is work."""
    assert work_classifier(NAMES)(WORK[case]) is True, WORK[case]


def test_item_b3_no_names_still_catches_effects():
    """With no schema objects known, writes and effects are still work and catalog
    reads still are not."""
    is_work = work_classifier([])
    assert is_work("INSERT INTO orgs VALUES (1)")
    assert is_work("SELECT pg_notify('c', 'x')")
    assert not is_work("SELECT * FROM orgs")
    assert not is_work("SELECT rolname FROM pg_roles")


@pytest.mark.postgres
def test_item_b3_schema_object_names_cover_the_migrated_schema(pg_url):
    """The names come from the migrated schema: its tables, a monthly partition of
    `usage_events`, and its functions; catalog objects are not among them."""
    with psycopg.connect(pg_url) as conn:
        names = set(schema_object_names(conn))
    for expected in ("orgs", "api_keys", "usage_events", "loom_ensure_usage_partitions"):
        assert expected in names, f"{expected} missing from {sorted(names)}"
    assert any(n.startswith("usage_events_") for n in names), "no usage_events partition"
    assert not {"pg_class", "pg_roles", "pg_has_role"} & names
