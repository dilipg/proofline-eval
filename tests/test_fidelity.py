import math

import proofline_eval as pe

TRUTH = dict(
    entities={"m1": ("Kavrel-7", "SamplingMethod", "Method"), "d1": ("Orvane-QA", "Benchmark", "Dataset")},
    mentions=[("c1", "m1", "Kavrel-7", "introduces"), ("c1", "d1", "Orvane-QA", "mentions"),
              ("c2", "m1", "KA-7", "uses")],
    facts=[("c1", "m1", "reports", "d1", "0.74"), ("c1", "m1", "evaluated_on", "d1", None)])


def perfect(**over):
    ext = dict(
        entities={"x_kavrel7": ("Kavrel-7", "SamplingMethod"), "x_orvaneqa": ("Orvane-QA", "Benchmark")},
        mentions=[("c1", "x_kavrel7", "Kavrel-7", "introduces"), ("c1", "x_orvaneqa", "Orvane-QA", "mentions"),
                  ("c2", "x_kavrel7", "KA-7", "uses")],
        facts=[("c1", "x_kavrel7", "reports", "x_orvaneqa", "0.74"),
               ("c1", "x_kavrel7", "evaluatedOn", "x_orvaneqa", None)],
        classes=[("Samplingmethod", None), ("Benchmark", None)], cards={"c1", "c2"})
    ext.update(over)
    return ext


def test_perfect_extraction_scores_one():
    g = pe.extraction_fidelity(TRUTH, perfect())
    for k in ("mention_precision", "mention_recall", "role_accuracy", "type_accuracy",
              "alias_resolution", "fact_precision", "fact_recall", "bridge_coverage"):
        assert g[k] == 1.0, k
    assert g["false_merges"] == 0 and g["domain_range_violations"] == 0 and g["untyped_links"] == 0
    assert g["class_recall"] == 0.2 and g["subclass_edge_recall"] == 0.0


def test_false_entity_split_alias_and_parent_type():
    ext = perfect(
        entities={"x_kavrel7": ("Kavrel-7", "Method"), "x_orvaneqa": ("Orvane-QA", "Benchmark"),
                  "x_ka7": ("KA-7", "Method"), "x_ssm": ("sparse sampling method", "SamplingMethod")},
        mentions=[("c1", "x_kavrel7", "Kavrel-7", "introduces"), ("c1", "x_orvaneqa", "Orvane-QA", "mentions"),
                  ("c2", "x_ka7", "KA-7", "uses"), ("c1", "x_ssm", "sparse sampling method", "introduces")])
    g = pe.extraction_fidelity(TRUTH, ext)
    assert g["mention_precision"] == 0.75 and g["mention_recall"] == 1.0
    assert g["alias_resolution"] == 0.0
    assert g["type_accuracy"] == 0.75                 # Kavrel-7 typed as parent Method: 0.5


def test_false_merge_violation_and_untyped():
    ext = perfect(
        entities={"x_both": ("Kavrel-7", "SamplingMethod"), "x_orvaneqa": ("Orvane-QA", "Benchmark")},
        mentions=[("c1", "x_both", "Kavrel-7", "introduces"), ("c1", "x_both", "Orvane-QA", "mentions")],
        facts=[("c1", "x_orvaneqa", "extends", "x_both", None), ("c1", "x_both", "related_to", "x_both", None)])
    g = pe.extraction_fidelity(TRUTH, ext)
    assert g["false_merges"] == 1
    assert g["domain_range_violations"] == 1
    assert g["untyped_links"] == 1


def test_fidelity_is_restricted_to_cards_read():
    ext = perfect(mentions=[("c1", "x_kavrel7", "Kavrel-7", "introduces"),
                            ("c1", "x_orvaneqa", "Orvane-QA", "mentions")], cards={"c1"})
    g = pe.extraction_fidelity(TRUTH, ext)
    assert g["mention_recall"] == 1.0
    assert math.isnan(g["bridge_coverage"])           # the user card c2 was never read


def test_grade_source_without_truth():
    g = pe.grade_source(dict(entities={}, mentions=[], facts=[]), perfect())
    assert g["measured"] is False and g["cards"] == 2 and g["mentions"] == 3


def test_extraction_block_not_ingested(seeded, capsys):
    assert pe.extraction_block(seeded) is None
    assert "NOT INGESTED" in capsys.readouterr().out


def test_extraction_block_grades_a_source(seeded, capsys):
    with seeded.conn.cursor() as cur:
        cur.execute("SELECT card_id FROM true_mentions GROUP BY card_id ORDER BY count(*) DESC LIMIT 1")
        cid = cur.fetchone()["card_id"]
        cur.execute("SELECT m.entity_id, m.surface, m.role, e.name, e.kind FROM true_mentions m "
                    "JOIN true_entities e ON e.id = m.entity_id WHERE m.card_id = %s", (cid,))
        rows = cur.fetchall()
        cur.execute("INSERT INTO extracted_cards VALUES (%s, 'test:copy')", (cid,))
        for r in rows:
            cur.execute("INSERT INTO entities VALUES (%s, %s, %s, 'test:copy') ON CONFLICT DO NOTHING",
                        ("x_" + r["entity_id"], r["name"], r["kind"]))
            cur.execute("INSERT INTO mentions VALUES (%s, %s, %s, %s, 'test:copy')",
                        (cid, "x_" + r["entity_id"], r["surface"], r["role"]))
    try:
        out = pe.extraction_block(seeded)
        g = out["test:copy"]
        assert g["measured"] and g["cards"] == 1
        assert g["mention_precision"] == 1.0 and g["mention_recall"] == 1.0
        # a coverage ratio prints beside the n it is over (1.000 over 5 is not a ceiling)
        assert "bridges graded (count)" in capsys.readouterr().out
    finally:
        with seeded.conn.cursor() as cur:
            for t in ("entities", "mentions", "extracted_cards"):
                cur.execute(f"DELETE FROM {t} WHERE source = 'test:copy'")
