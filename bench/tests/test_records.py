from loom_bench.records import RequestRecord, RequestStatus


def make(**kw):
    base = dict(
        request_id="r1",
        status=RequestStatus.OK,
        sent_at_s=1.0,
        first_token_at_s=1.2,
        finished_at_s=2.2,
        completion_tokens=11,
    )
    base.update(kw)
    return RequestRecord(**base)


def test_derived_latencies():
    r = make(scheduled_at_s=0.9)
    assert abs(r.ttft_s - 0.2) < 1e-9
    assert abs(r.e2e_s - 1.2) < 1e-9
    assert abs(r.tpot_s - 0.1) < 1e-9
    assert abs(r.queue_delay_s - 0.1) < 1e-9


def test_tpot_undefined_for_single_token_or_failure():
    assert make(completion_tokens=1).tpot_s is None
    assert make(first_token_at_s=None, status=RequestStatus.ERROR).ttft_s is None


def test_row_round_trip():
    r = make(itl_s=[0.1, 0.1], meta={"k": 1})
    row = r.to_row()
    assert row["status"] == "ok" and row["tpot_s"] is not None
    assert RequestRecord.from_row(row) == r
