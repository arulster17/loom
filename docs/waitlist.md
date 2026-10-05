# Waitlist signups

Signups from the waitlist form on the public results page, recorded as a demand signal.

**No waitlist backend is live yet.** Until `waitlist.action_url` is set in
`site/config.yaml`, the results page shows "Waitlist opens soon" and no form (see
[site.md](site.md#waitlist-endpoint)). The backend is still to be chosen: a form
service (Formspree, Buttondown) or a small Loom endpoint writing to the
`waitlist_signups` table.

Record a count with `loom_bench.site.waitlist.record_count`, taking it from
`waitlist_count(session)` when signups are in `waitlist_signups`, or from the form
service's dashboard otherwise. Recording the same source twice on one date replaces
that row.

```python
from loom_bench.site import record_count, waitlist_count
from loom_bench.store.db import session_scope

with session_scope() as s:
    record_count(None, waitlist_count(s), "waitlist_signups table")
```

| Date | Count | Source |
|---|---|---|
