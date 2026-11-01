-- Load an org's (and one key's) balance state from Postgres totals, unless present.
--
-- KEYS[1]  org prefix, 'loom:org:<org_id>'
-- ARGV[1]  credit total, micros (loom_org_credit_total)
-- ARGV[2]  usage total, micros (loom_org_usage_total)
-- ARGV[3]  api key id
-- ARGV[4]  key usage total, micros (loom_key_usage_total)
--
-- Org credit and usage are set together or not at all: a state where one exists without
-- the other could misstate the balance. Values already present are newer than any
-- Postgres snapshot (they include settled requests still in the stream), so they win.
-- Returns {org initialized now 1|0, key initialized now 1|0}.

local p = KEYS[1]
local kp = p .. ':key:' .. ARGV[3]
local org_new = 0
local key_new = 0
if redis.call('EXISTS', p .. ':usage') == 0 then
  redis.call('SET', p .. ':credit', ARGV[1])
  redis.call('SET', p .. ':usage', ARGV[2])
  org_new = 1
end
if redis.call('SET', kp .. ':usage', ARGV[4], 'NX') then
  key_new = 1
end
return {org_new, key_new}
