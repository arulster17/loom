-- Refresh an org's credit total from Postgres, unless Redis already holds a newer one.
--
-- KEYS[1]  org prefix, 'loom:org:<org_id>'
-- ARGV[1]  credit total, micros (loom_org_credit_total)
-- ARGV[2]  its version (loom_org_credit_version, read in the same snapshot)
--
-- Returns 1 when the credit now holds this version's total, 0 when the org is not loaded
-- (nothing written: a lone credit key would never be read), -1 when Redis already has a
-- newer version (a reader with an older snapshot lost the race; nothing written).

local p = KEYS[1]
if redis.call('EXISTS', p .. ':usage') == 0 then
  return 0
end
local current = redis.call('GET', p .. ':credit_v')
if current and tonumber(current) > tonumber(ARGV[2]) then
  return -1
end
redis.call('SET', p .. ':credit', ARGV[1])
redis.call('SET', p .. ':credit_v', ARGV[2])
return 1
