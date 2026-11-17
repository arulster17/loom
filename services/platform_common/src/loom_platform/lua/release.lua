-- Release a reservation that was never dispatched (the gateway answered 503
-- model_unavailable after reserve, or a reserve whose reply was lost): drop it and
-- refund its tpm estimate to the key and org buckets. The rpm token it took stays spent,
-- and nothing is charged or appended to the usage stream, so the request leaves no usage
-- event.
--
-- It also marks the request id done (the `:done:<rid>` marker reserve.lua refuses), even
-- when there is no reservation to drop: a reserve of the same id that runs after this
-- (queued behind a Redis stall) then answers duplicate_request and holds nothing. An
-- existing marker (a settled request) is left as it is.
--
-- KEYS[1]  org prefix, 'loom:org:<org_id>'
-- ARGV[1]  request id
-- ARGV[2]  seconds to remember the request id as done (at least the reservation TTL)
--
-- Returns 1 when a reservation was released, 0 when there was none.

local p = KEYS[1]
local rid = ARGV[1]
local done_ttl = tonumber(ARGV[2])
redis.call('SET', p .. ':done:' .. rid, '1', 'NX', 'EX', done_ttl)
local v = redis.call('HGET', p .. ':rsv', rid)
if not v then
  return 0
end
local key_id = string.match(v, '^%d+|([^|]+)|%d+$')
local tpm_est = release(p, rid)
if tpm_est and tpm_est > 0 then
  local now = now_ms()
  bucket_charge(p .. ':key:' .. key_id .. ':tb:tpm', -tpm_est, now)
  bucket_charge(p .. ':tb:tpm', -tpm_est, now)
end
return 1
