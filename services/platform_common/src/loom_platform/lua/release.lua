-- Release a reservation that was never dispatched (the gateway answered 503
-- model_unavailable after reserve, or lost the reply of the reserve call): drop it and
-- refund its tpm estimate to the key and org buckets. The rpm token it took stays spent,
-- and nothing is charged or appended to the usage stream, so the request leaves no usage
-- event.
--
-- Either way the request id is marked done (`:done:<rid>`, the marker reserve.lua refuses
-- and settle.lua writes), so a reserve script that runs after this release (it was queued
-- behind a Redis stall) answers duplicate_request and holds nothing.
--
-- KEYS[1]  org prefix, 'loom:org:<org_id>'
-- ARGV[1]  request id
-- ARGV[2]  done-marker TTL, seconds (at least the reservation TTL)
--
-- Returns 1 when a reservation was released, 0 when there was none.

local p = KEYS[1]
local rid = ARGV[1]
redis.call('SET', p .. ':done:' .. rid, '1', 'NX', 'EX', tonumber(ARGV[2]))
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
