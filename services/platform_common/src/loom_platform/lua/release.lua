-- Release a reservation that was never dispatched (the gateway answered 503
-- model_unavailable after reserve): drop it and refund its tpm estimate to the key and
-- org buckets. The rpm token it took stays spent, and nothing is charged or appended to
-- the usage stream, so the request leaves no usage event.
--
-- KEYS[1]  org prefix, 'loom:org:<org_id>'
-- ARGV[1]  request id
--
-- Returns 1 when a reservation was released, 0 when there was none.

local p = KEYS[1]
local rid = ARGV[1]
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
