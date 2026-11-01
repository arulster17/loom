-- Settle one request: release its reservation, charge its actual cost, and append its
-- usage event to the stream, all atomically, so the stream holds exactly the settled
-- requests and the org's usage counter always equals what the stream says was charged.
--
-- KEYS[1]  org prefix, 'loom:org:<org_id>'
-- KEYS[2]  usage stream, 'loom:usage'
-- ARGV[1]  request id      ARGV[2]  api key id
-- ARGV[3]  cost, micros    ARGV[4]  actual tokens, prompt + completion (for tpm)
-- ARGV[5]  seconds to remember the request as settled
-- ARGV[6..] usage event fields, name/value pairs (loom_platform.usage.UsageEvent)
--
-- Returns {1, org usage total after this request, stream entry id} or {0, 'duplicate'}
-- when the request was already settled (nothing is written then).
-- Settling works without a live reservation (expired, or a gateway restarted): the cost
-- is still charged and the event still written.

local p = KEYS[1]
local stream = KEYS[2]
local rid = ARGV[1]
local key_id = ARGV[2]
local cost = tonumber(ARGV[3])
local tokens = tonumber(ARGV[4])
local done_ttl = tonumber(ARGV[5])
local kp = p .. ':key:' .. key_id
local now = now_ms()

if not redis.call('SET', p .. ':done:' .. rid, '1', 'NX', 'EX', done_ttl) then
  return {0, 'duplicate'}
end

local tpm_est = release(p, rid) or 0
-- The reservation charged the tpm buckets its estimate; charge the difference now
-- (negative refunds an over-estimate).
local delta = tokens - tpm_est
if delta ~= 0 then
  bucket_charge(kp .. ':tb:tpm', delta, now)
  bucket_charge(p .. ':tb:tpm', delta, now)
end

local total = redis.call('INCRBY', p .. ':usage', cost)
redis.call('INCRBY', kp .. ':usage', cost)

local fields = {}
for i = 6, #ARGV do
  fields[#fields + 1] = ARGV[i]
end
fields[#fields + 1] = 'org_usage_total_micros'
fields[#fields + 1] = tostring(total)
local id = redis.call('XADD', stream, '*', unpack(fields))
return {1, total, id}
