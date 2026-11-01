-- Reserve credit and admit one request: every check and every write in one atomic call.
--
-- KEYS[1]  org prefix, 'loom:org:<org_id>'
-- ARGV[1]  request id            ARGV[2]  api key id
-- ARGV[3]  max cost, micros      ARGV[4]  tpm estimate (prompt-token upper bound)
-- ARGV[5]  reservation TTL, ms
-- ARGV[6]  key rpm   ARGV[7]  key tpm   ARGV[8]  org rpm   ARGV[9]  org tpm
-- ARGV[10] key concurrency   ARGV[11] org concurrency   ARGV[12] key spend limit, micros
--          (any limit < 0 means unlimited)
--
-- Returns {admitted 1|0, reason, retry_after_ms, available_micros}. reason is 'ok' or
-- one of: noinit, duplicate_request, insufficient_balance, key_spend_limit,
-- key_concurrency, org_concurrency, key_rpm, org_rpm, key_tpm, org_tpm.
-- 'noinit': the org or key has no balance state yet; load it from Postgres (init_org)
-- and call again. Nothing is written unless the request is admitted.

local p = KEYS[1]
local rid = ARGV[1]
local key_id = ARGV[2]
local max_cost = tonumber(ARGV[3])
local tpm_est = tonumber(ARGV[4])
local ttl_ms = tonumber(ARGV[5])
local key_rpm, key_tpm = tonumber(ARGV[6]), tonumber(ARGV[7])
local org_rpm, org_tpm = tonumber(ARGV[8]), tonumber(ARGV[9])
local key_conc, org_conc = tonumber(ARGV[10]), tonumber(ARGV[11])
local spend_limit = tonumber(ARGV[12])
local kp = p .. ':key:' .. key_id
local now = now_ms()

purge_expired(p, now, 64)

local credit = redis.call('GET', p .. ':credit')
local usage = redis.call('GET', p .. ':usage')
local key_usage = redis.call('GET', kp .. ':usage')
if not credit or not usage or not key_usage then
  return {0, 'noinit', 0, 0}
end
if redis.call('HEXISTS', p .. ':rsv', rid) == 1 or redis.call('EXISTS', p .. ':done:' .. rid) == 1 then
  return {0, 'duplicate_request', 0, 0}
end

local rsum = tonumber(redis.call('GET', p .. ':rsum') or '0')
local available = tonumber(credit) - tonumber(usage) - rsum

-- Money first: a 402 says what to fix; a 429 only says when to retry.
if available < max_cost then
  return {0, 'insufficient_balance', 0, available}
end
if spend_limit >= 0 then
  local key_rsum = tonumber(redis.call('GET', kp .. ':rsum') or '0')
  if tonumber(key_usage) + key_rsum + max_cost > spend_limit then
    return {0, 'key_spend_limit', 0, available}
  end
end

if key_conc >= 0 and tonumber(redis.call('GET', kp .. ':inflight') or '0') >= key_conc then
  return {0, 'key_concurrency', 1000, available}
end
if org_conc >= 0 and redis.call('ZCARD', p .. ':rsv:exp') >= org_conc then
  return {0, 'org_concurrency', 1000, available}
end

-- Token buckets: rpm needs 1 token; tpm needs the estimate, capped at the bucket size so
-- a large request can still pass when the bucket is full (it then runs into debt).
local buckets = {
  {'key_rpm', kp .. ':tb:rpm', key_rpm, 1},
  {'org_rpm', p .. ':tb:rpm', org_rpm, 1},
  {'key_tpm', kp .. ':tb:tpm', key_tpm, tpm_est},
  {'org_tpm', p .. ':tb:tpm', org_tpm, tpm_est},
}
local levels = {}
for i, b in ipairs(buckets) do
  local per_min = b[3]
  if per_min >= 0 then
    local level = bucket_level(b[2], per_min, now)
    local need = math.min(b[4], per_min)
    if level < need then
      local wait = math.ceil((need - level) * 60000 / per_min)
      return {0, b[1], wait, available}
    end
    levels[i] = level
  end
end

-- Admitted: charge buckets, record the reservation.
for i, b in ipairs(buckets) do
  if levels[i] then
    bucket_store(b[2], math.max(levels[i] - b[4], -b[3]), b[3], now)
  end
end
redis.call('HSET', p .. ':rsv', rid, string.format('%d|%s|%d', max_cost, key_id, tpm_est))
redis.call('ZADD', p .. ':rsv:exp', now + ttl_ms, rid)
redis.call('INCRBY', p .. ':rsum', max_cost)
redis.call('INCRBY', kp .. ':rsum', max_cost)
redis.call('INCR', kp .. ':inflight')
return {1, 'ok', 0, available - max_cost}
