-- Shared helpers, prepended to reserve.lua and settle.lua when they are loaded
-- (loom_platform.limiter). Key layout: docs/phase1-contracts.md, "Redis keys".
--
-- Scripts derive per-key names from the org prefix (KEYS[1]) and a key id, including
-- for expired reservations of other keys. That needs a single Redis node, which is
-- what Loom runs (no Redis Cluster).

local function now_ms()
  local t = redis.call('TIME')
  return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
end

-- Token bucket stored as a hash {t = tokens, ts = ms of t, c = per-minute rate}.
-- A missing bucket is full. Returns the tokens available now (may be negative: debt).
local function bucket_level(key, per_min, now)
  local h = redis.call('HMGET', key, 't', 'ts')
  if not h[1] then
    return per_min
  end
  local level = tonumber(h[1]) + (now - tonumber(h[2])) * per_min / 60000
  if level > per_min then
    level = per_min
  end
  return level
end

-- Store a bucket level. It refills completely within 2 minutes even from the deepest
-- allowed debt (-per_min), so an idle bucket can expire: missing means full.
local function bucket_store(key, level, per_min, now)
  redis.call('HSET', key, 't', tostring(level), 'ts', now, 'c', per_min)
  redis.call('PEXPIRE', key, 120000)
end

-- Charge `amount` tokens to an existing bucket, allowing debt down to -rate.
-- A bucket that has expired is full again, so a charge after expiry starts there.
local function bucket_charge(key, amount, now)
  local per_min = tonumber(redis.call('HGET', key, 'c'))
  if not per_min then
    return
  end
  local level = bucket_level(key, per_min, now) - amount
  if level < -per_min then
    level = -per_min
  end
  if level > per_min then
    level = per_min
  end
  bucket_store(key, level, per_min, now)
end

-- Drop reservation `rid` of org prefix `p`, undoing its counters.
-- Returns its tpm estimate, or nil when there was no such reservation.
local function release(p, rid)
  local v = redis.call('HGET', p .. ':rsv', rid)
  if not v then
    return nil
  end
  local amount, key_id, tpm_est = string.match(v, '^(%d+)|([^|]+)|(%d+)$')
  local kp = p .. ':key:' .. key_id
  redis.call('HDEL', p .. ':rsv', rid)
  redis.call('ZREM', p .. ':rsv:exp', rid)
  redis.call('DECRBY', p .. ':rsum', amount)
  redis.call('DECRBY', kp .. ':rsum', amount)
  redis.call('DECR', kp .. ':inflight')
  return tonumber(tpm_est)
end

-- Release up to `limit` reservations whose TTL has passed (crashed gateway, lost settle).
local function purge_expired(p, now, limit)
  local expired = redis.call('ZRANGEBYSCORE', p .. ':rsv:exp', '-inf', now, 'LIMIT', 0, limit)
  for _, rid in ipairs(expired) do
    if not release(p, rid) then
      redis.call('ZREM', p .. ':rsv:exp', rid)
    end
  end
end
