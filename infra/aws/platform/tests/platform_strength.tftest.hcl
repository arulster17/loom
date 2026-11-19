# Strength checks for the platform stack (docs/phase1-m3-spec.md §13), beside the
# per-area files:
#
# A5 / B4 (§5.2 secrets, "as infra/aws/bench does for the reaper key"): each secret's
#   resource policy denies secretsmanager:GetSecretValue to every principal except its
#   reader role (the instance role; for serving-runpod-key, the serving reaper's role):
#   Effect Deny, Principal "*" (or {AWS = "*"}), condition ArnNotEquals (or ArnNotLike)
#   on aws:PrincipalArn = the reader role, no NotAction / NotPrincipal. An Allow alone
#   would let any other principal the account's IAM policies allow read the value.
# A37 (§8.5, §5.2 healthcheck): the plan fails for every region but us-east-1, not just
#   one; it passes in us-east-1.
# A8 (§5.3): alarm statistics where §5.3 fixes them. §5.3: a counter alarm uses metric
#   math RATE()/DIFF() when the agent sends cumulative values, Sum when it sends deltas.
#   So a counter alarm that reads its metric directly (no metric_query) uses Sum; one
#   that uses metric_query applies RATE or DIFF. (P1a/P1b Sum, H3 Minimum and S5's
#   Maximum are pinned in platform_alarms / platform_healthcheck.)
#
# Addresses: as in platform_secrets, platform_healthcheck and platform_alarms.

mock_provider "aws" {
  source          = "./tests/mocks/aws"
  override_during = plan
}

mock_provider "aws" {
  alias           = "us_east_1"
  source          = "./tests/mocks/aws_us_east_1"
  override_during = plan
}

variables {
  domain          = "example.test"
  brand_name      = "Example Brand"
  ops_email       = "ops@example.test"
  budget_usd      = 75
  releases_bucket = "loom-test-releases"
}

# Distinct role ARNs (the mock gives every role the same one).
override_resource {
  target          = aws_iam_role.instance
  override_during = plan
  values = {
    arn = "arn:aws:iam::123456789012:role/mock-instance-role"
  }
}

override_resource {
  target          = aws_iam_role.serving_reaper
  override_during = plan
  values = {
    arn = "arn:aws:iam::123456789012:role/mock-serving-reaper-role"
  }
}

# ---- A5 / B4: the Deny that keeps every other principal out -----------------------

run "a5_each_secret_policy_denies_getsecretvalue_to_all_but_its_reader" {
  command = plan

  assert {
    condition = alltrue([
      for k, p in aws_secretsmanager_secret_policy.platform :
      anytrue([
        for s in flatten([jsondecode(p.policy).Statement]) :
        try(s.Effect, "") == "Deny"
        && contains(flatten([try(s.Action, [])]), "secretsmanager:GetSecretValue")
        && !can(s.NotAction)
        && !can(s.NotPrincipal)
        && (try(s.Principal == "*", false) || try(toset(flatten([s.Principal.AWS])) == toset(["*"]), false))
        && contains(
          [
            toset([k == "serving-runpod-key" ? aws_iam_role.serving_reaper.arn : aws_iam_role.instance.arn]),
            toset(["arn:aws:iam::123456789012:role/${k == "serving-runpod-key" ? aws_iam_role.serving_reaper.name : aws_iam_role.instance.name}"]),
          ],
          toset(flatten([try(s.Condition.ArnNotEquals["aws:PrincipalArn"], s.Condition.ArnNotLike["aws:PrincipalArn"], "no ArnNotEquals/ArnNotLike on aws:PrincipalArn")]))
        )
      ])
    ])
    error_message = "Each secret policy has a Deny of secretsmanager:GetSecretValue to Principal \"*\" with condition ArnNotEquals aws:PrincipalArn = its reader role (instance role; serving reaper for serving-runpod-key), as infra/aws/bench does. Deny statements found: ${jsonencode({ for k, p in aws_secretsmanager_secret_policy.platform : k => [for s in flatten([jsondecode(p.policy).Statement]) : s if try(s.Effect, "") == "Deny"] })}."
  }
}

# ---- A37: only us-east-1 ------------------------------------------------------------

run "a37_plan_passes_in_us_east_1" {
  command = plan

  override_data {
    target = data.aws_region.current
    values = {
      id     = "us-east-1"
      region = "us-east-1"
      name   = "us-east-1"
    }
  }

  assert {
    condition     = aws_route53_health_check.hooks.type == "HTTPS_STR_MATCH"
    error_message = "In us-east-1 the plan succeeds and includes the hooks. health check."
  }
}

run "a37_plan_fails_in_us_west_2" {
  command = plan

  override_data {
    target = data.aws_region.current
    values = {
      id     = "us-west-2"
      region = "us-west-2"
      name   = "us-west-2"
    }
  }

  expect_failures = [
    aws_route53_health_check.hooks,
  ]
}

run "a37_plan_fails_in_us_east_2" {
  command = plan

  override_data {
    target = data.aws_region.current
    values = {
      id     = "us-east-2"
      region = "us-east-2"
      name   = "us-east-2"
    }
  }

  expect_failures = [
    aws_route53_health_check.hooks,
  ]
}

run "a37_plan_fails_in_eu_central_1" {
  command = plan

  override_data {
    target = data.aws_region.current
    values = {
      id     = "eu-central-1"
      region = "eu-central-1"
      name   = "eu-central-1"
    }
  }

  expect_failures = [
    aws_route53_health_check.hooks,
  ]
}

run "a37_plan_fails_in_ap_southeast_1" {
  command = plan

  override_data {
    target = data.aws_region.current
    values = {
      id     = "ap-southeast-1"
      region = "ap-southeast-1"
      name   = "ap-southeast-1"
    }
  }

  expect_failures = [
    aws_route53_health_check.hooks,
  ]
}

# ---- A8: counter alarms' math (§5.3) ------------------------------------------------

run "a8_counter_alarms_sum_deltas_or_apply_rate_diff" {
  command = plan

  assert {
    condition = alltrue([
      for id, a in {
        p2a = aws_cloudwatch_metric_alarm.p2a
        g2  = aws_cloudwatch_metric_alarm.g2
        s1a = aws_cloudwatch_metric_alarm.s1a
        s2a = aws_cloudwatch_metric_alarm.s2a
        s2b = aws_cloudwatch_metric_alarm.s2b
        s2c = aws_cloudwatch_metric_alarm.s2c
        s4  = aws_cloudwatch_metric_alarm.s4
      } :
      length(a.metric_query) == 0
      ? a.statistic == "Sum"
      : anytrue([for q in a.metric_query : can(regex("(RATE|DIFF)\\(", q.expression))])
    ])
    error_message = "A counter alarm (P2a, G2, S1a, S2a, S2b, S2c, S4) reading its metric directly sums the deltas (statistic Sum, §5.3); with metric math it applies RATE() or DIFF(). Got: ${jsonencode({ for id, a in { p2a = aws_cloudwatch_metric_alarm.p2a, g2 = aws_cloudwatch_metric_alarm.g2, s1a = aws_cloudwatch_metric_alarm.s1a, s2a = aws_cloudwatch_metric_alarm.s2a, s2b = aws_cloudwatch_metric_alarm.s2b, s2c = aws_cloudwatch_metric_alarm.s2c, s4 = aws_cloudwatch_metric_alarm.s4 } : id => { statistic = a.statistic, metric_query = length(a.metric_query) } })}."
  }
}
