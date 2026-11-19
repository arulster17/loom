# One repository per image the release workflow builds (docs/phase1-m3-spec.md §5.2 ecr,
# §6.1): immutable tags (compose pins digests anyway), scan on push, the last 20 images.

locals {
  ecr_repositories = toset(["gateway", "billing", "web", "poolmgr", "devstack", "caddy"])
}

resource "aws_ecr_repository" "loom" {
  for_each = local.ecr_repositories

  name                 = "loom/${each.key}"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "loom" {
  for_each = local.ecr_repositories

  repository = aws_ecr_repository.loom[each.key].name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 20 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 20
      }
      action = { type = "expire" }
    }]
  })
}
