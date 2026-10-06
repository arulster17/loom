# How to add a GPU type (instance type)

A GPU type enters the Lab as an instance type with a price. On AWS that is a
configuration change: a `bench/prices.yaml` entry, the runner policy's allow-list, and a
quota (sections 1-6). On RunPod it is a price entry and a GPU type id
([RunPod](#runpod)). Other clouds have prices in `bench/prices.yaml` but no provider
yet, so they cannot be benchmarked in Phase 0.

## 1. Price it

Add the instance under `clouds.aws.<region>.instances` in `bench/prices.yaml`. All money
is integer micro-dollars per hour (`2_242_080` = $2.24208/h):

```yaml
        g6e.4xlarge:
          gpu: L40S                       # the name models use in hardware.gpu
          gpu_count: <int>
          gpu_memory_gb: <int>
          vcpus: <int>
          memory_gib: <int>
          local_nvme_gb: <int>            # 0 if the instance has no instance store
          on_demand_per_hour: <micros>
          spot_per_hour: <micros>         # optional; indicative only
          sources:
            - https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json
            - https://instances.vantage.sh/aws/ec2/g6e.4xlarge
          last_checked: 2026-10-05
```

Take the on-demand price from the AWS price list (first source above) and the specs and
spot average from the second (the existing entries record the same two sources). Rules
(`prices.py`, tested in `bench/tests/config/test_prices.py`):

| Rule | What happens otherwise |
|---|---|
| Prices are integers (a float like `2.24` is rejected) | validation error |
| `sources` has at least one URL; `last_checked` is a date | validation error |
| A verified entry has `vcpus`, `memory_gib` and `local_nvme_gb` | validation error |
| `verified: false` needs a `note` saying what to check | validation error |
| An unverified price is used only by the planner and the cloud providers (with a note); cost reports refuse it | `UnverifiedPriceError` in `bench report` |
| No spot price: spot runs are planned and accrued at on-demand | note in `bench plan` |

`committed_1y_per_hour` is optional and only used if a run's market is `committed_1y`
(none is today).

## 2. Allow it in the runner policy

The runner may launch only the types in `allowed_instance_types`
(`infra/aws/bench/variables.tf`; default g5.xlarge, g6.xlarge, g6e.xlarge, g6e.2xlarge,
g6e.12xlarge). Add the type to the default list, then re-apply with an admin identity:

```sh
terraform -chdir=infra/aws/bench apply -var owner=<your-name>
```

Without this, `RunInstances` is denied at provisioning time (after the plan was
accepted).

## 3. Quota

EC2 GPU quotas are per family, in vCPUs, and start at 0 (us-east-1):

| Family | Spot | On-demand |
|---|---|---|
| G and VT (g5, g6, g6e) | `L-3819A6DF` | `L-DB2E81BA` |
| P (p4d, p5, p5en) | `L-7212CCBC` | `L-417A185B` |

Request enough for the instance's vCPUs (commands in [aws-setup.md](../aws-setup.md),
step 1).

## 4. Host compatibility

- **Driver and CUDA.** Hosts boot the DLAMI "Base OSS Nvidia Driver" (Ubuntu 24.04,
  latest). The pinned vLLM v0.30.0 image needs driver ≥ 580 (CUDA 13.0); the engine image
  must also support the GPU's architecture.
- **Weights disk.** Weights go to `weights_dir` (default `/opt/dlami/nvme/loom-hf`, the
  instance-store NVMe). For an instance without local NVMe, point `weights_dir`
  (`LOOM_AWS_WEIGHTS_DIR`) at the root volume and raise the experiment's
  `provider.disk_gb` to fit images plus weights; EBS is accrued per GB.
- **Planner timings** (`plan.AWS_TIMING`) are the same for every instance type.

## 5. Use it

A model runs on the instance in `hardware.instance_types.aws` of its registry entry, or
the experiment's `provider.instance_type` override. The planner refuses a cell whose
instance has a different `gpu` name than `hardware.gpu`, or fewer GPUs than
`gpus_per_replica`. To benchmark an existing model on a different GPU, override both in
the experiment:

```yaml
provider: {kind: aws_ec2, region: us-east-1, instance_type: g6.xlarge, market: spot, disk_gb: 200}
variants:
  - name: vllm-l4
    hardware: {gpu: L4}
```

Every variant needs the override (a variant without it is refused with
`g6.xlarge has L4, spec wants L40S`). To make it the model's default, change
`hardware` in `config/models.yaml` instead.

## 6. Prove it

No test counts the price book's entries.
`test_prices.py::test_every_shipped_price_loads_and_is_consistent` checks every entry,
the new one included: it loads, `last_checked` is not after the book's, spot and
committed prices are below on-demand, and every AWS region has a verified storage price.

```bash
uv run python -c "
from loom_bench.prices import load_prices
from loom_bench.records import Market
p = load_prices()
print(p.instance_price('aws', 'us-east-1', 'g6e.4xlarge'))
print(p.instance_price('aws', 'us-east-1', 'g6e.4xlarge', Market.SPOT))"
uv run pytest -q bench/tests/config
uv run bench plan <experiment using the instance>.yaml     # exit 0, host priced as expected
```

The plan's host line shows the accrual rate: spot price × 1.25 (rounded up) plus the
root EBS volume ([cost-model.md](../cost-model.md#4-the-hourly-price-h)); reports price the
same host at its on-demand list price plus the volume. An instance type missing from
`bench/prices.yaml` is invalid input: `bench plan` and `bench run` exit 2 with
`instance type <type> has no price for aws/us-east-1 in bench/prices.yaml`.

## RunPod

RunPod pods are priced and created from the same price book, with no policy or quota
step (`provider.kind: runpod`, Secure Cloud on-demand only).

1. **Price it** under `clouds.runpod.secure.instances` in `bench/prices.yaml`, named
   `<gpu>-x<count>` like the existing `l40s-x1` and `l40s-x4`. Take the price and the
   minimum pod specs from the API (Secure Cloud, uninterruptible), with the GPU's RunPod
   id and count:

   ```sh
   curl -s https://api.runpod.io/graphql -H "Authorization: Bearer $RUNPOD_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"query":"query { gpuTypes(input:{id:\"NVIDIA L40S\"}) { id memoryInGb lowestPrice(input:{gpuCount:1, secureCloud:true}) { uninterruptablePrice minVcpu minMemory stockStatus } } }"}'
   ```

   `on_demand_per_hour` is `uninterruptablePrice` in micro-dollars, `vcpus` and
   `memory_gib` are `minVcpu` and `minMemory`, and `local_nvme_gb` is 0: the container
   disk is billed separately, by the region's `storage` entry. The same rules as on AWS
   apply (integers, sources, `last_checked`, verified specs).
2. **Map the GPU** to RunPod's GPU type id in `RUNPOD_GPU_TYPE_IDS`
   (`bench/src/loom_bench/experiment.py`; today `{"L40S": "NVIDIA L40S"}`), or set
   `provider.gpu_type_id` in the experiment.
3. **Use it**: `hardware.instance_types.runpod` (and `runpod` in `clouds`) of the model's
   registry entry, or the experiment's `provider.instance_type`. The planner refuses a
   cell whose GPU differs from the instance's or whose GPU count is not exactly the
   instance's `gpu_count`.
4. **Host compatibility.** `provider.allowed_cuda_versions` (default `["13.0"]`, which the
   pinned vLLM v0.30.0 image needs) restricts which hosts RunPod schedules. The container
   disk (`provider.container_disk_gb`, default 80) must hold the weights plus 35 GB, or
   the planner refuses. The planner timings are `plan.RUNPOD_TIMING` for every RunPod
   instance. Stock varies by datacenter (`stockStatus` above); `provider.data_center_ids`
   pins one.
5. **Prove it** as in section 6, with `instance_price('runpod', 'secure', '<name>')` and a
   `bench plan` of an experiment using it. The plan's host line shows the listed price
   plus container disk; at run time the provider accrues the larger of that and the pod's
   API `costPerHr`, and terminates a pod priced above the listed price ×
   `max_price_ratio` (1.25) before any work.
