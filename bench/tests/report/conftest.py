import pytest

from loom_bench.prices import PriceBook, load_competitors, load_prices
from loom_bench.records import Market
from loom_bench.store.models import BenchRun

from .factories import TTFT_SGLANG, make_runs


@pytest.fixture(scope="session")
def price_book() -> PriceBook:
    return load_prices()


@pytest.fixture(scope="session")
def competitors():
    return load_competitors()


@pytest.fixture(scope="session")
def vllm_runs() -> list[BenchRun]:
    return make_runs("vllm-bf16")


@pytest.fixture(scope="session")
def sglang_runs() -> list[BenchRun]:
    return make_runs("sglang-bf16", engine="sglang", ttft=TTFT_SGLANG)


@pytest.fixture(scope="session")
def awq_runs() -> list[BenchRun]:
    return make_runs("vllm-awq", quantization="awq", ttft=TTFT_SGLANG, market=Market.SPOT)


@pytest.fixture(scope="session")
def all_runs(vllm_runs, sglang_runs, awq_runs) -> list[BenchRun]:
    return [*vllm_runs, *sglang_runs, *awq_runs]
