import json
from pathlib import Path

from src.datasets.cmp_facade import CMPFacade
from src.datasets.hznu import Hznu
from src.datasets.irregular_facades import IrregularFacades
from src.train import TrainingConfig, train


def compare() -> None:
    """Compare different Configurations."""
    configs = [
        TrainingConfig(name="Baseline"),
        TrainingConfig(name="No Augmentation", augment=False),
        TrainingConfig(name="Only Hznu", datasets=[Hznu.download()]),
        TrainingConfig(name="Only Irregular Facades", datasets=[IrregularFacades.download()]),
        TrainingConfig(name="Only CMP", datasets=[CMPFacade.download()]),
    ]

    results = {}
    for config in configs:
        performance = train(config)[1]
        results[config.name] = performance
        _write_to_json(results)


def _write_to_json(results: dict[str, float]) -> None:
    output_path = Path(__file__).parent / "datasets" / ".data" / "compare.json"
    with output_path.open("w") as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == "__main__":
    compare()
