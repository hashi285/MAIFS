from pathlib import Path

project_root = Path(__file__).parent
dataset_root = project_root / "datasets"

dataset_paths = {
    # Specify where are the roots of the datasets.
    'FR'       : str(dataset_root / "FantasticReality_v1"),
    'IMD'      : str(dataset_root / "IMD2020"),
    'CA'       : str(dataset_root / "CASIA"),
    'tampCOCO' : str(dataset_root / "tampCOCO"),
    'compRAISE': str(dataset_root / "compRAISE"),
}
