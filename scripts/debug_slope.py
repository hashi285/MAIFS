"""
Debug Power Spectrum Slope values
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools.frequency_tool import FrequencyAnalysisTool

def debug_slope():
    """Print actual slope values for sample images"""
    dataset_path = Path("/root/Desktop/MAIFS/datasets/GenImage_subset/BigGAN/val")

    ai_images = list((dataset_path / "ai").glob("*.png"))[:5]
    nature_images = list((dataset_path / "nature").glob("*.JPEG"))[:5]

    tool = FrequencyAnalysisTool()

    print("AI Images:")
    print("=" * 60)
    for img_path in ai_images:
        image = np.array(Image.open(img_path).convert("RGB"))
        result = tool.analyze(image)
        slope_data = result.evidence["power_spectrum_slope_analysis"]
        print(f"{img_path.name}")
        print(f"  Slope: {slope_data['slope']:.3f}")
        print(f"  Score: {slope_data['slope_score']:.3f}")
        print()

    print("\nNatural Images:")
    print("=" * 60)
    for img_path in nature_images:
        image = np.array(Image.open(img_path).convert("RGB"))
        result = tool.analyze(image)
        slope_data = result.evidence["power_spectrum_slope_analysis"]
        print(f"{img_path.name}")
        print(f"  Slope: {slope_data['slope']:.3f}")
        print(f"  Score: {slope_data['slope_score']:.3f}")
        print()

if __name__ == "__main__":
    debug_slope()
