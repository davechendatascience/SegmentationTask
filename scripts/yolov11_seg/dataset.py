"""
Dataset utilities for Roboflow YOLOv11 segmentation training.

Handles dataset download from Roboflow and data loading for Ultralytics YOLO.
"""
import os
from pathlib import Path
from roboflow import Roboflow


def download_roboflow_dataset(workspace: str, project: str, version: int, api_key: str = None):
    """
    Download dataset from Roboflow.

    Args:
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version number
        api_key: Roboflow API key (if not set in environment)

    Returns:
        Path to downloaded dataset directory
    """
    if api_key is None:
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if api_key is None:
            raise ValueError("ROBOFLOW_API_KEY environment variable not set")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8")  # YOLOv8 format works with YOLOv11

    return Path(dataset.location)


def setup_data_yaml(data_dir: Path, output_yaml: str = "data.yaml"):
    """
    Create data.yaml file for YOLO training.

    Args:
        data_dir: Path to dataset directory
        output_yaml: Output YAML file path

    Returns:
        Path to created YAML file
    """
    yaml_path = data_dir / output_yaml

    # YOLO format expects data.yaml in the dataset directory
    if yaml_path.exists():
        print(f"data.yaml already exists at {yaml_path}")
        return yaml_path

    # If not exists, assume standard structure
    train_path = data_dir / "train"
    val_path = data_dir / "valid"
    test_path = data_dir / "test"

    # Create data.yaml content
    yaml_content = f"""
train: {train_path}
val: {val_path}
test: {test_path}

nc: 26  # number of classes (adjust based on your dataset)
names: ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11', 'class_12', 'class_13', 'class_14', 'class_15', 'class_16', 'class_17', 'class_18', 'class_19', 'class_20', 'class_21', 'class_22', 'class_23', 'class_24', 'class_25']  # class names
"""

    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())

    print(f"Created data.yaml at {yaml_path}")
    return yaml_path