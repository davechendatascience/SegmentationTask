from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


def build_processor(model_id: str, local_files_only: bool = False):
    return Mask2FormerImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)


def build_model(
    model_id: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
    local_files_only: bool = False,
):
    config = Mask2FormerConfig.from_pretrained(model_id, local_files_only=local_files_only)
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = label2id
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )
    return model

