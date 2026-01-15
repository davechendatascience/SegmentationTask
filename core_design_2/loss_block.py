"""
Mask2Former (HF) uses built-in losses from the model output.
This block exists to keep the same module skeleton as core_design.
"""


def get_loss(outputs):
    return outputs.loss

