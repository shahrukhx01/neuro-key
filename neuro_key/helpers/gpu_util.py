import torch


def is_gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


def get_device_id(device):
    """Get device id from device string.

    Args:
        device: Device string.

    Returns:
        Device id.
    """
    if device == "cpu":
        return -1
    elif device == "auto":
        return 0 if is_gpu_available() else -1
    elif device.startswith("cuda:"):
        device_no = device.replace("cuda:", "")
        if device_no.isnumeric():
            return int(device_no)

    raise Exception(f"Invalid device: '{device}'")
