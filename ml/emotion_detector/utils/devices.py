import os


def set_cuda_devices(gpu_device):
    """Masks the CUDA visible devices.
       Warning: Running this while another script is executed
                might end up in the other script to crash.
    Parameters
    ----------
    gpu_device: int
        The CUDA device to use, such as [0,1].
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
