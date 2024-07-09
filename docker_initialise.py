import sys
import subprocess
import re

def get_version_from_toml():
    try: 
        with open("pyproject.toml", "r") as file:
            content = file.read()
        version_match = re.search(r'version = "([^"]+)"', content)
        if version_match:
            return version_match.group(1)
    except Exception:
        return "Unknown"

def check_tensorflow():
    """
    To check tensorflow version and GPU Support and number of GPUs available
    """
    tensor_flow_version = "Not Found"
    gpu_support = "Not Found"
    number_of_gpus = "Not Found"
    
    tensorflow_version = subprocess.run(["python3", "-c", "import tensorflow as tf; print(tf.__version__)"], capture_output=True, text=True)
    gpu_support = subprocess.run(["python3", "-c", "import tensorflow as tf; print(tf.test.is_gpu_available())"], capture_output=True, text=True)
    number_of_gpus = subprocess.run(["python3", "-c", "import tensorflow as tf; print(len(tf.config.experimental.list_physical_devices('GPU')))"], capture_output=True, text=True)

    return tensorflow_version.stdout.strip(), gpu_support.stdout.strip(), number_of_gpus.stdout.strip()


def get_cuda_cudnn_nvidia_versions():

    cuda_version = 'Not found'

    # Get CUDA version
    try:
        cuda_version = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        cuda_version = re.search(r'release (\d+\.\d+)', cuda_version.stdout).group(1) if cuda_version.stdout else 'Not found'
    except Exception:
        pass

    # Get cuDNN version
    try:
        with open('/usr/local/cuda/include/cudnn_version.h', 'r') as f:
            cudnn_version = f.read()
        cudnn_version = re.search(r'#define CUDNN_MAJOR (\d+)\n#define CUDNN_MINOR (\d+)\n#define CUDNN_PATCHLEVEL (\d+)', cudnn_version)
        cudnn_version = f"{cudnn_version.group(1)}.{cudnn_version.group(2)}.{cudnn_version.group(3)}" if cudnn_version else 'Not found'
    except Exception:
        cudnn_version = 'Not found'

    # Get NVIDIA driver version
    nvidia_driver_version = 'Not found'
    try:
        nvidia_driver_version = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True)
        nvidia_driver_version = nvidia_driver_version.stdout.strip() if nvidia_driver_version.stdout else 'Not found'
    except Exception:
        pass

    return cuda_version, cudnn_version, nvidia_driver_version.split('\n')[0]



def main():
    version = get_version_from_toml()
    # run the ascii-image-converter in subprocess
    subprocess.run(["ascii-image-converter", "Fastvpinns_logo.png", "--braille", "-d" , "70,10"])
    print("**********************************************************")
    print(f"Official Docker Image for FastVPINNs - Version {version}")
    print(f"URL: https://cmgcds.github.io/fastvpinns/")
    print("Docker Image Author : Thivin Anandh")
    print("**********************************************************\n")
    
    # Execute any additional command passed to the Docker container
    # Should be a security risk, so commented out
    # if len(sys.argv) > 1:
    #     subprocess.run(sys.argv[1:])
    
    # obtain the cuda versions
    cuda_version, cudnn_version, nvidia_driver_version = get_cuda_cudnn_nvidia_versions()
    if cuda_version != 'Not found' and nvidia_driver_version != 'Not found':
        print(f"\033[92mGPU Checks Passed - GPU Acceleration is Available \033[0m")
    else :
        print(f"\033[91mGPU Checks Failed - Execution is available on CPU only\033[0m")


    # get tensorflow versions
    tensor_flow_version, gpu_support, number_of_gpus = check_tensorflow()

    column_width = 10
    print("-----------------------------------------------------------------------------------------------------")
    print(f"| CUDA Version:       {cuda_version:<{column_width}} || cuDNN Version: {cudnn_version:<{column_width}}   || NVIDIA Driver Version: {nvidia_driver_version:<{column_width}} |")
    print(f"| Tensorflow Version: {tensor_flow_version:<{column_width}} || GPU Support: {gpu_support:<{column_width}}     || Number of GPUs: {number_of_gpus:<{column_width}}        |")
    print("-----------------------------------------------------------------------------------------------------")


    

if __name__ == "__main__":
    main()