try:
    from colorama import init, Fore
    
    init(autoreset=True)
except ImportError:
    print("colorama is NOT installed! Did you activate the environment?")

    class FakeFore:
        GREEN = YELLOW = RED = ""
    
    Fore = FakeFore()

try:
    import torch
    
    if torch.cuda.is_available():
        print(Fore.GREEN + "torch is CUDA-enabled!")
        for i in range(torch.cuda.device_count()):
            print(Fore.GREEN + f"    {i} - {torch.cuda.get_device_name(i)}")
    else:
        print(Fore.RED + "torch is NOT CUDA-enabled!")
except ImportError:
    print(Fore.RED + "torch is NOT installed! Did you activate the environment?")


try:
    import slangpy_torch

    if slangpy_torch.get_api_ptr() != 0:
        print(Fore.GREEN + "slangpy-torch is installed and functional!")
    else:
        print(Fore.YELLOW + "slangpy-torch is installed but something is wrong!")
except ImportError:
    print(Fore.YELLOW + "slangpy-torch is NOT installed. Consider installing it for better performance!")
