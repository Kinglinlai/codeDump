import torch

# 1. Is CUDA available at all?
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    # 2. How many GPU devices?
    print("Number of GPUs:", torch.cuda.device_count())

    # 3. Index of the current default GPU
    cur_dev = torch.cuda.current_device()
    print("Current GPU index:", cur_dev)

    # 4. Name of that GPU
    print("Current GPU name:", torch.cuda.get_device_name(cur_dev))

    # 5. You can also check where a tensor or model lives:
    x = torch.randn(3, 3).to('cuda')   # move a tensor to GPU
    print("Tensor device:", x.device)

    # If you have a model:
    # model = MyModel().to('cuda')
    # print("Model parameters device:",
    #       next(model.parameters()).device)
