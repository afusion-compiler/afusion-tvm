# 获取torch model
import torch
import torchvision
import utils

def get_network(name, batch_size, dtype="float32"):
    """Get the symbol definition and random weight of a network

    Parameters
    ----------
    name: str
        The name of the network, can be 'resnet-18', 'resnet-50', 'vgg-16', 'inception_v3', 'mobilenet', ...
    batch_size: int
        batch size
    dtype: str
        Data type

    Returns
    -------
    net: tvm.IRModule
        The relay function of network definition
    params: dict
        The random parameters for benchmark
    input_shape: tuple
        The shape of input tensor
    output_shape: tuple
        The shape of output tensor
    """
    # input_shape = (batch_size, 3, 224, 224)
    # output_shape = (batch_size, 1000)
    if name == "mobilenet":
        model = getattr(torchvision.models, "mobilenet_v2")(pretrained=True)
        model = model.eval()
        input_shape = utils.get_shape_arr(name, batch_size)
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
    elif "resnet" in name:
        n_layer = int(name.split("-")[1])
        model_name = f"resnet{n_layer}"
        model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.eval()
        input_shape = utils.get_shape_arr(model_name, batch_size)
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        

    return scripted_model, input_shape

if __name__ == "__main__":
    print("torch workloads")
    model = get_network("resnet-18", 1)

