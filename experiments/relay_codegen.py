from tvm import relay
from tvm.relay import testing
import tvm

print(dir(tvm.target._ffi_api))
# Resnet18 workload
resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)


with relay.build_config(opt_level=0):
    graph, lib, params = relay.build_module.build(resnet18_mod, "llvm", params=resnet18_params)

# print relay ir
print(resnet18_mod.astext(show_meta_data=False))

# print source code
print(lib.get_source())
