import tvm
from tvm import relay
from tvm.relay.testing import mlp
from tvm.runtime import profiler_vm
import numpy as np

target = "cuda"
dev = tvm.cuda()
ctx = tvm.cpu(0)

mod, params = mlp.get_workload(1)

exe = relay.vm.compile(mod, target=target, params=params)
vm = profiler_vm.VirtualMachineProfiler(exe, dev)

data = tvm.nd.array(np.random.rand(1, 1, 28, 28).astype("float32"), device=dev)
report = vm.profile(
    data,
    func_name="main",
    collectors=[tvm.runtime.profiling.PAPIMetricCollector()],
)
print(report)