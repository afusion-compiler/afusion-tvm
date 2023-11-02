import os
from os import path as osp
import sys
  
from tvm import relay, runtime
from tvm.relay import testing
import tvm
from tvm import te


with tvm.transform.PassContext(opt_level=3):
    compiled_graph_lib = tvm.relay.build_module.build(net, "cuda", params=params)
from tvm.contrib.debugger import debug_executor as graph_runtime
dev = tvm.device(str("cuda"), 0)
## building runtime
debug_g_mod = graph_runtime.GraphModuleDebug(
    compiled_graph_lib["debug_create"]("default", dev),
    [dev],
    compiled_graph_lib.get_graph_json(),
    "."
)