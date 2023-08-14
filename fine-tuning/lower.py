import os
from os import path as osp
import sys

from tvm import relay, runtime
from tvm.relay import testing
import tvm
from tvm import te


lower_to_te = tvm._ffi.get_global_func("relay.backend.LowerToTE")