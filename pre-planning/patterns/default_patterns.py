import tvm
import logging
from tvm.relay.dataflow_pattern import *
from .pattern import *
from ..utils import *


# Warning(@Soo): note that we ignore tuplegetitem nodes in TVM Relay,
# because they are only used to extract result of Relay's batch_norm operator
# maps op type to pattern representing it
str_to_pattern = {
  # RESNE(X)T
  "ADD" : Pattern(is_op('add')(wildcard(), wildcard())),
  "CONV2D" : Pattern(is_op("nn.conv2d")(wildcard(), wildcard())),
  "CONV2D_WINOGRAD_WO_WT" : Pattern(is_op("nn.contrib_conv2d_winograd_without_weight_transform")(wildcard(), wildcard())),
  "RELU" : Pattern(is_op("nn.relu")(wildcard())),
  "CONV2D_RELU" : Pattern(is_op("nn.relu")(is_op("nn.conv2d")(wildcard(), wildcard()))),
  "CONV2D_WINOGRAD_WO_WT_RELU" : Pattern(is_op("nn.relu")(is_op("nn.contrib_conv2d_winograd_without_weight_transform")(wildcard(), wildcard()))),
  "CONV2D_ADD_RELU" : Pattern(is_op("nn.relu")(is_op("add")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard()))),
  # "ADD_RELU" : Pattern(is_op("nn.relu")(is_op("add")(wildcard(), wildcard()))),

  # BERT
  "DENSE" : Pattern(is_op("nn.dense")(wildcard(), wildcard())),
  "RESHAPE" : Pattern(is_op("reshape")(wildcard())),
  "TRANSPOSE" : Pattern(is_op("transpose")(wildcard())),
  "BATCH_MATMUL" : Pattern(is_op("nn.batch_matmul")(wildcard(),wildcard())),
  "RESHAPE_TRANSPOSE" : Pattern(is_op("transpose")(is_op("reshape")(wildcard()))),
  "TRANSPOSE_RESHAPE" : Pattern(is_op("reshape")(is_op("transpose")(wildcard()))),
  "DENSE_RELU": Pattern(is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard()))),

  # NASRNN
  "TANH" : Pattern(is_op("tanh")(wildcard())),
  "SIGMOID" : Pattern(is_op("sigmoid")(wildcard())),
  "MULTIPLY" : Pattern(is_op("multiply")(wildcard(), wildcard())),
  "TUPLE_GET_ITEM_0" : Pattern(is_tuple_get_item(wildcard(), 0)),
  "TUPLE_GET_ITEM_1" : Pattern(is_tuple_get_item(wildcard(), 1)),
  "TUPLE_TWO_IDX" : Pattern(is_tuple([wildcard(), wildcard()])),
  "DENSE_RELU_ADD_SIGMOID" : Pattern(is_op("sigmoid")(is_op("add")(is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard())),is_constant()))),
  "DENSE_RELU_ADD_TANH": Pattern(is_op("tanh")(is_op("add")(is_constant(), is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard()))))),
  "DENSE_RELU_ADD_RELU": Pattern(is_op("nn.relu")(is_op("add")(is_constant(), is_op("nn.relu")(is_op("nn.dense")(wildcard(), wildcard()))))),
  "MULTIPLY_TANH": Pattern(is_op("tanh")(is_op("multiply")(wildcard(), wildcard()))),
  "RELU_ADD_RELU": Pattern(is_op("nn.relu")(is_op("add")(is_constant(), is_op("nn.relu")(wildcard())))),
  "ADD_SIGMOID": Pattern(is_op("sigmoid")(is_op("add")(wildcard(), wildcard()))),
  "ADD_TANH": Pattern(is_op("tanh")(is_op("add")(wildcard(), wildcard()))),

  # NASNET-A
  "CONCAT" : Pattern(is_op("concatenate")(wildcard())),
  "BIAS_ADD" : Pattern(is_op("nn.bias_add")(wildcard(), wildcard())),
  "AVG_POOL2D" : Pattern(is_op("nn.avg_pool2d")(wildcard())),
  "MAX_POOL2D" : Pattern(is_op("nn.max_pool2d")(wildcard())),
  "TUPLE_FIVE_IDX" : Pattern(is_tuple([wildcard(), wildcard(), wildcard(), wildcard(), wildcard()])),
  "CONV2D_BIAS_RELU" : Pattern(is_op("nn.relu")(is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), wildcard()), is_constant()))),
  "CONV2D_ADD" : Pattern(is_op("add")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard())),
  "AVG_POOL2D_ADD" : Pattern(is_op("add")(is_op("nn.avg_pool2d")(wildcard()), wildcard())),
  "TUPLE_FIVE_IDX_CONCAT" : Pattern(is_op("concatenate")(is_tuple([wildcard(), wildcard(), wildcard(), wildcard(), wildcard()]))),

  # ResNet-3D
  "CONV3D": Pattern(is_op("nn.conv3d")(wildcard(), wildcard())),
  "CONV3D_RELU": Pattern(is_op("nn.relu")(is_op("nn.conv3d")(wildcard(), wildcard()))),
  "CONV3D_ADD" : Pattern(is_op("add")(is_op("nn.conv3d")(wildcard(), wildcard()), wildcard())),
  "CONV3D_ADD_RELU": Pattern(is_op("nn.relu")(is_op("add")(is_op("nn.conv3d")(wildcard(), wildcard()), wildcard()))),

  # Others
  "DIAMOND" : get_diamond(),
  "BATCHNORM" : Pattern(is_tuple_get_item(is_op("nn.batch_norm")(wildcard(), wildcard(), wildcard(), wildcard(), wildcard()), 0)),
  "SOFTMAX" : Pattern(is_op("nn.softmax")(wildcard())),
  "BATCH_FLATTEN" : Pattern(is_op("nn.batch_flatten")(wildcard())),
  "GLOBAL_AVG_POOL2D" : Pattern(is_op("nn.global_avg_pool2d")(wildcard())),
  "CONV3D_BIAS_RELU" : Pattern(is_op("nn.relu")(is_op("nn.bias_add")(is_op("nn.conv3d")(wildcard(), wildcard()), is_constant()))),

  # Other Fused Ops
  "CONV2D_BN": Pattern(is_tuple_get_item(is_op("nn.batch_norm")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard(), wildcard(), wildcard(), wildcard()), 0)),
  "BN_RELU" : Pattern(is_op("nn.relu")(is_tuple_get_item(is_op("nn.batch_norm")(wildcard(), wildcard(), wildcard(), wildcard(), wildcard()), 0))),
  "CONV2D_BN_RELU" : Pattern(is_op("nn.relu")(is_tuple_get_item(is_op("nn.batch_norm")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard(), wildcard(), wildcard(), wildcard()), 0))),
  "SUBTRACT" : Pattern(is_op("subtract")(wildcard(), wildcard())),
}


# maps relay operator type to names of input vars.
relayop_to_varnames = {
  # RESNE(X)T
  "add" : ["data", "data"],
  "nn.conv2d" : ["data", "weight"],
  "nn.contrib_conv2d_winograd_without_weight_transform" : ["data", "weight"],
  "nn.relu": ["data"],

  # BERT
  "nn.dense" : ["data", "weight"],
  "reshape": ["data"],
  "transpose": ["data"],
  "nn.batch_matmul" : ["data", "data"],
  #"nn.batch_matmul" : ["x", "y"],

  # NASRNN
  "tanh": ["data"],
  "multiply": ["data", "data"],
  # "multiply": ["lhs", "rhs"],
  "sigmoid": ["data"],
  # FIXME(@Soo): How should we deal with TUPLE and TUPLE_GET_ITEM?

  # NASNET-A
  "concatenate": ["data"],
  "nn.bias_add" : ["data", "bias"],
  "nn.avg_pool2d" : ["data"],
  "nn.max_pool2d" : ["data"],
  "tuple" : ["data", "data", "data", "data", "data"],

  # RESNET_3D
  "nn.conv3d": ["data", "weight"],

  # Others
  "nn.batch_norm" : ["data", "bn_data_gamma", "bn_data_beta", "bn_data_moving_mean", "bn_data_moving_var"],
  "nn.softmax" : ["data"],
  "nn.batch_flatten" : ["data"],
  "nn.global_avg_pool2d" : ["data"],

  # DCGAN
  "image.resize": ["data"],

  # BERT_FULL
  "divide": ["data", "data"],
  "subtract": ["data", "data"],
  "sqrt": ["data"],
  "variance": ["data", "data"],
  "mean": ["data"],

  # GPT2
  "power": ["data", "data"],
  "where": ["data", "data", "data"],
  "cast": ["data"],
  "take": ["data", "data"],
  "split": ["data"],
}


def convert_str_to_pattern(lst):
    from afusion.pattern_manager.pattern.default_patterns import str_to_pattern
    return [  
             tuple([str_to_pattern[name], constraint_func]) 
             for name, constraint_func in lst 
           ]


# cuDNN
# Set of pattern and its constraints
cudnn_default_patterns_str = [ 
    ["CONV2D", None], 
    ["CONV3D", None],
    ["SOFTMAX", None],
    ["MAX_POOL2D", None],
    ["AVG_POOL2D", None],
    #["CONV2D_ADD_RELU", None],
    #["RELU", None],
    ["CONV3D_BIAS_RELU", None],
    ["BATCHNORM", None],
]
cudnn_default_patterns = convert_str_to_pattern(cudnn_default_patterns_str)

# cuBLAS
cublas_default_patterns_str = [ 
    ["DENSE", None], 
    ["BATCH_MATMUL", None],
]
cublas_default_patterns = convert_str_to_pattern(cublas_default_patterns_str)

# DNNL
def dnnl_check_add(config):
  for idx_shape, shape in enumerate(config._data_shape):
    if len(shape) < 2:
        return False

    # Check if all inputs have same dimensionality
    if idx_shape > 0 and len(shape) != prev_shape:
        return False
    prev_shape = len(shape)

    if shape == [1, 64, 56, 56] or shape == [1,128,28,28] or shape == [1, 256, 14, 14]:
        return False
  return True

def dnnl_check_relu(config):
  for shape in config._data_shape:
      if len(shape) != 4:
          return False
  return True

dnnl_default_patterns_str = [
  ["CONV2D", None], 
  ["CONV3D", None], 
  ["BATCHNORM", None], 
  ["DENSE", None], 
  #["ADD", dnnl_check_add]
  ["RELU", dnnl_check_relu], 
]
dnnl_default_patterns = convert_str_to_pattern(dnnl_default_patterns_str)


# MKL
def mkl_check_dense(config):
    dim1 = len(config._data_shape[0])
    dim2 = len(config._data_shape[1])
    return dim1 == 2 and dim2 == 2

mkl_default_patterns_str = [ 
    ["DENSE", mkl_check_dense], 
    ["BATCH_MATMUL", None],
]
mkl_default_patterns = convert_str_to_pattern(mkl_default_patterns_str)

# Check all nodes between src and sink by using fcheck (checks fusion conditions)
# Start from sink
# Gotta be dfs
def check_path(src, node, fcheck, path = [], paths = []):
    path.append(node)
    if src == node:
        assert(len(path))
        paths.append(path.copy())

    elif is_var_node(node) or is_constant_node(node):
        pass
    elif fcheck(node, node==src):
        children = []
        if is_tuple_node(node):
            children = node.fields
        elif is_tuplegetitem_node(node):
            children = [ node.tuple_value ]
        elif is_call_node(node):
            children = node.args
        else:
            raise Exception(f"Unsupported type ({type(node)})")

        for child in children:
            check_path(src, child, fcheck, path, paths)

    out = path.pop()
    assert(node == out)
    


def generate_relay_pattern_node(node):
    # @sunggg: hacky solution to deal with tuple
    # Unlike is_op(), is_tuple() expects to accept operands at its initialization
    if is_tuple_node(node):
        return "tuple", len(node.fields)
    elif is_tuplegetitem_node(node):
        #print(node, dir(node), node.tuple_value)
        #is_tuple_get_item, 2
        return "tuplegetitem", 1
    elif is_call_node(node):
        return is_op(node.op.name), len(node.args)
    elif is_constant_node(node):
        return wildcard(), 0
        #return is_constant(), 0
    elif is_var_node(node):
        return wildcard(), 0
        #return is_var(), 0
    elif isinstance(node, tvm.ir.op.Op):
        return is_op(node.name), node.num_inputs
    else:
        raise Exception(f"Unsupported type ({type(node)})")


# NOTE: Seems like relay pattern matching considers pointer of pattern node.
#       Also, relay pattern string can be misleading
# for example,
#         is_conv2d = is_op('nn.conv2d')(is_var(), is_var())
#         path1 = is_op('nn.relu')(is_conv2d)
#         path2 = is_op('nn.leaky_relu')(is_conv2d)
#         diamond = is_op('add')(path1, path2)
#         --> CallPatternNode(Op(add), [CallPatternNode(Op(nn.relu), [CallPatternNode(Op(nn.conv2d), [VarPattern(),VarPattern()])]), CallPatternNode(Op(nn.leaky_relu), [CallPatternNode(Op(nn.conv2d), [VarPattern(),VarPattern()])])])
#
#         This diamond pattern does not match with the following expr
#            inp1 = relay.var('input1')
#            inp2 = relay.var('input2')
#            weight1 = relay.var('weight1')
#            weight2 = relay.var('weight2')
#            conv2d1 = relay.op.nn.conv2d(inp1, weight1)
#            conv2d2 = relay.op.nn.conv2d(inp2, weight2)
#            relu = relay.op.nn.relu(conv2d1)
#            leaky_relu = relay.op.nn.leaky_relu(conv2d2, alpha=0)
#            out = relu + leaky_relu

# 根据节点到模式映射构建模式对象。它可以用于在模式匹配过程中构建需要的模式。
def build_pattern_with_map(src, node, nodeToPatternMap):
    if node in nodeToPatternMap:
        rpattern = nodeToPatternMap[node][0]

        children = []
        if is_var_node(node) or is_constant_node(node):
            pass
        elif is_tuple_node(node):
            children = node.fields
        elif is_tuplegetitem_node(node):
            children = [ node.tuple_value ]
        elif is_call_node(node):
            children = node.args
        else:
            raise Exception(f"Unsupported type ({type(node)})")

        if node == src:
            return rpattern

        operands = [ build_pattern_with_map(src, child, nodeToPatternMap) for child in children ]
        return rpattern(*operands)
    else:
        return wildcard()

# 根据提供的节点和路径生成对应的Relay模式对象，并将其用于模式匹配和模式生成的过程中。
def generate_relay_pattern(src, sink, paths = None, cur_pattern_type = None, nodeToPatternMap = dict()):
    if paths is None:
        # Handle single node
        assert(not (is_constant_node(src) and is_var_node(src)))
        assert(cur_pattern_type is None)
        rpattern, num_operands = generate_relay_pattern_node(sink)

        if num_operands == 0:
            assert(rpattern != "tuple")
            return rpattern, get_op_pattern(sink), 1

        operands = [wildcard() for __ in range(num_operands)]
        # @sunggg: hacky solution to deal with tuple
        if rpattern == "tuple":
            return is_tuple(operands), get_op_pattern(sink), 1
        elif rpattern == "tuplegetitem":
            # is_tuple(None): match with any inputs
            return is_tuple_get_item(wildcard()), get_op_pattern(sink), 1
        else: 
            return rpattern(*operands), get_op_pattern(sink), 1

    else:
        # Handle multiple nodes
        # Create pattern node for all nodes in paths (sink~src)
        cnt = 0
        nodeToPatternMap = dict()
        for path in paths:
            for node in path:
                if node not in nodeToPatternMap:
                    nodeToPatternMap[node] = generate_relay_pattern_node(node)
                    cnt += 1

                # Create pattern node for const/var
                #for child in get_args(node):
                #    if is_constant_node(child) or is_var_node(child):
                #        if child not in nodeToPatternMap:
                #            nodeToPatternMap[child] = wildcard() #generate_relay_pattern_node(child)

        assert src in nodeToPatternMap, f"{src.op.name}"
        pnode, num_operands = nodeToPatternMap[src]

        if num_operands == 0:
            assert(pnode != "tuple")
            nodeToPatternMap[src] = (pnode, 0)
            rpattern = build_pattern_with_map(src, sink, nodeToPatternMap)
            return rpattern, cur_pattern_type, cnt

        operands = [wildcard() for __ in range(num_operands)]

        # @sunggg: hacky solution to deal with tuple
        if pnode == "tuple":
            nodeToPatternMap[src] = (is_tuple(operands), 0) # it's zero cause we already handled.    
        elif pnode == "tuplegetitem":
            nodeToPatternMap[src] = (is_tuple_get_item(wildcard()), 0) # it's zero cause we already handled.    
        else:
            nodeToPatternMap[src] = (pnode(*operands), 0) # it's zero cause we already handled.
        rpattern = build_pattern_with_map(src, sink, nodeToPatternMap)

        return rpattern, cur_pattern_type, cnt


# @sunggg: default pattern generator. 
# To simulate recent fusion engines, it generates patterns with dom tree.
# 基于pdt生成模式，并模拟融合操作，生成相应的模式列表
class DefaultPatternGenerator(BasePatternGenerator):
    def generate(self, post_dom_tree, expr):
        generated_patterns = list()
        if is_constant_node(expr) or is_var_node(expr):
            return generated_patterns # returns empty node

        # Check anchor node
        if is_tuple_node(expr) or self.pattern_rule.op_rule(expr):
            anchor_pattern, anchor_type, num_ops = generate_relay_pattern(expr, expr)
            
            # Verify if it is legitimate
            if self.pattern_rule.verify(anchor_pattern):
                generated_patterns.append(Pattern(anchor_pattern))
           
        
            def simulate_fusion(src, sink, cur_type, num_ops, nodeToPatternMap = dict()):
                assert(src is not None)
                if is_tuple_node(sink) and (sink in post_dom_tree):
                    simulate_fusion(src, post_dom_tree[sink], cur_type, num_ops, nodeToPatternMap)
                
                ops_to_fuse =  self.pattern_rule.fusion_rule(
                        src = src, 
                        sink = sink, 
                        cur_type = cur_type, 
                        num_ops = num_ops,
                    )
                if len(ops_to_fuse) and not isinstance(src, tvm.relay.expr.TupleGetItem):
                    fusion_pattern, cur_type, num_ops = generate_relay_pattern(src, sink, ops_to_fuse, cur_type, nodeToPatternMap)
                
                    # Append identified pattern
                    if self.pattern_rule.verify(fusion_pattern):
                        generated_patterns.append(Pattern(fusion_pattern))

                    # Go deeper
                    if sink in post_dom_tree:
                        simulate_fusion(src=src, sink=post_dom_tree[sink], cur_type=cur_type, num_ops=num_ops,  nodeToPatternMap = nodeToPatternMap)
                
            
            # Run fusion simulation
            if expr in post_dom_tree:
                simulate_fusion(src=expr, sink=post_dom_tree[expr], cur_type=anchor_type, num_ops=num_ops)
                
        return generated_patterns
        

# This pattern rule should be singleton
class TVM_PatternRule(BasePatternRule):
    enum2optype = {0:"kElemWise", 1:"kBroadcast", 2:"kInjective", 3:"kCommReduce", 4:"kOutEWiseFusable", 7:"kTuple", 8:"kOpaque"}
    optype2enum = {"kElemWise":0, "kBroadcast":1, "kInjective":2, "kCommReduce":3, "kOutEWiseFusable":4, "kTuple":7, "kOpaque":8}
    MAX_NUM_OPS = 256

    __instance = None
    @staticmethod
    def destroy():
        TVM_PatternRule.__instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if TVM_PatternRule.__instance != None:
            raise Exception("This class should be a singleton!")
        TVM_PatternRule.__instance = self

    
    @staticmethod
    def op_rule(expr):
        # @Sung: nn.softmax op type seems kOpaque...? Should revisit this.
        # - update: It seems like it is fixed in the latest tvm main. 
        #           Change the pattern type accordingly, but not sure this is enough change. 
        return (get_op_pattern(expr) != TVM_PatternRule.optype2enum["kOpaque"])
    
    @staticmethod
    def fusion_rule(src, sink, cur_type, num_ops):
        enum2optype = TVM_PatternRule.enum2optype
        optype2enum = TVM_PatternRule.optype2enum
        MAX_NUM_OPS  = TVM_PatternRule.MAX_NUM_OPS

        def _check_path(src, sink, fcheck):
            def helper(src, node, fcheck, path = [], paths = []):
                path.append(node)

                if src == node:
                    assert(len(path))    
                    paths.append(path.copy())
                elif is_var_node(node) or is_constant_node(node):
                    pass
                elif fcheck(node, node==src):
                    children = []
                    if is_tuple_node(node):
                        children = node.fields
                    elif is_tuplegetitem_node(node):
                        children = [ node.tuple_value ]
                    elif is_call_node(node):
                        children = node.args
                    else:
                        raise Exception(f"Unsupported type ({type(node)})")
                
                    for child in children:
                        helper(src, child, fcheck, path, paths)
            
                out = path.pop()
                assert(node == out)
            
            path, paths = [], []
            helper(src, sink, fcheck, path, paths)
            
            return paths

        if num_ops > MAX_NUM_OPS:
            return list()

        sink_type = get_op_pattern(sink)

        if cur_type == optype2enum["kOutEWiseFusable"]:
            def fcheck(node, is_sink):
                return get_op_pattern(node) <= optype2enum["kBroadcast"]

            if sink_type <= optype2enum["kInjective"]:
                return _check_path(src, sink, fcheck)

        elif cur_type <= optype2enum["kBroadcast"]:
            def fcheck(node, is_sink):
                kind = get_op_pattern(node)
                if not is_sink:
                    return kind <= optype2enum["kInjective"]
                else:
                    return kind <= optype2enum["kOutEWiseFusable"]

            if sink_type <= optype2enum["kCommReduce"]:
                return _check_path(src, sink, fcheck)

        elif cur_type == optype2enum["kInjective"] or cur_type == optype2enum["kTuple"]:
            def fcheck(node, is_sink):
                return get_op_pattern(node) <= optype2enum["kInjective"]
            return _check_path(src, sink, fcheck)

        elif cur_type == optype2enum["kCommReduce"] or cur_type == optype2enum["kOpaque"]:
            return list()

        else:
            raise Exception(f"Unsupported type ({type(sink)}, {enum2optype[cur_type]}, {src})")
        
        return list()
             
tvm_pattern_rule = TVM_PatternRule()
tvm_pattern_generator = DefaultPatternGenerator(tvm_pattern_rule)



class TRT_PatternRule(BasePatternRule):
    enum2optype = {0:"kElemWise", 1:"kBroadcast", 2:"kInjective", 3:"kCommReduce", 4:"kOutEWiseFusable", 7:"kTuple", 8:"kOpaque"}
    optype2enum = {"kElemWise":0, "kBroadcast":1, "kInjective":2, "kCommReduce":3, "kOutEWiseFusable":4, "kTuple":7, "kOpaque":8}
    MAX_NUM_OPS = 256
    ops_to_exclude = ["image.resize"] # Not supported in TensorRT
    __instance = None
    
    @staticmethod
    def destroy():
        TRT_PatternRule.__instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if TRT_PatternRule.__instance != None:
            raise Exception("This class should be a singleton!")
        TRT_PatternRule.__instance = self

    @staticmethod
    def op_rule(expr):
        optype2enum = TRT_PatternRule.optype2enum
        ops_to_exclude = TRT_PatternRule.ops_to_exclude

        if is_call_node(expr):
            return (expr.op.name not in ops_to_exclude)

        if isinstance(expr, tvm.ir.op.Op):
            return (expr.name in ops_to_exclude)

        return (get_op_pattern(expr) != optype2enum["kOpaque"]) 
    

    # Based on TensorRT documentation
    # It seems almost same with TVM's fusion 
    # TODO: Shuffle-shuffle, shuffle-reduce patterns
    @staticmethod
    def fusion_rule(src, sink, cur_type, num_ops):
        # Borrow type definitions from TVM to ease implemenation overhead
        # Users can define their own if they want
        enum2optype = TVM_PatternRule.enum2optype
        optype2enum = TVM_PatternRule.optype2enum
        MAX_NUM_OPS  = 256

        def _check_path(src, sink, fcheck):
            def helper(src, node, fcheck, path = [], paths = []):
                path.append(node)

                if src == node:
                    assert(len(path))    
                    paths.append(path.copy())
                elif is_var_node(node) or is_constant_node(node):
                    pass
                elif fcheck(node, node==src):
                    children = []
                    if is_tuple_node(node):
                        children = node.fields
                    elif is_tuplegetitem_node(node):
                        children = [ node.tuple_value ]
                    elif is_call_node(node):
                        children = node.args
                    else:
                        raise Exception(f"Unsupported type ({type(node)})")
                
                    for child in children:
                        helper(src, child, fcheck, path, paths)
            
                out = path.pop()
                assert(node == out)
            
            path, paths = [], []
            helper(src, sink, fcheck, path, paths)
            
            return paths

        if num_ops > MAX_NUM_OPS:
            return list()

        sink_type = get_op_pattern(sink)

        if cur_type == optype2enum["kOutEWiseFusable"]:
            def fcheck(node, is_sink):
                return get_op_pattern(node) <= optype2enum["kBroadcast"]

            if sink_type <= optype2enum["kInjective"]:
                return _check_path(src, sink, fcheck)

        elif cur_type <= optype2enum["kBroadcast"]:
            def fcheck(node, is_sink):
                kind = get_op_pattern(node)
                if not is_sink:
                    return kind <= optype2enum["kInjective"]
                else:
                    return kind <= optype2enum["kOutEWiseFusable"]

            if sink_type <= optype2enum["kCommReduce"]:
                return _check_path(src, sink, fcheck)

        elif cur_type == optype2enum["kInjective"] or cur_type == optype2enum["kTuple"]:
            def fcheck(node, is_sink):
                return get_op_pattern(node) <= optype2enum["kInjective"]
            return _check_path(src, sink, fcheck)

        elif cur_type == optype2enum["kCommReduce"] or cur_type == optype2enum["kOpaque"]:
            return list()

        else:
            raise Exception(f"Unsupported type ({type(sink)}, {enum2optype[cur_type]}, {src})")
        
        return list()

    @staticmethod
    def verify(pattern):
        q = [ pattern ]
        while len(q):
            cur = q.pop()
            if isinstance(cur, WildcardPattern):
                pass
            elif isinstance(cur, CallPattern):
                if ( 
                      isinstance(cur.op, ConstantPattern) 
                      or isinstance(cur.op, VarPattern)
                      or isinstance(cur.op, WildcardPattern)
                ):
                    pass
                else:
                    op_name = cur.op.expr.name

                    if op_name in TRT_PatternRule.ops_to_exclude:
                        return False
                    q.extend(cur.args)
            elif isinstance(cur, TuplePattern):
                q.extend(cur.fields)
            elif isinstance(cur, TupleGetItemPattern):
                q.append(cur.tuple)
            else:
                raise Exception(f"Unexpected expression type, {type(cur)}")

        return True

trt_pattern_rule = TRT_PatternRule() 
trt_pattern_generator = DefaultPatternGenerator(trt_pattern_rule)
