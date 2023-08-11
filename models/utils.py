WORKLOADS_DIC = {
    "resnet_block" : {1: {"input0": [1, 64, 56, 56]},
                      8: {"input0": [8, 64, 56, 56]},
                      16: {"input0": [16, 64, 56, 56]},
                      32: {"input0": [32, 64, 56, 56]},
                      64: {"input0": [64, 64, 56, 56]},
                      },
    "resnet18" : {1: {"input0": [1, 3, 224, 224]},
                  8: {"input0": [8, 64, 56, 56]},
                  16: {"input0": [16, 64, 56, 56]},
                  32: {"input0": [32, 64, 56, 56]},
                  64: {"input0": [64, 64, 56, 56]},
                  },
    "resnet50" : {1: {"input0": [1, 64, 56, 56]},
                  8: {"input0": [8, 64, 56, 56]},
                  16: {"input0": [16, 64, 56, 56]},
                  32: {"input0": [32, 64, 56, 56]},
                  64: {"input0": [64, 64, 56, 56]},
                  },
    "resnext50_32x4d" : {1: {"input0": [1, 64, 56, 56]},
                         4: {"input0": [4, 64, 56, 56]},
                         8: {"input0": [8, 64, 56, 56]},
                         16: {"input0": [16, 64, 56, 56]},
                         32: {"input0": [32, 64, 56, 56]},
                         64: {"input0": [64, 64, 56, 56]},
                         },
    "nasneta" : {1: {"input0": [1, 64, 56, 56]},
                 8: {"input0": [8, 64, 56, 56]},
                 16: {"input0": [16, 64, 56, 56]},
                 32: {"input0": [32, 64, 56, 56]},
                 64: {"input0": [64, 64, 56, 56]},
                 },
    # NasRNN always have some errors during autotuning operators with AutoTVM
    # "nasrnn": {'x.1': [1, 512]},
    # "nasrnn": {'x.1': [1, 1024]},
    # "nasrnn": {'x.1': [1, 2048]},
    "nasrnn": {
        1: {'x.1': [1, 2560]},
        8: {'x.1': [8, 2560]},
        16: {'x.1': [16, 2560]},
        32: {'x.1': [32, 2560]},
        64: {'x.1': [64, 2560]},
               },
    # "nasrnn": {'x.1': [1, 512], 'x.2': [1, 512], 'x.3': [1, 512], 'x.4': [1, 512], 'x': [1, 512]},
    "bert": {1: {"input0": [64, 1024]}},
    "bert_full": {1: {"input0": [1, 64, 256]}, # (batch_size, max_seq_len, n_hidden)
                  8: {"input0": [8, 64, 256]},
                  16: {"input0": [16, 64, 256]},
                  32: {"input0": [32, 64, 256]},
                  64: {"input0": [64, 64, 256]},
                  },
    "resnet50_3d": {1: {"input0": [1, 64, 3, 56, 56]},
                    8: {"input0": [8, 64, 3, 56, 56]},
                    16: {"input0": [16, 64, 3, 56, 56]},
                    32: {"input0": [32, 64, 3, 56, 56]},
                    64: {"input0": [64, 64, 3, 56, 56]},

                    },
    "mobilenetV2": {1: {"input0": [1, 32, 224, 224]},
                     8: {"input0": [8, 32, 224, 224]},
                     16: {"input0": [16, 32, 224, 224]},
                     32: {"input0": [32, 32, 224, 224]},
                     64: {"input0": [64, 32, 224, 224]},
                     
                     },
    # "mobilenetV2": {"input0": [1, 32, 56, 56]},
    "dcgan": {1: {"input0": [1, 100]},
              8: {"input0": [8, 100]},
              16: {"input0": [16, 100]},
              32: {"input0": [32, 100]},
              64: {"input0": [64, 100]},
              },
    "yolov3": {
        1: {"input0": [1,3,416,416]},
        8: {"input0": [8,3,416,416]},
        16: {"input0": [16,3,416,416]},
        32: {"input0": [32,3,416,416]},
        64: {"input0": [64,3,416,416]},
               
               },
    "gpt2":{1: {"input0":[1, 1024]}}, # 1 means # of sentences / 1024 means # of words.

    
    "squeezenet" : {1: {"input0": [1, 3, 224, 224]},
                  8: {"input0": [8, 3, 224, 224]},
                  16: {"input0": [16, 3, 224, 224]}},
    "vit" : {1: {"input0": [1, 3, 224, 224]},
                  8: {"input0": [8, 3, 224, 224]},
                  16: {"input0": [16, 3, 224, 224]}},
    "shufflenetV2" : {1: {"input0": [1, 3, 224, 224]},
                  8: {"input0": [8, 3, 224, 224]},
                  16: {"input0": [16, 3, 224, 224]}},
}

def get_shape_arr(model, batch_size):
    shape_dict = WORKLOADS_DIC[model][batch_size]
    shape_arr = list(shape_dict.items())
    return shape_arr[0][1]

def get_shape_list(model, batch_size):
    shape_dict = WORKLOADS_DIC[model][batch_size]
    shape_arr = list(shape_dict.items())
    return shape_arr