import torch as th
import torch.utils.dlpack
import graphpy as gpk
def gp_spmm(input1, input2, input3, input4, dim1_0, dim1_1, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    input3_dl = th.utils.dlpack.to_dlpack(input3)
    input4_dl = th.utils.dlpack.to_dlpack(input4)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.spmm(input1_dl, input2_dl, input3_dl, input4_dl, res_dl1)
    return res1
