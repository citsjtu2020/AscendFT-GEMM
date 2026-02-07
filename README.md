# AscendFT-GEMM
AscendFT-GEMM is a high-performance fault-tolerant GEMM design for Ascend NPUs

AscendFT-GEMM are coding based on the CATLASS, a leading open-source operator template library for Ascend NPUs. We completely implement the non-fault-tolerant GEMM and AscendFT-GEMM by ourselves but leveraging the template to standardize the code style. 

The core of our code to realize original GEMM and fault-tolerant GEMM can be found at the path: example/cube_op_self/gemm 

(The core block level code for original GEMM: examples/cube_op_self/gemm/block/block_mmad_pingpong_preload.hpp)

(The core block level code for fault-tolerant AscendFT-GEMM: examples/cube_op_self/gemm/block/block_mmad_pingpong_fault_abe_spec_no_splitk_robust.hpp)
