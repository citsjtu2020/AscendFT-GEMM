# AscendFT-GEMM
AscendFT-GEMM is a high-performance fault-tolerant GEMM design for Ascend NPUs

AscendFT-GEMM are coding based on the CATLASS, a leading open-source operator template library for Ascend NPUs. We completely implement the non-fault-tolerant GEMM and AscendFT-GEMM by ourselves but leveraging the template to standardize the code style. 

The core of our code to realize baseline GEMM and fault-tolerant GEMM can be found at the path: example/cube_op_self/gemm 

(The core block level code for baseline GEMM: examples/cube_op_self/gemm/block/block_mmad_pingpong_preload.hpp; Thre core kernel level code for baselien GEMM: examples/cube_op_self/gemm/kernel/matmul_epilogue_preload.hpp)


(The core block level code for fault-tolerant AscendFT-GEMM: examples/cube_op_self/gemm/block/block_mmad_pingpong_fault_abe_spec_no_splitk_robust.hpp)

## Running examples of Ascend-GEMM

### 0. Install CANN Environment
#### 0.1) prepare the hardware
Currently, our work supports the Ascend 910B/910C NPU devices on the Atlas A2/A3 Servers. Please run our code on these servers which have installed the npu-driver and npu-firmware environments.
#### 0.2) install the software environment
##### 0.2.1) Download the packages
Our work is recommended running at the CANN 8.2.1 environments. Please download the source at [Ascend resouce download center](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1.alpha003), and find the toolkit and kernel package as follows:
<div align="center">
  <img src="./docs/images/software_environment.png" alt="soft_source">
  <br>
  <em>Figure 1: Required Package In the List</em>
</div>
After that, please upload the packages to the specific path of the server (e.g., ${HOME})

##### 0.2.2) Increase the execution permissions for the software package

chmod +x Ascend-cann-toolkit_8.3.RC1.alpha003_linux-aarch64.run

chmod a+x Ascend-cann-kernels-910b_8.2.RC1.alpha003_linux-aarch64.run

##### 0.2.3) Install the packages:

./Ascend-cann-toolkit_8.3.RC1.alpha003_linux-aarch64.run --install

./Ascend-cann-kernels-910b_8.2.RC1.alpha003_linux-aarch64.run --install


##### 0.2.4) Add the configuration to the PATH:
The configuration script for environment variables is named "set_env.sh". The current installation path is exemplified as ${HOME}/Ascend:

###### a) Add the configuration to the ${HOME}/.bashrc:

export ASCEND_HOME_PATH=${HOME}/Ascend/ascend-toolkit/latest

export ASCEND_CANN_PACKAGE_PATH=${HOME}/Ascend/latest

export ASCEND_INSTALL_PATH=${HOME}/Ascend/ascend-toolkit/latest

export ASCEND_HOME_DIR=${HOME}/Ascend/ascend-toolkit/latest

source ${HOME}/Ascend/ascend-toolkit/set_env.sh

###### b) Make the configuration take effect:

source .bashrc (or just: source ${HOME}/Ascend/ascend-toolkit/set_env.sh)


### 1. Run the code for FP32 precision:

#### 1.1 Compiling Code

After clone the code to the server, e.g., ${HOME} in this example:

a) enter the path: ${HOME}/AscendFT-GEMM/

b) Build the executable file:

bash scripts/build.sh 03_matmul_add_self_preload_fp32

#### 1.2 Run Code

a) Params: 03_matmul_add m n k [device_id, make_golden]
| Item | Discription | Rquired/Optional|
|------|------------|------------|
| m | outer row dimension size|Rquired|
| n | outer column dimension size|Required|
| k | reduction dimension size |Rquired|
| device_id | ID of NPU card,0~7 |Optional|
| make_golden | 1: completely validate the correctness of each element; 0: validate the correctness of row checksum |Optional|










