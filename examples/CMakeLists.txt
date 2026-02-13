# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(CATLASS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(CATLASS_EXAMPLES_CUBESELF_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cube_op_self)

add_custom_target(catlass_examples)

macro(catlass_example_add_executable NAME AICORE_ARCH)
    set(EXAMPLE_DESTINATION bin)

    if(DEFINED ENABLE_SIMULATOR AND ENABLE_SIMULATOR)
        set(EXAMPLE_DESTINATION bin_sim)
    endif()

    # $<$<COMPILE_LANGUAGE:ASCEND>:-DASCENDC_TRACE_ON>
    add_executable(${NAME} ${ARGN})
    target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:ASCEND>:--cce-aicore-arch=${AICORE_ARCH}>)
    target_link_libraries(${NAME} PRIVATE
                      ${ASCEND_HOME_PATH}/lib64/libascendcl.so
                      ${ASCEND_HOME_PATH}/lib64/libnnopbase.so
                      ${ASCEND_HOME_PATH}/lib64/libopapi.so)
    add_dependencies(catlass_examples ${NAME})
    install(TARGETS ${NAME} DESTINATION ${EXAMPLE_DESTINATION} COMPONENT ${NAME})
    install(TARGETS ${NAME} DESTINATION ${EXAMPLE_DESTINATION} COMPONENT catlass_examples)
endmacro()

# $<$<COMPILE_LANGUAGE:ASCEND>:-DASCENDC_TRACE_ON>
add_compile_options(
    $<$<COMPILE_LANGUAGE:ASCEND>:-Wno-macro-redefined>
    $<$<COMPILE_LANGUAGE:ASCEND>:-Wno-ignored-attributes>
    # $<$<COMPILE_LANGUAGE:ASCEND>:"-Xaicore-start -O3 -Xaicore-end"

    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-stack-size=0x8000>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-function-stack-size=0x8000>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-record-overflow=true>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-addr-transform>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-mllvm -cce-aicore-dcci-insert-for-scalar=false>"
    # "SHELL:$<$<COMPILE_LANGUAGE:ASCEND>:-Xaicore-start -g -Xaicore-end"
    
)

# ascendc_compile_definitions(
#     ...
#     -DASCENDC_TRACE_ON
# )

set(INCLUDE_ACLNN_BASE_ROOT_DIR "${ASCEND_HOME_PATH}/include")
include_directories(
    ${CATLASS_EXAMPLES_COMMON_SOURCE_DIR}
    ${CATLASS_EXAMPLES_CUBESELF_SOURCE_DIR}
    ${CATLASS_INCLUDE_DIR}
    ${INCLUDE_ACLNN_BASE_ROOT_DIR}/aclnn
)
link_directories(${ASCEND_HOME_PATH}/lib64)
link_libraries(dl tiling_api platform c_sec nnopbase ascendcl m)

if(DEFINED ENABLE_ASCENDC_DUMP AND ENABLE_ASCENDC_DUMP)
    add_compile_definitions(ENABLE_ASCENDC_DUMP)
    add_compile_definitions(ASCENDC_DUMP=1)
    add_compile_definitions(ASCENDC_DEBUG)
    link_libraries(ascend_dump)
endif()

foreach(EXAMPLE
     03_matmul_add_self_preload_fp16
     03_matmul_add_self_preload_bf16
     03_matmul_add_self_preload_fp32
    18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_fp16_small_simple_inited
    18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_fp16_medium_simple_inited
    18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_fp16_medium_robust_inited
    18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_fp16_large_robust_inited
    18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_bf16_small_simple_inited
    18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_bf16_small_robust_inited
    18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_bf16_medium_robust_inited
    18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_bf16_medium_simple_inited
    18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_bf16_large_robust_inited
    18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_fp32_large_robust_inited
    18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_fp32_medium_robust_inited
    18_matmul_ft_abe_aic_thre_no_splitk_asvar_ft_spec_fp32_small_simple_inited
    )
    add_subdirectory(${EXAMPLE})
endforeach()

add_subdirectory(shared_lib)
add_subdirectory(python_extension)
