/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP
#define EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP

#include <cmath>
#include <vector>
#include <string>
#include <cstdio>

#include "catlass/gemv_coord.hpp"
#include "catlass/gemm_coord.hpp"

namespace Catlass::golden {

template<class ElementResult, class ElementCompare>
std::vector<uint64_t> CompareData(const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect,
    uint32_t computeNum)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = static_cast<ElementCompare>(std::fabs(static_cast<float>(actualValue) - static_cast<float>(expectValue)));
        ElementCompare threshold = 0;
        threshold = static_cast<ElementCompare>(rtol * (std::max(1.0f, std::fabs(static_cast<float>(expectValue)))));
        if (diff > threshold) {
            errorIndices.push_back(i);
            std::cout << "Error at index " << i << ": "
                      << "actual = " << actualValue << ", expect = " << expectValue
                      << ", diff = " << diff << std::endl;
        }
    }
    return errorIndices;
}

template<>
std::vector<uint64_t> CompareData(const std::vector<half>& result, const std::vector<half>& expect,
    uint32_t computeNum)
{
    // class ElementResult, class ElementCompare
    using ElementCompare = half;
    using ElementResult = half;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = static_cast<ElementCompare>(std::fabs(static_cast<float>(actualValue) - static_cast<float>(expectValue)));
        ElementCompare threshold = 0;
        threshold = static_cast<ElementCompare>(rtol * (std::max(1.0f, std::fabs(static_cast<float>(expectValue)))));
        if (diff > threshold) {
            errorIndices.push_back(i);
            std::cout << "Error at index " << i << ": "
                      << "actual = " << (float)actualValue << ", expect = " << (float)expectValue
                      << ", diff = " << (float)diff << std::endl;
        }
    }
    return errorIndices;
}

template<>
std::vector<uint64_t> CompareData(const std::vector<uint8_t>& result, const std::vector<uint8_t>& expect,
    uint32_t computeNum)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;
        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
        }
    }
    return errorIndices;
}

template<class ElementIndex>
void ComputeErrorIndex(uint64_t basic_id, ElementIndex raw_index,
    std::vector<uint64_t> & error_indies)
{
    uint64_t bit_size = sizeof(ElementIndex) * 8;
    uint64_t start_base = basic_id * bit_size;

    error_indies.clear();
    std::cout<<"Bit Test: ";
    for (int i = 0; i < 8; i++) {
        std::cout << ((raw_index >> i) & 1);
        if (i < 7) std::cout<<" ";
        if ((raw_index & (1 << i)) == 0) {
            error_indies.push_back((uint64_t)i + start_base);
        }
    }
    std::cout<<std::endl;
}

template<class ElementIndex>
void GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
    std::vector<uint64_t> & error_indies,bool do_print)
{
    uint64_t bit_size = sizeof(ElementIndex) * 8;
    uint64_t start_base = basic_id * bit_size;

    error_indies.clear();
    if(do_print){
        std::cout<<"Bit Test: ";
    }
    
    for (int i = 0; i < 8; i++) {
        
        if(do_print){
            std::cout << ((raw_index >> i) & 1);
            if (i < 7) std::cout<<" ";
        }
        
        if ((raw_index & (1 << i)) == 0) {
            error_indies.push_back((uint64_t)i + start_base);
        }
    }
    if(do_print){
        std::cout<<std::endl;
    }
}

template<class ElementData>
std::vector<uint64_t> CompareDataWithIndex(
    const std::vector<uint8_t>& result, const std::vector<uint8_t>& expect,
    const std::vector<ElementData> &actualdata, const std::vector<ElementData> &expectdata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = sizeof(uint8_t) * 8;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    std::vector<uint64_t> unit_error_indices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
            
            start_base = i*bit_size;
            end_base = start_base + bit_size - 1;
            std::cout<<IdNameAct<<" ("<<start_base<<"~"<<end_base<<"): ";
            for(uint64_t idx = start_base; idx<=end_base; idx++){
                printf("%f ",actualdata[idx]);
            }
            printf("\n");
            std::cout<<IdNameExp<<"("<<start_base<<"~"<<end_base<<"): ";
            for(uint64_t idx = start_base; idx<=end_base; idx++){
                printf("%f ",expectdata[idx]);
            }
            printf("\n");
            ComputeErrorIndex<ElementCompare>(i, actualValue, unit_error_indices);
            printf("Detected Index: ");
            for(int32_t j = 0; j < unit_error_indices.size(); j++){
                std::cout<<unit_error_indices[j];
                if(j < unit_error_indices.size() - 1){
                    std::cout<<" ";
                }
            }
            printf("\n");
        }
    }
    return errorIndices;
}


template<class ElementData>
std::vector<uint64_t> GetErrorDataWithIndexTotal(
    const std::vector<uint8_t>& result, const std::vector<uint8_t>& expect,
    const std::vector<ElementData> &actualdata, const std::vector<ElementData> &expectdata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies, std::vector<ElementData>& total_error_data 
)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = sizeof(uint8_t) * 8;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    std::vector<uint64_t> unit_error_indices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            if(errorIndices.size() < 2){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
            }
            
            start_base = i*bit_size;
            end_base = start_base + bit_size - 1;
            if(errorIndices.size() < 2){
                std::cout<<IdNameAct<<" ("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if(errorIndices.size() < 2){
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                printf("Detected Index: ");
                for(int32_t j = 0; j < unit_error_indices.size(); j++){
                    std::cout<<unit_error_indices[j];
                    if(j < unit_error_indices.size() - 1){
                        std::cout<<" ";
                    }
                }
                printf("\n");
            }else{
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }
            total_error_idies.insert(total_error_idies.end(), unit_error_indices.begin(), unit_error_indices.end()); 
            for(int32_t j = 0; j < unit_error_indices.size(); j++){
                total_error_data.push_back(actualdata[unit_error_indices[j]]);
            }
        }
    }
    return errorIndices;
}

template<class ElementData>
std::vector<uint64_t> GetErrorDataAndIndexTotalWithThreshold(
    const std::vector<uint8_t>& result, const std::vector<uint8_t>& expect,
    const std::vector<ElementData> &actualdata, 
    const std::vector<ElementData> &expectdata,
    const std::vector<ElementData> &thresholddata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies, 
    std::vector<ElementData>& total_error_data,
    std::vector<ElementData>& total_fail_threshold_data 
)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = sizeof(uint8_t) * 8;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    std::vector<uint64_t> unit_error_indices;
    uint64_t show_cases = 0;
    for (uint64_t i = 0; i < result.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            if(errorIndices.size() < 8){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
            }
            
            start_base = i*bit_size;
            end_base = start_base + bit_size - 1;
            if(errorIndices.size() < 8){
                std::cout<<IdNameAct<<" ("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
                std::cout<<"Threshold"<<"("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if(errorIndices.size() < 8){
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                printf("Detected Index: ");
                for(int32_t j = 0; j < unit_error_indices.size(); j++){
                    std::cout<<unit_error_indices[j];
                    if(j < unit_error_indices.size() - 1){
                        std::cout<<" ";
                    }
                }
                printf("\n");
            }else{
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }
            total_error_idies.insert(total_error_idies.end(), unit_error_indices.begin(), unit_error_indices.end()); 
            for(int32_t j = 0; j < unit_error_indices.size(); j++){
                total_error_data.push_back(actualdata[unit_error_indices[j]]);
                total_fail_threshold_data.push_back(thresholddata[unit_error_indices[j]]);
            }
        }
        else{
            if(show_cases < 1){
                std::cout << "Correct at Index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
                start_base = i*bit_size;
                end_base = start_base + bit_size - 1;
                std::cout<<IdNameAct<<" ("<<start_base<<"~"<<end_base<<"): ";
                
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }

                printf("\n");
                
                std::cout<<IdNameExp<<"("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                
                printf("\n");
                
                std::cout<<"Threshold"<<"("<<start_base<<"~"<<end_base<<"): ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
                std::cout<<"Bit Test: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%d ", 1);
                }
                printf("\n");
            }
            show_cases += 1;
        }
    }
    return errorIndices;
}


template<class ElementData>
std::vector<uint64_t> GetErrorDataAndIndexSliceWithThreshold(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint8_t>& result, 
    const std::vector<uint8_t>& expect,
    const std::vector<ElementData> &actualdata, 
    const std::vector<ElementData> &expectdata,
    const std::vector<ElementData> &thresholddata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies,
    std::vector<uint64_t>& total_error_idies_m,
    std::vector<uint64_t>& total_error_idies_n, 
    std::vector<ElementData>& total_error_data,
    std::vector<ElementData>& total_fail_threshold_data)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    uint32_t slice_row_size = problemShape.m();

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = sizeof(uint8_t) * 8;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    uint64_t start_row_base = 0;
    uint64_t start_slice_col_base = 0;
    uint64_t end_row_base = 0;
    uint64_t end_slice_col_base = 0;
    std::vector<uint64_t> unit_error_indices;
    uint64_t show_cases = 0;
    for (uint64_t i = 0; i < result.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            if(errorIndices.size() < 8){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
            }
            
            start_base = i*bit_size;
            end_base = start_base + bit_size - 1;

            start_row_base = start_base % slice_row_size;
            start_slice_col_base = start_base / slice_row_size;

            end_row_base = end_base % slice_row_size;
            end_slice_col_base = end_base / slice_row_size;

            if(errorIndices.size() < 8){
                std::cout<<IdNameAct<<"["<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"["<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
                std::cout<<"Threshold"<<"["<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if(errorIndices.size() < 8){
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                printf("Detected Index: ");

                for(int32_t j = 0; j < unit_error_indices.size(); j++){
                    uint32_t tmp_err_index = unit_error_indices[j];
                    uint32_t tmp_err_row_idx = tmp_err_index % slice_row_size;
                    uint32_t tmp_err_col_idx = tmp_err_index / slice_row_size;
                    std::cout<<"("<<tmp_err_row_idx<<","<<tmp_err_col_idx<<")";
                    if(j < unit_error_indices.size() - 1){
                        std::cout<<" ";
                    }
                }
                printf("\n");
            }else{
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }
            total_error_idies.insert(total_error_idies.end(), unit_error_indices.begin(), unit_error_indices.end()); 
            for(int32_t j = 0; j < unit_error_indices.size(); j++){

                uint32_t tmp_err_index = unit_error_indices[j];
                uint32_t tmp_err_row_idx = tmp_err_index % slice_row_size;
                uint32_t tmp_err_col_idx = tmp_err_index / slice_row_size;

                total_error_idies_m.push_back(tmp_err_row_idx);
                total_error_idies_n.push_back(tmp_err_col_idx);

                total_error_data.push_back(actualdata[unit_error_indices[j]]);
                total_fail_threshold_data.push_back(thresholddata[unit_error_indices[j]]);
            }
        }
        else{
            if(show_cases < 1){
                std::cout << "Correct at Index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
                start_base = i*bit_size;
                end_base = start_base + bit_size - 1;

                start_row_base = start_base % slice_row_size;
                start_slice_col_base = start_base / slice_row_size;

                end_row_base = end_base % slice_row_size;
                end_slice_col_base = end_base / slice_row_size;

                std::cout<<IdNameAct<<"["<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_col_base<<")]: ";
                
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }

                printf("\n");
                
                std::cout<<IdNameExp<<"["<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                
                printf("\n");
                
                std::cout<<"Threshold"<<"["<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
                std::cout<<"Bit Test: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%d ", 1);
                }
                printf("\n");
            }
            show_cases += 1;
        }
    }
    return errorIndices;
}

template<class ElementData>
std::vector<uint64_t> GetErrorDataAndIndexSliceSplitKWithThreshold(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitNnum,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<uint8_t>& result, 
    const std::vector<uint8_t>& expect,
    const std::vector<ElementData> &actualdata, 
    const std::vector<ElementData> &expectdata,
    const std::vector<ElementData> &thresholddata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies,
    std::vector<uint64_t>& total_error_idies_m,
    std::vector<uint64_t>& total_error_idies_n, 
    std::vector<uint64_t>& total_error_idies_k,
    std::vector<ElementData>& total_error_data,
    std::vector<ElementData>& total_fail_threshold_data)
{
    using ElementCompare = uint8_t;
    using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    uint32_t slice_row_size = problemShape.m();
    uint32_t slice_total_size = problemShape.m() * SplitNnum;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = sizeof(uint8_t) * 8;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    uint64_t start_KSlice_base = 0;
    uint64_t end_KSlice_base = 0;
    uint64_t start_base_inSlice = 0;
    uint64_t end_base_inSlice = 0; 
    uint64_t start_row_base = 0;
    uint64_t start_slice_col_base = 0;
    uint64_t end_row_base = 0;
    uint64_t end_slice_col_base = 0;
    std::vector<uint64_t> unit_error_indices;
    uint64_t show_cases = 0;
    for (uint64_t i = 0; i < result.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            if(errorIndices.size() < 8){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
            }
            
            start_base = i * bit_size;
            end_base = start_base + bit_size - 1;

            start_KSlice_base = start_base  / slice_total_size;
            start_base_inSlice = start_base % slice_total_size;

            start_row_base = start_base_inSlice % slice_row_size;
            start_slice_col_base = start_base_inSlice / slice_row_size;

            end_KSlice_base = end_base  / slice_total_size;
            end_base_inSlice = end_base % slice_total_size;

            end_row_base = end_base_inSlice % slice_row_size;
            end_slice_col_base = end_base_inSlice / slice_row_size;

            if(errorIndices.size() < 8){
                std::cout<<IdNameAct<<"["<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_KSlice_base<<","<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"["<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_KSlice_base<<","<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
                std::cout<<"Threshold"<<"["<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_KSlice_base<<","<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if(errorIndices.size() < 8){
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                printf("Detected Index: ");

                for(int32_t j = 0; j < unit_error_indices.size(); j++){
                    uint32_t tmp_err_index = unit_error_indices[j];
                    uint32_t tmp_err_kslice_idx = tmp_err_index / slice_total_size;
                    uint32_t tmp_err_idx_mn = tmp_err_index % slice_total_size;

                    uint32_t tmp_err_row_idx = tmp_err_idx_mn % slice_row_size;
                    uint32_t tmp_err_col_idx = tmp_err_idx_mn / slice_row_size;
                    std::cout<<"("<<tmp_err_kslice_idx<<","<<tmp_err_row_idx<<","<<tmp_err_col_idx<<")";
                    if(j < unit_error_indices.size() - 1){
                        std::cout<<" ";
                    }
                }
                printf("\n");
            }else{
                GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }

            total_error_idies.insert(total_error_idies.end(), unit_error_indices.begin(), unit_error_indices.end()); 
            for(int32_t j = 0; j < unit_error_indices.size(); j++){

                uint32_t tmp_err_index = unit_error_indices[j];
                uint32_t tmp_err_kslice_idx = tmp_err_index / slice_total_size;
                uint32_t tmp_err_idx_mn = tmp_err_index % slice_total_size;

                uint32_t tmp_err_row_idx = tmp_err_idx_mn % slice_row_size;
                uint32_t tmp_err_col_idx = tmp_err_idx_mn / slice_row_size;

                total_error_idies_m.push_back(tmp_err_row_idx);
                total_error_idies_n.push_back(tmp_err_col_idx);
                total_error_idies_k.push_back(tmp_err_kslice_idx);

                total_error_data.push_back(actualdata[unit_error_indices[j]]);
                total_fail_threshold_data.push_back(thresholddata[unit_error_indices[j]]);
            }
        }
        else{
            if(show_cases < 1){
                std::cout << "Correct at Index " << i << ": "
                      << "actual = " << static_cast<unsigned int>(actualValue) << ", expect = " << static_cast<unsigned int>(expectValue)
                      << ", diff = " << static_cast<unsigned int>(diff) << std::endl;
                

                end_row_base = end_base % slice_row_size;
                end_slice_col_base = end_base / slice_row_size;

                start_base = i*bit_size;
                end_base = start_base + bit_size - 1;

                start_KSlice_base = start_base  / slice_total_size;
                start_base_inSlice = start_base % slice_total_size;

                start_row_base = start_base_inSlice % slice_row_size;
                start_slice_col_base = start_base_inSlice / slice_row_size;

                end_KSlice_base = end_base  / slice_total_size;
                end_base_inSlice = end_base % slice_total_size;

                end_row_base = end_base_inSlice % slice_row_size;
                end_slice_col_base = end_base_inSlice / slice_row_size;

                std::cout<<IdNameAct<<"["<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_KSlice_base<<","<<end_row_base<<","<<end_slice_col_base<<")]: ";
                
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }

                printf("\n");
                
                std::cout<<IdNameExp<<"["<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_KSlice_base<<","<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                
                printf("\n");
                
                std::cout<<"Threshold"<<"["<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<"~("<<end_KSlice_base<<","<<end_row_base<<","<<end_slice_col_base<<")]: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
                std::cout<<"Bit Test: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%d ", 1);
                }
                printf("\n");
            }
            show_cases += 1;
        }
    }
    return errorIndices;
}


template<class ElementData>
std::vector<uint64_t> CompareDataAndIndexSliceWithThreshold(
    const Catlass::GemvCoord &problemShape,
    const std::vector<ElementData> &actualdata, 
    const std::vector<ElementData> &expectdata,
    const std::vector<ElementData> &thresholddata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies,
    std::vector<uint64_t>& total_error_idies_m,
    std::vector<uint64_t>& total_error_idies_n, 
    std::vector<ElementData>& total_error_data,
    std::vector<ElementData>& total_fail_threshold_data)
{
    // using ElementCompare = uint8_t;
    // using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    uint32_t slice_row_size = problemShape.m();

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = 1;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    uint64_t start_row_base = 0;
    uint64_t start_slice_col_base = 0;
    uint64_t end_row_base = 0;
    uint64_t end_slice_col_base = 0;
    std::vector<uint64_t> unit_error_indices;
    uint64_t show_cases = 0;
    for (uint64_t i = 0; i < expectdata.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementData actualValue = (actualdata[i]);
        ElementData expectValue = expectdata[i];
        ElementData diff = std::fabs(actualValue - expectValue);
        ElementData tmp_threshold = thresholddata[i];

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > tmp_threshold) {
            errorIndices.push_back(i);
            if(errorIndices.size() < 8){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << (actualValue) << ", expect = " << (expectValue)
                      << ", diff = " << (diff) << ", threshold = "<<(tmp_threshold)<< std::endl;
            }
            
            start_base = i;
            end_base = start_base + bit_size - 1;

            start_row_base = start_base % slice_row_size;
            start_slice_col_base = start_base / slice_row_size;

            end_row_base = end_base % slice_row_size;
            end_slice_col_base = end_base / slice_row_size;

            if(errorIndices.size() < 8){
                std::cout<<IdNameAct<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
                std::cout<<"Threshold"<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if(errorIndices.size() < 8){
                // GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                // printf("Detected Index: ");
                std::cout<<"Detected Index: "<<i<<"("<<start_row_base<<","<<start_slice_col_base<<")";
                printf("\n");
            }else{
                // GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }
            total_error_idies.push_back(i);
            total_error_idies_m.push_back(start_row_base);
            total_error_idies_n.push_back(start_slice_col_base);

            total_error_data.push_back(actualdata[i]);
            total_fail_threshold_data.push_back(thresholddata[i]);
        }
        else{
            if(show_cases < 1){
                std::cout << "Correct at Index " << i << ": "
                      << "actual = " << (actualValue) << ", expect = " << (expectValue)
                      << ", diff = " << (diff)<< ", threshold = " <<(tmp_threshold) << std::endl;
                start_base = i*bit_size;
                end_base = start_base + bit_size - 1;

                start_row_base = start_base % slice_row_size;
                start_slice_col_base = start_base / slice_row_size;

                end_row_base = end_base % slice_row_size;
                end_slice_col_base = end_base / slice_row_size;

                std::cout<<IdNameAct<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }

                printf("\n");
                
                std::cout<<IdNameExp<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                
                printf("\n");
                
                std::cout<<"Threshold"<<"("<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
                std::cout<<"Bit Test: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%d ", 1);
                }
                printf("\n");
            }
            show_cases += 1;
        }
    }
    return errorIndices;
}

template<class ElementData>
std::vector<uint64_t> CompareDataAndIndexSliceSplitKWithThreshold(
    const Catlass::GemvCoord &problemShape,
    const std::vector<uint32_t> &actualKSliceSize,
    uint32_t SplitNnum,
    uint32_t SplitKNum, uint32_t HeadKSliceNum,
    const std::vector<ElementData> &actualdata, 
    const std::vector<ElementData> &expectdata,
    const std::vector<ElementData> &thresholddata, 
    uint32_t computeNum, const char* IdNameAct, const char* IdNameExp,
    std::vector<uint64_t>& total_error_idies,
    std::vector<uint64_t>& total_error_idies_m,
    std::vector<uint64_t>& total_error_idies_n,
    std::vector<uint64_t>& total_error_idies_k, 
    std::vector<ElementData>& total_error_data,
    std::vector<ElementData>& total_fail_threshold_data)
{
    // using ElementCompare = uint8_t;
    // using ElementResult = uint8_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    uint32_t slice_row_size = problemShape.m();
    uint32_t slice_total_size = problemShape.m() * SplitNnum;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    uint64_t bit_size = 1;
    uint64_t start_base = 0;
    uint64_t end_base = 0;
    uint64_t start_KSlice_base = 0;
    uint64_t end_KSlice_base = 0;
    uint64_t start_base_inSlice = 0;
    uint64_t end_base_inSlice = 0; 
    uint64_t start_row_base = 0;
    uint64_t start_slice_col_base = 0;
    uint64_t end_row_base = 0;
    uint64_t end_slice_col_base = 0;
    std::vector<uint64_t> unit_error_indices;
    uint64_t show_cases = 0;
    for (uint64_t i = 0; i < expectdata.size(); ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementData actualValue = (actualdata[i]);
        ElementData expectValue = expectdata[i];
        ElementData diff = std::fabs(actualValue - expectValue);
        ElementData tmp_threshold = thresholddata[i];

        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > tmp_threshold) {
            errorIndices.push_back(i);
            if(errorIndices.size() < 8){
                std::cout << "Error at index " << i << ": "
                      << "actual = " << (actualValue) << ", expect = " << (expectValue)
                      << ", diff = " << (diff) << ", threshold = "<<(tmp_threshold)<< std::endl;
            }

            start_base = i;
            end_base = start_base + bit_size - 1;

            start_KSlice_base = start_base  / slice_total_size;
            start_base_inSlice = start_base % slice_total_size;

            start_row_base = start_base_inSlice % slice_row_size;
            start_slice_col_base = start_base_inSlice / slice_row_size;

            end_KSlice_base = end_base  / slice_total_size;
            end_base_inSlice = end_base % slice_total_size;

            end_row_base = end_base_inSlice % slice_row_size;
            end_slice_col_base = end_base_inSlice / slice_row_size;

            if(errorIndices.size() < 8){
                std::cout<<IdNameAct<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }
                printf("\n");
                std::cout<<IdNameExp<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                printf("\n");
                std::cout<<"Threshold"<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
            }
            /*
            GetErrorIndexPart(uint64_t basic_id, ElementIndex raw_index,
            std::vector<uint64_t> & error_indies,bool do_print)
            */
            if(errorIndices.size() < 8){
                // GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,true);
                // printf("Detected Index: ");
                std::cout<<"Detected Index: "<<i<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")";
                printf("\n");
            }else{
                // GetErrorIndexPart<ElementCompare>(i, actualValue, unit_error_indices,false);
            }
            total_error_idies.push_back(i);
            total_error_idies_m.push_back(start_row_base);
            total_error_idies_n.push_back(start_slice_col_base);
            total_error_idies_k.push_back(start_KSlice_base);

            total_error_data.push_back(actualdata[i]);
            total_fail_threshold_data.push_back(thresholddata[i]);
        }
        else{
            if(show_cases < 1){
                std::cout << "Correct at Index " << i << ": "
                      << "actual = " << (actualValue) << ", expect = " << (expectValue)
                      << ", diff = " << (diff)<< ", threshold = " <<(tmp_threshold) << std::endl;

                start_base = i*bit_size;
                end_base = start_base + bit_size - 1;

                start_KSlice_base = start_base  / slice_total_size;
                start_base_inSlice = start_base % slice_total_size;

                start_row_base = start_base_inSlice % slice_row_size;
                start_slice_col_base = start_base_inSlice / slice_row_size;

                end_KSlice_base = end_base  / slice_total_size;
                end_base_inSlice = end_base % slice_total_size;

                end_row_base = end_base_inSlice % slice_row_size;
                end_slice_col_base = end_base_inSlice / slice_row_size;

                std::cout<<IdNameAct<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",actualdata[idx]);
                }

                printf("\n");
                
                std::cout<<IdNameExp<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ",expectdata[idx]);
                }
                
                printf("\n");
                
                std::cout<<"Threshold"<<"("<<start_KSlice_base<<","<<start_row_base<<","<<start_slice_col_base<<")"<<": ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%f ", thresholddata[idx]);
                }
                printf("\n");
                std::cout<<"Bit Test: ";
                for(uint64_t idx = start_base; idx<=end_base; idx++){
                    printf("%d ", 1);
                }
                printf("\n");
            }
            show_cases += 1;
        }
    }
    return errorIndices;
}


template<>
std::vector<uint64_t> CompareData(const std::vector<int32_t>& result, const std::vector<int32_t>& expect,
    uint32_t computeNum)
{
    using ElementCompare = int32_t;
    using ElementResult = int32_t;
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        if(errorIndices.size() >= 1000) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        ElementCompare threshold = 0;
        // threshold = rtol * std::max(1.0f, std::fabs(expectValue));
        if (diff > threshold) {
            errorIndices.push_back(i);
            std::cout << "Error at index " << i << ": "
                      << "actual = " << actualValue << ", expect = " << expectValue
                      << ", diff = " << diff << std::endl;
        }
    }
    return errorIndices;
}

// Compare for GroupedMatmul slicing M
template<class ElementResult, class ElementCompare>
std::vector<uint64_t> CompareData(const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect,
    uint32_t computeNum, uint32_t validNum)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < validNum; ++i) {
        // if(errorIndices.size() >= 64) break;
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
            errorIndices.push_back(i);
            std::cout << "Error at index " << i << ": "
                      << "actual = " << actualValue << ", expect = " << expectValue
                      << ", diff = " << diff << std::endl;
        }
    }
    return errorIndices;
}

// Compare for GroupedMatmul slicing K
template<class ElementResult, class ElementCompare, class T>
std::vector<uint64_t> CompareData(const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect,
    uint32_t computeNum, const std::vector<T>& groupList, uint32_t stride)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    T prevGroupValue = 0;
    uint64_t currentIndex = 0;
    for (const auto& groupValue : groupList) {
        // if(errorIndices.size() >= 64) break;
        if (groupValue == prevGroupValue) {
            currentIndex += stride;
            prevGroupValue = groupValue;
            continue;
        }
        for (uint64_t i = 0; i < stride; ++i) {
            if (currentIndex >= result.size()) break;
            // if(errorIndices.size() >= 64) break;
            ElementCompare actualValue = static_cast<ElementCompare>(result[currentIndex]);
            ElementCompare expectValue = expect[currentIndex];
            ElementCompare diff = std::fabs(actualValue - expectValue);
            if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
                errorIndices.push_back(i);
                std::cout << "Error at index " << i << ": "
                        << "actual = " << actualValue << ", expect = " << expectValue
                        << ", diff = " << diff << std::endl;
            }
            currentIndex++;
        }
        prevGroupValue = groupValue;
    }
    return errorIndices;
}

}  // namespace Catlass::golden

#endif  // EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP
