#ifndef APT_SDDMM_H_
#define APT_SDDMM_H_

#include <torch/torch.h>

#include "./utils.h"

namespace apt {
torch::Tensor UAddV(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor lhs,
    torch::Tensor rhs, torch::Tensor coo_offset, torch::Tensor lhs_offset,
    torch::Tensor rhs_offset);

torch::Tensor UMulV(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor lhs,
    torch::Tensor rhs, torch::Tensor coo_offset, torch::Tensor lhs_offset,
    torch::Tensor rhs_offset);
}  // namespace apt

#endif