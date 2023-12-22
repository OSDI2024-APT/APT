#ifndef APT_FEAT_SHUFFLE_H_
#define APT_FEAT_SHUFFLE_H_

#include <torch/script.h>

#include <string>

#include "./utils.h"

namespace apt {

// APT feature shuffle
torch::Tensor FeatShuffle(
    torch::Tensor inputs, torch::Tensor send_offset, torch::Tensor recv_offset,
    torch::Tensor permutation, IdType feat_dim, IdType fwd_flag);

torch::Tensor SPFeatShuffle(
    torch::Tensor inputs, torch::Tensor send_sizes, torch::Tensor recv_sizes,
    IdType total_recv_size, IdType expand, IdType shuffle_with_dst);

// MP feature shuffle forward
torch::Tensor MPFeatShuffleFwd(
    torch::Tensor inputs, torch::Tensor send_size, torch::Tensor recv_size,
    IdType feat_dim);
// MP feature shuffle backward
torch::Tensor MPFeatShuffleBwd(
    torch::Tensor inputs, torch::Tensor send_size, torch::Tensor recv_size,
    IdType feat_dim);

}  // namespace apt

#endif