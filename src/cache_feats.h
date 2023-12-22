#ifndef APT_CACHE_FEATS_H_
#define APT_CACHE_FEATS_H_

#include <torch/script.h>

#include <string>

#include "./utils.h"

namespace apt {

void CacheFeatsShared(
    IdType num_total_nodes, torch::Tensor localnode_feats,
    torch::Tensor cached_feats, torch::Tensor cached_idx,
    torch::Tensor localnode_idx, IdType feat_dim_offset = 0);
}  // namespace apt

#endif