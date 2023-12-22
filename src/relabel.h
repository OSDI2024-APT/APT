#ifndef APT_RELABEL_H_
#define APT_RELABEL_H_

#include <torch/torch.h>

#include "./utils.h"
#include "glog/logging.h"

namespace apt {
std::vector<torch::Tensor> RelabelCSC(
    torch::Tensor seeds, torch::Tensor neighbors);
}  // namespace apt

#endif