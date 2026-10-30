/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TORCH_TPU_CSRC_COMMON_PROCESS_H_
#define TORCH_TPU_CSRC_COMMON_PROCESS_H_

#include <sys/types.h>

namespace torch_tpu {

// Returns the PID of the root TorchTPU process in this process lineage.
//
// On the first call in a root process (where `TORCH_TPU_INTERNAL_ROOT_PID` is
// not set in the environment), records `getpid()` into
// `TORCH_TPU_INTERNAL_ROOT_PID` so descendant processes inherit it.
//
// Does not depend on or trigger PjRt initialization.
[[nodiscard]] pid_t GetRootProcessPid();

// Returns true if the current process is the root (parent) TorchTPU process,
// or false if it was forked or spawned from an ancestor process that already
// initialized TorchTPU.
//
// Does not depend on or trigger PjRt initialization.
[[nodiscard]] bool IsParentProcess();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_PROCESS_H_
