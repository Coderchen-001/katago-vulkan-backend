#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// 🔴 P1-c5/P1-c6 (2026-10-13): count the user evals that are executing, so the tuning probe's window predicate can
// ask the question it actually needs ("is the device exclusively mine right now?") instead of the proxy it used
// before ("has any eval already run?").  The new tuner flow measures BETWEEN evals, where the device is
// exclusive but an eval has already run, so the proxy was refusing a legitimate window.
// A COUNTER rather than a flag because the NN server evaluates on several threads, and the state machine that
// waits for "the first eval" must wait until none is in flight.
// Declared here, OUTSIDE the `extern "C"` block below, because the definition in ggml-vulkan.cpp is C++.
// The tuner is the only caller; production paths never touch it.
void ggml_vk_add_user_eval_in_flight(ggml_backend_t backend, int delta);

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

#ifdef  __cplusplus
}
#endif
