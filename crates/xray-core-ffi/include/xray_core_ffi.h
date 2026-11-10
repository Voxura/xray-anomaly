#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t XrConfigHandle;
#define XR_INVALID_HANDLE ((XrConfigHandle)0)

enum XrStatus {
    XR_STATUS_SUCCESS = 0,
    XR_STATUS_INVALID_PARAM = -1,
    XR_STATUS_INVALID_HANDLE = -2,
    XR_STATUS_BUFFER_TOO_SMALL = -3,
    XR_STATUS_NOT_FOUND = -4,
    XR_STATUS_PARSE_ERROR = -5,
    XR_STATUS_PANIC = -99
};

// Open configuration from memory buffer
XrConfigHandle xr_config_open_memory(const char* data, size_t len, int32_t* out_status);

// Read string into caller-owned buffer
int32_t xr_config_read_string_buf(
    XrConfigHandle handle,
    const char* section,
    const char* key,
    char* out_buf,
    size_t max_len
);

// Typed scalar readers
int32_t xr_config_read_int(XrConfigHandle handle, const char* section, const char* key, int32_t def_val);
float xr_config_read_float(XrConfigHandle handle, const char* section, const char* key, float def_val);
bool xr_config_read_bool(XrConfigHandle handle, const char* section, const char* key, bool def_val);

// Existence checks
bool xr_config_section_exists(XrConfigHandle handle, const char* section);
bool xr_config_line_exists(XrConfigHandle handle, const char* section, const char* key);

// Free configuration handle
int32_t xr_config_free(XrConfigHandle handle);

#ifdef __cplusplus
}
#endif
