#ifndef XRAY_RENDER_DX12_H
#define XRAY_RENDER_DX12_H

#include <stdint.h>

#if defined(_MSC_VER)
#define XR_DX12_ALIGN_256 __declspec(align(256))
#else
#define XR_DX12_ALIGN_256 __attribute__((aligned(256)))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Dx12Renderer Dx12Renderer;
typedef uint64_t XRayMeshHandle;

typedef struct XRayVertex {
    float position[3];
    float normal[3];
    float uv[2];
    float tangent[3];
} XRayVertex;

/* Column-major matrices; view_proj multiplies a column position vector. */
typedef struct XR_DX12_ALIGN_256 CameraConstants {
    float view[4][4];
    float proj[4][4];
    float view_proj[4][4];
    float cam_pos[4];
    float padding[12];
} CameraConstants;

enum {
    XR_DX12_OK = 0,
    XR_DX12_ERR_NULL_PTR = -1,
    XR_DX12_ERR_PANIC = -2,
    XR_DX12_ERR_D3D12 = -3,
    XR_DX12_ERR_INVALID_STATE = -4,
    XR_DX12_ERR_INVALID_ARGUMENT = -5,
    XR_DX12_ERR_INVALID_HANDLE = -6,
    XR_DX12_INDEX_U16 = 57, /* DXGI_FORMAT_R16_UINT */
    XR_DX12_INDEX_U32 = 42  /* DXGI_FORMAT_R32_UINT */
};

int32_t xr_dx12_renderer_create(uint32_t worker_count, Dx12Renderer **out_ptr);
int32_t xr_dx12_renderer_destroy(Dx12Renderer *renderer);
int32_t xr_dx12_renderer_attach_window(Dx12Renderer *renderer, void *hwnd, uint32_t width, uint32_t height);
int32_t xr_dx12_renderer_begin_frame(Dx12Renderer *renderer);
int32_t xr_dx12_renderer_clear(Dx12Renderer *renderer, float r, float g, float b, float a);
int32_t xr_dx12_renderer_end_frame(Dx12Renderer *renderer, uint32_t vsync);
int32_t xr_dx12_renderer_wait_for_gpu(Dx12Renderer *renderer);

/* Serialize calls per renderer. All pointers must designate live sized memory.
 * Data is copied before return. On failure a valid out_handle is set to zero.
 * vert_stride must be 44; idx_format is XR_DX12_INDEX_U16 or XR_DX12_INDEX_U32.
 * Handles are process-local and valid only for their creating renderer. */
int32_t xr_dx12_create_mesh(Dx12Renderer *renderer, const void *vertices,
    uint32_t vert_count, uint32_t vert_stride, const void *indices,
    uint32_t idx_count, uint32_t idx_format, XRayMeshHandle *out_handle);
int32_t xr_dx12_destroy_mesh(Dx12Renderer *renderer, XRayMeshHandle handle);
int32_t xr_dx12_update_camera(Dx12Renderer *renderer, const CameraConstants *camera_constants_ptr);
int32_t xr_dx12_draw_indexed(Dx12Renderer *renderer, XRayMeshHandle mesh_handle);

#ifdef __cplusplus
}
static_assert(sizeof(XRayVertex) == 44, "XRayVertex ABI mismatch");
static_assert(sizeof(CameraConstants) == 256, "CameraConstants ABI mismatch");
static_assert(alignof(CameraConstants) == 256, "CameraConstants alignment mismatch");
#else
_Static_assert(sizeof(XRayVertex) == 44, "XRayVertex ABI mismatch");
_Static_assert(sizeof(CameraConstants) == 256, "CameraConstants ABI mismatch");
_Static_assert(_Alignof(CameraConstants) == 256, "CameraConstants alignment mismatch");
#endif

#undef XR_DX12_ALIGN_256
#endif
