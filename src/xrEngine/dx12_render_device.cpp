#include "stdafx.h"
#include "dx12_render_device.h"

dx12RenderDeviceRender::dx12RenderDeviceRender()
    : m_pRustRenderer(nullptr),
      m_hRustModule(nullptr),
      m_fGamma(1.0f),
      m_fBrightness(1.0f),
      m_fContrast(1.0f),
      m_hTestMesh(0),
      m_dwWidth(0),
      m_dwHeight(0),
      pfn_create(nullptr),
      pfn_destroy(nullptr),
      pfn_attach_window(nullptr),
      pfn_resize(nullptr),
      pfn_begin_frame(nullptr),
      pfn_clear(nullptr),
      pfn_end_frame(nullptr),
      pfn_wait_for_gpu(nullptr),
      pfn_get_device_state(nullptr),
      pfn_create_mesh(nullptr),
      pfn_destroy_mesh(nullptr),
      pfn_update_camera(nullptr),
      pfn_draw_indexed(nullptr)
{
}

dx12RenderDeviceRender::~dx12RenderDeviceRender()
{
    DestroyHW();
    UnloadRustBackend();
}

bool dx12RenderDeviceRender::LoadRustBackend()
{
    if (m_hRustModule)
        return true;

    // Search candidates: same directory as xrEngine, or current directory, or bin/
    const char* dll_names[] = {
        "xray_render_dx12.dll",
        "bin\\xray_render_dx12.dll",
        "..\\bin\\xray_render_dx12.dll",
        "crates\\xray-render-dx12\\target\\release\\xray_render_dx12.dll",
        "crates\\target\\release\\xray_render_dx12.dll",
        "target\\release\\xray_render_dx12.dll"
    };

    for (const char* path : dll_names)
    {
        m_hRustModule = LoadLibraryA(path);
        if (m_hRustModule)
        {
            Log("Loaded Pure Rust DirectX 12 Backend module:", path);
            break;
        }
    }

    if (!m_hRustModule)
    {
        Msg("! [DirectX 12] Failed to find or load xray_render_dx12.dll");
        return false;
    }

    pfn_create = (PFN_xr_dx12_renderer_create)GetProcAddress(m_hRustModule, "xr_dx12_renderer_create");
    pfn_destroy = (PFN_xr_dx12_renderer_destroy)GetProcAddress(m_hRustModule, "xr_dx12_renderer_destroy");
    pfn_attach_window = (PFN_xr_dx12_renderer_attach_window)GetProcAddress(m_hRustModule, "xr_dx12_renderer_attach_window");
    pfn_resize = (PFN_xr_dx12_renderer_resize)GetProcAddress(m_hRustModule, "xr_dx12_renderer_resize");
    pfn_begin_frame = (PFN_xr_dx12_renderer_begin_frame)GetProcAddress(m_hRustModule, "xr_dx12_renderer_begin_frame");
    pfn_clear = (PFN_xr_dx12_renderer_clear)GetProcAddress(m_hRustModule, "xr_dx12_renderer_clear");
    pfn_end_frame = (PFN_xr_dx12_renderer_end_frame)GetProcAddress(m_hRustModule, "xr_dx12_renderer_end_frame");
    pfn_wait_for_gpu = (PFN_xr_dx12_renderer_wait_for_gpu)GetProcAddress(m_hRustModule, "xr_dx12_renderer_wait_for_gpu");
    pfn_get_device_state = (PFN_xr_dx12_renderer_get_device_state)GetProcAddress(m_hRustModule, "xr_dx12_renderer_get_device_state");
    pfn_create_mesh = (PFN_xr_dx12_create_mesh)GetProcAddress(m_hRustModule, "xr_dx12_create_mesh");
    pfn_destroy_mesh = (PFN_xr_dx12_destroy_mesh)GetProcAddress(m_hRustModule, "xr_dx12_destroy_mesh");
    pfn_update_camera = (PFN_xr_dx12_update_camera)GetProcAddress(m_hRustModule, "xr_dx12_update_camera");
    pfn_draw_indexed = (PFN_xr_dx12_draw_indexed)GetProcAddress(m_hRustModule, "xr_dx12_draw_indexed");

    if (!pfn_create || !pfn_destroy || !pfn_attach_window || !pfn_resize ||
        !pfn_begin_frame || !pfn_clear || !pfn_end_frame || !pfn_wait_for_gpu || !pfn_get_device_state ||
        !pfn_create_mesh || !pfn_destroy_mesh || !pfn_update_camera || !pfn_draw_indexed)
    {
        Msg("! [DirectX 12] Failed to resolve one or more required entry points in xray_render_dx12.dll");
        UnloadRustBackend();
        return false;
    }

    return true;
}

void dx12RenderDeviceRender::UnloadRustBackend()
{
    DestroyTestGeometry();

    if (m_hRustModule)
    {
        FreeLibrary(m_hRustModule);
        m_hRustModule = nullptr;
    }
    pfn_create = nullptr;
    pfn_destroy = nullptr;
    pfn_attach_window = nullptr;
    pfn_resize = nullptr;
    pfn_begin_frame = nullptr;
    pfn_clear = nullptr;
    pfn_end_frame = nullptr;
    pfn_wait_for_gpu = nullptr;
    pfn_get_device_state = nullptr;
    pfn_create_mesh = nullptr;
    pfn_destroy_mesh = nullptr;
    pfn_update_camera = nullptr;
    pfn_draw_indexed = nullptr;
}

void dx12RenderDeviceRender::Copy(IRenderDeviceRender& _in)
{
    *this = *(dx12RenderDeviceRender*)&_in;
}

void dx12RenderDeviceRender::setGamma(float fGamma)
{
    m_fGamma = fGamma;
}

void dx12RenderDeviceRender::setBrightness(float fGamma)
{
    m_fBrightness = fGamma;
}

void dx12RenderDeviceRender::setContrast(float fGamma)
{
    m_fContrast = fGamma;
}

void dx12RenderDeviceRender::updateGamma()
{
}

void dx12RenderDeviceRender::OnDeviceDestroy(BOOL bKeepTextures)
{
}

void dx12RenderDeviceRender::ValidateHW()
{
}

void dx12RenderDeviceRender::DestroyHW()
{
    DestroyTestGeometry();

    if (m_pRustRenderer && pfn_destroy)
    {
        pfn_destroy(m_pRustRenderer);
        m_pRustRenderer = nullptr;
    }
}

void dx12RenderDeviceRender::Reset(HWND hWnd, u32& dwWidth, u32& dwHeight, float& fWidth_2, float& fHeight_2)
{
    m_dwWidth = dwWidth;
    m_dwHeight = dwHeight;

    if (m_pRustRenderer && pfn_resize)
    {
        pfn_resize(m_pRustRenderer, dwWidth, dwHeight);
        fWidth_2 = float(dwWidth / 2);
        fHeight_2 = float(dwHeight / 2);
    }
}

void dx12RenderDeviceRender::SetupStates()
{
}

void dx12RenderDeviceRender::OnDeviceCreate(LPCSTR shName)
{
    Log("DirectX 12 Pure Rust Device Created. Shader cache:", shName ? shName : "default");
}

#pragma pack(push, 1)
struct XRayVertexWire {
    float position[3];
    float normal[3];
    float uv[2];
    float tangent[3];
};
#pragma pack(pop)
static_assert(sizeof(XRayVertexWire) == 44, "XRayVertexWire ABI mismatch");

struct CameraConstantsWire {
    float view[4][4];
    float proj[4][4];
    float view_proj[4][4];
    float cam_pos[4];
    float padding[12];
};
static_assert(sizeof(CameraConstantsWire) == 256, "CameraConstantsWire ABI mismatch");

void dx12RenderDeviceRender::CreateTestGeometry()
{
    if (!m_pRustRenderer || !pfn_create_mesh)
        return;

    // 3D Tetrahedron / Pyramid centered around origin with distinct normals
    XRayVertexWire vertices[] = {
        // Base triangle (facing -Y)
        { {-0.6f, -0.4f, -0.6f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },
        { { 0.6f, -0.4f, -0.6f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },
        { { 0.0f, -0.4f,  0.6f}, {0.0f, -1.0f, 0.0f}, {0.5f, 1.0f}, {1.0f, 0.0f, 0.0f} },
        // Front face (facing +Z, bright red/magenta)
        { {-0.6f, -0.4f, -0.6f}, {-0.577f, 0.577f, -0.577f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },
        { { 0.0f,  0.6f,  0.0f}, {-0.577f, 0.577f, -0.577f}, {0.5f, 1.0f}, {1.0f, 0.0f, 0.0f} },
        { { 0.6f, -0.4f, -0.6f}, {-0.577f, 0.577f, -0.577f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },
        // Right face (facing +X, yellow/green)
        { { 0.6f, -0.4f, -0.6f}, {0.816f, 0.408f, 0.408f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },
        { { 0.0f,  0.6f,  0.0f}, {0.816f, 0.408f, 0.408f}, {0.5f, 1.0f}, {0.0f, 0.0f, 1.0f} },
        { { 0.0f, -0.4f,  0.6f}, {0.816f, 0.408f, 0.408f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },
        // Left face (facing -X, cyan/blue)
        { { 0.0f, -0.4f,  0.6f}, {-0.816f, 0.408f, 0.408f}, {0.0f, 0.0f}, {0.0f, 0.0f, -1.0f} },
        { { 0.0f,  0.6f,  0.0f}, {-0.816f, 0.408f, 0.408f}, {0.5f, 1.0f}, {-0.707f, 0.577f, 0.0f} },
        { {-0.6f, -0.4f, -0.6f}, {-0.816f, 0.408f, 0.408f}, {1.0f, 0.0f}, {0.0f, 0.0f, -1.0f} }
    };
    uint16_t indices[] = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11
    };

    uint64_t handle = 0;
    int status = pfn_create_mesh(m_pRustRenderer, vertices, 12, sizeof(XRayVertexWire), indices, 12, 57, &handle);
    if (status == 0 && handle != 0)
    {
        m_hTestMesh = handle;
        Msg("* [DirectX 12] Phase 4 3D test mesh created successfully (handle: %llu).", handle);
    }
    else
    {
        Msg("! [DirectX 12] Failed to create Phase 4 3D test mesh (status: %d)", status);
    }
}

void dx12RenderDeviceRender::DestroyTestGeometry()
{
    if (m_pRustRenderer && m_hTestMesh != 0 && pfn_destroy_mesh)
    {
        pfn_destroy_mesh(m_pRustRenderer, m_hTestMesh);
        m_hTestMesh = 0;
    }
}

void dx12RenderDeviceRender::UpdateTestCamera()
{
    if (!m_pRustRenderer || !pfn_update_camera)
        return;

    static float s_angle = 0.0f;
    s_angle += 0.025f;
    if (s_angle > 6.2831853f) s_angle -= 6.2831853f;

    float aspect = (m_dwHeight > 0) ? (float)m_dwWidth / (float)m_dwHeight : 1.333f;
    float fov = 1.04719755f; // 60 deg
    float f = 1.0f / tanf(fov * 0.5f);
    float nearZ = 0.1f;
    float farZ = 100.0f;
    float dist = 2.2f;

    float c = cosf(s_angle);
    float s = sinf(s_angle);

    CameraConstantsWire cam;
    memset(&cam, 0, sizeof(cam));

    // Column-major view_proj matrix:
    cam.view_proj[0][0] = c * (f / aspect);
    cam.view_proj[1][1] = f;
    cam.view_proj[2][0] = s * (f / aspect);

    cam.view_proj[0][2] = -s * (farZ / (farZ - nearZ));
    cam.view_proj[2][2] = c * (farZ / (farZ - nearZ));
    cam.view_proj[3][2] = (dist - nearZ) * (farZ / (farZ - nearZ));

    cam.view_proj[0][3] = -s;
    cam.view_proj[2][3] = c;
    cam.view_proj[3][3] = dist;

    cam.cam_pos[0] = 0.0f;
    cam.cam_pos[1] = 0.0f;
    cam.cam_pos[2] = -dist;
    cam.cam_pos[3] = 1.0f;

    pfn_update_camera(m_pRustRenderer, &cam);
}

void dx12RenderDeviceRender::Create(HWND hWnd, u32& dwWidth, u32& dwHeight, float& fWidth_2, float& fHeight_2, bool)
{
    Log("Creating DirectX 12 Pure Rust Backend Device Context...");
    R_ASSERT2(LoadRustBackend(), "Failed to load pure Rust DirectX 12 backend dynamic library");

    if (dwWidth == 0 || dwHeight == 0)
    {
        RECT rc;
        if (GetClientRect(hWnd, &rc) && (rc.right > rc.left) && (rc.bottom > rc.top))
        {
            dwWidth = (u32)(rc.right - rc.left);
            dwHeight = (u32)(rc.bottom - rc.top);
        }
        else
        {
            dwWidth = psCurrentVidMode[0] ? psCurrentVidMode[0] : 1280;
            dwHeight = psCurrentVidMode[1] ? psCurrentVidMode[1] : 720;
        }
    }

    int status = pfn_create(4, &m_pRustRenderer);
    R_ASSERT2(status == 0, "Failed to create pure Rust DirectX 12 renderer context");

    status = pfn_attach_window(m_pRustRenderer, (void*)hWnd, dwWidth, dwHeight);
    R_ASSERT2(status == 0, "Failed to attach Win32 HWND to DirectX 12 swapchain");

    m_dwWidth = dwWidth;
    m_dwHeight = dwHeight;
    fWidth_2 = float(dwWidth / 2);
    fHeight_2 = float(dwHeight / 2);

    CreateTestGeometry();
    Msg("DirectX 12 Presentation SwapChain initialized successfully (%ux%u).", dwWidth, dwHeight);
}

void dx12RenderDeviceRender::SetupGPU(BOOL bForceGPU_SW, BOOL bForceGPU_NonPure, BOOL bForceGPU_REF)
{
}

void dx12RenderDeviceRender::overdrawBegin()
{
}

void dx12RenderDeviceRender::overdrawEnd()
{
}

void dx12RenderDeviceRender::DeferredLoad(BOOL E)
{
}

void dx12RenderDeviceRender::ResourcesDeferredUpload()
{
}

void dx12RenderDeviceRender::ResourcesDeferredUnload()
{
}

void dx12RenderDeviceRender::ResourcesGetMemoryUsage(u32& m_base, u32& c_base, u32& m_lmaps, u32& c_lmaps)
{
    m_base = 0;
    c_base = 0;
    m_lmaps = 0;
    c_lmaps = 0;
}

void dx12RenderDeviceRender::ResourcesDestroyNecessaryTextures()
{
}

void dx12RenderDeviceRender::ResourcesStoreNecessaryTextures()
{
}

void dx12RenderDeviceRender::ResourcesDumpMemoryUsage()
{
}

void dx12RenderDeviceRender::ResourcesPrefetchCreateTexture(LPCSTR name)
{
}

bool dx12RenderDeviceRender::HWSupportsShaderYUV2RGB()
{
    return true;
}

IRenderDeviceRender::DeviceState dx12RenderDeviceRender::GetDeviceState()
{
    if (!m_pRustRenderer || !pfn_get_device_state)
        return dsOK;

    int state = 0;
    pfn_get_device_state(m_pRustRenderer, &state);
    switch (state)
    {
    case 1:
        return dsLost;
    case 2:
        return dsNeedReset;
    default:
        return dsOK;
    }
}

BOOL dx12RenderDeviceRender::GetForceGPU_REF()
{
    return FALSE;
}

u32 dx12RenderDeviceRender::GetCacheStatPolys()
{
    return 0;
}

void dx12RenderDeviceRender::Begin()
{
    if (m_pRustRenderer)
    {
        UpdateTestCamera();
        if (pfn_begin_frame)
            pfn_begin_frame(m_pRustRenderer);
    }
}

void dx12RenderDeviceRender::Clear()
{
    if (m_pRustRenderer && pfn_clear)
        pfn_clear(m_pRustRenderer, 0.05f, 0.05f, 0.08f, 1.0f);
}

void dx12RenderDeviceRender::End()
{
    if (m_pRustRenderer)
    {
        if (m_hTestMesh != 0 && pfn_draw_indexed)
        {
            pfn_draw_indexed(m_pRustRenderer, m_hTestMesh);
        }

        if (pfn_end_frame)
        {
            bool use_vsync = !!psDeviceFlags.test(rsVSync);
            pfn_end_frame(m_pRustRenderer, use_vsync ? 1 : 0);
        }
    }
}

void dx12RenderDeviceRender::ClearTarget()
{
    if (m_pRustRenderer && pfn_clear)
        pfn_clear(m_pRustRenderer, 0.0f, 0.0f, 0.0f, 1.0f);
}

void dx12RenderDeviceRender::SetCacheXform(Fmatrix& mView, Fmatrix& mProject)
{
}

void dx12RenderDeviceRender::SetCacheXform_prev(Fmatrix& mView, Fmatrix& mProject)
{
}

void dx12RenderDeviceRender::OnAssetsChanged()
{
}
