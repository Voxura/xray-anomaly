#include "stdafx.h"
#include "dx12_render_device.h"

dx12RenderDeviceRender::dx12RenderDeviceRender()
    : m_pRustRenderer(nullptr),
      m_hRustModule(nullptr),
      m_fGamma(1.0f),
      m_fBrightness(1.0f),
      m_fContrast(1.0f),
      pfn_create(nullptr),
      pfn_destroy(nullptr),
      pfn_attach_window(nullptr),
      pfn_resize(nullptr),
      pfn_begin_frame(nullptr),
      pfn_clear(nullptr),
      pfn_end_frame(nullptr),
      pfn_wait_for_gpu(nullptr),
      pfn_get_device_state(nullptr)
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

    if (!pfn_create || !pfn_destroy || !pfn_attach_window || !pfn_resize ||
        !pfn_begin_frame || !pfn_clear || !pfn_end_frame || !pfn_wait_for_gpu || !pfn_get_device_state)
    {
        Msg("! [DirectX 12] Failed to resolve one or more required entry points in xray_render_dx12.dll");
        UnloadRustBackend();
        return false;
    }

    return true;
}

void dx12RenderDeviceRender::UnloadRustBackend()
{
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
    if (m_pRustRenderer && pfn_destroy)
    {
        pfn_destroy(m_pRustRenderer);
        m_pRustRenderer = nullptr;
    }
}

void dx12RenderDeviceRender::Reset(HWND hWnd, u32& dwWidth, u32& dwHeight, float& fWidth_2, float& fHeight_2)
{
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

void dx12RenderDeviceRender::Create(HWND hWnd, u32& dwWidth, u32& dwHeight, float& fWidth_2, float& fHeight_2, bool)
{
    Log("Creating DirectX 12 Pure Rust Backend Device Context...");
    R_ASSERT2(LoadRustBackend(), "Failed to load pure Rust DirectX 12 backend dynamic library");

    int status = pfn_create(4, &m_pRustRenderer);
    R_ASSERT2(status == 0, "Failed to create pure Rust DirectX 12 renderer context");

    status = pfn_attach_window(m_pRustRenderer, (void*)hWnd, dwWidth, dwHeight);
    R_ASSERT2(status == 0, "Failed to attach Win32 HWND to DirectX 12 swapchain");

    fWidth_2 = float(dwWidth / 2);
    fHeight_2 = float(dwHeight / 2);
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
    if (m_pRustRenderer && pfn_begin_frame)
        pfn_begin_frame(m_pRustRenderer);
}

void dx12RenderDeviceRender::Clear()
{
    if (m_pRustRenderer && pfn_clear)
        pfn_clear(m_pRustRenderer, 0.0f, 0.0f, 0.0f, 1.0f);
}

void dx12RenderDeviceRender::End()
{
    if (m_pRustRenderer && pfn_end_frame)
    {
        bool use_vsync = !!psDeviceFlags.test(rsVSync);
        pfn_end_frame(m_pRustRenderer, use_vsync ? 1 : 0);
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
