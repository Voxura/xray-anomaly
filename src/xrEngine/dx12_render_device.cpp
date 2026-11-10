#include "stdafx.h"
#include "dx12_render_device.h"

dx12RenderDeviceRender::dx12RenderDeviceRender()
    : m_pRustRenderer(nullptr),
      m_fGamma(1.0f),
      m_fBrightness(1.0f),
      m_fContrast(1.0f)
{
}

dx12RenderDeviceRender::~dx12RenderDeviceRender()
{
    DestroyHW();
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
    if (m_pRustRenderer)
    {
        xr_dx12_renderer_destroy(m_pRustRenderer);
        m_pRustRenderer = nullptr;
    }
}

void dx12RenderDeviceRender::Reset(HWND hWnd, u32& dwWidth, u32& dwHeight, float& fWidth_2, float& fHeight_2)
{
    if (m_pRustRenderer)
    {
        xr_dx12_renderer_resize(m_pRustRenderer, dwWidth, dwHeight);
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
    int status = xr_dx12_renderer_create(4, &m_pRustRenderer);
    R_ASSERT2(status == 0, "Failed to create pure Rust DirectX 12 renderer context");

    status = xr_dx12_renderer_attach_window(m_pRustRenderer, (void*)hWnd, dwWidth, dwHeight);
    R_ASSERT2(status == 0, "Failed to attach Win32 HWND to DirectX 12 swapchain");

    fWidth_2 = float(dwWidth / 2);
    fHeight_2 = float(dwHeight / 2);
    Log("DirectX 12 Presentation SwapChain initialized successfully (%ux%u).", dwWidth, dwHeight);
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
    if (!m_pRustRenderer)
        return dsOK;

    int state = 0;
    xr_dx12_renderer_get_device_state(m_pRustRenderer, &state);
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
        xr_dx12_renderer_begin_frame(m_pRustRenderer);
}

void dx12RenderDeviceRender::Clear()
{
    if (m_pRustRenderer)
        xr_dx12_renderer_clear(m_pRustRenderer, 0.0f, 0.0f, 0.0f, 1.0f);
}

void dx12RenderDeviceRender::End()
{
    if (m_pRustRenderer)
    {
        bool use_vsync = !!psDeviceFlags.test(rsVSync);
        xr_dx12_renderer_end_frame(m_pRustRenderer, use_vsync ? 1 : 0);
    }
}

void dx12RenderDeviceRender::ClearTarget()
{
    if (m_pRustRenderer)
        xr_dx12_renderer_clear(m_pRustRenderer, 0.0f, 0.0f, 0.0f, 1.0f);
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
