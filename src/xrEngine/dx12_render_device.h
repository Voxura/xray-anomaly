#pragma once

#include "stdafx.h"
#include "../Include/xrRender/RenderDeviceRender.h"

// Forward declaration of Rust opaque renderer handle
struct XrDx12Renderer;

// Function pointer signatures for dynamic loading
typedef int (*PFN_xr_dx12_renderer_create)(unsigned int worker_count, XrDx12Renderer** out_ptr);
typedef int (*PFN_xr_dx12_renderer_destroy)(XrDx12Renderer* renderer);
typedef int (*PFN_xr_dx12_renderer_attach_window)(XrDx12Renderer* renderer, void* hwnd, unsigned int width, unsigned int height);
typedef int (*PFN_xr_dx12_renderer_resize)(XrDx12Renderer* renderer, unsigned int width, unsigned int height);
typedef int (*PFN_xr_dx12_renderer_begin_frame)(XrDx12Renderer* renderer);
typedef int (*PFN_xr_dx12_renderer_clear)(XrDx12Renderer* renderer, float r, float g, float b, float a);
typedef int (*PFN_xr_dx12_renderer_end_frame)(XrDx12Renderer* renderer, unsigned int vsync);
typedef int (*PFN_xr_dx12_renderer_wait_for_gpu)(XrDx12Renderer* renderer);
typedef int (*PFN_xr_dx12_renderer_get_device_state)(XrDx12Renderer* renderer, int* out_state);

class ENGINE_API dx12RenderDeviceRender : public IRenderDeviceRender
{
private:
    XrDx12Renderer* m_pRustRenderer;
    HMODULE m_hRustModule;
    float m_fGamma;
    float m_fBrightness;
    float m_fContrast;

    // Dynamically resolved function pointers
    PFN_xr_dx12_renderer_create pfn_create;
    PFN_xr_dx12_renderer_destroy pfn_destroy;
    PFN_xr_dx12_renderer_attach_window pfn_attach_window;
    PFN_xr_dx12_renderer_resize pfn_resize;
    PFN_xr_dx12_renderer_begin_frame pfn_begin_frame;
    PFN_xr_dx12_renderer_clear pfn_clear;
    PFN_xr_dx12_renderer_end_frame pfn_end_frame;
    PFN_xr_dx12_renderer_wait_for_gpu pfn_wait_for_gpu;
    PFN_xr_dx12_renderer_get_device_state pfn_get_device_state;

    bool LoadRustBackend();
    void UnloadRustBackend();

public:
    dx12RenderDeviceRender();
    virtual ~dx12RenderDeviceRender();

    virtual void Copy(IRenderDeviceRender& _in) override;

    // Gamma correction
    virtual void setGamma(float fGamma) override;
    virtual void setBrightness(float fGamma) override;
    virtual void setContrast(float fGamma) override;
    virtual void updateGamma() override;

    // Device lifecycle
    virtual void OnDeviceDestroy(BOOL bKeepTextures) override;
    virtual void ValidateHW() override;
    virtual void DestroyHW() override;
    virtual void Reset(HWND hWnd, u32& dwWidth, u32& dwHeight, float& fWidth_2, float& fHeight_2) override;
    virtual void SetupStates() override;
    virtual void OnDeviceCreate(LPCSTR shName) override;
    virtual void Create(HWND hWnd, u32& dwWidth, u32& dwHeight, float& fWidth_2, float& fHeight_2, bool) override;
    virtual void SetupGPU(BOOL bForceGPU_SW, BOOL bForceGPU_NonPure, BOOL bForceGPU_REF) override;

    // Overdraw
    virtual void overdrawBegin() override;
    virtual void overdrawEnd() override;

    // Resources
    virtual void DeferredLoad(BOOL E) override;
    virtual void ResourcesDeferredUpload() override;
    virtual void ResourcesDeferredUnload() override;
    virtual void ResourcesGetMemoryUsage(u32& m_base, u32& c_base, u32& m_lmaps, u32& c_lmaps) override;
    virtual void ResourcesDestroyNecessaryTextures() override;
    virtual void ResourcesStoreNecessaryTextures() override;
    virtual void ResourcesDumpMemoryUsage() override;
    virtual void ResourcesPrefetchCreateTexture(LPCSTR name) override;

    // HW Support
    virtual bool HWSupportsShaderYUV2RGB() override;

    // Frame Presentation
    virtual DeviceState GetDeviceState() override;
    virtual BOOL GetForceGPU_REF() override;
    virtual u32 GetCacheStatPolys() override;
    virtual void Begin() override;
    virtual void Clear() override;
    virtual void End() override;
    virtual void ClearTarget() override;
    virtual void SetCacheXform(Fmatrix& mView, Fmatrix& mProject) override;
    virtual void SetCacheXform_prev(Fmatrix& mView, Fmatrix& mProject) override;
    virtual void OnAssetsChanged() override;
};
