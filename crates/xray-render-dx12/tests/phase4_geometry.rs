//! Real GPU readback, not just successful command recording. Requires DXC/SM6.6.
use std::mem::ManuallyDrop;
use std::ptr;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, WS_OVERLAPPEDWINDOW,
};
use windows::core::w;
use xray_render_dx12::ffi::*;
use xray_render_dx12::geometry_abi::{INDEX_FORMAT_U16, INDEX_FORMAT_U32};
use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::{CameraConstants, XRayVertex};

struct Window(HWND);
impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.0);
        }
    }
}

fn camera() -> CameraConstants {
    let mut c = CameraConstants::default();
    for i in 0..4 {
        c.view[i][i] = 1.0;
        c.proj[i][i] = 1.0;
        c.view_proj[i][i] = 1.0;
    }
    c
}

unsafe fn transition(
    cmd: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
        ..Default::default()
    };
    unsafe {
        cmd.ResourceBarrier(std::slice::from_ref(&barrier));
        ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
    }
}

#[test]
fn phase4_indexed_draw_readback_and_lifetimes() {
    unsafe {
        let mut debug: Option<windows::Win32::Graphics::Direct3D12::ID3D12Debug> = None;
        if windows::Win32::Graphics::Direct3D12::D3D12GetDebugInterface(&mut debug).is_ok() {
            if let Some(d) = debug {
                d.EnableDebugLayer();
            }
        }
    }
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("Phase 4 readback"),
            WS_OVERLAPPEDWINDOW,
            0,
            0,
            64,
            64,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    });
    let mut renderer = Dx12Renderer::new(1).expect("DX12 hardware required");
    println!("Phase 4 readback adapter: {}", renderer.adapter_name());
    renderer.attach_window(window.0, 64, 64).unwrap();
    let vertices =
        [[-0.8, -0.8, 0.5], [0.0, 0.8, 0.5], [0.8, -0.8, 0.5]].map(|position| XRayVertex {
            position,
            normal: [1.0, 0.0, 0.0],
            ..Default::default()
        });
    let indices16 = [0u16, 1, 2];
    let indices32 = [0u32, 1, 2];
    let readback: ID3D12Resource = unsafe {
        let mut resource = None;
        renderer
            .device_ctx
            .device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 64 * 256,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut resource,
            )
            .unwrap();
        resource.unwrap()
    };
    for (format, indices) in [
        (INDEX_FORMAT_U16, indices16.as_ptr().cast()),
        (INDEX_FORMAT_U32, indices32.as_ptr().cast()),
    ] {
        unsafe {
            let r = ptr::addr_of_mut!(renderer);
            let mut handle = 0;
            assert_eq!(
                xr_dx12_create_mesh(
                    r,
                    vertices.as_ptr().cast(),
                    3,
                    48,
                    indices,
                    3,
                    format,
                    &mut handle
                ),
                XR_DX12_ERR_INVALID_ARGUMENT
            );
            assert_eq!(handle, 0);
            assert_eq!(
                xr_dx12_create_mesh(
                    r,
                    vertices.as_ptr().cast(),
                    3,
                    44,
                    indices,
                    3,
                    format,
                    &mut handle
                ),
                XR_DX12_OK
            );
            assert_ne!(handle, 0);
            assert_eq!(xr_dx12_draw_indexed(r, handle), XR_DX12_ERR_INVALID_STATE);
            assert_eq!(xr_dx12_update_camera(r, &camera()), XR_DX12_OK);
            assert_eq!(xr_dx12_renderer_begin_frame(r), XR_DX12_OK);
            assert_ne!(xr_dx12_renderer_begin_frame(r), XR_DX12_OK);
            assert!(renderer.resize(32, 32).is_err());
            renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);
            assert_eq!(xr_dx12_draw_indexed(r, handle), XR_DX12_OK);
            let mut moved = camera();
            moved.view_proj[3][0] = 10.0; // Column-major translation out of view.
            assert_eq!(xr_dx12_update_camera(r, &moved), XR_DX12_OK);
            assert_eq!(xr_dx12_draw_indexed(r, handle), XR_DX12_OK);
            assert_eq!(xr_dx12_destroy_mesh(r, handle), XR_DX12_OK);
            assert_eq!(xr_dx12_destroy_mesh(r, handle), XR_DX12_ERR_INVALID_HANDLE);
            assert_eq!(xr_dx12_draw_indexed(r, handle), XR_DX12_ERR_INVALID_HANDLE);

            let cmd = renderer.current_cmd_list.as_ref().unwrap();
            let backbuffer = renderer
                .swapchain
                .as_ref()
                .unwrap()
                .current_backbuffer_resource();
            transition(
                cmd,
                backbuffer,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            let mut source = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(backbuffer.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let mut destination = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(readback.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            Width: 64,
                            Height: 64,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            cmd.CopyTextureRegion(&destination, 0, 0, 0, &source, None);
            ManuallyDrop::drop(&mut source.pResource);
            ManuallyDrop::drop(&mut destination.pResource);
            transition(
                cmd,
                backbuffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            renderer.end_frame(false).unwrap();
            renderer.wait_for_gpu().unwrap();
            let mut mapped = ptr::null_mut();
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 64 * 256,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), 64 * 256);
            let center = bytes[32 * 256 + 32 * 4..32 * 256 + 32 * 4 + 4].to_vec();
            let corner = bytes[..4].to_vec();
            readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            assert_eq!(
                center,
                [255, 0, 0, 255],
                "format {format}: triangle/camera lifetime"
            );
            assert_eq!(
                corner,
                [0, 0, 0, 255],
                "format {format}: clear outside triangle"
            );
            println!("Phase 4 format {format}: red center and black corner verified");
        }
    }
}

#[test]
fn phase4_null_ffi_arguments() {
    unsafe {
        let mut handle = 123;
        assert_eq!(
            xr_dx12_create_mesh(
                ptr::null_mut(),
                ptr::null(),
                3,
                44,
                ptr::null(),
                3,
                INDEX_FORMAT_U16,
                &mut handle
            ),
            XR_DX12_ERR_NULL_PTR
        );
        assert_eq!(handle, 0);
        assert_eq!(
            xr_dx12_destroy_mesh(ptr::null_mut(), 1),
            XR_DX12_ERR_NULL_PTR
        );
        assert_eq!(
            xr_dx12_update_camera(ptr::null_mut(), ptr::null()),
            XR_DX12_ERR_NULL_PTR
        );
        assert_eq!(
            xr_dx12_draw_indexed(ptr::null_mut(), 1),
            XR_DX12_ERR_NULL_PTR
        );
    }
}
