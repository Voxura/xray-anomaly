//! Windows WASAPI Shared-Mode Audio Backend & Headless Software Fallback (`xray-audio`)
//!
//! Spawns a dedicated MTA COM worker thread owning the Windows WASAPI `IMMDeviceEnumerator` /
//! `IAudioClient` / `IAudioRenderClient` lifecycle (`48000 Hz` stereo `f32` with automatic
//! Windows SRC), pulling spatialized audio directly from [`SpatialAudioMixer`]. When running
//! in `--headless` mode or in environments with no active audio endpoint, gracefully selects
//! `AudioOutputBackend::HeadlessNull` so engine loops continue deterministically.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use windows::Win32::Media::Audio::{
    eConsole, eRender, IAudioClient, IAudioRenderClient, IMMDeviceEnumerator, MMDeviceEnumerator,
    AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
    AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, WAVEFORMATEX,
};
use windows::Win32::Media::Multimedia::WAVE_FORMAT_IEEE_FLOAT;
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED,
};

use crate::mixer::SpatialAudioMixer;

/// Active Windows WASAPI hardware output stream driving [`SpatialAudioMixer`].
pub struct WasapiAudioOutput {
    sample_rate: u32,
    stop_flag: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl WasapiAudioOutput {
    /// Initialize Windows WASAPI shared-mode float32 output feeding from `mixer`.
    pub fn start(mixer: Arc<Mutex<SpatialAudioMixer>>) -> Result<Self, String> {
        let target_rate = mixer
            .lock()
            .map(|m| m.sample_rate)
            .unwrap_or(crate::mixer::DEFAULT_MIXER_SAMPLE_RATE);

        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_worker = Arc::clone(&stop_flag);
        let (init_tx, init_rx) = mpsc::sync_channel::<Result<u32, String>>(1);

        let handle = thread::Builder::new()
            .name("xray-wasapi-audio".to_string())
            .spawn(move || {
                wasapi_worker_thread(mixer, target_rate, stop_worker, init_tx);
            })
            .map_err(|e| format!("failed to spawn WASAPI worker thread: {e}"))?;

        match init_rx.recv() {
            Ok(Ok(rate)) => Ok(Self {
                sample_rate: rate,
                stop_flag,
                thread: Some(handle),
            }),
            Ok(Err(err)) => {
                stop_flag.store(true, Ordering::Release);
                let _ = handle.join();
                Err(err)
            }
            Err(_) => {
                stop_flag.store(true, Ordering::Release);
                let _ = handle.join();
                Err("WASAPI initialization channel disconnected".to_string())
            }
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl Drop for WasapiAudioOutput {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

/// Active audio output mode selected by [`crate::manager::AudioManager`].
pub enum AudioOutputBackend {
    /// Live Windows WASAPI shared-mode float32 output stream.
    Wasapi(WasapiAudioOutput),
    /// Headless software-only backend (tests, `--headless` server, or environments with no endpoint).
    HeadlessNull {
        sample_rate: u32,
        fallback_reason: Option<String>,
    },
}

impl AudioOutputBackend {
    /// Open hardware WASAPI output if `prefer_hardware == true`, falling back cleanly to
    /// [`AudioOutputBackend::HeadlessNull`] if no endpoint is available.
    pub fn open_default(mixer: Arc<Mutex<SpatialAudioMixer>>, prefer_hardware: bool) -> Self {
        let rate = mixer
            .lock()
            .map(|m| m.sample_rate)
            .unwrap_or(crate::mixer::DEFAULT_MIXER_SAMPLE_RATE);

        if !prefer_hardware {
            return Self::HeadlessNull {
                sample_rate: rate,
                fallback_reason: None,
            };
        }

        match WasapiAudioOutput::start(mixer) {
            Ok(wasapi) => Self::Wasapi(wasapi),
            Err(reason) => Self::HeadlessNull {
                sample_rate: rate,
                fallback_reason: Some(reason),
            },
        }
    }

    pub fn is_hardware_active(&self) -> bool {
        matches!(self, Self::Wasapi(_))
    }

    pub fn sample_rate(&self) -> u32 {
        match self {
            Self::Wasapi(w) => w.sample_rate(),
            Self::HeadlessNull { sample_rate, .. } => *sample_rate,
        }
    }

    pub fn fallback_reason(&self) -> Option<&str> {
        match self {
            Self::Wasapi(_) => None,
            Self::HeadlessNull {
                fallback_reason, ..
            } => fallback_reason.as_deref(),
        }
    }
}

fn wasapi_worker_thread(
    mixer: Arc<Mutex<SpatialAudioMixer>>,
    sample_rate: u32,
    stop_flag: Arc<AtomicBool>,
    init_tx: mpsc::SyncSender<Result<u32, String>>,
) {
    // Initialize MTA COM apartment strictly on the worker thread so Win32 STA threads remain untouched.
    let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    let com_initialized = hr.is_ok();

    let setup = unsafe { setup_wasapi_client(sample_rate) };
    let (client, render_client, buffer_frames) = match setup {
        Ok(res) => {
            let _ = init_tx.send(Ok(sample_rate));
            res
        }
        Err(err) => {
            let _ = init_tx.send(Err(err));
            if com_initialized {
                unsafe { CoUninitialize() };
            }
            return;
        }
    };

    let mut scratch = vec![0.0f32; buffer_frames as usize * 2];

    // Prime initial buffer cleanly before starting playback
    if let Ok(ptr) = unsafe { render_client.GetBuffer(buffer_frames) } {
        let len = buffer_frames as usize * 2;
        if let Ok(mut guard) = mixer.lock() {
            guard.mix_frames(&mut scratch[..len]);
        } else {
            scratch[..len].fill(0.0);
        }
        unsafe {
            let dst = std::slice::from_raw_parts_mut(ptr as *mut f32, len);
            dst.copy_from_slice(&scratch[..len]);
            let _ = render_client.ReleaseBuffer(buffer_frames, 0);
        }
    }

    if unsafe { client.Start() }.is_err() {
        if com_initialized {
            unsafe { CoUninitialize() };
        }
        return;
    }

    while !stop_flag.load(Ordering::Acquire) {
        let padding = match unsafe { client.GetCurrentPadding() } {
            Ok(p) => p,
            Err(_) => break,
        };
        let available = buffer_frames.saturating_sub(padding);
        if available > 0 {
            let len = available as usize * 2;
            if scratch.len() < len {
                scratch.resize(len, 0.0);
            }
            if let Ok(mut guard) = mixer.lock() {
                guard.mix_frames(&mut scratch[..len]);
            } else {
                scratch[..len].fill(0.0);
            }
            if let Ok(ptr) = unsafe { render_client.GetBuffer(available) } {
                unsafe {
                    let dst = std::slice::from_raw_parts_mut(ptr as *mut f32, len);
                    dst.copy_from_slice(&scratch[..len]);
                    let _ = render_client.ReleaseBuffer(available, 0);
                }
            }
        }
        thread::sleep(Duration::from_millis(5));
    }

    unsafe {
        let _ = client.Stop();
        if com_initialized {
            CoUninitialize();
        }
    }
}

unsafe fn setup_wasapi_client(
    sample_rate: u32,
) -> Result<(IAudioClient, IAudioRenderClient, u32), String> {
    let enumerator: IMMDeviceEnumerator =
        unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
            .map_err(|e| format!("CoCreateInstance(MMDeviceEnumerator) failed: {e}"))?;

    let device = unsafe { enumerator.GetDefaultAudioEndpoint(eRender, eConsole) }
        .map_err(|e| format!("GetDefaultAudioEndpoint failed: {e}"))?;

    let client: IAudioClient = unsafe { device.Activate(CLSCTX_ALL, None) }
        .map_err(|e| format!("IMMDevice::Activate(IAudioClient) failed: {e}"))?;

    let wfex = WAVEFORMATEX {
        wFormatTag: WAVE_FORMAT_IEEE_FLOAT as u16,
        nChannels: 2,
        nSamplesPerSec: sample_rate,
        nAvgBytesPerSec: sample_rate * 8,
        nBlockAlign: 8,
        wBitsPerSample: 32,
        cbSize: 0,
    };

    let stream_flags =
        AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    // 40 ms shared-mode buffer (in 100ns units)
    let hns_buffer_duration: i64 = 400_000;

    unsafe {
        client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            stream_flags,
            hns_buffer_duration,
            0,
            &wfex,
            None,
        )
    }
    .map_err(|e| format!("IAudioClient::Initialize failed: {e}"))?;

    let buffer_frames =
        unsafe { client.GetBufferSize() }.map_err(|e| format!("GetBufferSize failed: {e}"))?;
    let render_client: IAudioRenderClient =
        unsafe { client.GetService() }.map_err(|e| format!("GetService failed: {e}"))?;

    Ok((client, render_client, buffer_frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_headless_and_hardware_fallback() {
        let mixer = Arc::new(Mutex::new(SpatialAudioMixer::default()));
        let headless = AudioOutputBackend::open_default(Arc::clone(&mixer), false);
        assert!(!headless.is_hardware_active());
        assert_eq!(headless.sample_rate(), crate::mixer::DEFAULT_MIXER_SAMPLE_RATE);

        // Hardware open either activates live WASAPI on Windows workstations or falls back cleanly
        let auto_backend = AudioOutputBackend::open_default(mixer, true);
        assert_eq!(
            auto_backend.sample_rate(),
            crate::mixer::DEFAULT_MIXER_SAMPLE_RATE
        );
    }
}
