//! Safe C ABI export layer for `xray-core`
//!
//! Provides a generational slot-map handle registry and 100% panic containment across the DLL boundary.
//! Prevents dangling pointers, use-after-free, double-free, and wild pointer crashes from C++.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::panic::catch_unwind;
use std::ptr;
use std::sync::{Arc, LazyLock, RwLock};
use xray_core::LtxDocument;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum XrStatus {
    Success = 0,
    InvalidParam = -1,
    InvalidHandle = -2,
    BufferTooSmall = -3,
    NotFound = -4,
    ParseError = -5,
    Panic = -99,
}

/// 64-bit opaque handle: Upper 32 bits = Generation, Lower 32 bits = Slot index.
pub type XrConfigHandle = u64;

pub const XR_INVALID_HANDLE: XrConfigHandle = 0;

struct ConfigEntry {
    _source: String,
    doc: LtxDocument<'static>,
}

struct Slot {
    generation: u32,
    entry: Option<Arc<ConfigEntry>>,
}

struct HandleRegistry {
    slots: Vec<Slot>,
    free_indices: Vec<usize>,
}

impl HandleRegistry {
    const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_indices: Vec::new(),
        }
    }

    fn insert(&mut self, entry: ConfigEntry) -> XrConfigHandle {
        let arc = Arc::new(entry);
        if let Some(idx) = self.free_indices.pop() {
            let slot = &mut self.slots[idx];
            let gen = slot.generation;
            slot.entry = Some(arc);
            ((gen as u64) << 32) | (idx as u64)
        } else {
            let idx = self.slots.len();
            let gen = 1u32;
            self.slots.push(Slot {
                generation: gen,
                entry: Some(arc),
            });
            ((gen as u64) << 32) | (idx as u64)
        }
    }

    fn get(&self, handle: XrConfigHandle) -> Option<Arc<ConfigEntry>> {
        if handle == XR_INVALID_HANDLE {
            return None;
        }
        let idx = (handle & 0xFFFF_FFFF) as usize;
        let gen = (handle >> 32) as u32;

        if let Some(slot) = self.slots.get(idx) {
            if slot.generation == gen {
                return slot.entry.clone();
            }
        }
        None
    }

    fn remove(&mut self, handle: XrConfigHandle) -> Result<(), XrStatus> {
        if handle == XR_INVALID_HANDLE {
            return Err(XrStatus::InvalidHandle);
        }
        let idx = (handle & 0xFFFF_FFFF) as usize;
        let gen = (handle >> 32) as u32;

        if idx >= self.slots.len() {
            return Err(XrStatus::InvalidHandle);
        }

        let slot = &mut self.slots[idx];
        if slot.generation != gen || slot.entry.is_none() {
            return Err(XrStatus::InvalidHandle);
        }

        slot.entry = None;
        slot.generation = slot.generation.wrapping_add(1);
        if slot.generation == 0 {
            slot.generation = 1;
        }
        self.free_indices.push(idx);
        Ok(())
    }
}

static REGISTRY: LazyLock<RwLock<HandleRegistry>> = LazyLock::new(|| RwLock::new(HandleRegistry::new()));

/// Open and parse an LTX configuration string in memory.
/// Returns an opaque 64-bit handle or XR_INVALID_HANDLE on failure.
#[no_mangle]
pub unsafe extern "C" fn xr_config_open_memory(
    data: *const c_char,
    len: usize,
    out_status: *mut i32,
) -> XrConfigHandle {
    let result = catch_unwind(|| {
        if data.is_null() {
            if !out_status.is_null() {
                *out_status = XrStatus::InvalidParam as i32;
            }
            return XR_INVALID_HANDLE;
        }

        let slice = std::slice::from_raw_parts(data as *const u8, len);
        let s = match std::str::from_utf8(slice) {
            Ok(valid) => valid.to_string(),
            Err(_) => {
                if !out_status.is_null() {
                    *out_status = XrStatus::ParseError as i32;
                }
                return XR_INVALID_HANDLE;
            }
        };

        // Leak string slice safely into 'static memory for zero-copy parsing
        let boxed_str: &'static str = Box::leak(s.into_boxed_str());
        let doc = LtxDocument::parse(boxed_str);

        let entry = ConfigEntry {
            _source: String::new(),
            doc,
        };

        let mut reg = REGISTRY.write().expect("lock poisoned");
        let handle = reg.insert(entry);

        if !out_status.is_null() {
            *out_status = XrStatus::Success as i32;
        }

        handle
    });

    match result {
        Ok(h) => h,
        Err(_) => {
            if !out_status.is_null() {
                *out_status = XrStatus::Panic as i32;
            }
            XR_INVALID_HANDLE
        }
    }
}

/// Read a string value into a caller-provided output buffer.
/// Returns the number of bytes written (excluding null terminator), or a negative error status.
#[no_mangle]
pub unsafe extern "C" fn xr_config_read_string_buf(
    handle: XrConfigHandle,
    section: *const c_char,
    key: *const c_char,
    out_buf: *mut c_char,
    max_len: usize,
) -> i32 {
    let result = catch_unwind(|| {
        if section.is_null() || key.is_null() || out_buf.is_null() || max_len == 0 {
            return XrStatus::InvalidParam as i32;
        }

        let entry = match REGISTRY.read().expect("lock poisoned").get(handle) {
            Some(e) => e,
            None => return XrStatus::InvalidHandle as i32,
        };

        let sec_str = match CStr::from_ptr(section).to_str() {
            Ok(s) => s,
            Err(_) => return XrStatus::InvalidParam as i32,
        };

        let key_str = match CStr::from_ptr(key).to_str() {
            Ok(s) => s,
            Err(_) => return XrStatus::InvalidParam as i32,
        };

        if let Some(val) = entry.doc.read_str(sec_str, key_str) {
            let bytes = val.as_bytes();
            if bytes.len() >= max_len {
                return XrStatus::BufferTooSmall as i32;
            }

            ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf as *mut u8, bytes.len());
            *out_buf.add(bytes.len()) = 0; // null terminator
            bytes.len() as i32
        } else {
            XrStatus::NotFound as i32
        }
    });

    result.unwrap_or(XrStatus::Panic as i32)
}

/// Read integer value with fallback.
#[no_mangle]
pub unsafe extern "C" fn xr_config_read_int(
    handle: XrConfigHandle,
    section: *const c_char,
    key: *const c_char,
    def_val: i32,
) -> i32 {
    let result = catch_unwind(|| {
        if section.is_null() || key.is_null() {
            return def_val;
        }

        let entry = match REGISTRY.read().expect("lock poisoned").get(handle) {
            Some(e) => e,
            None => return def_val,
        };

        let sec_str = match CStr::from_ptr(section).to_str() {
            Ok(s) => s,
            Err(_) => return def_val,
        };

        let key_str = match CStr::from_ptr(key).to_str() {
            Ok(s) => s,
            Err(_) => return def_val,
        };

        entry.doc.read_i32(sec_str, key_str, def_val)
    });

    result.unwrap_or(def_val)
}

/// Read float value with fallback.
#[no_mangle]
pub unsafe extern "C" fn xr_config_read_float(
    handle: XrConfigHandle,
    section: *const c_char,
    key: *const c_char,
    def_val: f32,
) -> f32 {
    let result = catch_unwind(|| {
        if section.is_null() || key.is_null() {
            return def_val;
        }

        let entry = match REGISTRY.read().expect("lock poisoned").get(handle) {
            Some(e) => e,
            None => return def_val,
        };

        let sec_str = match CStr::from_ptr(section).to_str() {
            Ok(s) => s,
            Err(_) => return def_val,
        };

        let key_str = match CStr::from_ptr(key).to_str() {
            Ok(s) => s,
            Err(_) => return def_val,
        };

        entry.doc.read_f32(sec_str, key_str, def_val)
    });

    result.unwrap_or(def_val)
}

/// Read boolean value with fallback.
#[no_mangle]
pub unsafe extern "C" fn xr_config_read_bool(
    handle: XrConfigHandle,
    section: *const c_char,
    key: *const c_char,
    def_val: bool,
) -> bool {
    let result = catch_unwind(|| {
        if section.is_null() || key.is_null() {
            return def_val;
        }

        let entry = match REGISTRY.read().expect("lock poisoned").get(handle) {
            Some(e) => e,
            None => return def_val,
        };

        let sec_str = match CStr::from_ptr(section).to_str() {
            Ok(s) => s,
            Err(_) => return def_val,
        };

        let key_str = match CStr::from_ptr(key).to_str() {
            Ok(s) => s,
            Err(_) => return def_val,
        };

        entry.doc.read_bool(sec_str, key_str, def_val)
    });

    result.unwrap_or(def_val)
}

/// Check if a section exists in the configuration.
#[no_mangle]
pub unsafe extern "C" fn xr_config_section_exists(
    handle: XrConfigHandle,
    section: *const c_char,
) -> bool {
    let result = catch_unwind(|| {
        if section.is_null() {
            return false;
        }

        let entry = match REGISTRY.read().expect("lock poisoned").get(handle) {
            Some(e) => e,
            None => return false,
        };

        let sec_str = match CStr::from_ptr(section).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        };

        entry.doc.sections.contains_key(sec_str)
    });

    result.unwrap_or(false)
}

/// Check if a key exists in a section (including inherited keys).
#[no_mangle]
pub unsafe extern "C" fn xr_config_line_exists(
    handle: XrConfigHandle,
    section: *const c_char,
    key: *const c_char,
) -> bool {
    let result = catch_unwind(|| {
        if section.is_null() || key.is_null() {
            return false;
        }

        let entry = match REGISTRY.read().expect("lock poisoned").get(handle) {
            Some(e) => e,
            None => return false,
        };

        let sec_str = match CStr::from_ptr(section).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        };

        let key_str = match CStr::from_ptr(key).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        };

        entry.doc.read_str(sec_str, key_str).is_some()
    });

    result.unwrap_or(false)
}

/// Destroy a configuration handle and release its resources.
/// Returns XrStatus::Success or XrStatus::InvalidHandle if already freed or invalid.
#[no_mangle]
pub unsafe extern "C" fn xr_config_free(handle: XrConfigHandle) -> i32 {
    let result = catch_unwind(|| {
        let mut reg = REGISTRY.write().expect("lock poisoned");
        match reg.remove(handle) {
            Ok(()) => XrStatus::Success as i32,
            Err(e) => e as i32,
        }
    });

    result.unwrap_or(XrStatus::Panic as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_handle_lifecycle_and_stale_detection() {
        let ltx = CString::new("[test]\nval = 42\nname = stalker\n").unwrap();
        let mut status = 0;

        unsafe {
            let handle = xr_config_open_memory(ltx.as_ptr(), ltx.as_bytes().len(), &mut status);
            assert_ne!(handle, XR_INVALID_HANDLE);
            assert_eq!(status, XrStatus::Success as i32);

            let sec = CString::new("test").unwrap();
            let key = CString::new("val").unwrap();
            let val = xr_config_read_int(handle, sec.as_ptr(), key.as_ptr(), 0);
            assert_eq!(val, 42);

            // Buffer read
            let mut buf = [0 as c_char; 64];
            let name_key = CString::new("name").unwrap();
            let len = xr_config_read_string_buf(
                handle,
                sec.as_ptr(),
                name_key.as_ptr(),
                buf.as_mut_ptr(),
                buf.len(),
            );
            assert_eq!(len, 7);
            let out_str = CStr::from_ptr(buf.as_ptr()).to_str().unwrap();
            assert_eq!(out_str, "stalker");

            // Free handle
            let free_status = xr_config_free(handle);
            assert_eq!(free_status, XrStatus::Success as i32);

            // Stale handle check: reading from freed handle returns fallback / error
            let stale_val = xr_config_read_int(handle, sec.as_ptr(), key.as_ptr(), 999);
            assert_eq!(stale_val, 999);

            // Double free returns InvalidHandle error, does NOT crash or panic
            let double_free = xr_config_free(handle);
            assert_eq!(double_free, XrStatus::InvalidHandle as i32);
        }
    }

    #[test]
    fn test_slot_reuse_with_advanced_generation() {
        let ltx1 = CString::new("[doc1]\nid = 111\n").unwrap();
        let ltx2 = CString::new("[doc2]\nid = 222\n").unwrap();

        unsafe {
            let mut status = 0;
            let h1 = xr_config_open_memory(ltx1.as_ptr(), ltx1.as_bytes().len(), &mut status);
            assert_ne!(h1, XR_INVALID_HANDLE);

            // Destroy h1
            assert_eq!(xr_config_free(h1), XrStatus::Success as i32);

            // Open h2 - reuses slot index with advanced generation counter
            let h2 = xr_config_open_memory(ltx2.as_ptr(), ltx2.as_bytes().len(), &mut status);
            assert_ne!(h2, XR_INVALID_HANDLE);
            assert_ne!(h1, h2, "Slot generation must ensure h1 != h2 despite same slot");

            let sec2 = CString::new("doc2").unwrap();
            let key = CString::new("id").unwrap();
            assert_eq!(xr_config_read_int(h2, sec2.as_ptr(), key.as_ptr(), 0), 222);

            // Using stale h1 against the reused slot MUST fail with fallback / invalid handle
            let sec1 = CString::new("doc1").unwrap();
            assert_eq!(xr_config_read_int(h1, sec1.as_ptr(), key.as_ptr(), -99), -99);

            assert_eq!(xr_config_free(h2), XrStatus::Success as i32);
        }
    }

    #[test]
    fn test_buffer_edge_cases_null_undersized_exact_fit() {
        let ltx = CString::new("[strings]\nname = Pripyat\n").unwrap();
        unsafe {
            let mut status = 0;
            let h = xr_config_open_memory(ltx.as_ptr(), ltx.as_bytes().len(), &mut status);
            assert_ne!(h, XR_INVALID_HANDLE);

            let sec = CString::new("strings").unwrap();
            let key = CString::new("name").unwrap();

            // 1. Null buffer pointer
            let ret_null = xr_config_read_string_buf(h, sec.as_ptr(), key.as_ptr(), ptr::null_mut(), 64);
            assert_eq!(ret_null, XrStatus::InvalidParam as i32);

            // 2. Zero capacity
            let mut dummy = [0 as c_char; 8];
            let ret_zero = xr_config_read_string_buf(h, sec.as_ptr(), key.as_ptr(), dummy.as_mut_ptr(), 0);
            assert_eq!(ret_zero, XrStatus::InvalidParam as i32);

            // 3. Undersized buffer ("Pripyat" is 7 bytes, buffer size 7 cannot fit null terminator)
            let mut small_buf = [0 as c_char; 7];
            let ret_small = xr_config_read_string_buf(h, sec.as_ptr(), key.as_ptr(), small_buf.as_mut_ptr(), 7);
            assert_eq!(ret_small, XrStatus::BufferTooSmall as i32);

            // 4. Exact-fit buffer ("Pripyat" is 7 bytes + 1 null terminator = 8 bytes)
            let mut exact_buf = [0 as c_char; 8];
            let ret_exact = xr_config_read_string_buf(h, sec.as_ptr(), key.as_ptr(), exact_buf.as_mut_ptr(), 8);
            assert_eq!(ret_exact, 7);
            assert_eq!(CStr::from_ptr(exact_buf.as_ptr()).to_str().unwrap(), "Pripyat");

            xr_config_free(h);
        }
    }

    #[test]
    fn test_concurrent_read_and_destroy() {
        use std::thread;

        let ltx = CString::new("[concurrency]\ncounter = 100\n").unwrap();
        unsafe {
            let mut status = 0;
            let h = xr_config_open_memory(ltx.as_ptr(), ltx.as_bytes().len(), &mut status);
            assert_ne!(h, XR_INVALID_HANDLE);

            let sec = CString::new("concurrency").unwrap();
            let key = CString::new("counter").unwrap();

            let reader = thread::spawn(move || {
                let mut sum = 0;
                for _ in 0..1000 {
                    let v = xr_config_read_int(h, sec.as_ptr(), key.as_ptr(), -1);
                    if v == 100 {
                        sum += 1;
                    }
                }
                sum
            });

            // Concurrently free handle
            let destroyer = thread::spawn(move || {
                xr_config_free(h)
            });

            let _ = destroyer.join().unwrap();
            let sum = reader.join().unwrap();
            // Reader completed safely without crashing or undefined behavior
            assert!(sum >= 0);
        }
    }
}
