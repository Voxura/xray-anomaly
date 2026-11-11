//! Zero-Copy Read-Only Memory-Mapped File Abstraction
//!
//! Uses native Win32 file mapping APIs (`CreateFileW` / `CreateFileMappingW` / `MapViewOfFile`)
//! with zero external crate dependencies so large `.db` / `.xdb` X-Ray archives map directly
//! into virtual address space and can be queried across worker threads without heap copies.

use std::io;
use std::ops::Deref;
use std::path::Path;

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;
    use std::ptr::NonNull;
    use std::{io, slice};

    type Handle = isize;
    const INVALID_HANDLE_VALUE: Handle = -1;
    const GENERIC_READ: u32 = 0x8000_0000;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;
    const OPEN_EXISTING: u32 = 3;
    const FILE_ATTRIBUTE_NORMAL: u32 = 0x0000_0080;
    const PAGE_READONLY: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x0004;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileW(
            lp_file_name: *const u16,
            dw_desired_access: u32,
            dw_share_mode: u32,
            lp_security_attributes: *const c_void,
            dw_creation_disposition: u32,
            dw_flags_and_attributes: u32,
            h_template_file: Handle,
        ) -> Handle;
        fn GetFileSizeEx(h_file: Handle, lp_file_size: *mut i64) -> i32;
        fn CreateFileMappingW(
            h_file: Handle,
            lp_file_mapping_attributes: *const c_void,
            fl_protect: u32,
            dw_maximum_size_high: u32,
            dw_maximum_size_low: u32,
            lp_name: *const u16,
        ) -> Handle;
        fn MapViewOfFile(
            h_file_mapping_object: Handle,
            dw_desired_access: u32,
            dw_file_offset_high: u32,
            dw_file_offset_low: u32,
            dw_number_of_bytes_to_map: usize,
        ) -> *const c_void;
        fn UnmapViewOfFile(lp_base_address: *const c_void) -> i32;
        fn CloseHandle(h_object: Handle) -> i32;
    }

    pub struct MmapInner {
        view_ptr: NonNull<u8>,
        len: usize,
        file_handle: Handle,
        mapping_handle: Handle,
        mapped: bool,
    }

    // SAFETY: The mapped view is strictly read-only (PAGE_READONLY + FILE_MAP_READ)
    // and its backing Win32 section handles remain valid until `MmapInner` is dropped.
    unsafe impl Send for MmapInner {}
    unsafe impl Sync for MmapInner {}

    impl MmapInner {
        pub fn open(path: &Path) -> io::Result<Self> {
            let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
            if wide.contains(&0) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "path contains interior NUL byte",
                ));
            }
            wide.push(0);

            unsafe {
                let file_handle = CreateFileW(
                    wide.as_ptr(),
                    GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    std::ptr::null(),
                    OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL,
                    0,
                );
                if file_handle == INVALID_HANDLE_VALUE {
                    return Err(io::Error::last_os_error());
                }

                let mut raw_size: i64 = 0;
                if GetFileSizeEx(file_handle, &mut raw_size) == 0 {
                    let err = io::Error::last_os_error();
                    CloseHandle(file_handle);
                    return Err(err);
                }

                if raw_size < 0 {
                    CloseHandle(file_handle);
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "negative file size reported by OS",
                    ));
                }

                if raw_size == 0 {
                    CloseHandle(file_handle);
                    return Ok(Self {
                        view_ptr: NonNull::dangling(),
                        len: 0,
                        file_handle: 0,
                        mapping_handle: 0,
                        mapped: false,
                    });
                }

                let len = usize::try_from(raw_size).map_err(|_| {
                    CloseHandle(file_handle);
                    io::Error::new(
                        io::ErrorKind::FileTooLarge,
                        "mapped file exceeds virtual address space",
                    )
                })?;

                if len > isize::MAX as usize {
                    CloseHandle(file_handle);
                    return Err(io::Error::new(
                        io::ErrorKind::FileTooLarge,
                        "mapped file exceeds Rust slice limit",
                    ));
                }

                let mapping_handle = CreateFileMappingW(
                    file_handle,
                    std::ptr::null(),
                    PAGE_READONLY,
                    0,
                    0,
                    std::ptr::null(),
                );
                if mapping_handle == 0 {
                    let err = io::Error::last_os_error();
                    CloseHandle(file_handle);
                    return Err(err);
                }

                let raw_ptr = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, len);
                if raw_ptr.is_null() {
                    let err = io::Error::last_os_error();
                    CloseHandle(mapping_handle);
                    CloseHandle(file_handle);
                    return Err(err);
                }

                Ok(Self {
                    view_ptr: NonNull::new_unchecked(raw_ptr as *mut u8),
                    len,
                    file_handle,
                    mapping_handle,
                    mapped: true,
                })
            }
        }

        #[inline]
        pub fn as_slice(&self) -> &[u8] {
            if self.len == 0 {
                &[]
            } else {
                // SAFETY: `view_ptr` points to `len` read-only bytes mapped until Drop.
                unsafe { slice::from_raw_parts(self.view_ptr.as_ptr(), self.len) }
            }
        }
    }

    impl Drop for MmapInner {
        fn drop(&mut self) {
            if self.mapped {
                unsafe {
                    UnmapViewOfFile(self.view_ptr.as_ptr() as *const c_void);
                    CloseHandle(self.mapping_handle);
                    CloseHandle(self.file_handle);
                }
            }
        }
    }
}

#[cfg(not(windows))]
mod fallback {
    use std::path::Path;
    use std::{fs, io};

    pub struct MmapInner {
        data: Vec<u8>,
    }

    impl MmapInner {
        pub fn open(path: &Path) -> io::Result<Self> {
            Ok(Self {
                data: fs::read(path)?,
            })
        }

        #[inline]
        pub fn as_slice(&self) -> &[u8] {
            &self.data
        }
    }
}

#[cfg(windows)]
use win::MmapInner;
#[cfg(not(windows))]
use fallback::MmapInner;

/// Read-only memory-mapped file exposing a zero-copy `&[u8]` byte slice.
pub struct MappedFile {
    inner: MmapInner,
}

impl MappedFile {
    /// Open and map `path` read-only into process address space.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Self {
            inner: MmapInner::open(path.as_ref())?,
        })
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.inner.as_slice()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.as_slice().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Deref for MappedFile {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for MappedFile {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_file(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("xray_vfs_mmap_{}_{}_{}", std::process::id(), nanos, name))
    }

    #[test]
    fn test_mmap_read_only_and_empty_file() {
        let empty_path = unique_temp_file("empty.bin");
        std::fs::write(&empty_path, b"").unwrap();
        {
            let mapped = MappedFile::open(&empty_path).expect("empty file should map cleanly");
            assert!(mapped.is_empty());
            assert_eq!(&*mapped, b"");
        }
        let _ = std::fs::remove_file(&empty_path);

        let payload = b"X-Ray Anomaly Zero-Copy Mmap Payload 0123456789ABCDEF";
        let path = unique_temp_file("data.bin");
        std::fs::write(&path, payload).unwrap();
        {
            let mapped = MappedFile::open(&path).expect("file should map read-only");
            assert_eq!(mapped.len(), payload.len());
            assert_eq!(mapped.as_slice(), payload);
        }
        std::fs::remove_file(&path).unwrap();
    }
}
