use core::ffi::c_void;
use core::fmt::{self, Write};
use core::sync::atomic::{AtomicBool, Ordering};

static RUNNING: AtomicBool = AtomicBool::new(true);

#[inline]
pub fn is_running() -> bool {
    RUNNING.load(Ordering::Acquire)
}

#[inline]
pub fn request_stop() {
    RUNNING.store(false, Ordering::Release);
}

pub struct StackBuf<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> StackBuf<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            buf: [0; N],
            len: 0,
        }
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> bool {
        if self.len + bytes.len() > N {
            return false;
        }
        let mut i = 0;
        while i < bytes.len() {
            self.buf[self.len + i] = bytes[i];
            i += 1;
        }
        self.len += bytes.len();
        true
    }

    pub fn push_cstr_end(&mut self) -> bool {
        self.push_bytes(&[0])
    }
}

impl<const N: usize> Write for StackBuf<N> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if self.push_bytes(s.as_bytes()) {
            Ok(())
        } else {
            Err(fmt::Error)
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessStats {
    pub cpu_time_ns: u64,
    pub private_bytes: u64,
    pub write_bytes: u64,
    pub write_ops: u64,
}

/// Extracts the first string entry inside `<key> : [ "..." ]` of SteamVR's
/// `openvrpaths.vrpath` without allocating or requiring a full JSON parser.
fn parse_vrpath_array_first<'a>(vrpath: &[u8], key: &[u8], out: &'a mut [u8]) -> Option<&'a [u8]> {
    let key_pos = find_subslice(vrpath, key)?;
    let after_key = &vrpath[key_pos + key.len()..];
    let bracket_pos = find_byte(after_key, b'[')?;
    let after_bracket = &after_key[bracket_pos + 1..];
    let quote_start = find_byte(after_bracket, b'"')?;
    let content = &after_bracket[quote_start + 1..];

    let mut out_len = 0usize;
    let mut i = 0usize;
    while i < content.len() {
        let b = content[i];
        if b == b'"' {
            return Some(&out[..out_len]);
        }
        if b == b'\\' {
            i += 1;
            if i >= content.len() {
                return None;
            }
            let esc = content[i];
            let decoded = match esc {
                b'\\' => b'\\',
                b'/' => b'/',
                b'"' => b'"',
                other => other,
            };
            if out_len >= out.len() {
                return None;
            }
            out[out_len] = decoded;
            out_len += 1;
        } else {
            if out_len >= out.len() {
                return None;
            }
            out[out_len] = b;
            out_len += 1;
        }
        i += 1;
    }
    None
}

pub fn parse_vrpath_runtime<'a>(vrpath: &[u8], out: &'a mut [u8]) -> Option<&'a [u8]> {
    parse_vrpath_array_first(vrpath, b"\"runtime\"", out)
}

pub fn parse_vrpath_config<'a>(vrpath: &[u8], out: &'a mut [u8]) -> Option<&'a [u8]> {
    parse_vrpath_array_first(vrpath, b"\"config\"", out)
}

fn find_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    let mut i = 0;
    while i < haystack.len() {
        if haystack[i] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    let limit = haystack.len() - needle.len() + 1;
    let mut i = 0;
    while i < limit {
        let mut matched = true;
        let mut j = 0;
        while j < needle.len() {
            if haystack[i + j] != needle[j] {
                matched = false;
                break;
            }
            j += 1;
        }
        if matched {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn pop_path_component(path: &[u8]) -> &[u8] {
    let mut end = path.len();
    while end > 0 && (path[end - 1] == b'\\' || path[end - 1] == b'/') {
        end -= 1;
    }
    while end > 0 && path[end - 1] != b'\\' && path[end - 1] != b'/' {
        end -= 1;
    }
    if end > 0 && (path[end - 1] == b'\\' || path[end - 1] == b'/') {
        end -= 1;
    }
    &path[..end]
}

// ============================================================================
// Windows implementation
// ============================================================================
#[cfg(target_os = "windows")]
mod imp {
    use super::*;

    const STD_OUTPUT_HANDLE: u32 = (-11i32) as u32;
    const STD_ERROR_HANDLE: u32 = (-12i32) as u32;
    const GENERIC_READ: u32 = 0x8000_0000;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;
    const FILE_SHARE_DELETE: u32 = 0x0000_0004;
    const OPEN_EXISTING: u32 = 3;
    const FILE_ATTRIBUTE_NORMAL: u32 = 0x0000_0080;
    const INVALID_HANDLE_VALUE: isize = -1;
    const INVALID_FILE_ATTRIBUTES: u32 = 0xFFFF_FFFF;
    const CREATE_WAITABLE_TIMER_HIGH_RESOLUTION: u32 = 0x0000_0002;
    const TIMER_ALL_ACCESS: u32 = 0x001F_0003;
    const INFINITE: u32 = 0xFFFF_FFFF;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    impl FileTime {
        #[inline]
        fn to_ns(self) -> u64 {
            let ticks100ns = ((self.high as u64) << 32) | (self.low as u64);
            ticks100ns.saturating_mul(100)
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct ProcessMemoryCountersEx {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Win32FileAttributeData {
        file_attributes: u32,
        creation_time: FileTime,
        last_access_time: FileTime,
        last_write_time: FileTime,
        file_size_high: u32,
        file_size_low: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct IoCounters {
        read_operation_count: u64,
        write_operation_count: u64,
        other_operation_count: u64,
        read_transfer_count: u64,
        write_transfer_count: u64,
        other_transfer_count: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(nStdHandle: u32) -> isize;
        fn WriteFile(
            hFile: isize,
            lpBuffer: *const u8,
            nNumberOfBytesToWrite: u32,
            lpNumberOfBytesWritten: *mut u32,
            lpOverlapped: *mut c_void,
        ) -> i32;
        fn ExitProcess(uExitCode: u32) -> !;
        fn GetCommandLineW() -> *const u16;
        fn GetModuleFileNameW(hModule: isize, lpFilename: *mut u16, nSize: u32) -> u32;
        fn GetCurrentDirectoryW(nBufferLength: u32, lpBuffer: *mut u16) -> u32;
        fn GetEnvironmentVariableW(lpName: *const u16, lpBuffer: *mut u16, nSize: u32) -> u32;
        fn CreateFileW(
            lpFileName: *const u16,
            dwDesiredAccess: u32,
            dwShareMode: u32,
            lpSecurityAttributes: *const c_void,
            dwCreationDisposition: u32,
            dwFlagsAndAttributes: u32,
            hTemplateFile: isize,
        ) -> isize;
        fn ReadFile(
            hFile: isize,
            lpBuffer: *mut u8,
            nNumberOfBytesToRead: u32,
            lpNumberOfBytesRead: *mut u32,
            lpOverlapped: *mut c_void,
        ) -> i32;
        fn GetFileAttributesW(lpFileName: *const u16) -> u32;
        fn GetFileAttributesExW(
            lpFileName: *const u16,
            fInfoLevelId: i32,
            lpFileInformation: *mut Win32FileAttributeData,
        ) -> i32;
        fn CloseHandle(hObject: isize) -> i32;
        fn LoadLibraryW(lpLibFileName: *const u16) -> isize;
        fn GetProcAddress(hModule: isize, lpProcName: *const u8) -> *const c_void;
        fn FreeLibrary(hLibModule: isize) -> i32;
        fn QueryPerformanceCounter(lpPerformanceCount: *mut i64) -> i32;
        fn QueryPerformanceFrequency(lpFrequency: *mut i64) -> i32;
        fn Sleep(dwMilliseconds: u32);
        fn CreateWaitableTimerExW(
            lpTimerAttributes: *const c_void,
            lpTimerName: *const u16,
            dwFlags: u32,
            dwDesiredAccess: u32,
        ) -> isize;
        fn SetWaitableTimer(
            hTimer: isize,
            lpDueTime: *const i64,
            lPeriod: i32,
            pfnCompletionRoutine: *const c_void,
            lpArgToCompletionRoutine: *const c_void,
            fResume: i32,
        ) -> i32;
        fn WaitForSingleObject(hHandle: isize, dwMilliseconds: u32) -> u32;
        fn SetConsoleCtrlHandler(
            handlerRoutine: Option<unsafe extern "system" fn(u32) -> i32>,
            add: i32,
        ) -> i32;
        fn GetCurrentProcess() -> isize;
        fn GetProcessTimes(
            hProcess: isize,
            lpCreationTime: *mut FileTime,
            lpExitTime: *mut FileTime,
            lpKernelTime: *mut FileTime,
            lpUserTime: *mut FileTime,
        ) -> i32;
        fn K32GetProcessMemoryInfo(
            process: isize,
            ppsmemCounters: *mut ProcessMemoryCountersEx,
            cb: u32,
        ) -> i32;
        fn GetProcessIoCounters(hProcess: isize, lpIoCounters: *mut IoCounters) -> i32;
    }

    unsafe extern "system" fn ctrl_handler(_ctrl_type: u32) -> i32 {
        request_stop();
        1
    }

    pub fn install_signal_handlers() {
        unsafe {
            let _ = SetConsoleCtrlHandler(Some(ctrl_handler), 1);
        }
    }

    pub fn write_stdout(bytes: &[u8]) {
        unsafe {
            let h = GetStdHandle(STD_OUTPUT_HANDLE);
            if h != 0 && h != INVALID_HANDLE_VALUE {
                let mut written = 0u32;
                let _ = WriteFile(
                    h,
                    bytes.as_ptr(),
                    bytes.len() as u32,
                    &mut written,
                    core::ptr::null_mut(),
                );
            }
        }
    }

    pub fn write_stderr(bytes: &[u8]) {
        unsafe {
            let h = GetStdHandle(STD_ERROR_HANDLE);
            if h != 0 && h != INVALID_HANDLE_VALUE {
                let mut written = 0u32;
                let _ = WriteFile(
                    h,
                    bytes.as_ptr(),
                    bytes.len() as u32,
                    &mut written,
                    core::ptr::null_mut(),
                );
            }
        }
    }

    pub fn exit_process(code: u32) -> ! {
        unsafe { ExitProcess(code) }
    }

    pub fn abort() -> ! {
        exit_process(101)
    }

    pub struct MonotonicClock {
        freq: u64,
        start: i64,
    }

    impl MonotonicClock {
        pub fn new() -> Self {
            let mut freq = 1i64;
            let mut start = 0i64;
            unsafe {
                let _ = QueryPerformanceFrequency(&mut freq);
                let _ = QueryPerformanceCounter(&mut start);
            }
            Self {
                freq: (freq.max(1)) as u64,
                start,
            }
        }

        #[inline]
        pub fn now_ns(&self) -> u64 {
            let mut now = 0i64;
            unsafe {
                let _ = QueryPerformanceCounter(&mut now);
            }
            let ticks = (now.wrapping_sub(self.start)) as u64;
            ((ticks as u128 * 1_000_000_000u128) / (self.freq as u128)) as u64
        }
    }

    pub struct WaitTimer {
        handle: isize,
    }

    impl WaitTimer {
        pub fn new() -> Self {
            let mut h = unsafe {
                CreateWaitableTimerExW(
                    core::ptr::null(),
                    core::ptr::null(),
                    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                    TIMER_ALL_ACCESS,
                )
            };
            if h == 0 {
                // Fallback to standard waitable timer on pre-Windows 10 1803 systems.
                h = unsafe {
                    CreateWaitableTimerExW(
                        core::ptr::null(),
                        core::ptr::null(),
                        0,
                        TIMER_ALL_ACCESS,
                    )
                };
            }
            Self { handle: h }
        }

        #[inline]
        pub fn sleep_idle_ms(&self, ms: u32) {
            unsafe { Sleep(ms) };
        }

        #[inline]
        pub fn sleep_precise_ns(&self, ns: u64) {
            if ns == 0 {
                return;
            }
            if self.handle != 0 {
                let intervals = -((ns / 100).max(1) as i64);
                let ok = unsafe {
                    SetWaitableTimer(
                        self.handle,
                        &intervals,
                        0,
                        core::ptr::null(),
                        core::ptr::null(),
                        0,
                    )
                };
                if ok != 0 {
                    unsafe {
                        let _ = WaitForSingleObject(self.handle, INFINITE);
                    }
                    return;
                }
            }
            let ms = ((ns + 999_999) / 1_000_000).max(1) as u32;
            unsafe { Sleep(ms) };
        }
    }

    impl Drop for WaitTimer {
        fn drop(&mut self) {
            if self.handle != 0 {
                unsafe {
                    let _ = CloseHandle(self.handle);
                }
            }
        }
    }

    pub struct DynamicLibrary {
        handle: isize,
    }

    impl DynamicLibrary {
        pub fn open_utf8(path_nul: &[u8]) -> Option<Self> {
            let mut wide = [0u16; 520];
            let len = utf8_nul_to_utf16(path_nul, &mut wide)?;
            if len == 0 {
                return None;
            }
            let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
            if handle != 0 {
                Some(Self { handle })
            } else {
                None
            }
        }

        /// # Safety
        /// `symbol_nul` must be a valid NUL-terminated ASCII symbol name.
        pub unsafe fn get_symbol(&self, symbol_nul: &[u8]) -> Option<*const c_void> {
            if symbol_nul.last() != Some(&0) {
                return None;
            }
            let ptr = unsafe { GetProcAddress(self.handle, symbol_nul.as_ptr()) };
            if ptr.is_null() { None } else { Some(ptr) }
        }
    }

    impl Drop for DynamicLibrary {
        fn drop(&mut self) {
            if self.handle != 0 {
                unsafe {
                    let _ = FreeLibrary(self.handle);
                }
            }
        }
    }

    fn read_vrpath_file(out: &mut [u8]) -> Option<usize> {
        let env_name: [u16; 13] = [
            b'L' as u16, b'O' as u16, b'C' as u16, b'A' as u16, b'L' as u16, b'A' as u16,
            b'P' as u16, b'P' as u16, b'D' as u16, b'A' as u16, b'T' as u16, b'A' as u16, 0,
        ];
        let mut local_app_data = [0u16; 520];
        let n = unsafe {
            GetEnvironmentVariableW(
                env_name.as_ptr(),
                local_app_data.as_mut_ptr(),
                local_app_data.len() as u32,
            )
        };
        if n == 0 || (n as usize) >= local_app_data.len() {
            return None;
        }
        let mut vrpath_utf8 = StackBuf::<520>::new();
        if !push_utf16_as_utf8(&local_app_data[..n as usize], &mut vrpath_utf8) {
            return None;
        }
        let _ = vrpath_utf8.push_bytes(b"\\openvr\\openvrpaths.vrpath\0");
        let mut wide = [0u16; 520];
        utf8_nul_to_utf16(vrpath_utf8.as_bytes(), &mut wide)?;
        read_file_head(&wide, out)
    }

    pub fn load_openvr_library() -> Option<DynamicLibrary> {
        if let Some(lib) = DynamicLibrary::open_utf8(b"openvr_api.dll\0") {
            return Some(lib);
        }

        // Try alongside the executable.
        if let Some(exe_dir) = exe_dir_utf8() {
            let mut buf = StackBuf::<520>::new();
            let _ = buf.push_bytes(exe_dir.as_bytes());
            let _ = buf.push_bytes(b"\\openvr_api.dll\0");
            if let Some(lib) = DynamicLibrary::open_utf8(buf.as_bytes()) {
                return Some(lib);
            }
        }

        // Read %LOCALAPPDATA%\openvr\openvrpaths.vrpath to locate SteamVR's runtime DLL.
        let mut file_buf = [0u8; 4096];
        if let Some(read_len) = read_vrpath_file(&mut file_buf) {
            let mut runtime_buf = [0u8; 512];
            if let Some(rt) = parse_vrpath_runtime(&file_buf[..read_len], &mut runtime_buf) {
                let mut dll_path = StackBuf::<600>::new();
                let _ = dll_path.push_bytes(rt);
                let _ = dll_path.push_bytes(b"\\bin\\win64\\openvr_api.dll\0");
                if let Some(lib) = DynamicLibrary::open_utf8(dll_path.as_bytes()) {
                    return Some(lib);
                }
            }
        }
        None
    }

    pub fn query_chaperone_file_stamp() -> Option<(u64, u64)> {
        let mut file_buf = [0u8; 4096];
        let read_len = read_vrpath_file(&mut file_buf)?;
        let mut cfg_buf = [0u8; 512];
        let cfg = parse_vrpath_config(&file_buf[..read_len], &mut cfg_buf)?;
        let mut chap_path = StackBuf::<600>::new();
        let _ = chap_path.push_bytes(cfg);
        let _ = chap_path.push_bytes(b"\\chaperone_info.vrchap\0");
        let mut wide = [0u16; 600];
        utf8_nul_to_utf16(chap_path.as_bytes(), &mut wide)?;
        let mut attr = Win32FileAttributeData::default();
        let ok = unsafe { GetFileAttributesExW(wide.as_ptr(), 0, &mut attr) };
        if ok == 0 {
            return None;
        }
        let size = ((attr.file_size_high as u64) << 32) | (attr.file_size_low as u64);
        Some((attr.last_write_time.to_ns(), size))
    }

    pub fn resolve_action_manifest_path(out: &mut StackBuf<520>) -> bool {
        if let Some(exe_dir) = exe_dir_utf8() {
            let exe_bytes = exe_dir.as_bytes();
            // <exe_dir>\manifest\actions.json
            out.clear();
            let _ = out.push_bytes(exe_bytes);
            let _ = out.push_bytes(b"\\manifest\\actions.json\0");
            if path_exists_utf8_nul(out.as_bytes()) {
                return true;
            }

            // <exe_dir>\..\..\manifest\actions.json (typical cargo target/release layout)
            let p1 = pop_path_component(exe_bytes);
            let p2 = pop_path_component(p1);
            if !p2.is_empty() {
                out.clear();
                let _ = out.push_bytes(p2);
                let _ = out.push_bytes(b"\\manifest\\actions.json\0");
                if path_exists_utf8_nul(out.as_bytes()) {
                    return true;
                }
            }

            // <exe_dir>\actions.json
            out.clear();
            let _ = out.push_bytes(exe_bytes);
            let _ = out.push_bytes(b"\\actions.json\0");
            if path_exists_utf8_nul(out.as_bytes()) {
                return true;
            }
        }

        if let Some(cwd) = cwd_utf8() {
            out.clear();
            let _ = out.push_bytes(cwd.as_bytes());
            let _ = out.push_bytes(b"\\manifest\\actions.json\0");
            if path_exists_utf8_nul(out.as_bytes()) {
                return true;
            }

            out.clear();
            let _ = out.push_bytes(cwd.as_bytes());
            let _ = out.push_bytes(b"\\actions.json\0");
            if path_exists_utf8_nul(out.as_bytes()) {
                return true;
            }
        }

        false
    }

    pub fn query_process_stats() -> ProcessStats {
        let mut stats = ProcessStats::default();
        unsafe {
            let proc_handle = GetCurrentProcess();
            let mut create = FileTime::default();
            let mut exit = FileTime::default();
            let mut kernel = FileTime::default();
            let mut user = FileTime::default();
            if GetProcessTimes(proc_handle, &mut create, &mut exit, &mut kernel, &mut user) != 0 {
                stats.cpu_time_ns = kernel.to_ns().saturating_add(user.to_ns());
            }

            let mut mem = ProcessMemoryCountersEx::default();
            mem.cb = core::mem::size_of::<ProcessMemoryCountersEx>() as u32;
            if K32GetProcessMemoryInfo(proc_handle, &mut mem, mem.cb) != 0 {
                stats.private_bytes = mem.private_usage as u64;
            }

            let mut io = IoCounters::default();
            if GetProcessIoCounters(proc_handle, &mut io) != 0 {
                stats.write_bytes = io.write_transfer_count;
                stats.write_ops = io.write_operation_count;
            }
        }
        stats
    }

    pub fn command_line_has_flag(flag: &[u8]) -> bool {
        let ptr = unsafe { GetCommandLineW() };
        if ptr.is_null() {
            return false;
        }
        let mut len = 0usize;
        unsafe {
            while len < 1024 && *ptr.add(len) != 0 {
                len += 1;
            }
        }
        let slice = unsafe { core::slice::from_raw_parts(ptr, len) };
        let mut i = 0usize;
        while i < slice.len() {
            // Skip whitespace.
            while i < slice.len() && (slice[i] == b' ' as u16 || slice[i] == b'\t' as u16) {
                i += 1;
            }
            if i >= slice.len() {
                break;
            }
            let token_start;
            let token_end;
            if slice[i] == b'"' as u16 {
                i += 1;
                token_start = i;
                while i < slice.len() && slice[i] != b'"' as u16 {
                    i += 1;
                }
                token_end = i;
                if i < slice.len() {
                    i += 1;
                }
            } else {
                token_start = i;
                while i < slice.len() && slice[i] != b' ' as u16 && slice[i] != b'\t' as u16 {
                    i += 1;
                }
                token_end = i;
            }
            let tok = &slice[token_start..token_end];
            if tok.len() == flag.len() {
                let mut eq = true;
                let mut k = 0;
                while k < tok.len() {
                    if tok[k] != flag[k] as u16 {
                        eq = false;
                        break;
                    }
                    k += 1;
                }
                if eq {
                    return true;
                }
            }
        }
        false
    }

    fn path_exists_utf8_nul(path_nul: &[u8]) -> bool {
        let mut wide = [0u16; 520];
        let Some(_) = utf8_nul_to_utf16(path_nul, &mut wide) else {
            return false;
        };
        let attrs = unsafe { GetFileAttributesW(wide.as_ptr()) };
        attrs != INVALID_FILE_ATTRIBUTES
    }

    fn exe_dir_utf8() -> Option<StackBuf<520>> {
        let mut wide = [0u16; 520];
        let n = unsafe { GetModuleFileNameW(0, wide.as_mut_ptr(), wide.len() as u32) };
        if n == 0 || (n as usize) >= wide.len() {
            return None;
        }
        let mut utf8 = StackBuf::<520>::new();
        if !push_utf16_as_utf8(&wide[..n as usize], &mut utf8) {
            return None;
        }
        let dir_len = pop_path_component(utf8.as_bytes()).len();
        utf8.len = dir_len;
        Some(utf8)
    }

    fn cwd_utf8() -> Option<StackBuf<520>> {
        let mut wide = [0u16; 520];
        let n = unsafe { GetCurrentDirectoryW(wide.len() as u32, wide.as_mut_ptr()) };
        if n == 0 || (n as usize) >= wide.len() {
            return None;
        }
        let mut utf8 = StackBuf::<520>::new();
        if !push_utf16_as_utf8(&wide[..n as usize], &mut utf8) {
            return None;
        }
        Some(utf8)
    }

    fn read_file_head(path_wide_nul: &[u16], out: &mut [u8]) -> Option<usize> {
        let h = unsafe {
            CreateFileW(
                path_wide_nul.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                core::ptr::null(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                0,
            )
        };
        if h == INVALID_HANDLE_VALUE || h == 0 {
            return None;
        }
        let mut read = 0u32;
        let ok = unsafe {
            ReadFile(
                h,
                out.as_mut_ptr(),
                out.len() as u32,
                &mut read,
                core::ptr::null_mut(),
            )
        };
        unsafe {
            let _ = CloseHandle(h);
        }
        if ok != 0 { Some(read as usize) } else { None }
    }

    fn push_utf16_as_utf8<const N: usize>(src: &[u16], dst: &mut StackBuf<N>) -> bool {
        let mut i = 0;
        while i < src.len() {
            let unit = src[i];
            let cp = if (0xD800..=0xDBFF).contains(&unit) && i + 1 < src.len() {
                let low = src[i + 1];
                if (0xDC00..=0xDFFF).contains(&low) {
                    i += 1;
                    0x10000 + (((unit as u32 - 0xD800) << 10) | (low as u32 - 0xDC00))
                } else {
                    0xFFFD
                }
            } else {
                unit as u32
            };
            let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
            let mut tmp = [0u8; 4];
            let encoded = ch.encode_utf8(&mut tmp);
            if !dst.push_bytes(encoded.as_bytes()) {
                return false;
            }
            i += 1;
        }
        true
    }

    fn utf8_nul_to_utf16(src_nul: &[u8], dst: &mut [u16]) -> Option<usize> {
        let end = src_nul.iter().position(|&b| b == 0).unwrap_or(src_nul.len());
        let s = core::str::from_utf8(&src_nul[..end]).ok()?;
        let mut pos = 0usize;
        for ch in s.chars() {
            let mut pair = [0u16; 2];
            let enc = ch.encode_utf16(&mut pair);
            if pos + enc.len() >= dst.len() {
                return None;
            }
            for u in enc.iter() {
                dst[pos] = *u;
                pos += 1;
            }
        }
        if pos >= dst.len() {
            return None;
        }
        dst[pos] = 0;
        Some(pos)
    }
}

// ============================================================================
// Linux implementation
// ============================================================================
#[cfg(target_os = "linux")]
mod imp {
    use super::*;

    const CLOCK_MONOTONIC: i32 = 1;
    const CLOCK_PROCESS_CPUTIME_ID: i32 = 2;
    const RTLD_NOW: i32 = 2;
    const F_OK: i32 = 0;
    const O_RDONLY: i32 = 0;
    const SIGINT: i32 = 2;
    const SIGTERM: i32 = 15;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Timespec {
        tv_sec: i64,
        tv_nsec: i64,
    }

    #[link(name = "c")]
    unsafe extern "C" {
        fn write(fd: i32, buf: *const c_void, count: usize) -> isize;
        fn _exit(status: i32) -> !;
        fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
        fn nanosleep(req: *const Timespec, rem: *mut Timespec) -> i32;
        fn readlink(pathname: *const u8, buf: *mut u8, bufsiz: usize) -> isize;
        fn getcwd(buf: *mut u8, size: usize) -> *mut u8;
        fn getenv(name: *const u8) -> *const u8;
        fn access(pathname: *const u8, mode: i32) -> i32;
        fn open(pathname: *const u8, flags: i32) -> i32;
        fn read(fd: i32, buf: *mut c_void, count: usize) -> isize;
        fn lseek(fd: i32, offset: i64, whence: i32) -> i64;
        fn close(fd: i32) -> i32;
        fn signal(signum: i32, handler: usize) -> usize;
    }

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(filename: *const u8, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> i32;
    }

    extern "C" fn posix_signal_handler(_sig: i32) {
        request_stop();
    }

    pub fn install_signal_handlers() {
        unsafe {
            let _ = signal(SIGINT, posix_signal_handler as *const () as usize);
            let _ = signal(SIGTERM, posix_signal_handler as *const () as usize);
        }
    }

    pub fn write_stdout(bytes: &[u8]) {
        unsafe {
            let _ = write(1, bytes.as_ptr().cast(), bytes.len());
        }
    }

    pub fn write_stderr(bytes: &[u8]) {
        unsafe {
            let _ = write(2, bytes.as_ptr().cast(), bytes.len());
        }
    }

    pub fn exit_process(code: u32) -> ! {
        unsafe { _exit(code as i32) }
    }

    pub fn abort() -> ! {
        exit_process(101)
    }

    pub struct MonotonicClock {
        start_ns: u64,
    }

    impl MonotonicClock {
        pub fn new() -> Self {
            Self {
                start_ns: raw_clock_ns(CLOCK_MONOTONIC),
            }
        }

        #[inline]
        pub fn now_ns(&self) -> u64 {
            raw_clock_ns(CLOCK_MONOTONIC).saturating_sub(self.start_ns)
        }
    }

    fn raw_clock_ns(clock_id: i32) -> u64 {
        let mut ts = Timespec::default();
        if unsafe { clock_gettime(clock_id, &mut ts) } != 0 {
            return 0;
        }
        (ts.tv_sec as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add(ts.tv_nsec as u64)
    }

    pub struct WaitTimer;

    impl WaitTimer {
        pub fn new() -> Self {
            Self
        }

        #[inline]
        pub fn sleep_idle_ms(&self, ms: u32) {
            self.sleep_precise_ns((ms as u64).saturating_mul(1_000_000));
        }

        #[inline]
        pub fn sleep_precise_ns(&self, ns: u64) {
            if ns == 0 {
                return;
            }
            let req = Timespec {
                tv_sec: (ns / 1_000_000_000) as i64,
                tv_nsec: (ns % 1_000_000_000) as i64,
            };
            unsafe {
                let _ = nanosleep(&req, core::ptr::null_mut());
            }
        }
    }

    pub struct DynamicLibrary {
        handle: *mut c_void,
    }

    impl DynamicLibrary {
        pub fn open_utf8(path_nul: &[u8]) -> Option<Self> {
            if path_nul.last() != Some(&0) {
                return None;
            }
            let handle = unsafe { dlopen(path_nul.as_ptr(), RTLD_NOW) };
            if handle.is_null() {
                None
            } else {
                Some(Self { handle })
            }
        }

        /// # Safety
        /// `symbol_nul` must be a valid NUL-terminated ASCII symbol name.
        pub unsafe fn get_symbol(&self, symbol_nul: &[u8]) -> Option<*const c_void> {
            if symbol_nul.last() != Some(&0) {
                return None;
            }
            let ptr = unsafe { dlsym(self.handle, symbol_nul.as_ptr()) };
            if ptr.is_null() { None } else { Some(ptr) }
        }
    }

    impl Drop for DynamicLibrary {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    let _ = dlclose(self.handle);
                }
            }
        }
    }

    fn read_vrpath_file(out: &mut [u8]) -> Option<usize> {
        let mut vrpath = StackBuf::<520>::new();
        let xdg = unsafe { getenv(b"XDG_CONFIG_HOME\0".as_ptr()) };
        if !xdg.is_null() {
            let s = cstr_bytes(xdg);
            if !s.is_empty() {
                let _ = vrpath.push_bytes(s);
                let _ = vrpath.push_bytes(b"/openvr/openvrpaths.vrpath\0");
            }
        }
        if vrpath.len == 0 {
            let home = unsafe { getenv(b"HOME\0".as_ptr()) };
            if !home.is_null() {
                let s = cstr_bytes(home);
                if !s.is_empty() {
                    let _ = vrpath.push_bytes(s);
                    let _ = vrpath.push_bytes(b"/.config/openvr/openvrpaths.vrpath\0");
                }
            }
        }
        if vrpath.len == 0 {
            return None;
        }
        read_file_head(vrpath.as_bytes(), out)
    }

    pub fn load_openvr_library() -> Option<DynamicLibrary> {
        if let Some(lib) = DynamicLibrary::open_utf8(b"libopenvr_api.so\0") {
            return Some(lib);
        }

        if let Some(exe_dir) = exe_dir_utf8() {
            let mut buf = StackBuf::<520>::new();
            let _ = buf.push_bytes(exe_dir.as_bytes());
            let _ = buf.push_bytes(b"/libopenvr_api.so\0");
            if let Some(lib) = DynamicLibrary::open_utf8(buf.as_bytes()) {
                return Some(lib);
            }
        }

        let mut file_buf = [0u8; 4096];
        if let Some(read_len) = read_vrpath_file(&mut file_buf) {
            let mut runtime_buf = [0u8; 512];
            if let Some(rt) = parse_vrpath_runtime(&file_buf[..read_len], &mut runtime_buf) {
                let mut so_path = StackBuf::<600>::new();
                let _ = so_path.push_bytes(rt);
                let _ = so_path.push_bytes(b"/bin/linux64/libopenvr_api.so\0");
                if let Some(lib) = DynamicLibrary::open_utf8(so_path.as_bytes()) {
                    return Some(lib);
                }
            }
        }

        None
    }

    pub fn query_chaperone_file_stamp() -> Option<(u64, u64)> {
        let mut file_buf = [0u8; 4096];
        let read_len = read_vrpath_file(&mut file_buf)?;
        let mut cfg_buf = [0u8; 512];
        let cfg = parse_vrpath_config(&file_buf[..read_len], &mut cfg_buf)?;
        let mut chap_path = StackBuf::<600>::new();
        let _ = chap_path.push_bytes(cfg);
        let _ = chap_path.push_bytes(b"/chaperone_info.vrchap\0");
        let fd = unsafe { open(chap_path.as_bytes().as_ptr(), O_RDONLY) };
        if fd < 0 {
            return None;
        }
        let size = unsafe { lseek(fd, 0, 2) };
        unsafe {
            let _ = close(fd);
        }
        if size >= 0 { Some((0, size as u64)) } else { None }
    }

    pub fn resolve_action_manifest_path(out: &mut StackBuf<520>) -> bool {
        if let Some(exe_dir) = exe_dir_utf8() {
            let exe_bytes = exe_dir.as_bytes();
            out.clear();
            let _ = out.push_bytes(exe_bytes);
            let _ = out.push_bytes(b"/manifest/actions.json\0");
            if path_exists_utf8_nul(out.as_bytes()) {
                return true;
            }

            let p1 = pop_path_component(exe_bytes);
            let p2 = pop_path_component(p1);
            if !p2.is_empty() {
                out.clear();
                let _ = out.push_bytes(p2);
                let _ = out.push_bytes(b"/manifest/actions.json\0");
                if path_exists_utf8_nul(out.as_bytes()) {
                    return true;
                }
            }

            out.clear();
            let _ = out.push_bytes(exe_bytes);
            let _ = out.push_bytes(b"/actions.json\0");
            if path_exists_utf8_nul(out.as_bytes()) {
                return true;
            }
        }

        if let Some(cwd) = cwd_utf8() {
            out.clear();
            let _ = out.push_bytes(cwd.as_bytes());
            let _ = out.push_bytes(b"/manifest/actions.json\0");
            if path_exists_utf8_nul(out.as_bytes()) {
                return true;
            }

            out.clear();
            let _ = out.push_bytes(cwd.as_bytes());
            let _ = out.push_bytes(b"/actions.json\0");
            if path_exists_utf8_nul(out.as_bytes()) {
                return true;
            }
        }

        false
    }

    pub fn query_process_stats() -> ProcessStats {
        let mut stats = ProcessStats::default();
        stats.cpu_time_ns = raw_clock_ns(CLOCK_PROCESS_CPUTIME_ID);

        let mut buf = [0u8; 2048];
        if let Some(n) = read_file_head(b"/proc/self/status\0", &mut buf) {
            if let Some(kb) = parse_proc_kv_u64(&buf[..n], b"VmRSS:") {
                stats.private_bytes = kb.saturating_mul(1024);
            }
        }
        if let Some(n) = read_file_head(b"/proc/self/io\0", &mut buf) {
            if let Some(wb) = parse_proc_kv_u64(&buf[..n], b"write_bytes:") {
                stats.write_bytes = wb;
            }
            if let Some(wo) = parse_proc_kv_u64(&buf[..n], b"syscr:") {
                let _ = wo;
            }
            if let Some(wo) = parse_proc_kv_u64(&buf[..n], b"syscw:") {
                stats.write_ops = wo;
            }
        }
        stats
    }

    pub fn command_line_has_flag(flag: &[u8]) -> bool {
        let mut buf = [0u8; 1024];
        let Some(n) = read_file_head(b"/proc/self/cmdline\0", &mut buf) else {
            return false;
        };
        let slice = &buf[..n];
        let mut start = 0usize;
        let mut i = 0usize;
        while i <= slice.len() {
            if i == slice.len() || slice[i] == 0 {
                if &slice[start..i] == flag {
                    return true;
                }
                start = i + 1;
            }
            i += 1;
        }
        false
    }

    fn parse_proc_kv_u64(buf: &[u8], key: &[u8]) -> Option<u64> {
        let pos = find_subslice(buf, key)?;
        let mut i = pos + key.len();
        while i < buf.len() && (buf[i] == b' ' || buf[i] == b'\t') {
            i += 1;
        }
        let mut val = 0u64;
        let mut any = false;
        while i < buf.len() && buf[i].is_ascii_digit() {
            any = true;
            val = val.saturating_mul(10).saturating_add((buf[i] - b'0') as u64);
            i += 1;
        }
        if any { Some(val) } else { None }
    }

    fn cstr_bytes<'a>(ptr: *const u8) -> &'a [u8] {
        let mut len = 0usize;
        unsafe {
            while len < 512 && *ptr.add(len) != 0 {
                len += 1;
            }
            core::slice::from_raw_parts(ptr, len)
        }
    }

    fn path_exists_utf8_nul(path_nul: &[u8]) -> bool {
        if path_nul.last() != Some(&0) {
            return false;
        }
        unsafe { access(path_nul.as_ptr(), F_OK) == 0 }
    }

    fn exe_dir_utf8() -> Option<StackBuf<520>> {
        let mut raw = [0u8; 512];
        let n = unsafe { readlink(b"/proc/self/exe\0".as_ptr(), raw.as_mut_ptr(), raw.len()) };
        if n <= 0 || (n as usize) >= raw.len() {
            return None;
        }
        let dir = pop_path_component(&raw[..n as usize]);
        let mut out = StackBuf::<520>::new();
        let _ = out.push_bytes(dir);
        Some(out)
    }

    fn cwd_utf8() -> Option<StackBuf<520>> {
        let mut raw = [0u8; 512];
        let ptr = unsafe { getcwd(raw.as_mut_ptr(), raw.len()) };
        if ptr.is_null() {
            return None;
        }
        let s = cstr_bytes(raw.as_ptr());
        let mut out = StackBuf::<520>::new();
        let _ = out.push_bytes(s);
        Some(out)
    }

    fn read_file_head(path_nul: &[u8], out: &mut [u8]) -> Option<usize> {
        if path_nul.last() != Some(&0) {
            return None;
        }
        let fd = unsafe { open(path_nul.as_ptr(), O_RDONLY) };
        if fd < 0 {
            return None;
        }
        let n = unsafe { read(fd, out.as_mut_ptr().cast(), out.len()) };
        unsafe {
            let _ = close(fd);
        }
        if n >= 0 { Some(n as usize) } else { None }
    }
}

pub use imp::{
    DynamicLibrary, MonotonicClock, WaitTimer, abort, command_line_has_flag, exit_process,
    install_signal_handlers, load_openvr_library, query_chaperone_file_stamp, query_process_stats,
    resolve_action_manifest_path, write_stderr, write_stdout,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vrpath_runtime_and_config_windows_and_linux() {
        let win_json = br#"{
            "config": ["E:\\Steam\\config"],
            "runtime": [
                "E:\\Program Files (x86)\\Steam\\steamapps\\common\\SteamVR"
            ],
            "version": 1
        }"#;
        let mut out = [0u8; 256];
        let cfg = parse_vrpath_config(win_json, &mut out).expect("should parse config");
        assert_eq!(cfg, b"E:\\Steam\\config");
        let rt = parse_vrpath_runtime(win_json, &mut out).expect("should parse runtime");
        assert_eq!(
            rt,
            b"E:\\Program Files (x86)\\Steam\\steamapps\\common\\SteamVR"
        );

        let lin_json = br#"{"config":["\/home\/user\/.steam\/config"],"runtime":["\/home\/user\/.local\/share\/Steam\/steamapps\/common\/SteamVR"]}"#;
        let cfg_lin = parse_vrpath_config(lin_json, &mut out).expect("should parse linux config");
        assert_eq!(cfg_lin, b"/home/user/.steam/config");
        let rt_lin = parse_vrpath_runtime(lin_json, &mut out).expect("should parse linux runtime");
        assert_eq!(
            rt_lin,
            b"/home/user/.local/share/Steam/steamapps/common/SteamVR"
        );
    }
}
