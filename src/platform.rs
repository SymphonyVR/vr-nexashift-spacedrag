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
    imp::signal_stop_event();
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
    pub cpu_cycles: u64,
    pub private_bytes: u64,
    pub write_bytes: u64,
    pub write_ops: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitOutcome {
    Tick,
    Stop,
    ServerExited,
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

    use core::cell::Cell;
    use core::sync::atomic::AtomicIsize;

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
    const INFINITE: u32 = 0xFFFF_FFFF;
    const WAIT_OBJECT_0: u32 = 0;
    const CREATE_WAITABLE_TIMER_HIGH_RESOLUTION: u32 = 0x0000_0002;
    const TIMER_ALL_ACCESS: u32 = 0x001F_0003;
    const FILE_MAP_READ: u32 = 0x0004;
    const SYNCHRONIZE: u32 = 0x0010_0000;
    const TH32CS_SNAPPROCESS: u32 = 0x0000_0002;

    static STOP_EVENT: AtomicIsize = AtomicIsize::new(0);

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

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(super) struct MemoryBasicInformation {
        base_address: *const c_void,
        allocation_base: *const c_void,
        allocation_protect: u32,
        partition_id: u16,
        region_size: usize,
        state: u32,
        protect: u32,
        mem_type: u32,
    }

    impl Default for MemoryBasicInformation {
        fn default() -> Self {
            Self {
                base_address: core::ptr::null(),
                allocation_base: core::ptr::null(),
                allocation_protect: 0,
                partition_id: 0,
                region_size: 0,
                state: 0,
                protect: 0,
                mem_type: 0,
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(super) struct ProcessEntry32W {
        dw_size: u32,
        cnt_usage: u32,
        th32_process_id: u32,
        th32_default_heap_id: usize,
        th32_module_id: u32,
        cnt_threads: u32,
        th32_parent_process_id: u32,
        pc_pri_class_base: i32,
        dw_flags: u32,
        sz_exe_file: [u16; 260],
    }

    impl Default for ProcessEntry32W {
        fn default() -> Self {
            Self {
                dw_size: core::mem::size_of::<Self>() as u32,
                cnt_usage: 0,
                th32_process_id: 0,
                th32_default_heap_id: 0,
                th32_module_id: 0,
                cnt_threads: 0,
                th32_parent_process_id: 0,
                pc_pri_class_base: 0,
                dw_flags: 0,
                sz_exe_file: [0; 260],
            }
        }
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
        fn CreateEventW(
            lpEventAttributes: *const c_void,
            bManualReset: i32,
            bInitialState: i32,
            lpName: *const u16,
        ) -> isize;
        fn SetEvent(hEvent: isize) -> i32;
        fn WaitForSingleObject(hHandle: isize, dwMilliseconds: u32) -> u32;
        fn WaitForMultipleObjects(
            nCount: u32,
            lpHandles: *const isize,
            bWaitAll: i32,
            dwMilliseconds: u32,
        ) -> u32;
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
        fn CancelWaitableTimer(hTimer: isize) -> i32;
        fn OpenFileMappingW(dwDesiredAccess: u32, bInheritHandle: i32, lpName: *const u16)
            -> isize;
        fn MapViewOfFile(
            hFileMappingObject: isize,
            dwDesiredAccess: u32,
            dwFileOffsetHigh: u32,
            dwFileOffsetLow: u32,
            dwNumberOfBytesToMap: usize,
        ) -> *const c_void;
        fn UnmapViewOfFile(lpBaseAddress: *const c_void) -> i32;
        fn VirtualQuery(
            lpAddress: *const c_void,
            lpBuffer: *mut MemoryBasicInformation,
            dwLength: usize,
        ) -> usize;
        fn OpenProcess(dwDesiredAccess: u32, bInheritHandle: i32, dwProcessId: u32) -> isize;
        fn GetCurrentProcessId() -> u32;
        fn CreateToolhelp32Snapshot(dwFlags: u32, th32ProcessID: u32) -> isize;
        fn Process32FirstW(hSnapshot: isize, lppe: *mut ProcessEntry32W) -> i32;
        fn Process32NextW(hSnapshot: isize, lppe: *mut ProcessEntry32W) -> i32;
        fn SetConsoleCtrlHandler(
            handlerRoutine: Option<unsafe extern "system" fn(u32) -> i32>,
            add: i32,
        ) -> i32;
        fn GetCurrentProcess() -> isize;
        fn GetCurrentThread() -> isize;
        fn QueryThreadCycleTime(ThreadHandle: isize, CycleTime: *mut u64) -> i32;
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

    #[inline]
    pub fn signal_stop_event() {
        let h = STOP_EVENT.load(Ordering::Acquire);
        if h != 0 {
            unsafe {
                let _ = SetEvent(h);
            }
        }
    }

    pub struct ReactiveWaiter {
        event_handle: isize,
        timer_handle: isize,
        armed_period_ms: Cell<u32>,
    }

    impl ReactiveWaiter {
        pub fn new() -> Self {
            let h = unsafe { CreateEventW(core::ptr::null(), 1, 0, core::ptr::null()) };
            if h != 0 {
                STOP_EVENT.store(h, Ordering::Release);
                if !is_running() {
                    unsafe {
                        let _ = SetEvent(h);
                    }
                }
            }
            let mut timer = unsafe {
                CreateWaitableTimerExW(
                    core::ptr::null(),
                    core::ptr::null(),
                    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                    TIMER_ALL_ACCESS,
                )
            };
            if timer == 0 {
                timer = unsafe {
                    CreateWaitableTimerExW(core::ptr::null(), core::ptr::null(), 0, TIMER_ALL_ACCESS)
                };
            }
            Self {
                event_handle: h,
                timer_handle: timer,
                armed_period_ms: Cell::new(0),
            }
        }

        #[inline]
        pub fn wait_stop_or_timeout_ms(&self, timeout_ms: u32) {
            if self.event_handle != 0 && is_running() {
                unsafe {
                    let _ = WaitForSingleObject(self.event_handle, timeout_ms);
                }
            }
        }

        #[inline]
        pub fn disarm_idle_timer(&self) {
            if self.timer_handle != 0 && self.armed_period_ms.get() != 0 {
                unsafe {
                    let _ = CancelWaitableTimer(self.timer_handle);
                }
                self.armed_period_ms.set(0);
            }
        }

        #[inline]
        pub fn wait_idle_tick(&self, watch_handle: Option<isize>, period_ms: u32) -> WaitOutcome {
            if !is_running() {
                return WaitOutcome::Stop;
            }
            let period_ms = period_ms.max(1);
            if self.timer_handle == 0 || self.event_handle == 0 {
                self.wait_stop_or_timeout_ms(period_ms);
                if !is_running() {
                    return WaitOutcome::Stop;
                }
                if let Some(wh) = watch_handle {
                    if wh != 0 && unsafe { WaitForSingleObject(wh, 0) } == WAIT_OBJECT_0 {
                        return WaitOutcome::ServerExited;
                    }
                }
                return WaitOutcome::Tick;
            }

            if self.armed_period_ms.get() != period_ms {
                let due: i64 = -(period_ms as i64) * 10_000;
                let ok = unsafe {
                    SetWaitableTimer(
                        self.timer_handle,
                        &due,
                        period_ms as i32,
                        core::ptr::null(),
                        core::ptr::null(),
                        0,
                    )
                };
                if ok != 0 {
                    self.armed_period_ms.set(period_ms);
                }
            }

            let mut handles = [self.event_handle, self.timer_handle, 0isize];
            let count = if let Some(wh) = watch_handle {
                if wh != 0 {
                    handles[2] = wh;
                    3u32
                } else {
                    2u32
                }
            } else {
                2u32
            };

            let res = unsafe { WaitForMultipleObjects(count, handles.as_ptr(), 0, INFINITE) };
            match res {
                r if r == WAIT_OBJECT_0 => WaitOutcome::Stop,
                r if r == WAIT_OBJECT_0 + 1 => WaitOutcome::Tick,
                r if count == 3 && r == WAIT_OBJECT_0 + 2 => WaitOutcome::ServerExited,
                _ => {
                    if is_running() {
                        WaitOutcome::Tick
                    } else {
                        WaitOutcome::Stop
                    }
                }
            }
        }
    }

    impl Drop for ReactiveWaiter {
        fn drop(&mut self) {
            self.disarm_idle_timer();
            if self.timer_handle != 0 {
                unsafe {
                    let _ = CloseHandle(self.timer_handle);
                }
            }
            if self.event_handle != 0 {
                let _ = STOP_EVENT.compare_exchange(
                    self.event_handle,
                    0,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                unsafe {
                    let _ = CloseHandle(self.event_handle);
                }
            }
        }
    }

    pub struct MappedReadOnlyShm {
        map_handle: isize,
        view: *const u8,
        len: usize,
    }

    impl MappedReadOnlyShm {
        pub fn open_utf8(name_nul: &[u8], min_len: usize) -> Option<Self> {
            let mut wide = [0u16; 260];
            utf8_nul_to_utf16(name_nul, &mut wide)?;
            let map_handle = unsafe { OpenFileMappingW(FILE_MAP_READ, 0, wide.as_ptr()) };
            if map_handle == 0 {
                return None;
            }
            // Map the full section (size = 0) and inspect the committed region size via VirtualQuery.
            let view = unsafe { MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, 0) } as *const u8;
            if view.is_null() {
                unsafe {
                    let _ = CloseHandle(map_handle);
                }
                return None;
            }
            let mut mbi = MemoryBasicInformation::default();
            let queried = unsafe {
                VirtualQuery(
                    view.cast(),
                    &mut mbi,
                    core::mem::size_of::<MemoryBasicInformation>(),
                )
            };
            if queried < core::mem::size_of::<MemoryBasicInformation>()
                || mbi.region_size < min_len
                || ((view as usize) & 7) != 0
            {
                unsafe {
                    let _ = UnmapViewOfFile(view.cast());
                    let _ = CloseHandle(map_handle);
                }
                return None;
            }
            Some(Self {
                map_handle,
                view,
                len: mbi.region_size,
            })
        }

        #[inline]
        pub fn len(&self) -> usize {
            self.len
        }

        #[inline]
        pub fn read_volatile_u8(&self, byte_offset: usize) -> Option<u8> {
            if byte_offset >= self.len {
                return None;
            }
            let ptr = unsafe { self.view.add(byte_offset) };
            Some(unsafe { core::ptr::read_volatile(ptr) })
        }

        #[inline]
        pub fn read_volatile_u32(&self, byte_offset: usize) -> Option<u32> {
            if (byte_offset & 3) != 0 || byte_offset.checked_add(4)? > self.len {
                return None;
            }
            let ptr = unsafe { self.view.add(byte_offset) } as *const u32;
            Some(unsafe { core::ptr::read_volatile(ptr) })
        }

        #[inline]
        pub fn read_volatile_u64(&self, byte_offset: usize) -> Option<u64> {
            if (byte_offset & 7) != 0 || byte_offset.checked_add(8)? > self.len {
                return None;
            }
            let ptr = unsafe { self.view.add(byte_offset) } as *const u64;
            Some(unsafe { core::ptr::read_volatile(ptr) })
        }

        #[inline]
        pub fn copy_volatile_u32_words(&self, byte_offset: usize, dst: &mut [u32]) -> bool {
            if (byte_offset & 3) != 0 {
                return false;
            }
            let Some(byte_len) = dst.len().checked_mul(4) else {
                return false;
            };
            let Some(end) = byte_offset.checked_add(byte_len) else {
                return false;
            };
            if end > self.len {
                return false;
            }
            let src_ptr = unsafe { self.view.add(byte_offset) } as *const u32;
            let mut i = 0usize;
            while i < dst.len() {
                dst[i] = unsafe { core::ptr::read_volatile(src_ptr.add(i)) };
                i += 1;
            }
            true
        }
    }

    impl Drop for MappedReadOnlyShm {
        fn drop(&mut self) {
            if !self.view.is_null() {
                unsafe {
                    let _ = UnmapViewOfFile(self.view.cast());
                }
            }
            if self.map_handle != 0 {
                unsafe {
                    let _ = CloseHandle(self.map_handle);
                }
            }
        }
    }

    pub struct ProcessWaitHandle {
        handle: isize,
    }

    impl ProcessWaitHandle {
        pub fn open_sync(pid: u32) -> Option<Self> {
            let h = unsafe { OpenProcess(SYNCHRONIZE, 0, pid) };
            if h != 0 {
                Some(Self { handle: h })
            } else {
                None
            }
        }

        #[inline]
        pub fn raw(&self) -> isize {
            self.handle
        }
    }

    impl Drop for ProcessWaitHandle {
        fn drop(&mut self) {
            if self.handle != 0 {
                unsafe {
                    let _ = CloseHandle(self.handle);
                }
            }
        }
    }

    fn utf16_eq_ascii_ignore_case(wide: &[u16], ascii: &[u8]) -> bool {
        let mut len = 0usize;
        while len < wide.len() && wide[len] != 0 {
            len += 1;
        }
        if len != ascii.len() {
            return false;
        }
        let mut i = 0;
        while i < len {
            let w = wide[i];
            if w > 0x7F {
                return false;
            }
            let a = (w as u8).to_ascii_lowercase();
            let b = ascii[i].to_ascii_lowercase();
            if a != b {
                return false;
            }
            i += 1;
        }
        true
    }

    pub fn find_connected_vrserver_shm(
    ) -> Option<(u32, Option<ProcessWaitHandle>, MappedReadOnlyShm, MappedReadOnlyShm)> {
        let self_pid = unsafe { GetCurrentProcessId() };
        let snap = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
        if snap == INVALID_HANDLE_VALUE || snap == 0 {
            return None;
        }

        let mut entry = ProcessEntry32W::default();
        let mut found = None;
        let mut has_item = unsafe { Process32FirstW(snap, &mut entry) != 0 };
        while has_item {
            if utf16_eq_ascii_ignore_case(&entry.sz_exe_file, b"vrserver.exe") {
                let server_pid = entry.th32_process_id;
                let mut conn_name = StackBuf::<128>::new();
                let _ = write!(
                    conn_name,
                    "VR_InputConnectionState_{}_{}-IPCWrapper\0",
                    self_pid, server_pid
                );
                let mut state_name = StackBuf::<128>::new();
                let _ = write!(state_name, "VR_InputState_{}-IPCWrapper\0", server_pid);

                if let (Some(conn_shm), Some(state_shm)) = (
                    MappedReadOnlyShm::open_utf8(conn_name.as_bytes(), 64),
                    MappedReadOnlyShm::open_utf8(state_name.as_bytes(), 64),
                ) {
                    let wait_handle = ProcessWaitHandle::open_sync(server_pid);
                    found = Some((server_pid, wait_handle, state_shm, conn_shm));
                    break;
                }
            }
            entry.dw_size = core::mem::size_of::<ProcessEntry32W>() as u32;
            has_item = unsafe { Process32NextW(snap, &mut entry) != 0 };
        }

        unsafe {
            let _ = CloseHandle(snap);
        }
        found
    }

    pub fn open_property_container_shm(
        server_pid: u32,
        container_id: u64,
    ) -> Option<(MappedReadOnlyShm, MappedReadOnlyShm, MappedReadOnlyShm)> {
        let mut hdr_name = StackBuf::<128>::new();
        let _ = write!(
            hdr_name,
            "PC_{:x}_header_{}-IPCWrapper\0",
            container_id, server_pid
        );
        let hdr_shm = MappedReadOnlyShm::open_utf8(hdr_name.as_bytes(), 16)?;

        let mut tbl_name = StackBuf::<128>::new();
        let _ = write!(
            tbl_name,
            "PC_{:x}_table0_{}-IPCWrapper\0",
            container_id, server_pid
        );
        let tbl_shm = MappedReadOnlyShm::open_utf8(tbl_name.as_bytes(), 16)?;

        let mut sto_name = StackBuf::<128>::new();
        let _ = write!(
            sto_name,
            "PC_{:x}_storage0_{}-IPCWrapper\0",
            container_id, server_pid
        );
        let sto_shm = MappedReadOnlyShm::open_utf8(sto_name.as_bytes(), 16)?;
        Some((hdr_shm, tbl_shm, sto_shm))
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

    fn resolve_manifest_file(file_name: &[u8], out: &mut StackBuf<520>) -> bool {
        let mut check = |base: &[u8], sub: &[u8]| -> bool {
            out.clear();
            let ok = out.push_bytes(base)
                && out.push_bytes(sub)
                && out.push_bytes(file_name)
                && out.push_cstr_end();
            ok && path_exists_utf8_nul(out.as_bytes())
        };

        if let Some(exe_dir) = exe_dir_utf8() {
            let exe_bytes = exe_dir.as_bytes();
            if check(exe_bytes, b"\\manifest\\") {
                return true;
            }
            let p1 = pop_path_component(exe_bytes);
            let p2 = pop_path_component(p1);
            if !p2.is_empty() && check(p2, b"\\manifest\\") {
                return true;
            }
            if check(exe_bytes, b"\\") {
                return true;
            }
        }

        if let Some(cwd) = cwd_utf8() {
            let cwd_bytes = cwd.as_bytes();
            if check(cwd_bytes, b"\\manifest\\") || check(cwd_bytes, b"\\") {
                return true;
            }
        }

        false
    }

    #[inline]
    pub fn resolve_action_manifest_path(out: &mut StackBuf<520>) -> bool {
        resolve_manifest_file(b"actions.json", out)
    }

    #[inline]
    pub fn resolve_vrmanifest_path(out: &mut StackBuf<520>) -> bool {
        resolve_manifest_file(b"nexashift.vrmanifest", out)
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

            let mut cycles = 0u64;
            if QueryThreadCycleTime(GetCurrentThread(), &mut cycles) != 0 {
                stats.cpu_cycles = cycles;
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

    use core::sync::atomic::AtomicI32;

    const CLOCK_MONOTONIC: i32 = 1;
    const CLOCK_PROCESS_CPUTIME_ID: i32 = 2;
    const RTLD_NOW: i32 = 2;
    const F_OK: i32 = 0;
    const O_RDONLY: i32 = 0;
    const O_NONBLOCK_CLOEXEC: i32 = 0x0008_0800;
    const POLLIN: i16 = 1;
    const SIGINT: i32 = 2;
    const SIGTERM: i32 = 15;

    static STOP_PIPE_WR: AtomicI32 = AtomicI32::new(-1);

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Timespec {
        tv_sec: i64,
        tv_nsec: i64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct PollFd {
        fd: i32,
        events: i16,
        revents: i16,
    }

    #[link(name = "c")]
    unsafe extern "C" {
        fn write(fd: i32, buf: *const c_void, count: usize) -> isize;
        fn _exit(status: i32) -> !;
        fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
        fn pipe2(pipefd: *mut i32, flags: i32) -> i32;
        fn poll(fds: *mut PollFd, nfds: usize, timeout: i32) -> i32;
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

    #[inline]
    pub fn signal_stop_event() {
        let fd = STOP_PIPE_WR.load(Ordering::Acquire);
        if fd >= 0 {
            let b = 1u8;
            unsafe {
                let _ = write(fd, (&b as *const u8).cast(), 1);
            }
        }
    }

    pub struct ReactiveWaiter {
        read_fd: i32,
        write_fd: i32,
    }

    impl ReactiveWaiter {
        pub fn new() -> Self {
            let mut fds = [-1i32; 2];
            if unsafe { pipe2(fds.as_mut_ptr(), O_NONBLOCK_CLOEXEC) } == 0 {
                STOP_PIPE_WR.store(fds[1], Ordering::Release);
                if !is_running() {
                    signal_stop_event();
                }
                Self {
                    read_fd: fds[0],
                    write_fd: fds[1],
                }
            } else {
                Self {
                    read_fd: -1,
                    write_fd: -1,
                }
            }
        }

        #[inline]
        pub fn wait_stop_or_timeout_ms(&self, timeout_ms: u32) {
            if self.read_fd >= 0 && is_running() {
                let mut pfd = PollFd {
                    fd: self.read_fd,
                    events: POLLIN,
                    revents: 0,
                };
                unsafe {
                    let _ = poll(&mut pfd, 1, timeout_ms as i32);
                }
            }
        }

        #[inline]
        pub fn disarm_idle_timer(&self) {}

        #[inline]
        pub fn wait_idle_tick(&self, _watch_handle: Option<isize>, period_ms: u32) -> WaitOutcome {
            if !is_running() {
                return WaitOutcome::Stop;
            }
            if self.read_fd >= 0 {
                let mut pfd = PollFd {
                    fd: self.read_fd,
                    events: POLLIN,
                    revents: 0,
                };
                let ret = unsafe { poll(&mut pfd, 1, period_ms as i32) };
                if !is_running() || (ret > 0 && (pfd.revents & POLLIN) != 0) {
                    return WaitOutcome::Stop;
                }
            }
            WaitOutcome::Tick
        }
    }

    pub struct MappedReadOnlyShm;

    impl MappedReadOnlyShm {
        #[inline]
        pub fn len(&self) -> usize {
            0
        }

        #[inline]
        pub fn read_volatile_u8(&self, _byte_offset: usize) -> Option<u8> {
            None
        }

        #[inline]
        pub fn read_volatile_u32(&self, _byte_offset: usize) -> Option<u32> {
            None
        }

        #[inline]
        pub fn read_volatile_u64(&self, _byte_offset: usize) -> Option<u64> {
            None
        }

        #[inline]
        pub fn copy_volatile_u32_words(&self, _byte_offset: usize, _dst: &mut [u32]) -> bool {
            false
        }
    }

    pub struct ProcessWaitHandle;

    impl ProcessWaitHandle {
        #[inline]
        pub fn raw(&self) -> isize {
            0
        }
    }

    pub fn find_connected_vrserver_shm(
    ) -> Option<(u32, Option<ProcessWaitHandle>, MappedReadOnlyShm, MappedReadOnlyShm)> {
        None
    }

    pub fn open_property_container_shm(
        _server_pid: u32,
        _container_id: u64,
    ) -> Option<(MappedReadOnlyShm, MappedReadOnlyShm, MappedReadOnlyShm)> {
        None
    }

    impl Drop for ReactiveWaiter {
        fn drop(&mut self) {
            if self.write_fd >= 0 {
                let _ = STOP_PIPE_WR.compare_exchange(
                    self.write_fd,
                    -1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                unsafe {
                    let _ = close(self.write_fd);
                }
            }
            if self.read_fd >= 0 {
                unsafe {
                    let _ = close(self.read_fd);
                }
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

    fn resolve_manifest_file(file_name: &[u8], out: &mut StackBuf<520>) -> bool {
        let mut check = |base: &[u8], sub: &[u8]| -> bool {
            out.clear();
            let ok = out.push_bytes(base)
                && out.push_bytes(sub)
                && out.push_bytes(file_name)
                && out.push_cstr_end();
            ok && path_exists_utf8_nul(out.as_bytes())
        };

        if let Some(exe_dir) = exe_dir_utf8() {
            let exe_bytes = exe_dir.as_bytes();
            if check(exe_bytes, b"/manifest/") {
                return true;
            }
            let p1 = pop_path_component(exe_bytes);
            let p2 = pop_path_component(p1);
            if !p2.is_empty() && check(p2, b"/manifest/") {
                return true;
            }
            if check(exe_bytes, b"/") {
                return true;
            }
        }

        if let Some(cwd) = cwd_utf8() {
            let cwd_bytes = cwd.as_bytes();
            if check(cwd_bytes, b"/manifest/") || check(cwd_bytes, b"/") {
                return true;
            }
        }

        false
    }

    #[inline]
    pub fn resolve_action_manifest_path(out: &mut StackBuf<520>) -> bool {
        resolve_manifest_file(b"actions.json", out)
    }

    #[inline]
    pub fn resolve_vrmanifest_path(out: &mut StackBuf<520>) -> bool {
        resolve_manifest_file(b"nexashift.vrmanifest", out)
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
    DynamicLibrary, MappedReadOnlyShm, MonotonicClock, ProcessWaitHandle, ReactiveWaiter, abort,
    command_line_has_flag, exit_process, find_connected_vrserver_shm, install_signal_handlers,
    load_openvr_library, open_property_container_shm, query_chaperone_file_stamp,
    query_process_stats, resolve_action_manifest_path, resolve_vrmanifest_path, write_stderr,
    write_stdout,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_process_entry32w_size() {
        assert_eq!(core::mem::size_of::<imp::ProcessEntry32W>(), 568);
        assert_eq!(core::mem::size_of::<imp::MemoryBasicInformation>(), 48);
    }

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
