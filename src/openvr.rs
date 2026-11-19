use core::cell::Cell;
use core::ffi::c_void;
use core::fmt::Write;
use core::mem::{MaybeUninit, size_of};

use crate::math::{HmdMatrix34_t, SimdMath, Vec3};
use crate::platform::{
    DynamicLibrary, MappedReadOnlyShm, ProcessWaitHandle, StackBuf, find_connected_vrserver_shm,
    open_property_container_shm, resolve_action_manifest_path, resolve_vrmanifest_path,
    write_stderr,
};
use crate::state::{Hand, HandInput, INVALID_DEVICE_INDEX, InputFrame};

pub const K_UN_TRACKED_DEVICE_INDEX_HMD: u32 = 0;
pub const K_UN_MAX_TRACKED_DEVICE_COUNT: usize = 64;
pub const K_UN_TRACKED_DEVICE_INDEX_INVALID: u32 = 0xFFFF_FFFF;

pub const EVR_APPLICATION_BACKGROUND: i32 = 3;
pub const E_TRACKING_UNIVERSE_RAW_AND_UNCALIBRATED: i32 = 2;
pub const E_TRACKING_RESULT_RUNNING_OK: i32 = 200;
pub const E_TRACKED_CONTROLLER_ROLE_LEFT_HAND: i32 = 1;
pub const E_TRACKED_CONTROLLER_ROLE_RIGHT_HAND: i32 = 2;
pub const PROP_DISPLAY_FREQUENCY_FLOAT: i32 = 2002;
pub const VR_EVENT_QUIT: u32 = 700;
pub const VR_EVENT_CHAPERONE_DATA_HAS_CHANGED: u32 = 800;
pub const VR_EVENT_CHAPERONE_UNIVERSE_HAS_CHANGED: u32 = 801;
pub const VR_EVENT_CHAPERONE_TEMP_DATA_HAS_CHANGED: u32 = 802;
pub const VR_EVENT_CHAPERONE_ROOM_SETUP_COMMITTED: u32 = 805;
pub const VR_EVENT_STANDING_ZERO_POSE_RESET: u32 = 808;
pub const VR_EVENT_INPUT_BINDING_LOAD_SUCCESSFUL: u32 = 1702;
pub const VR_EVENT_INPUT_ACTION_MANIFEST_RELOADED: u32 = 1703;
pub const VR_EVENT_INPUT_BINDINGS_UPDATED: u32 = 1707;
pub const E_CHAPERONE_CONFIG_FILE_LIVE: i32 = 1;
pub const K_N_ACTION_SET_OVERLAY_GLOBAL_PRIORITY_MAX: i32 = 0x01FF_FFFF;

pub const VR_INIT_ERROR_NONE: i32 = 0;
pub const VR_INPUT_ERROR_NONE: i32 = 0;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TrackedDevicePose_t {
    pub m_device_to_absolute_tracking: HmdMatrix34_t,
    pub v_velocity: Vec3,
    pub v_angular_velocity: Vec3,
    pub e_tracking_result: i32,
    pub b_pose_is_valid: bool,
    pub b_device_is_connected: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct InputDigitalActionData_t {
    pub b_active: bool,
    pub active_origin: u64,
    pub b_state: bool,
    pub b_changed: bool,
    pub f_update_time: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct InputOriginInfo_t {
    pub device_path: u64,
    pub tracked_device_index: u32,
    pub rch_render_model_component_name: [u8; 128],
}

impl Default for InputOriginInfo_t {
    fn default() -> Self {
        Self {
            device_path: 0,
            tracked_device_index: K_UN_TRACKED_DEVICE_INDEX_INVALID,
            rch_render_model_component_name: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct InputBindingInfo_t {
    pub rch_device_path_name: [u8; 128],
    pub rch_input_path_name: [u8; 128],
    pub rch_mode_name: [u8; 128],
    pub rch_slot_name: [u8; 128],
    pub rch_input_source_type: [u8; 32],
}

impl Default for InputBindingInfo_t {
    fn default() -> Self {
        Self {
            rch_device_path_name: [0; 128],
            rch_input_path_name: [0; 128],
            rch_mode_name: [0; 128],
            rch_slot_name: [0; 128],
            rch_input_source_type: [0; 32],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VRActiveActionSet_t {
    pub ul_action_set: u64,
    pub ul_restricted_to_device: u64,
    pub ul_secondary_action_set: u64,
    pub un_padding: u32,
    pub n_priority: i32,
}

#[cfg_attr(target_os = "linux", repr(C, packed(4)))]
#[cfg_attr(not(target_os = "linux"), repr(C))]
#[derive(Clone, Copy)]
pub struct VREvent_t {
    pub event_type: u32,
    pub tracked_device_index: u32,
    pub event_age_seconds: f32,
    pub data: [u64; 6],
}

#[repr(C)]
pub struct VR_IVRSystem_FnTable {
    pub get_recommended_render_target_size: *const c_void,
    pub get_projection_matrix: *const c_void,
    pub get_projection_raw: *const c_void,
    pub compute_distortion: *const c_void,
    pub get_eye_to_head_transform: *const c_void,
    pub get_time_since_last_vsync: *const c_void,
    pub get_d3d9_adapter_index: *const c_void,
    pub get_dxgi_output_info: *const c_void,
    pub get_output_device: *const c_void,
    pub is_display_on_desktop: *const c_void,
    pub set_display_visibility: *const c_void,
    pub get_device_to_absolute_tracking_pose: unsafe extern "system" fn(
        e_origin: i32,
        f_predicted_seconds_to_photons_from_now: f32,
        p_tracked_device_pose_array: *mut TrackedDevicePose_t,
        un_tracked_device_pose_array_count: u32,
    ),
    pub get_seated_zero_pose_to_standing_absolute_tracking_pose: *const c_void,
    pub get_raw_zero_pose_to_standing_absolute_tracking_pose: *const c_void,
    pub get_sorted_tracked_device_indices_of_class: *const c_void,
    pub get_tracked_device_activity_level: *const c_void,
    pub apply_transform: *const c_void,
    pub get_tracked_device_index_for_controller_role:
        unsafe extern "system" fn(un_device_type: i32) -> u32,
    pub get_controller_role_for_tracked_device_index: *const c_void,
    pub get_tracked_device_class: *const c_void,
    pub is_tracked_device_connected: *const c_void,
    pub get_bool_tracked_device_property: *const c_void,
    pub get_float_tracked_device_property: unsafe extern "system" fn(
        un_device_index: u32,
        prop: i32,
        p_error: *mut i32,
    ) -> f32,
    pub get_int32_tracked_device_property: *const c_void,
    pub get_uint64_tracked_device_property: *const c_void,
    pub get_matrix34_tracked_device_property: *const c_void,
    pub get_array_tracked_device_property: *const c_void,
    pub get_string_tracked_device_property: *const c_void,
    pub get_prop_error_name_from_enum: *const c_void,
    pub poll_next_event:
        unsafe extern "system" fn(p_event: *mut VREvent_t, uncb_vr_event: u32) -> bool,
    pub poll_next_event_with_pose: *const c_void,
    pub get_event_type_name_from_enum: *const c_void,
    pub get_hidden_area_mesh: *const c_void,
    pub get_controller_state: *const c_void,
    pub get_controller_state_with_pose: *const c_void,
    pub trigger_haptic_pulse: *const c_void,
    pub get_button_id_name_from_enum: *const c_void,
    pub get_controller_axis_type_name_from_enum: *const c_void,
    pub is_input_available: *const c_void,
    pub is_steamvr_drawing_controllers: *const c_void,
    pub should_application_pause: *const c_void,
    pub should_application_reduce_rendering_work: *const c_void,
    pub perform_firmware_update: *const c_void,
    pub acknowledge_quit_exiting: unsafe extern "system" fn(),
    pub get_app_container_file_paths: *const c_void,
    pub get_runtime_version: unsafe extern "system" fn() -> *const u8,
}

#[repr(C)]
pub struct VR_IVRChaperoneSetup_FnTable {
    pub commit_working_copy: unsafe extern "system" fn(config_file: i32) -> bool,
    pub revert_working_copy: unsafe extern "system" fn(),
    pub get_working_play_area_size: *const c_void,
    pub get_working_play_area_rect: *const c_void,
    pub get_working_collision_bounds_info: *const c_void,
    pub get_live_collision_bounds_info: *const c_void,
    pub get_working_seated_zero_pose_to_raw_tracking_pose: *const c_void,
    pub get_working_standing_zero_pose_to_raw_tracking_pose:
        unsafe extern "system" fn(pmat_standing_zero_pose_to_raw_tracking_pose: *mut HmdMatrix34_t) -> bool,
    pub set_working_play_area_size: *const c_void,
    pub set_working_collision_bounds_info: *const c_void,
    pub set_working_perimeter: *const c_void,
    pub set_working_seated_zero_pose_to_raw_tracking_pose: *const c_void,
    pub set_working_standing_zero_pose_to_raw_tracking_pose:
        unsafe extern "system" fn(p_mat_standing_zero_pose_to_raw_tracking_pose: *const HmdMatrix34_t),
    pub reload_from_disk: *const c_void,
    pub get_live_seated_zero_pose_to_raw_tracking_pose: *const c_void,
    pub export_live_to_buffer: *const c_void,
    pub import_from_buffer_to_working: *const c_void,
    pub show_working_set_preview: unsafe extern "system" fn(),
    pub hide_working_set_preview: unsafe extern "system" fn(),
    pub room_setup_starting: *const c_void,
}

#[repr(C)]
pub struct VR_IVRInput_FnTable {
    pub set_action_manifest_path:
        unsafe extern "system" fn(pch_action_manifest_path: *const u8) -> i32,
    pub get_action_set_handle: unsafe extern "system" fn(
        pch_action_set_name: *const u8,
        p_handle: *mut u64,
    ) -> i32,
    pub get_action_handle:
        unsafe extern "system" fn(pch_action_name: *const u8, p_handle: *mut u64) -> i32,
    pub get_input_source_handle: unsafe extern "system" fn(
        pch_input_source_path: *const u8,
        p_handle: *mut u64,
    ) -> i32,
    pub update_action_state: unsafe extern "system" fn(
        p_sets: *mut VRActiveActionSet_t,
        un_size_of_vr_selected_action_set_t: u32,
        un_set_count: u32,
    ) -> i32,
    pub get_digital_action_data: unsafe extern "system" fn(
        action: u64,
        p_action_data: *mut InputDigitalActionData_t,
        un_action_data_size: u32,
        ul_restrict_to_device: u64,
    ) -> i32,
    pub get_analog_action_data: *const c_void,
    pub get_pose_action_data_relative_to_now: *const c_void,
    pub get_pose_action_data_for_next_frame: *const c_void,
    pub get_skeletal_action_data: *const c_void,
    pub get_dominant_hand: *const c_void,
    pub set_dominant_hand: *const c_void,
    pub get_bone_count: *const c_void,
    pub get_bone_hierarchy: *const c_void,
    pub get_bone_name: *const c_void,
    pub get_skeletal_reference_transforms: *const c_void,
    pub get_skeletal_tracking_level: *const c_void,
    pub get_skeletal_bone_data: *const c_void,
    pub get_skeletal_summary_data: *const c_void,
    pub get_skeletal_bone_data_compressed: *const c_void,
    pub decompress_skeletal_bone_data: *const c_void,
    pub trigger_haptic_vibration_action: *const c_void,
    pub get_action_origins: *const c_void,
    pub get_origin_localized_name: *const c_void,
    pub get_origin_tracked_device_info: unsafe extern "system" fn(
        origin: u64,
        p_origin_info: *mut InputOriginInfo_t,
        un_origin_info_size: u32,
    ) -> i32,
    pub get_action_binding_info: unsafe extern "system" fn(
        action: u64,
        p_origin_info: *mut InputBindingInfo_t,
        un_binding_info_size: u32,
        un_binding_info_count: u32,
        pun_returned_binding_info_count: *mut u32,
    ) -> i32,
    pub show_action_origins: *const c_void,
    pub show_bindings_for_action_set: *const c_void,
    pub get_component_state_for_binding: *const c_void,
    pub is_using_legacy_input: *const c_void,
    pub open_binding_ui: *const c_void,
    pub get_binding_variant: *const c_void,
}

#[repr(C)]
pub struct VR_IVROverlay_FnTable {
    pub _before_wait_frame_sync: [*const c_void; 47],
    pub wait_frame_sync: unsafe extern "system" fn(n_timeout_ms: u32) -> i32,
    pub _between_wait_and_dashboard: [*const c_void; 20],
    pub is_dashboard_visible: unsafe extern "system" fn() -> bool,
    pub _after_dashboard: [*const c_void; 13],
}

#[repr(C)]
pub struct VR_IVRSettings_FnTable {
    pub get_settings_error_name_from_enum: *const c_void,
    pub set_bool: unsafe extern "system" fn(
        pch_section: *const u8,
        pch_settings_key: *const u8,
        b_value: bool,
        pe_error: *mut i32,
    ),
    pub set_int32: *const c_void,
    pub set_float: *const c_void,
    pub set_string: *const c_void,
    pub get_bool: unsafe extern "system" fn(
        pch_section: *const u8,
        pch_settings_key: *const u8,
        pe_error: *mut i32,
    ) -> bool,
    pub get_int32: *const c_void,
    pub get_float: *const c_void,
    pub get_string: *const c_void,
    pub remove_section: *const c_void,
    pub remove_key_in_section: *const c_void,
}

#[repr(C)]
pub struct VR_IVRApplications_FnTable {
    pub add_application_manifest: unsafe extern "system" fn(
        pch_application_manifest_full_path: *const u8,
        b_temporary: bool,
    ) -> i32,
    pub remove_application_manifest:
        unsafe extern "system" fn(pch_application_manifest_full_path: *const u8) -> i32,
    pub _middle_slots: [*const c_void; 15],
    pub set_application_auto_launch:
        unsafe extern "system" fn(pch_app_key: *const u8, b_auto_launch: bool) -> i32,
    pub get_application_auto_launch: unsafe extern "system" fn(pch_app_key: *const u8) -> bool,
}

type FnVrInitInternal = unsafe extern "C" fn(pe_error: *mut i32, e_type: i32) -> isize;
type FnVrShutdownInternal = unsafe extern "C" fn();
type FnVrGetGenericInterface =
    unsafe extern "C" fn(pch_interface_version: *const u8, pe_error: *mut i32) -> isize;

#[derive(Clone, Copy, Debug)]
pub enum OpenVrError {
    LibraryNotFound,
    MissingEntryPoint(&'static str),
    VrInitFailed(i32),
    InterfaceFailed(&'static str, i32),
    ManifestNotFound,
    InputInitFailed(&'static str, i32),
    AppInstallFailed(&'static str, i32),
}

pub struct OpenVrSession {
    _lib: DynamicLibrary,
    fn_shutdown: FnVrShutdownInternal,
    vr_system: *const VR_IVRSystem_FnTable,
    chaperone_setup: *const VR_IVRChaperoneSetup_FnTable,
    vr_input: *const VR_IVRInput_FnTable,
    vr_overlay: *const VR_IVROverlay_FnTable,
    action_set_main: u64,
    action_space_drag: u64,
    source_left: u64,
    source_right: u64,
    hmd_refresh_hz: f32,
    working_set_synced: Cell<bool>,
    dirty_uncommitted: Cell<bool>,
    bindings_dirty: Cell<bool>,
}

impl OpenVrSession {
    pub fn init(register_actions: bool) -> Result<Self, OpenVrError> {
        let lib = crate::platform::load_openvr_library().ok_or(OpenVrError::LibraryNotFound)?;

        let fn_init: FnVrInitInternal = unsafe {
            let ptr = lib
                .get_symbol(b"VR_InitInternal\0")
                .ok_or(OpenVrError::MissingEntryPoint("VR_InitInternal"))?;
            core::mem::transmute(ptr)
        };
        let fn_shutdown: FnVrShutdownInternal = unsafe {
            let ptr = lib
                .get_symbol(b"VR_ShutdownInternal\0")
                .ok_or(OpenVrError::MissingEntryPoint("VR_ShutdownInternal"))?;
            core::mem::transmute(ptr)
        };
        let fn_get_interface: FnVrGetGenericInterface = unsafe {
            let ptr = lib
                .get_symbol(b"VR_GetGenericInterface\0")
                .ok_or(OpenVrError::MissingEntryPoint("VR_GetGenericInterface"))?;
            core::mem::transmute(ptr)
        };

        let mut init_err = VR_INIT_ERROR_NONE;
        let _token = unsafe { fn_init(&mut init_err, EVR_APPLICATION_BACKGROUND) };
        if init_err != VR_INIT_ERROR_NONE {
            return Err(OpenVrError::VrInitFailed(init_err));
        }

        // Helper closure that shuts down VR cleanly if interface acquisition fails.
        let get_table = |ver_nul: &'static [u8], name: &'static str| -> Result<*const c_void, OpenVrError> {
            let mut err = VR_INIT_ERROR_NONE;
            let ptr = unsafe { fn_get_interface(ver_nul.as_ptr(), &mut err) };
            if ptr == 0 || err != VR_INIT_ERROR_NONE {
                unsafe { fn_shutdown() };
                return Err(OpenVrError::InterfaceFailed(name, err));
            }
            Ok(ptr as *const c_void)
        };

        let vr_system = get_table(b"FnTable:IVRSystem_022\0", "IVRSystem_022")?
            as *const VR_IVRSystem_FnTable;
        let chaperone_setup = get_table(
            b"FnTable:IVRChaperoneSetup_006\0",
            "IVRChaperoneSetup_006",
        )? as *const VR_IVRChaperoneSetup_FnTable;
        let vr_input =
            get_table(b"FnTable:IVRInput_010\0", "IVRInput_010")? as *const VR_IVRInput_FnTable;
        let vr_overlay = get_table(b"FnTable:IVROverlay_026\0", "IVROverlay_026")?
            as *const VR_IVROverlay_FnTable;

        let mut action_set_main = 0u64;
        let mut action_space_drag = 0u64;
        let mut source_left = 0u64;
        let mut source_right = 0u64;

        if register_actions {
            let mut manifest_path = StackBuf::<520>::new();
            if !resolve_action_manifest_path(&mut manifest_path) {
                unsafe { fn_shutdown() };
                return Err(OpenVrError::ManifestNotFound);
            }

            let err = unsafe {
                ((*vr_input).set_action_manifest_path)(manifest_path.as_bytes().as_ptr())
            };
            if err != VR_INPUT_ERROR_NONE {
                unsafe { fn_shutdown() };
                return Err(OpenVrError::InputInitFailed("SetActionManifestPath", err));
            }

            let err = unsafe {
                ((*vr_input).get_action_set_handle)(
                    b"/actions/main\0".as_ptr(),
                    &mut action_set_main,
                )
            };
            if err != VR_INPUT_ERROR_NONE || action_set_main == 0 {
                unsafe { fn_shutdown() };
                return Err(OpenVrError::InputInitFailed("GetActionSetHandle", err));
            }

            let err = unsafe {
                ((*vr_input).get_action_handle)(
                    b"/actions/main/in/space_drag\0".as_ptr(),
                    &mut action_space_drag,
                )
            };
            if err != VR_INPUT_ERROR_NONE || action_space_drag == 0 {
                unsafe { fn_shutdown() };
                return Err(OpenVrError::InputInitFailed("GetActionHandle", err));
            }

            let err = unsafe {
                ((*vr_input).get_input_source_handle)(
                    b"/user/hand/left\0".as_ptr(),
                    &mut source_left,
                )
            };
            if err != VR_INPUT_ERROR_NONE || source_left == 0 {
                unsafe { fn_shutdown() };
                return Err(OpenVrError::InputInitFailed(
                    "GetInputSourceHandle(left)",
                    err,
                ));
            }

            let err = unsafe {
                ((*vr_input).get_input_source_handle)(
                    b"/user/hand/right\0".as_ptr(),
                    &mut source_right,
                )
            };
            if err != VR_INPUT_ERROR_NONE || source_right == 0 {
                unsafe { fn_shutdown() };
                return Err(OpenVrError::InputInitFailed(
                    "GetInputSourceHandle(right)",
                    err,
                ));
            }
        }

        let mut prop_err = 0i32;
        let raw_hz = unsafe {
            ((*vr_system).get_float_tracked_device_property)(
                K_UN_TRACKED_DEVICE_INDEX_HMD,
                PROP_DISPLAY_FREQUENCY_FLOAT,
                &mut prop_err,
            )
        };
        let hmd_refresh_hz = if prop_err == 0 && raw_hz.is_finite() && (60.0..=240.0).contains(&raw_hz) {
            raw_hz.clamp(90.0, 144.0)
        } else {
            120.0
        };

        Ok(Self {
            _lib: lib,
            fn_shutdown,
            vr_system,
            chaperone_setup,
            vr_input,
            vr_overlay,
            action_set_main,
            action_space_drag,
            source_left,
            source_right,
            hmd_refresh_hz,
            working_set_synced: Cell::new(false),
            dirty_uncommitted: Cell::new(false),
            bindings_dirty: Cell::new(false),
        })
    }

    #[inline]
    pub fn take_bindings_dirty(&self) -> bool {
        let dirty = self.bindings_dirty.get();
        if dirty {
            self.bindings_dirty.set(false);
        }
        dirty
    }

    #[inline]
    pub fn wait_frame_sync(&self, timeout_ms: u32) -> bool {
        unsafe { ((*self.vr_overlay).wait_frame_sync)(timeout_ms) == 0 }
    }

    #[inline]
    pub fn is_dashboard_visible(&self) -> bool {
        unsafe { ((*self.vr_overlay).is_dashboard_visible)() }
    }

    #[inline]
    pub fn hmd_refresh_hz(&self) -> f32 {
        self.hmd_refresh_hz
    }

    #[inline]
    pub fn drag_interval_ns(&self) -> u64 {
        (1_000_000_000.0f32 / self.hmd_refresh_hz) as u64
    }

    pub fn runtime_version(&self) -> &str {
        let ptr = unsafe { ((*self.vr_system).get_runtime_version)() };
        if ptr.is_null() {
            return "unknown";
        }
        let mut len = 0usize;
        unsafe {
            while len < 64 && *ptr.add(len) != 0 {
                len += 1;
            }
            core::str::from_utf8(core::slice::from_raw_parts(ptr, len)).unwrap_or("unknown")
        }
    }

    pub fn controller_device_index(&self, hand: Hand) -> u32 {
        let role = match hand {
            Hand::Left => E_TRACKED_CONTROLLER_ROLE_LEFT_HAND,
            Hand::Right => E_TRACKED_CONTROLLER_ROLE_RIGHT_HAND,
        };
        unsafe { ((*self.vr_system).get_tracked_device_index_for_controller_role)(role) }
    }

    fn query_openvr_digital_actions(
        &self,
    ) -> (i32, i32, i32, InputDigitalActionData_t, InputDigitalActionData_t) {
        // Keep priority at 0 so SteamVR never inserts our bindings into globalPriorityMap and
        // never blocks foreground games (e.g. VRChat B/Y/Joystick) from receiving input.
        let mut active_set = VRActiveActionSet_t {
            ul_action_set: self.action_set_main,
            ul_restricted_to_device: 0,
            ul_secondary_action_set: 0,
            un_padding: 0,
            n_priority: 0,
        };
        let err = unsafe {
            ((*self.vr_input).update_action_state)(
                &mut active_set,
                size_of::<VRActiveActionSet_t>() as u32,
                1,
            )
        };

        let mut left_data = InputDigitalActionData_t::default();
        let mut right_data = InputDigitalActionData_t::default();
        let data_size = size_of::<InputDigitalActionData_t>() as u32;
        let (left_err, right_err) = if err == VR_INPUT_ERROR_NONE {
            let l = unsafe {
                ((*self.vr_input).get_digital_action_data)(
                    self.action_space_drag,
                    &mut left_data,
                    data_size,
                    self.source_left,
                )
            };
            let r = unsafe {
                ((*self.vr_input).get_digital_action_data)(
                    self.action_space_drag,
                    &mut right_data,
                    data_size,
                    self.source_right,
                )
            };
            (l, r)
        } else {
            (-1, -1)
        };

        (err, left_err, right_err, left_data, right_data)
    }

    pub fn poll_should_quit(&self) -> bool {
        let mut event = MaybeUninit::<VREvent_t>::zeroed();
        let event_size = size_of::<VREvent_t>() as u32;
        let mut should_quit = false;
        unsafe {
            while ((*self.vr_system).poll_next_event)(event.as_mut_ptr(), event_size) {
                let ev = event.assume_init();
                match ev.event_type {
                    VR_EVENT_QUIT => {
                        should_quit = true;
                    }
                    VR_EVENT_CHAPERONE_TEMP_DATA_HAS_CHANGED => {
                        if !self.dirty_uncommitted.get() {
                            self.working_set_synced.set(false);
                        }
                    }
                    VR_EVENT_CHAPERONE_DATA_HAS_CHANGED
                    | VR_EVENT_CHAPERONE_UNIVERSE_HAS_CHANGED
                    | VR_EVENT_CHAPERONE_ROOM_SETUP_COMMITTED
                    | VR_EVENT_STANDING_ZERO_POSE_RESET => {
                        self.working_set_synced.set(false);
                        self.dirty_uncommitted.set(false);
                    }
                    VR_EVENT_INPUT_BINDING_LOAD_SUCCESSFUL
                    | VR_EVENT_INPUT_ACTION_MANIFEST_RELOADED
                    | VR_EVENT_INPUT_BINDINGS_UPDATED => {
                        self.bindings_dirty.set(true);
                    }
                    _ => {}
                }
            }
        }
        should_quit
    }

    #[inline]
    pub fn acknowledge_quit_exiting(&self) {
        unsafe {
            ((*self.vr_system).acknowledge_quit_exiting)();
        }
    }

    pub fn sample_inputs(
        &self,
        active_device_hint: Option<(Hand, u32)>,
        mut monitor: Option<&mut BareMetalInputMonitor>,
        now_ns: u64,
    ) -> InputFrame {
        let mut used_shm_fast_path = false;
        let mut left_pressed = false;
        let mut left_shm_dev = None;
        let mut right_pressed = false;
        let mut right_shm_dev = None;
        let mut left_ov_changed = false;
        let mut right_ov_changed = false;
        let mut left_origin = 0u64;
        let mut right_origin = 0u64;

        if let Some(m) = monitor.as_deref_mut()
            && m.has_validated_bindings()
        {
            let l_eval = m.eval_hand_status(Hand::Left, now_ns);
            let r_eval = m.eval_hand_status(Hand::Right, now_ns);
            if l_eval != ShmEval::Invalid && r_eval != ShmEval::Invalid {
                used_shm_fast_path = true;
                match l_eval {
                    ShmEval::Pressed(dev) => {
                        left_pressed = true;
                        left_shm_dev = Some(dev);
                    }
                    ShmEval::Unpressed | ShmEval::Unbound => {}
                    ShmEval::Invalid => {}
                }
                match r_eval {
                    ShmEval::Pressed(dev) => {
                        right_pressed = true;
                        right_shm_dev = Some(dev);
                    }
                    ShmEval::Unpressed | ShmEval::Unbound => {}
                    ShmEval::Invalid => {}
                }
            }
        }

        if !used_shm_fast_path {
            let (err, left_err, right_err, left_data, right_data) =
                self.query_openvr_digital_actions();
            let left_ov =
                err == VR_INPUT_ERROR_NONE && left_err == VR_INPUT_ERROR_NONE && left_data.b_active;
            let right_ov = err == VR_INPUT_ERROR_NONE
                && right_err == VR_INPUT_ERROR_NONE
                && right_data.b_active;
            left_ov_changed = left_ov && left_data.b_changed;
            right_ov_changed = right_ov && right_data.b_changed;
            left_origin = left_data.active_origin;
            right_origin = right_data.active_origin;

            let eval_side = |hand: Hand,
                             ov_active: bool,
                             ov_data: &InputDigitalActionData_t,
                             mon: &mut Option<&mut BareMetalInputMonitor>|
             -> (bool, Option<u32>) {
                if ov_active {
                    if let Some(m) = mon.as_deref_mut() {
                        m.sync_hand_detectors(hand, now_ns);
                    }
                    (ov_data.b_state, None)
                } else if let Some(m) = mon.as_deref_mut() {
                    // Dashboard / focus suppression fallback: read bound hardware components from SHM.
                    match m.eval_hand_status(hand, now_ns) {
                        ShmEval::Pressed(dev) => (true, Some(dev)),
                        ShmEval::Unpressed | ShmEval::Unbound | ShmEval::Invalid => (false, None),
                    }
                } else {
                    (false, None)
                }
            };

            (left_pressed, left_shm_dev) =
                eval_side(Hand::Left, left_ov, &left_data, &mut monitor);
            (right_pressed, right_shm_dev) =
                eval_side(Hand::Right, right_ov, &right_data, &mut monitor);
        }

        // Idle fast path: skip all device pose queries when neither trigger is held or changed.
        if !left_pressed
            && !right_pressed
            && !left_ov_changed
            && !right_ov_changed
            && active_device_hint.is_none()
        {
            return InputFrame::IDLE;
        }

        let left_dev_idx = if left_pressed {
            self.pick_device_index(Hand::Left, left_origin, left_shm_dev, active_device_hint)
        } else {
            INVALID_DEVICE_INDEX
        };
        let right_dev_idx = if right_pressed {
            self.pick_device_index(Hand::Right, right_origin, right_shm_dev, active_device_hint)
        } else {
            INVALID_DEVICE_INDEX
        };

        let mut max_idx: Option<u32> = None;
        if (left_dev_idx as usize) < K_UN_MAX_TRACKED_DEVICE_COUNT {
            max_idx = Some(left_dev_idx);
        }
        if (right_dev_idx as usize) < K_UN_MAX_TRACKED_DEVICE_COUNT {
            max_idx = Some(match max_idx {
                Some(m) => m.max(right_dev_idx),
                None => right_dev_idx,
            });
        }

        let Some(max_dev) = max_idx else {
            return InputFrame {
                left: HandInput {
                    pressed: left_pressed,
                    device_index: INVALID_DEVICE_INDEX,
                    pose_valid: false,
                    position_raw: Vec3::ZERO,
                },
                right: HandInput {
                    pressed: right_pressed,
                    device_index: INVALID_DEVICE_INDEX,
                    pose_valid: false,
                    position_raw: Vec3::ZERO,
                },
            };
        };

        let pose_count = ((max_dev as usize) + 1).min(K_UN_MAX_TRACKED_DEVICE_COUNT);
        let mut poses = MaybeUninit::<[TrackedDevicePose_t; K_UN_MAX_TRACKED_DEVICE_COUNT]>::uninit();
        unsafe {
            ((*self.vr_system).get_device_to_absolute_tracking_pose)(
                E_TRACKING_UNIVERSE_RAW_AND_UNCALIBRATED,
                0.0,
                poses.as_mut_ptr().cast::<TrackedDevicePose_t>(),
                pose_count as u32,
            );
        }
        // SAFETY: GetDeviceToAbsoluteTrackingPose writes elements 0..pose_count; build_hand_input
        // reads only verified indices < pose_count via raw pointer and never assumes the whole array init.
        let poses_ptr = poses.as_ptr().cast::<TrackedDevicePose_t>();

        let left = self.build_hand_input(left_pressed, left_dev_idx, poses_ptr, pose_count);
        let right = self.build_hand_input(right_pressed, right_dev_idx, poses_ptr, pose_count);

        InputFrame { left, right }
    }

    #[inline]
    fn pick_device_index(
        &self,
        hand: Hand,
        active_origin: u64,
        shm_dev: Option<u32>,
        active_device_hint: Option<(Hand, u32)>,
    ) -> u32 {
        match active_device_hint {
            Some((active_hand, idx)) if active_hand == hand => idx,
            _ => match shm_dev {
                Some(idx) if (idx as usize) < K_UN_MAX_TRACKED_DEVICE_COUNT => idx,
                _ => self.resolve_device_index(hand, active_origin),
            },
        }
    }

    fn build_hand_input(
        &self,
        pressed: bool,
        device_index: u32,
        poses_ptr: *const TrackedDevicePose_t,
        pose_count: usize,
    ) -> HandInput {
        if !pressed {
            return HandInput::UNPRESSED;
        }

        if (device_index as usize) >= pose_count {
            return HandInput {
                pressed: true,
                device_index: INVALID_DEVICE_INDEX,
                pose_valid: false,
                position_raw: Vec3::ZERO,
            };
        }

        let pose = unsafe { poses_ptr.add(device_index as usize).read() };
        let pose_valid = pose.b_device_is_connected
            && pose.b_pose_is_valid
            && pose.e_tracking_result == E_TRACKING_RESULT_RUNNING_OK;
        let position_raw = if pose_valid {
            pose.m_device_to_absolute_tracking.translation()
        } else {
            Vec3::ZERO
        };

        HandInput {
            pressed: true,
            device_index,
            pose_valid,
            position_raw,
        }
    }

    fn resolve_device_index(&self, hand: Hand, active_origin: u64) -> u32 {
        if active_origin != 0 {
            let mut info = InputOriginInfo_t::default();
            let err = unsafe {
                ((*self.vr_input).get_origin_tracked_device_info)(
                    active_origin,
                    &mut info,
                    size_of::<InputOriginInfo_t>() as u32,
                )
            };
            if err == VR_INPUT_ERROR_NONE && (info.tracked_device_index as usize) < K_UN_MAX_TRACKED_DEVICE_COUNT {
                return info.tracked_device_index;
            }
        }

        let role = match hand {
            Hand::Left => E_TRACKED_CONTROLLER_ROLE_LEFT_HAND,
            Hand::Right => E_TRACKED_CONTROLLER_ROLE_RIGHT_HAND,
        };
        let idx = unsafe { ((*self.vr_system).get_tracked_device_index_for_controller_role)(role) };
        if (idx as usize) < K_UN_MAX_TRACKED_DEVICE_COUNT {
            idx
        } else {
            INVALID_DEVICE_INDEX
        }
    }

    pub fn get_working_standing_pose(&self) -> Option<HmdMatrix34_t> {
        let mut mat = HmdMatrix34_t::IDENTITY;
        if !self.working_set_synced.get() {
            unsafe {
                ((*self.chaperone_setup).revert_working_copy)();
            }
            self.working_set_synced.set(true);
        }
        let ok = unsafe {
            ((*self.chaperone_setup).get_working_standing_zero_pose_to_raw_tracking_pose)(&mut mat)
        };
        if ok && mat.is_finite() {
            return Some(mat);
        }

        unsafe {
            ((*self.chaperone_setup).revert_working_copy)();
        }
        self.working_set_synced.set(true);
        let ok_retry = unsafe {
            ((*self.chaperone_setup).get_working_standing_zero_pose_to_raw_tracking_pose)(&mut mat)
        };
        if ok_retry && mat.is_finite() {
            Some(mat)
        } else {
            None
        }
    }

    pub fn peek_working_standing_pose(&self) -> Option<HmdMatrix34_t> {
        let mut mat = HmdMatrix34_t::IDENTITY;
        let ok = unsafe {
            ((*self.chaperone_setup).get_working_standing_zero_pose_to_raw_tracking_pose)(&mut mat)
        };
        if ok && mat.is_finite() {
            Some(mat)
        } else {
            None
        }
    }

    #[inline]
    pub fn apply_working_standing_pose(&self, mat: &HmdMatrix34_t) {
        unsafe {
            ((*self.chaperone_setup).set_working_standing_zero_pose_to_raw_tracking_pose)(mat);
            ((*self.chaperone_setup).show_working_set_preview)();
        }
        self.working_set_synced.set(true);
        self.dirty_uncommitted.set(true);
    }

    /// Commits `mat` into SteamVR's live chaperone config and clears `showing_working_pid` so
    /// closing or killing the daemon never makes `vrserver` reload the pre-drag pose.
    pub fn commit_working_standing_pose(&self, mat: &HmdMatrix34_t) -> bool {
        let committed = unsafe {
            ((*self.chaperone_setup).set_working_standing_zero_pose_to_raw_tracking_pose)(mat);
            ((*self.chaperone_setup).commit_working_copy)(E_CHAPERONE_CONFIG_FILE_LIVE)
        };
        if committed {
            unsafe {
                ((*self.chaperone_setup).hide_working_set_preview)();
            }
            self.dirty_uncommitted.set(false);
            self.working_set_synced.set(false);
        }
        committed
    }

    pub fn restore_working_preview(&self) {
        if self.dirty_uncommitted.get() {
            unsafe {
                ((*self.chaperone_setup).hide_working_set_preview)();
                ((*self.chaperone_setup).revert_working_copy)();
            }
            self.dirty_uncommitted.set(false);
            self.working_set_synced.set(false);
        }
    }

    pub fn open_bare_metal_monitor(&self) -> Option<BareMetalInputMonitor> {
        match BareMetalInputMonitor::open_and_validate(self, 0) {
            OpenMonitorResult::Ready(m) => Some(m),
            _ => None,
        }
    }
}

const SHM_ACTIVE_SET_COUNT_MAX: u32 = 128;
const SHM_PROP_COUNT_MAX: u32 = 4096;
const MAX_WATCHED_CONTAINERS: usize = 8;
const MAX_SLOTS_PER_CONTAINER: usize = 32;
const MAX_TOTAL_SLOTS: usize = 128;
const MAX_BOUND_SLOTS: usize = 16;
const PROP_TYPE_INPUT_COMPONENT: u32 = 0x21;
const POST_EDGE_GRACE_NS: u64 = 750_000_000;
const DOUBLE_CLICK_WINDOW_NS: u64 = 500_000_000;
const SHM_ANALOG_CLICK_THRESHOLD: f32 = 0.65;
const REPROBE_BACKOFF_NS: [u64; 3] = [
    5_000_000_000,
    15_000_000_000,
    60_000_000_000,
];
const HEALTH_CHECK_INTERVAL_NS: u64 = 60_000_000_000;
const NO_SLOTS_CONTROLLER_POLL_NS: u64 = 2_000_000_000;

#[inline]
pub const fn reprobe_backoff_ns(stage: u8) -> u64 {
    let idx = if (stage as usize) < REPROBE_BACKOFF_NS.len() {
        stage as usize
    } else {
        REPROBE_BACKOFF_NS.len() - 1
    };
    REPROBE_BACKOFF_NS[idx]
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayoutTrust {
    VerifiedKnown,
    DiscoveredValidated,
    Invalid,
}

impl LayoutTrust {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            LayoutTrust::VerifiedKnown => "verified_known",
            LayoutTrust::DiscoveredValidated => "discovered_validated",
            LayoutTrust::Invalid => "invalid",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RebuildOutcome {
    Ready,
    BindingsNotReady,
    LayoutInvalid,
}

enum OpenMonitorResult {
    Ready(BareMetalInputMonitor),
    BindingsNotReady,
    LayoutInvalid,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShmEval {
    Pressed(u32),
    Unpressed,
    Unbound,
    Invalid,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CrossCheckOutcome {
    Matched,
    Skipped,
    Mismatch,
}

#[inline]
const fn spans_disjoint(a_off: usize, a_len: usize, b_off: usize, b_len: usize) -> bool {
    let Some(a_end) = a_off.checked_add(a_len) else {
        return false;
    };
    let Some(b_end) = b_off.checked_add(b_len) else {
        return false;
    };
    a_end <= b_off || b_end <= a_off
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SteamVrLayout {
    pub name: &'static str,
    pub conn_seq_off: usize,
    pub input_array_off: usize,
    pub input_entry_bytes: usize,
    pub max_components: usize,
    pub input_count_off: usize,
    pub active_set_count_off: usize,
    pub entry_ctype_off: usize,
    pub entry_handle_off: usize,
    pub entry_dev_idx_off: usize,
    pub entry_prop_id_off: usize,
    pub hdr_prop_count_off: usize,
    pub hdr_storage_sz_off: usize,
    pub prop_table_entry_bytes: usize,
    pub prop_entry_id_off: usize,
    pub prop_entry_type_off: usize,
    pub prop_entry_offset_off: usize,
    pub prop_entry_len_off: usize,
    pub prop_type_input_component: u32,
    pub storage_component_bytes: usize,
    pub storage_ctype_off: usize,
    pub storage_value_off: usize,
}

impl SteamVrLayout {
    pub fn is_structurally_sane(&self) -> bool {
        if (self.conn_seq_off & 7) != 0
            || (self.input_array_off & 3) != 0
            || self.input_entry_bytes < 0x34
            || (self.input_entry_bytes & 3) != 0
            || self.max_components == 0
            || self.max_components > 2048
            || (self.input_count_off & 3) != 0
            || (self.active_set_count_off & 3) != 0
        {
            return false;
        }
        let Some(arr_bytes) = self.max_components.checked_mul(self.input_entry_bytes) else {
            return false;
        };
        if self.input_array_off.checked_add(arr_bytes).is_none() {
            return false;
        }
        if !spans_disjoint(self.input_count_off, 4, self.input_array_off, arr_bytes)
            || !spans_disjoint(self.active_set_count_off, 4, self.input_array_off, arr_bytes)
            || !spans_disjoint(self.input_count_off, 4, self.active_set_count_off, 4)
        {
            return false;
        }

        let entry_u32s = [
            (self.entry_ctype_off, 4usize),
            (self.entry_handle_off, 8usize),
            (self.entry_dev_idx_off, 4usize),
            (self.entry_prop_id_off, 4usize),
        ];
        let mut i = 0usize;
        while i < entry_u32s.len() {
            let (off, sz) = entry_u32s[i];
            if (off & 3) != 0 || off + sz > self.input_entry_bytes {
                return false;
            }
            let mut j = i + 1;
            while j < entry_u32s.len() {
                if !spans_disjoint(off, sz, entry_u32s[j].0, entry_u32s[j].1) {
                    return false;
                }
                j += 1;
            }
            i += 1;
        }

        if (self.hdr_prop_count_off & 3) != 0
            || (self.hdr_storage_sz_off & 3) != 0
            || !spans_disjoint(self.hdr_prop_count_off, 4, self.hdr_storage_sz_off, 4)
        {
            return false;
        }

        if self.prop_table_entry_bytes < 24 || (self.prop_table_entry_bytes & 3) != 0 {
            return false;
        }
        let tbl_offs = [
            self.prop_entry_id_off,
            self.prop_entry_type_off,
            self.prop_entry_offset_off,
            self.prop_entry_len_off,
        ];
        i = 0;
        while i < tbl_offs.len() {
            let off = tbl_offs[i];
            if (off & 3) != 0 || off + 4 > self.prop_table_entry_bytes {
                return false;
            }
            let mut j = i + 1;
            while j < tbl_offs.len() {
                if !spans_disjoint(off, 4, tbl_offs[j], 4) {
                    return false;
                }
                j += 1;
            }
            i += 1;
        }

        if self.storage_component_bytes < 0x1c
            || (self.storage_component_bytes & 3) != 0
            || (self.storage_ctype_off & 3) != 0
            || self.storage_ctype_off + 4 > self.storage_component_bytes
            || (self.storage_value_off & 3) != 0
            || self.storage_value_off + 4 > self.storage_component_bytes
            || !spans_disjoint(self.storage_ctype_off, 4, self.storage_value_off, 4)
        {
            return false;
        }

        true
    }
}

pub const STEAMVR_LAYOUT_V2: SteamVrLayout = SteamVrLayout {
    name: "steamvr_v2",
    conn_seq_off: 0x408,
    input_array_off: 0,
    input_entry_bytes: 72,
    max_components: 512,
    input_count_off: 0x9000,
    active_set_count_off: 0x9808,
    entry_ctype_off: 0x00,
    entry_handle_off: 0x10,
    entry_dev_idx_off: 0x20,
    entry_prop_id_off: 0x30,
    hdr_prop_count_off: 4,
    hdr_storage_sz_off: 8,
    prop_table_entry_bytes: 28,
    prop_entry_id_off: 0,
    prop_entry_type_off: 8,
    prop_entry_offset_off: 16,
    prop_entry_len_off: 20,
    prop_type_input_component: PROP_TYPE_INPUT_COMPONENT,
    storage_component_bytes: 72,
    storage_ctype_off: 0x10,
    storage_value_off: 0x18,
};

pub const KNOWN_LAYOUTS: &[SteamVrLayout] = &[STEAMVR_LAYOUT_V2];

trait ShmReadView {
    fn len(&self) -> usize;
    fn read_u8(&self, byte_offset: usize) -> Option<u8>;
    fn read_u32(&self, byte_offset: usize) -> Option<u32>;
    fn read_u64(&self, byte_offset: usize) -> Option<u64>;
}

impl ShmReadView for MappedReadOnlyShm {
    #[inline]
    fn len(&self) -> usize {
        MappedReadOnlyShm::len(self)
    }

    #[inline]
    fn read_u8(&self, byte_offset: usize) -> Option<u8> {
        self.read_volatile_u8(byte_offset)
    }

    #[inline]
    fn read_u32(&self, byte_offset: usize) -> Option<u32> {
        self.read_volatile_u32(byte_offset)
    }

    #[inline]
    fn read_u64(&self, byte_offset: usize) -> Option<u64> {
        self.read_volatile_u64(byte_offset)
    }
}

struct ContainerViews<V> {
    hdr: V,
    tbl: V,
    sto: V,
}

trait ContainerOpener {
    type View: ShmReadView;
    fn open(&mut self, dev_idx: u32) -> Option<ContainerViews<Self::View>>;
}

struct MappedContainerOpener {
    server_pid: u32,
}

impl ContainerOpener for MappedContainerOpener {
    type View = MappedReadOnlyShm;

    fn open(&mut self, dev_idx: u32) -> Option<ContainerViews<Self::View>> {
        let cid = 0x1_0000_0000u64 + (dev_idx as u64);
        let (hdr, tbl, sto) = open_property_container_shm(self.server_pid, cid)?;
        Some(ContainerViews { hdr, tbl, sto })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TriggerKind {
    Direct,
    Double,
}

#[derive(Clone, Copy, Default)]
struct DoubleClickDetector {
    prev_raw: bool,
    awaiting_second: bool,
    latched_double: bool,
    last_first_press_ns: u64,
}

impl DoubleClickDetector {
    #[inline]
    fn step(&mut self, raw: bool, now_ns: u64) -> bool {
        if raw && !self.prev_raw {
            if self.awaiting_second
                && now_ns.saturating_sub(self.last_first_press_ns) <= DOUBLE_CLICK_WINDOW_NS
            {
                self.latched_double = true;
                self.awaiting_second = false;
            } else {
                self.last_first_press_ns = now_ns;
                self.awaiting_second = false;
            }
        } else if !raw && self.prev_raw {
            if self.latched_double {
                self.latched_double = false;
                self.awaiting_second = false;
            } else if now_ns.saturating_sub(self.last_first_press_ns) <= DOUBLE_CLICK_WINDOW_NS {
                self.awaiting_second = true;
            }
        } else if !raw
            && self.awaiting_second
            && now_ns.saturating_sub(self.last_first_press_ns) > DOUBLE_CLICK_WINDOW_NS
        {
            self.awaiting_second = false;
        }
        self.prev_raw = raw;
        self.latched_double && raw
    }

    #[inline]
    fn sync_inactive(&mut self, raw: bool) {
        self.prev_raw = raw;
        if !raw {
            self.latched_double = false;
            self.awaiting_second = false;
            self.last_first_press_ns = 0;
        }
    }
}

#[derive(Clone, Copy)]
struct BoundShmSlot {
    hand: Hand,
    dev_idx: u32,
    container_idx: usize,
    storage_off: usize,
    ctype: u32,
    kind: TriggerKind,
    detector: DoubleClickDetector,
}

#[derive(Clone, Copy)]
struct BindingTarget {
    hand: Hand,
    kind: TriggerKind,
    primary_handle: u64,
    fallback_handle: u64,
    expected_dev_idx: Option<u32>,
    ctype_hint: Option<u32>,
}

#[derive(Clone, Copy)]
struct DiscoveryAnchor {
    handle: u64,
    expected_dev_idx: Option<u32>,
    ctype_hint: Option<u32>,
}

#[inline]
fn cstr_slice(buf: &[u8]) -> &[u8] {
    let mut len = 0usize;
    while len < buf.len() && buf[len] != 0 {
        len += 1;
    }
    &buf[..len]
}

fn parse_hand(dev_path: &[u8]) -> Option<Hand> {
    if dev_path == b"/user/hand/left" {
        Some(Hand::Left)
    } else if dev_path == b"/user/hand/right" {
        Some(Hand::Right)
    } else {
        None
    }
}

fn ends_with(haystack: &[u8], suffix: &[u8]) -> bool {
    haystack.len() >= suffix.len() && &haystack[haystack.len() - suffix.len()..] == suffix
}

/// Resolves an OpenVR binding entry (`dev`, `inp`, `slot`) into its NUL-terminated component path,
/// optional analog fallback path (`/value`), hand, and trigger kind.
fn resolve_binding_component_path(
    dev: &[u8],
    inp: &[u8],
    slot: &[u8],
    primary: &mut StackBuf<256>,
    fallback: &mut StackBuf<256>,
) -> Option<(Hand, TriggerKind, bool)> {
    let hand = parse_hand(dev)?;
    let (kind, suffix) = match slot {
        b"double" => (TriggerKind::Double, b"/click".as_slice()),
        b"touch" => (TriggerKind::Direct, b"/touch".as_slice()),
        b"pull" | b"force" | b"value" => (TriggerKind::Direct, b"/value".as_slice()),
        _ => (TriggerKind::Direct, b"/click".as_slice()),
    };

    let mut ok = primary.push_bytes(dev) && primary.push_bytes(inp);
    let already_has_component =
        ends_with(inp, b"/click") || ends_with(inp, b"/touch") || ends_with(inp, b"/value");
    if !already_has_component {
        ok = ok && primary.push_bytes(suffix);
    }
    if !(ok && primary.push_cstr_end()) {
        return None;
    }

    let mut has_value_fallback = !already_has_component && suffix == b"/click";
    if has_value_fallback {
        has_value_fallback = fallback.push_bytes(dev)
            && fallback.push_bytes(inp)
            && fallback.push_bytes(b"/value")
            && fallback.push_cstr_end();
    }

    Some((hand, kind, has_value_fallback))
}

fn collect_binding_targets(
    session: &OpenVrSession,
    out: &mut [Option<BindingTarget>; MAX_BOUND_SLOTS],
) -> (usize, bool) {
    for slot in out.iter_mut() {
        *slot = None;
    }
    if session.action_space_drag == 0 {
        return (0, false);
    }
    let mut infos = [InputBindingInfo_t::default(); MAX_BOUND_SLOTS];
    let mut raw_count = 0u32;
    let err = unsafe {
        ((*session.vr_input).get_action_binding_info)(
            session.action_space_drag,
            infos.as_mut_ptr(),
            size_of::<InputBindingInfo_t>() as u32,
            infos.len() as u32,
            &mut raw_count,
        )
    };
    if err != VR_INPUT_ERROR_NONE || raw_count == 0 {
        return (0, false);
    }

    let left_dev = session.controller_device_index(Hand::Left);
    let right_dev = session.controller_device_index(Hand::Right);
    let dev_for_hand = |hand: Hand| -> Option<u32> {
        let d = match hand {
            Hand::Left => left_dev,
            Hand::Right => right_dev,
        };
        if (1..K_UN_MAX_TRACKED_DEVICE_COUNT as u32).contains(&d) {
            Some(d)
        } else {
            None
        }
    };

    let n = (raw_count as usize).min(infos.len());
    let mut count = 0usize;
    let mut any_incomplete = (raw_count as usize) > infos.len();
    let mut i = 0usize;
    while i < n && count < MAX_BOUND_SLOTS {
        let dev = cstr_slice(&infos[i].rch_device_path_name);
        let inp = cstr_slice(&infos[i].rch_input_path_name);
        let slot = cstr_slice(&infos[i].rch_slot_name);
        let mut primary = StackBuf::<256>::new();
        let mut fallback = StackBuf::<256>::new();
        if let Some((hand, kind, has_fallback)) =
            resolve_binding_component_path(dev, inp, slot, &mut primary, &mut fallback)
        {
            let mut ph = 0u64;
            let primary_handle = if unsafe {
                ((*session.vr_input).get_input_source_handle)(
                    primary.as_bytes().as_ptr(),
                    &mut ph,
                )
            } == VR_INPUT_ERROR_NONE
            {
                ph
            } else {
                0
            };

            let fallback_handle = if has_fallback {
                let mut fh = 0u64;
                if unsafe {
                    ((*session.vr_input).get_input_source_handle)(
                        fallback.as_bytes().as_ptr(),
                        &mut fh,
                    )
                } == VR_INPUT_ERROR_NONE
                {
                    fh
                } else {
                    0
                }
            } else {
                0
            };

            let ctype_hint = if ends_with(primary.as_bytes(), b"/value\0") {
                Some(2)
            } else if ends_with(primary.as_bytes(), b"/click\0")
                || ends_with(primary.as_bytes(), b"/touch\0")
            {
                if fallback_handle == 0 {
                    Some(1)
                } else {
                    None
                }
            } else {
                None
            };

            if primary_handle != 0 || fallback_handle != 0 {
                out[count] = Some(BindingTarget {
                    hand,
                    kind,
                    primary_handle,
                    fallback_handle,
                    expected_dev_idx: dev_for_hand(hand),
                    ctype_hint,
                });
                count += 1;
            } else if dev_for_hand(hand).is_some() {
                any_incomplete = true;
            }
        }
        i += 1;
    }
    (count, any_incomplete)
}

fn build_discovery_anchors(
    targets: &[Option<BindingTarget>],
    out: &mut [DiscoveryAnchor; 32],
) -> usize {
    let mut n = 0usize;
    let mut i = 0usize;
    while i < targets.len() {
        if let Some(t) = targets[i] {
            if t.primary_handle != 0 && n < out.len() {
                out[n] = DiscoveryAnchor {
                    handle: t.primary_handle,
                    expected_dev_idx: t.expected_dev_idx,
                    ctype_hint: match t.ctype_hint {
                        Some(c) => Some(c),
                        None if t.fallback_handle != 0 => Some(1),
                        None => None,
                    },
                };
                n += 1;
            }
            if t.fallback_handle != 0 && n < out.len() {
                out[n] = DiscoveryAnchor {
                    handle: t.fallback_handle,
                    expected_dev_idx: t.expected_dev_idx,
                    ctype_hint: Some(2),
                };
                n += 1;
            }
        }
        i += 1;
    }
    n
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ShmMetadataSnapshot {
    conn_seq: u64,
    c1: u32,
    c2: u32,
    container_props: [(u32, u32); MAX_WATCHED_CONTAINERS],
}

fn read_global_shm_meta<C: ShmReadView, I: ShmReadView>(
    conn_view: &C,
    input_view: &I,
    layout: &SteamVrLayout,
) -> Option<(u64, u32, u32)> {
    let conn_seq = conn_view.read_u64(layout.conn_seq_off)?;
    let c1 = input_view.read_u32(layout.input_count_off)?;
    let c2 = input_view.read_u32(layout.active_set_count_off)?;
    if (c1 as usize) > layout.max_components || c2 > SHM_ACTIVE_SET_COUNT_MAX {
        return None;
    }
    let array_end = layout
        .input_array_off
        .checked_add((c1 as usize).checked_mul(layout.input_entry_bytes)?)?;
    if array_end > input_view.len() {
        return None;
    }
    Some((conn_seq, c1, c2))
}

fn lookup_input_state_in_view<I: ShmReadView>(
    input_view: &I,
    layout: &SteamVrLayout,
    c1: u32,
    handle: u64,
) -> Option<(u32, u32, u32)> {
    if handle == 0 {
        return None;
    }
    let target_lo = handle as u32;
    let target_hi = (handle >> 32) as u32;
    let mut i = 0usize;
    while i < (c1 as usize) {
        let entry_off = layout.input_array_off + i * layout.input_entry_bytes;
        let lo = input_view.read_u32(entry_off + layout.entry_handle_off)?;
        let hi = input_view.read_u32(entry_off + layout.entry_handle_off + 4)?;
        if lo == target_lo && hi == target_hi {
            let ctype = input_view.read_u32(entry_off + layout.entry_ctype_off)?;
            let dev_idx = input_view.read_u32(entry_off + layout.entry_dev_idx_off)?;
            let prop_lo = input_view.read_u32(entry_off + layout.entry_prop_id_off)?;
            return Some((ctype, dev_idx, 0x1000_0000u32 | prop_lo));
        }
        i += 1;
    }
    None
}

fn lookup_prop_in_table<V: ShmReadView>(
    tbl: &V,
    sto: &V,
    layout: &SteamVrLayout,
    prop_count: u32,
    storage_sz: u32,
    prop_id: u32,
    expected_ctype: u32,
) -> Option<usize> {
    if (prop_count as usize) * layout.prop_table_entry_bytes > tbl.len()
        || (storage_sz as usize) > sto.len()
    {
        return None;
    }
    let mut p = 0usize;
    while p < (prop_count as usize) {
        let toff = p * layout.prop_table_entry_bytes;
        let prop = tbl.read_u32(toff + layout.prop_entry_id_off)?;
        let ptype = tbl.read_u32(toff + layout.prop_entry_type_off)?;
        let soff = tbl.read_u32(toff + layout.prop_entry_offset_off)?;
        let slen = tbl.read_u32(toff + layout.prop_entry_len_off)?;
        if prop == prop_id
            && ptype == layout.prop_type_input_component
            && (slen as usize) == layout.storage_component_bytes
            && (soff as usize) + layout.storage_component_bytes <= (storage_sz as usize)
        {
            let sto_ctype = sto.read_u32(soff as usize + layout.storage_ctype_off)?;
            if sto_ctype == expected_ctype {
                return Some(soff as usize);
            }
            return None;
        }
        p += 1;
    }
    None
}

fn read_raw_bool_view<V: ShmReadView>(
    sto: &V,
    off: usize,
    ctype: u32,
    layout: &SteamVrLayout,
) -> Option<bool> {
    if ctype == 1 {
        Some(sto.read_u8(off + layout.storage_value_off)? != 0)
    } else {
        let raw = sto.read_u32(off + layout.storage_value_off)?;
        let f = f32::from_bits(raw);
        Some(f.is_finite() && f >= SHM_ANALOG_CLICK_THRESHOLD)
    }
}

fn validate_layout_on_views<C: ShmReadView, I: ShmReadView, O: ContainerOpener>(
    layout: &SteamVrLayout,
    conn_view: &C,
    input_view: &I,
    opener: &mut O,
    targets: &[Option<BindingTarget>],
    target_count: usize,
    controller_devs: (u32, u32),
) -> RebuildOutcome {
    if !layout.is_structurally_sane() {
        return RebuildOutcome::LayoutInvalid;
    }
    let Some((_, c1, _)) = read_global_shm_meta(conn_view, input_view, layout) else {
        return RebuildOutcome::LayoutInvalid;
    };

    let mut devs = [u32::MAX; MAX_WATCHED_CONTAINERS];
    let mut dev_count = 0usize;
    let mut i = 0usize;
    while i < (c1 as usize) {
        let entry_off = layout.input_array_off + i * layout.input_entry_bytes;
        let Some(comp_type) = input_view.read_u32(entry_off + layout.entry_ctype_off) else {
            return RebuildOutcome::LayoutInvalid;
        };
        let Some(dev_idx) = input_view.read_u32(entry_off + layout.entry_dev_idx_off) else {
            return RebuildOutcome::LayoutInvalid;
        };
        if comp_type <= 2 && (1..K_UN_MAX_TRACKED_DEVICE_COUNT as u32).contains(&dev_idx) {
            let mut seen = false;
            let mut k = 0usize;
            while k < dev_count {
                if devs[k] == dev_idx {
                    seen = true;
                    break;
                }
                k += 1;
            }
            if !seen && dev_count < MAX_WATCHED_CONTAINERS {
                devs[dev_count] = dev_idx;
                dev_count += 1;
            }
        }
        i += 1;
    }

    let any_controller_role = (1..K_UN_MAX_TRACKED_DEVICE_COUNT as u32).contains(&controller_devs.0)
        || (1..K_UN_MAX_TRACKED_DEVICE_COUNT as u32).contains(&controller_devs.1);
    if dev_count == 0 {
        if any_controller_role && (c1 > 0 || target_count > 0) {
            return RebuildOutcome::LayoutInvalid;
        }
        return if any_controller_role {
            RebuildOutcome::BindingsNotReady
        } else {
            RebuildOutcome::Ready
        };
    }

    if target_count == 0 {
        return RebuildOutcome::BindingsNotReady;
    }

    let mut resolved = 0usize;
    let mut b = 0usize;
    while b < target_count {
        if let Some(t) = targets[b] {
            let found = lookup_input_state_in_view(input_view, layout, c1, t.primary_handle)
                .or_else(|| lookup_input_state_in_view(input_view, layout, c1, t.fallback_handle));
            if let Some((ctype, dev_idx, prop_id)) = found {
                if let Some(exp) = t.expected_dev_idx
                    && dev_idx != exp
                {
                    return RebuildOutcome::LayoutInvalid;
                }
                if ctype == 1 || ctype == 2 {
                    let Some(views) = opener.open(dev_idx) else {
                        return RebuildOutcome::LayoutInvalid;
                    };
                    let Some(prop_count) = views.hdr.read_u32(layout.hdr_prop_count_off) else {
                        return RebuildOutcome::LayoutInvalid;
                    };
                    let Some(storage_sz) = views.hdr.read_u32(layout.hdr_storage_sz_off) else {
                        return RebuildOutcome::LayoutInvalid;
                    };
                    if prop_count == 0 || prop_count > SHM_PROP_COUNT_MAX {
                        return RebuildOutcome::LayoutInvalid;
                    }
                    let Some(soff) = lookup_prop_in_table(
                        &views.tbl,
                        &views.sto,
                        layout,
                        prop_count,
                        storage_sz,
                        prop_id,
                        ctype,
                    ) else {
                        return RebuildOutcome::LayoutInvalid;
                    };
                    if read_raw_bool_view(&views.sto, soff, ctype, layout).is_none() {
                        return RebuildOutcome::LayoutInvalid;
                    }
                    resolved += 1;
                }
            }
        }
        b += 1;
    }

    if resolved == 0 {
        RebuildOutcome::BindingsNotReady
    } else {
        RebuildOutcome::Ready
    }
}

fn discover_layout_with_views<C: ShmReadView, I: ShmReadView, O: ContainerOpener>(
    conn_view: &C,
    input_view: &I,
    opener: &mut O,
    anchors: &[DiscoveryAnchor],
) -> Option<SteamVrLayout> {
    if anchors.is_empty() || input_view.len() < 0x100 || conn_view.len() < 8 {
        return None;
    }

    let scan_end = input_view.len().min(0x3_0000).saturating_sub(8);
    let mut hits = [(0usize, 0usize); 64];
    let mut hit_count = 0usize;
    let mut off = 0usize;
    while off <= scan_end && hit_count < hits.len() {
        let Some(lo) = input_view.read_u32(off) else {
            break;
        };
        let Some(hi) = input_view.read_u32(off + 4) else {
            break;
        };
        if lo != 0 || hi != 0 {
            let val = (lo as u64) | ((hi as u64) << 32);
            let mut a = 0usize;
            while a < anchors.len() {
                if anchors[a].handle == val {
                    hits[hit_count] = (off, a);
                    hit_count += 1;
                    break;
                }
                a += 1;
            }
        }
        off += 4;
    }
    if hit_count == 0 {
        return None;
    }

    const CANDIDATE_STRIDES: [usize; 19] = [
        72, 80, 64, 76, 84, 88, 96, 60, 68, 56, 92, 100, 104, 108, 112, 116, 120, 124, 128,
    ];
    const CANDIDATE_ARR_OFFS: [usize; 4] = [0, 4, 8, 16];
    const CANDIDATE_TBL_STRIDES: [usize; 7] = [28, 32, 24, 36, 40, 44, 48];
    const CANDIDATE_HDR_OFFS: [(usize, usize); 3] = [(4, 8), (0, 4), (8, 12)];
    const CANDIDATE_TBL_FIELDS: [(usize, usize, usize, usize); 6] = [
        (0, 8, 16, 20),
        (0, 8, 20, 24),
        (0, 12, 20, 24),
        (0, 12, 24, 28),
        (4, 12, 20, 24),
        (4, 12, 24, 28),
    ];
    const CANDIDATE_STO_CTYPE_OFFS: [usize; 5] = [0x10, 0x08, 0x0c, 0x14, 0x18];
    const CANDIDATE_MAX_COMPS: [usize; 3] = [512, 256, 1024];

    let mut arr_i = 0usize;
    while arr_i < CANDIDATE_ARR_OFFS.len() {
        let array_off = CANDIDATE_ARR_OFFS[arr_i];
        let mut st_i = 0usize;
        while st_i < CANDIDATE_STRIDES.len() {
            let stride = CANDIDATE_STRIDES[st_i];
            if hits[0].0 < array_off {
                st_i += 1;
                continue;
            }
            let entry_handle_off = (hits[0].0 - array_off) % stride;
            if (entry_handle_off & 3) != 0 || entry_handle_off + 8 > stride {
                st_i += 1;
                continue;
            }

            let mut max_hit_idx = 0usize;
            let mut all_aligned = true;
            let mut h = 0usize;
            while h < hit_count {
                let pos = hits[h].0;
                if pos < array_off || (pos - array_off) % stride != entry_handle_off {
                    all_aligned = false;
                    break;
                }
                let eidx = (pos - array_off) / stride;
                if eidx > max_hit_idx {
                    max_hit_idx = eidx;
                }
                h += 1;
            }
            if !all_aligned || max_hit_idx >= 1024 {
                st_i += 1;
                continue;
            }

            // Probe entry_dev_idx_off, entry_ctype_off, entry_prop_id_off (default V2 offsets first).
            let mut dev_pass = 0usize;
            while dev_pass < stride {
                let dev_off = if dev_pass == 0 {
                    0x20usize
                } else {
                    let raw = dev_pass - 4;
                    if raw == 0x20 {
                        stride
                    } else {
                        raw
                    }
                };
                if dev_off + 4 > stride
                    || !spans_disjoint(dev_off, 4, entry_handle_off, 8)
                {
                    dev_pass += 4;
                    continue;
                }
                let mut dev_ok = true;
                h = 0;
                while h < hit_count {
                    let e_off = hits[h].0 - entry_handle_off;
                    let a_idx = hits[h].1;
                    let Some(dv) = input_view.read_u32(e_off + dev_off) else {
                        dev_ok = false;
                        break;
                    };
                    if !(1..K_UN_MAX_TRACKED_DEVICE_COUNT as u32).contains(&dv) {
                        dev_ok = false;
                        break;
                    }
                    if let Some(exp) = anchors[a_idx].expected_dev_idx
                        && dv != exp
                    {
                        dev_ok = false;
                        break;
                    }
                    h += 1;
                }
                if !dev_ok {
                    dev_pass += 4;
                    continue;
                }

                let mut ct_pass = 0usize;
                while ct_pass < stride {
                    let ctype_off = if ct_pass == 0 {
                        0x00usize
                    } else if ct_pass == 4 {
                        stride
                    } else {
                        ct_pass - 4
                    };
                    if ctype_off + 4 > stride
                        || !spans_disjoint(ctype_off, 4, entry_handle_off, 8)
                        || !spans_disjoint(ctype_off, 4, dev_off, 4)
                    {
                        ct_pass += 4;
                        continue;
                    }
                    let mut ct_ok = true;
                    h = 0;
                    while h < hit_count {
                        let e_off = hits[h].0 - entry_handle_off;
                        let a_idx = hits[h].1;
                        let Some(ct) = input_view.read_u32(e_off + ctype_off) else {
                            ct_ok = false;
                            break;
                        };
                        if ct != 1 && ct != 2 {
                            ct_ok = false;
                            break;
                        }
                        if let Some(hint) = anchors[a_idx].ctype_hint
                            && ct != hint
                        {
                            ct_ok = false;
                            break;
                        }
                        h += 1;
                    }
                    if !ct_ok {
                        ct_pass += 4;
                        continue;
                    }

                    let mut pr_pass = 0usize;
                    while pr_pass < stride {
                        let prop_off = if pr_pass == 0 {
                            0x30usize
                        } else {
                            let raw = pr_pass - 4;
                            if raw == 0x30 { stride } else { raw }
                        };
                        if prop_off + 4 > stride
                            || !spans_disjoint(prop_off, 4, entry_handle_off, 8)
                            || !spans_disjoint(prop_off, 4, dev_off, 4)
                            || !spans_disjoint(prop_off, 4, ctype_off, 4)
                        {
                            pr_pass += 4;
                            continue;
                        }
                        let mut prop_ok = true;
                        h = 0;
                        while h < hit_count {
                            let e_off = hits[h].0 - entry_handle_off;
                            let Some(pr) = input_view.read_u32(e_off + prop_off) else {
                                prop_ok = false;
                                break;
                            };
                            if pr == 0 || (pr & 0xF000_0000) != 0 {
                                prop_ok = false;
                                break;
                            }
                            h += 1;
                        }
                        if !prop_ok {
                            pr_pass += 4;
                            continue;
                        }

                        // Discover count/active-set offsets and container table geometry for this entry layout.
                        if let Some(layout) = probe_container_and_counts(
                            conn_view,
                            input_view,
                            opener,
                            &hits[..hit_count],
                            array_off,
                            stride,
                            ctype_off,
                            entry_handle_off,
                            dev_off,
                            prop_off,
                            max_hit_idx,
                            &CANDIDATE_HDR_OFFS,
                            &CANDIDATE_TBL_STRIDES,
                            &CANDIDATE_TBL_FIELDS,
                            &CANDIDATE_STO_CTYPE_OFFS,
                            &CANDIDATE_MAX_COMPS,
                        ) {
                            return Some(layout);
                        }

                        pr_pass += 4;
                    }
                    ct_pass += 4;
                }
                dev_pass += 4;
            }
            st_i += 1;
        }
        arr_i += 1;
    }

    None
}

#[allow(clippy::too_many_arguments)]
fn probe_container_and_counts<C: ShmReadView, I: ShmReadView, O: ContainerOpener>(
    conn_view: &C,
    input_view: &I,
    opener: &mut O,
    hits: &[(usize, usize)],
    array_off: usize,
    stride: usize,
    entry_ctype_off: usize,
    entry_handle_off: usize,
    entry_dev_idx_off: usize,
    entry_prop_id_off: usize,
    max_hit_idx: usize,
    hdr_candidates: &[(usize, usize)],
    tbl_strides: &[usize],
    tbl_fields: &[(usize, usize, usize, usize)],
    sto_ctype_offs: &[usize],
    max_comp_candidates: &[usize],
) -> Option<SteamVrLayout> {
    // Locate (max_components, input_count_off, active_set_count_off).
    let (max_components, input_count_off) = {
        let mut found = None;
        let mut mc_i = 0usize;
        while mc_i < max_comp_candidates.len() {
            let max_c = max_comp_candidates[mc_i];
            if let Some(arr_b) = max_c.checked_mul(stride)
                && let Some(c1_off) = array_off.checked_add(arr_b)
                && let Some(c1) = input_view.read_u32(c1_off)
                && (c1 as usize) > max_hit_idx
                && (c1 as usize) <= max_c
            {
                let prev_e = array_off + ((c1 as usize) - 1) * stride + entry_handle_off;
                let prev_nonzero = input_view.read_u32(prev_e).unwrap_or(0) != 0
                    || input_view.read_u32(prev_e + 4).unwrap_or(0) != 0;
                let next_zero = if (c1 as usize) < max_c {
                    let next_e = array_off + (c1 as usize) * stride + entry_handle_off;
                    input_view.read_u32(next_e) == Some(0)
                        && input_view.read_u32(next_e + 4) == Some(0)
                } else {
                    true
                };
                if prev_nonzero && next_zero {
                    found = Some((max_c, c1_off));
                    break;
                }
            }
            mc_i += 1;
        }
        found?
    };

    let active_set_count_off = {
        let rel_offs = [0x808usize, 0x804, 0x800, 0x810, 4];
        let mut found = None;
        let mut r = 0usize;
        while r < rel_offs.len() {
            let off = input_count_off + rel_offs[r];
            if let Some(c2) = input_view.read_u32(off)
                && c2 <= SHM_ACTIVE_SET_COUNT_MAX
            {
                found = Some(off);
                break;
            }
            r += 1;
        }
        found?
    };

    let conn_seq_off = if conn_view.len() >= 0x410 {
        0x408usize
    } else if conn_view.len() >= 0x408 {
        0x400usize
    } else {
        0usize
    };
    if conn_view.read_u64(conn_seq_off).is_none() {
        return None;
    }

    // Open container for first hit and match property table + storage geometry across all hits.
    let first_e = hits[0].0 - entry_handle_off;
    let first_dev = input_view.read_u32(first_e + entry_dev_idx_off)?;
    let first_ctype = input_view.read_u32(first_e + entry_ctype_off)?;
    let first_prop_id = 0x1000_0000u32 | input_view.read_u32(first_e + entry_prop_id_off)?;
    let first_views = opener.open(first_dev)?;

    let mut hdr_i = 0usize;
    while hdr_i < hdr_candidates.len() {
        let (hdr_cnt_off, hdr_sz_off) = hdr_candidates[hdr_i];
        let Some(prop_count) = first_views.hdr.read_u32(hdr_cnt_off) else {
            hdr_i += 1;
            continue;
        };
        let Some(storage_sz) = first_views.hdr.read_u32(hdr_sz_off) else {
            hdr_i += 1;
            continue;
        };
        if prop_count == 0
            || prop_count > SHM_PROP_COUNT_MAX
            || storage_sz == 0
            || (storage_sz as usize) > first_views.sto.len()
        {
            hdr_i += 1;
            continue;
        }

        let mut tf_i = 0usize;
        while tf_i < tbl_fields.len() {
            let (id_off, type_off, off_off, len_off) = tbl_fields[tf_i];
            let mut ts_i = 0usize;
            while ts_i < tbl_strides.len() {
                let tbl_stride = tbl_strides[ts_i];
                if len_off + 4 > tbl_stride || (prop_count as usize) * tbl_stride > first_views.tbl.len() {
                    ts_i += 1;
                    continue;
                }
                let mut table_rows_sane = true;
                let mut matched_entry = None;
                let mut p = 0usize;
                while p < (prop_count as usize) {
                    let toff = p * tbl_stride;
                    let Some(pid) = first_views.tbl.read_u32(toff + id_off) else {
                        table_rows_sane = false;
                        break;
                    };
                    let Some(ptype) = first_views.tbl.read_u32(toff + type_off) else {
                        table_rows_sane = false;
                        break;
                    };
                    let Some(soff) = first_views.tbl.read_u32(toff + off_off) else {
                        table_rows_sane = false;
                        break;
                    };
                    let Some(slen) = first_views.tbl.read_u32(toff + len_off) else {
                        table_rows_sane = false;
                        break;
                    };
                    if pid == 0
                        || ptype == 0
                        || ptype > 0x100
                        || (soff as usize).saturating_add(slen as usize) > (storage_sz as usize)
                    {
                        table_rows_sane = false;
                        break;
                    }
                    if pid == first_prop_id && ptype == PROP_TYPE_INPUT_COMPONENT {
                        if (slen as usize) >= 0x1c
                            && (slen as usize) <= 256
                            && (slen as usize & 3) == 0
                        {
                            matched_entry = Some((soff as usize, slen as usize));
                        } else {
                            table_rows_sane = false;
                            break;
                        }
                    }
                    p += 1;
                }

                if table_rows_sane
                    && let Some((first_soff, storage_bytes)) = matched_entry
                {
                    let mut sc_i = 0usize;
                    while sc_i < sto_ctype_offs.len() {
                        let sto_ctype_off = sto_ctype_offs[sc_i];
                        let sto_value_off = sto_ctype_off + 8;
                        if sto_value_off + 4 > storage_bytes
                            || first_views.sto.read_u32(first_soff + sto_ctype_off)
                                != Some(first_ctype)
                        {
                            sc_i += 1;
                            continue;
                        }

                        let candidate = SteamVrLayout {
                            name: "discovered_shm",
                            conn_seq_off,
                            input_array_off: array_off,
                            input_entry_bytes: stride,
                            max_components,
                            input_count_off,
                            active_set_count_off,
                            entry_ctype_off,
                            entry_handle_off,
                            entry_dev_idx_off,
                            entry_prop_id_off,
                            hdr_prop_count_off: hdr_cnt_off,
                            hdr_storage_sz_off: hdr_sz_off,
                            prop_table_entry_bytes: tbl_stride,
                            prop_entry_id_off: id_off,
                            prop_entry_type_off: type_off,
                            prop_entry_offset_off: off_off,
                            prop_entry_len_off: len_off,
                            prop_type_input_component: PROP_TYPE_INPUT_COMPONENT,
                            storage_component_bytes: storage_bytes,
                            storage_ctype_off: sto_ctype_off,
                            storage_value_off: sto_value_off,
                        };
                        if !candidate.is_structurally_sane() {
                            sc_i += 1;
                            continue;
                        }

                        // Verify every anchor hit resolves cleanly in its property container under this layout.
                        let mut all_hits_ok = true;
                        let mut h = 1usize;
                        while h < hits.len() {
                            let e = hits[h].0 - entry_handle_off;
                            let Some(dv) = input_view.read_u32(e + entry_dev_idx_off) else {
                                all_hits_ok = false;
                                break;
                            };
                            let Some(ct) = input_view.read_u32(e + entry_ctype_off) else {
                                all_hits_ok = false;
                                break;
                            };
                            let Some(pr_lo) = input_view.read_u32(e + entry_prop_id_off) else {
                                all_hits_ok = false;
                                break;
                            };
                            let Some(v) = opener.open(dv) else {
                                all_hits_ok = false;
                                break;
                            };
                            let Some(pc) = v.hdr.read_u32(hdr_cnt_off) else {
                                all_hits_ok = false;
                                break;
                            };
                            let Some(ssz) = v.hdr.read_u32(hdr_sz_off) else {
                                all_hits_ok = false;
                                break;
                            };
                            if pc == 0
                                || pc > SHM_PROP_COUNT_MAX
                                || lookup_prop_in_table(
                                    &v.tbl,
                                    &v.sto,
                                    &candidate,
                                    pc,
                                    ssz,
                                    0x1000_0000u32 | pr_lo,
                                    ct,
                                )
                                .is_none()
                            {
                                all_hits_ok = false;
                                break;
                            }
                            h += 1;
                        }
                        if all_hits_ok {
                            return Some(candidate);
                        }
                        sc_i += 1;
                    }
                }
                ts_i += 1;
            }
            tf_i += 1;
        }
        hdr_i += 1;
    }

    None
}

struct WatchedContainer {
    dev_idx: u32,
    hdr_shm: MappedReadOnlyShm,
    tbl_shm: MappedReadOnlyShm,
    storage_shm: MappedReadOnlyShm,
    prop_count: u32,
    storage_sz: u32,
    slot_count: usize,
    slots: [(usize, u32); MAX_SLOTS_PER_CONTAINER],
}

/// Bare-metal shared-memory input watcher for SteamVR controller property containers.
///
/// Validates a `SteamVrLayout` against mapped region bounds and OpenVR binding handles, reads
/// bound controller components directly from shared memory with SIMD, and verifies layout
/// coherence before/after snapshot reads.
pub struct BareMetalInputMonitor {
    server_pid: u32,
    server_wait_handle: Option<ProcessWaitHandle>,
    input_state_shm: MappedReadOnlyShm,
    conn_state_shm: MappedReadOnlyShm,
    layout: SteamVrLayout,
    trust: LayoutTrust,
    shm_valid: bool,
    all_connected_bindings_mapped: bool,
    last_conn_seq: u64,
    last_c1: u32,
    last_c2: u32,
    grace_until_ns: u64,
    containers: [Option<WatchedContainer>; MAX_WATCHED_CONTAINERS],
    total_slots: usize,
    prev_vals: [u32; MAX_TOTAL_SLOTS],
    curr_vals: [u32; MAX_TOTAL_SLOTS],
    bound_slots: [Option<BoundShmSlot>; MAX_BOUND_SLOTS],
    bound_count: usize,
}

impl BareMetalInputMonitor {
    fn open_and_validate(session: &OpenVrSession, now_ns: u64) -> OpenMonitorResult {
        let Some((server_pid, server_wait_handle, input_state_shm, conn_state_shm)) =
            find_connected_vrserver_shm()
        else {
            return OpenMonitorResult::LayoutInvalid;
        };

        let mut monitor = BareMetalInputMonitor {
            server_pid,
            server_wait_handle,
            input_state_shm,
            conn_state_shm,
            layout: STEAMVR_LAYOUT_V2,
            trust: LayoutTrust::Invalid,
            shm_valid: false,
            all_connected_bindings_mapped: false,
            last_conn_seq: 0,
            last_c1: 0,
            last_c2: 0,
            grace_until_ns: 0,
            containers: [const { None }; MAX_WATCHED_CONTAINERS],
            total_slots: 0,
            prev_vals: [0; MAX_TOTAL_SLOTS],
            curr_vals: [0; MAX_TOTAL_SLOTS],
            bound_slots: [None; MAX_BOUND_SLOTS],
            bound_count: 0,
        };

        match monitor.rebuild_and_validate(session, now_ns) {
            RebuildOutcome::Ready => OpenMonitorResult::Ready(monitor),
            RebuildOutcome::BindingsNotReady => OpenMonitorResult::BindingsNotReady,
            RebuildOutcome::LayoutInvalid => OpenMonitorResult::LayoutInvalid,
        }
    }

    #[inline]
    pub fn vrserver_wait_handle(&self) -> Option<isize> {
        self.server_wait_handle.as_ref().map(|h| h.raw())
    }

    #[inline]
    pub fn watched_slot_count(&self) -> usize {
        self.total_slots
    }

    #[inline]
    pub fn watched_bound_count(&self) -> usize {
        self.bound_count
    }

    #[inline]
    pub fn layout_name(&self) -> &'static str {
        self.layout.name
    }

    #[inline]
    pub fn trust(&self) -> LayoutTrust {
        self.trust
    }

    #[inline]
    pub fn is_shm_valid(&self) -> bool {
        self.shm_valid
    }

    #[inline]
    pub fn has_validated_bindings(&self) -> bool {
        self.shm_valid
            && self.trust != LayoutTrust::Invalid
            && self.bound_count > 0
            && self.all_connected_bindings_mapped
    }

    #[inline]
    pub fn extend_grace(&mut self, now_ns: u64) {
        self.grace_until_ns = now_ns.saturating_add(POST_EDGE_GRACE_NS);
    }

    fn clear_containers(&mut self) {
        for slot in &mut self.containers {
            *slot = None;
        }
        self.total_slots = 0;
        self.bound_slots = [None; MAX_BOUND_SLOTS];
        self.bound_count = 0;
        self.all_connected_bindings_mapped = false;
    }

    fn select_and_populate_layout(
        &mut self,
        session: &OpenVrSession,
    ) -> RebuildOutcome {
        let mut targets = [None; MAX_BOUND_SLOTS];
        let (target_count, any_incomplete) = collect_binding_targets(session, &mut targets);
        let controller_devs = (
            session.controller_device_index(Hand::Left),
            session.controller_device_index(Hand::Right),
        );
        let mut opener = MappedContainerOpener {
            server_pid: self.server_pid,
        };

        // 1. Try known verified descriptors first.
        let mut k = 0usize;
        let mut last_outcome = RebuildOutcome::LayoutInvalid;
        while k < KNOWN_LAYOUTS.len() {
            let outcome = validate_layout_on_views(
                &KNOWN_LAYOUTS[k],
                &self.conn_state_shm,
                &self.input_state_shm,
                &mut opener,
                &targets,
                target_count,
                controller_devs,
            );
            match outcome {
                RebuildOutcome::Ready => {
                    return self.populate_validated_layout(
                        KNOWN_LAYOUTS[k],
                        LayoutTrust::VerifiedKnown,
                        &targets,
                        target_count,
                        any_incomplete,
                    );
                }
                RebuildOutcome::BindingsNotReady => {
                    self.layout = KNOWN_LAYOUTS[k];
                    self.trust = LayoutTrust::VerifiedKnown;
                    last_outcome = RebuildOutcome::BindingsNotReady;
                }
                RebuildOutcome::LayoutInvalid => {}
            }
            k += 1;
        }
        if last_outcome == RebuildOutcome::BindingsNotReady {
            return RebuildOutcome::BindingsNotReady;
        }

        // 2. Try structural discovery anchored on public OpenVR component handles.
        if target_count > 0 {
            let mut anchors = [DiscoveryAnchor {
                handle: 0,
                expected_dev_idx: None,
                ctype_hint: None,
            }; 32];
            let anchor_count = build_discovery_anchors(&targets[..target_count], &mut anchors);
            if anchor_count > 0
                && let Some(discovered) = discover_layout_with_views(
                    &self.conn_state_shm,
                    &self.input_state_shm,
                    &mut opener,
                    &anchors[..anchor_count],
                )
            {
                let outcome = validate_layout_on_views(
                    &discovered,
                    &self.conn_state_shm,
                    &self.input_state_shm,
                    &mut opener,
                    &targets,
                    target_count,
                    controller_devs,
                );
                if outcome == RebuildOutcome::Ready {
                    return self.populate_validated_layout(
                        discovered,
                        LayoutTrust::DiscoveredValidated,
                        &targets,
                        target_count,
                        any_incomplete,
                    );
                }
            }
        }

        self.trust = LayoutTrust::Invalid;
        self.shm_valid = false;
        self.clear_containers();
        if target_count == 0 {
            RebuildOutcome::BindingsNotReady
        } else {
            RebuildOutcome::LayoutInvalid
        }
    }

    fn populate_validated_layout(
        &mut self,
        layout: SteamVrLayout,
        trust: LayoutTrust,
        targets: &[Option<BindingTarget>; MAX_BOUND_SLOTS],
        target_count: usize,
        any_incomplete: bool,
    ) -> RebuildOutcome {
        let Some((conn_seq, c1, c2)) =
            read_global_shm_meta(&self.conn_state_shm, &self.input_state_shm, &layout)
        else {
            self.shm_valid = false;
            self.trust = LayoutTrust::Invalid;
            return RebuildOutcome::LayoutInvalid;
        };

        self.clear_containers();
        self.layout = layout;
        self.trust = trust;

        let mut devs = [u32::MAX; MAX_WATCHED_CONTAINERS];
        let mut dev_count = 0usize;
        let mut i = 0usize;
        while i < (c1 as usize) {
            let entry_off = layout.input_array_off + i * layout.input_entry_bytes;
            let Some(comp_type) = self
                .input_state_shm
                .read_volatile_u32(entry_off + layout.entry_ctype_off)
            else {
                self.shm_valid = false;
                return RebuildOutcome::LayoutInvalid;
            };
            let Some(dev_idx) = self
                .input_state_shm
                .read_volatile_u32(entry_off + layout.entry_dev_idx_off)
            else {
                self.shm_valid = false;
                return RebuildOutcome::LayoutInvalid;
            };
            if comp_type <= 2 && (1..K_UN_MAX_TRACKED_DEVICE_COUNT as u32).contains(&dev_idx) {
                let mut seen = false;
                let mut k = 0usize;
                while k < dev_count {
                    if devs[k] == dev_idx {
                        seen = true;
                        break;
                    }
                    k += 1;
                }
                if !seen && dev_count < MAX_WATCHED_CONTAINERS {
                    devs[dev_count] = dev_idx;
                    dev_count += 1;
                }
            }
            i += 1;
        }

        let mut d = 0usize;
        while d < dev_count {
            let dev_idx = devs[d];
            let cid = 0x1_0000_0000u64 + (dev_idx as u64);
            let Some((hdr_shm, tbl_shm, sto_shm)) = open_property_container_shm(self.server_pid, cid)
            else {
                self.shm_valid = false;
                self.clear_containers();
                return RebuildOutcome::LayoutInvalid;
            };
            let Some(prop_count) = hdr_shm.read_volatile_u32(layout.hdr_prop_count_off) else {
                self.shm_valid = false;
                self.clear_containers();
                return RebuildOutcome::LayoutInvalid;
            };
            let Some(storage_sz) = hdr_shm.read_volatile_u32(layout.hdr_storage_sz_off) else {
                self.shm_valid = false;
                self.clear_containers();
                return RebuildOutcome::LayoutInvalid;
            };
            if prop_count == 0
                || prop_count > SHM_PROP_COUNT_MAX
                || (prop_count as usize) * layout.prop_table_entry_bytes > tbl_shm.len()
                || (storage_sz as usize) > sto_shm.len()
            {
                self.shm_valid = false;
                self.clear_containers();
                return RebuildOutcome::LayoutInvalid;
            }
            self.containers[d] = Some(WatchedContainer {
                dev_idx,
                hdr_shm,
                tbl_shm,
                storage_shm: sto_shm,
                prop_count,
                storage_sz,
                slot_count: 0,
                slots: [(0, 0); MAX_SLOTS_PER_CONTAINER],
            });
            d += 1;
        }

        let mut all_connected_ok = !any_incomplete;
        let mut b = 0usize;
        while b < target_count && self.bound_count < MAX_BOUND_SLOTS {
            if let Some(t) = targets[b] {
                let found = lookup_input_state_in_view(
                    &self.input_state_shm,
                    &layout,
                    c1,
                    t.primary_handle,
                )
                .or_else(|| {
                    lookup_input_state_in_view(
                        &self.input_state_shm,
                        &layout,
                        c1,
                        t.fallback_handle,
                    )
                });
                if let Some((ctype, dev_idx, prop_id)) = found {
                    if let Some(exp) = t.expected_dev_idx
                        && dev_idx != exp
                    {
                        self.shm_valid = false;
                        self.trust = LayoutTrust::Invalid;
                        self.clear_containers();
                        return RebuildOutcome::LayoutInvalid;
                    }
                    if ctype == 1 || ctype == 2 {
                        let mut matched_loc = None;
                        let mut cidx = 0usize;
                        while cidx < dev_count {
                            if let Some(ref c) = self.containers[cidx]
                                && c.dev_idx == dev_idx
                            {
                                if let Some(soff) = lookup_prop_in_table(
                                    &c.tbl_shm,
                                    &c.storage_shm,
                                    &layout,
                                    c.prop_count,
                                    c.storage_sz,
                                    prop_id,
                                    ctype,
                                ) {
                                    matched_loc = Some((cidx, soff));
                                }
                                break;
                            }
                            cidx += 1;
                        }
                        if let Some((container_idx, storage_off)) = matched_loc {
                            self.bound_slots[self.bound_count] = Some(BoundShmSlot {
                                hand: t.hand,
                                dev_idx,
                                container_idx,
                                storage_off,
                                ctype,
                                kind: t.kind,
                                detector: DoubleClickDetector::default(),
                            });
                            self.bound_count += 1;
                        } else if t.expected_dev_idx.is_some() {
                            all_connected_ok = false;
                        }
                    } else if t.expected_dev_idx.is_some() {
                        all_connected_ok = false;
                    }
                } else if t.expected_dev_idx.is_some() {
                    all_connected_ok = false;
                }
            }
            b += 1;
        }

        // Only watch bound slots (never enumerate unrelated components).
        b = 0;
        while b < self.bound_count {
            if let Some(bs) = self.bound_slots[b]
                && let Some(ref mut c) = self.containers[bs.container_idx]
            {
                let mut dup = false;
                let mut k = 0usize;
                while k < c.slot_count {
                    if c.slots[k].0 == bs.storage_off {
                        dup = true;
                        break;
                    }
                    k += 1;
                }
                if !dup
                    && c.slot_count < MAX_SLOTS_PER_CONTAINER
                    && self.total_slots < MAX_TOTAL_SLOTS
                {
                    c.slots[c.slot_count] = (bs.storage_off, bs.ctype);
                    c.slot_count += 1;
                    self.total_slots += 1;
                }
            }
            b += 1;
        }

        if dev_count > 0 && self.bound_count == 0 {
            self.shm_valid = false;
            self.clear_containers();
            return RebuildOutcome::BindingsNotReady;
        }

        self.last_conn_seq = conn_seq;
        self.last_c1 = c1;
        self.last_c2 = c2;
        self.all_connected_bindings_mapped = all_connected_ok;

        if !self.read_all_slots_into_prev() {
            self.shm_valid = false;
            self.trust = LayoutTrust::Invalid;
            self.clear_containers();
            return RebuildOutcome::LayoutInvalid;
        }

        // Verify metadata stayed coherent during rebuild.
        let Some(after) = self.read_metadata_snapshot() else {
            self.shm_valid = false;
            self.clear_containers();
            return RebuildOutcome::BindingsNotReady;
        };
        if !self.snapshot_matches_cached(&after) {
            self.shm_valid = false;
            self.clear_containers();
            return RebuildOutcome::BindingsNotReady;
        }

        self.shm_valid = true;
        RebuildOutcome::Ready
    }

    pub fn rebuild_and_validate(
        &mut self,
        session: &OpenVrSession,
        now_ns: u64,
    ) -> RebuildOutcome {
        let mut attempt = 0usize;
        let mut outcome = RebuildOutcome::LayoutInvalid;
        while attempt < 2 {
            outcome = self.select_and_populate_layout(session);
            if outcome == RebuildOutcome::Ready {
                break;
            }
            attempt += 1;
        }
        if outcome != RebuildOutcome::Ready {
            return outcome;
        }

        // Discovered unknown layouts require live confirmation against GetDigitalActionData before use.
        if self.bound_count > 0 {
            match self.cross_check_with_openvr(session, now_ns) {
                CrossCheckOutcome::Mismatch => {
                    self.shm_valid = false;
                    self.trust = LayoutTrust::Invalid;
                    self.clear_containers();
                    return RebuildOutcome::LayoutInvalid;
                }
                CrossCheckOutcome::Skipped if self.trust == LayoutTrust::DiscoveredValidated => {
                    // Dashboard is currently suppressing OpenVR action data; retry via reprobe
                    // once active action comparison is possible.
                    self.shm_valid = false;
                    self.trust = LayoutTrust::Invalid;
                    self.clear_containers();
                    return RebuildOutcome::BindingsNotReady;
                }
                CrossCheckOutcome::Matched | CrossCheckOutcome::Skipped => {}
            }
        }

        RebuildOutcome::Ready
    }

    pub fn rebuild_for_session(&mut self, session: &OpenVrSession) -> bool {
        self.rebuild_and_validate(session, 0) == RebuildOutcome::Ready
    }

    fn read_metadata_snapshot(&self) -> Option<ShmMetadataSnapshot> {
        let (conn_seq, c1, c2) =
            read_global_shm_meta(&self.conn_state_shm, &self.input_state_shm, &self.layout)?;
        let mut container_props = [(0u32, 0u32); MAX_WATCHED_CONTAINERS];
        let mut cidx = 0usize;
        while cidx < self.containers.len() {
            if let Some(ref c) = self.containers[cidx] {
                let cur_count = c.hdr_shm.read_volatile_u32(self.layout.hdr_prop_count_off)?;
                let cur_sz = c.hdr_shm.read_volatile_u32(self.layout.hdr_storage_sz_off)?;
                if cur_count == 0
                    || cur_count > SHM_PROP_COUNT_MAX
                    || (cur_count as usize) * self.layout.prop_table_entry_bytes > c.tbl_shm.len()
                    || (cur_sz as usize) > c.storage_shm.len()
                {
                    return None;
                }
                container_props[cidx] = (cur_count, cur_sz);
            }
            cidx += 1;
        }
        Some(ShmMetadataSnapshot {
            conn_seq,
            c1,
            c2,
            container_props,
        })
    }

    fn snapshot_matches_cached(&self, snap: &ShmMetadataSnapshot) -> bool {
        if snap.conn_seq != self.last_conn_seq || snap.c1 != self.last_c1 || snap.c2 != self.last_c2
        {
            return false;
        }
        let mut cidx = 0usize;
        while cidx < self.containers.len() {
            if let Some(ref c) = self.containers[cidx] {
                let (cnt, sz) = snap.container_props[cidx];
                if cnt != c.prop_count || sz != c.storage_sz {
                    return false;
                }
            }
            cidx += 1;
        }
        true
    }

    #[inline]
    pub fn reseed_current_slots(&mut self) -> bool {
        if !self.shm_valid {
            return false;
        }
        let Some(before) = self.read_metadata_snapshot() else {
            self.shm_valid = false;
            return false;
        };
        if !self.snapshot_matches_cached(&before) {
            return false;
        }
        if !self.read_all_slots_into_prev() {
            self.shm_valid = false;
            return false;
        }
        let Some(after) = self.read_metadata_snapshot() else {
            self.shm_valid = false;
            return false;
        };
        before == after
    }

    #[inline]
    fn has_bound_hand(&self, hand: Hand) -> bool {
        let mut b = 0usize;
        while b < self.bound_count {
            if let Some(bs) = self.bound_slots[b]
                && bs.hand == hand
            {
                return true;
            }
            b += 1;
        }
        false
    }

    fn cross_check_with_openvr(&mut self, session: &OpenVrSession, now_ns: u64) -> CrossCheckOutcome {
        let mut attempt = 0usize;
        while attempt < 2 {
            let (err, left_err, right_err, left_data, right_data) =
                session.query_openvr_digital_actions();
            if err != VR_INPUT_ERROR_NONE {
                return CrossCheckOutcome::Skipped;
            }

            let l_active = left_err == VR_INPUT_ERROR_NONE && left_data.b_active;
            let r_active = right_err == VR_INPUT_ERROR_NONE && right_data.b_active;
            if !l_active && !r_active {
                return CrossCheckOutcome::Skipped;
            }

            let mut mismatch = false;
            let mut compared_any = false;
            let checks = [(Hand::Left, l_active, left_data), (Hand::Right, r_active, right_data)];
            let mut k = 0usize;
            while k < checks.len() {
                let (hand, active, data) = checks[k];
                if active && self.has_bound_hand(hand) {
                    if !data.b_changed {
                        let ev = self.eval_hand_direct_state(hand);
                        match ev {
                            ShmEval::Invalid => return CrossCheckOutcome::Mismatch,
                            ShmEval::Unbound => {}
                            ShmEval::Pressed(_) => {
                                compared_any = true;
                                if !data.b_state {
                                    mismatch = true;
                                }
                            }
                            ShmEval::Unpressed => {
                                compared_any = true;
                                // Note: if user is holding a double-click binding (`b_state == true`),
                                // wait/skip comparing `Direct` only vs `Double` unless `!data.b_state`
                                // where SHM raw must also be false for Direct slots.
                                let has_double = self.hand_has_double_binding(hand);
                                if data.b_state && !has_double {
                                    mismatch = true;
                                }
                            }
                        }
                    }
                    self.sync_hand_detectors(hand, now_ns);
                }
                k += 1;
            }

            if !mismatch {
                return if compared_any {
                    CrossCheckOutcome::Matched
                } else {
                    CrossCheckOutcome::Skipped
                };
            }
            attempt += 1;
        }
        CrossCheckOutcome::Mismatch
    }

    #[inline]
    fn hand_has_double_binding(&self, hand: Hand) -> bool {
        let mut b = 0usize;
        while b < self.bound_count {
            if let Some(bs) = self.bound_slots[b]
                && bs.hand == hand
                && bs.kind == TriggerKind::Double
            {
                return true;
            }
            b += 1;
        }
        false
    }

    /// Evaluates raw direct-press state without mutating `DoubleClickDetector` timing state.
    fn eval_hand_direct_state(&self, hand: Hand) -> ShmEval {
        if !self.shm_valid || self.bound_count == 0 {
            return ShmEval::Invalid;
        }
        let mut any_bound = false;
        let mut any_direct_pressed = false;
        let mut pressed_dev = u32::MAX;
        let mut b = 0usize;
        while b < self.bound_count {
            if let Some(bs) = self.bound_slots[b]
                && bs.hand == hand
            {
                any_bound = true;
                let Some(ref c) = self.containers[bs.container_idx] else {
                    return ShmEval::Invalid;
                };
                let Some(raw) =
                    read_raw_bool_view(&c.storage_shm, bs.storage_off, bs.ctype, &self.layout)
                else {
                    return ShmEval::Invalid;
                };
                if bs.kind == TriggerKind::Direct && raw {
                    any_direct_pressed = true;
                    pressed_dev = bs.dev_idx;
                }
            }
            b += 1;
        }
        if !any_bound {
            ShmEval::Unbound
        } else if any_direct_pressed {
            ShmEval::Pressed(pressed_dev)
        } else {
            ShmEval::Unpressed
        }
    }

    pub fn sync_hand_detectors(&mut self, hand: Hand, _now_ns: u64) {
        let mut b = 0usize;
        while b < self.bound_count {
            if let Some(mut bs) = self.bound_slots[b]
                && bs.hand == hand
                && let Some(ref c) = self.containers[bs.container_idx]
                && let Some(raw) =
                    read_raw_bool_view(&c.storage_shm, bs.storage_off, bs.ctype, &self.layout)
            {
                bs.detector.sync_inactive(raw);
                self.bound_slots[b] = Some(bs);
            }
            b += 1;
        }
    }

    pub fn eval_hand_status(&mut self, hand: Hand, now_ns: u64) -> ShmEval {
        if !self.shm_valid || self.bound_count == 0 {
            return ShmEval::Invalid;
        }
        let mut any_bound_for_hand = false;
        let mut any_pressed = false;
        let mut dev_idx = u32::MAX;
        let mut b = 0usize;
        while b < self.bound_count {
            if let Some(mut bs) = self.bound_slots[b]
                && bs.hand == hand
            {
                any_bound_for_hand = true;
                let Some(ref c) = self.containers[bs.container_idx] else {
                    self.shm_valid = false;
                    return ShmEval::Invalid;
                };
                let Some(raw) =
                    read_raw_bool_view(&c.storage_shm, bs.storage_off, bs.ctype, &self.layout)
                else {
                    self.shm_valid = false;
                    return ShmEval::Invalid;
                };
                let pressed = match bs.kind {
                    TriggerKind::Direct => {
                        bs.detector.sync_inactive(raw);
                        raw
                    }
                    TriggerKind::Double => bs.detector.step(raw, now_ns),
                };
                self.bound_slots[b] = Some(bs);
                if dev_idx == u32::MAX {
                    dev_idx = bs.dev_idx;
                }
                if pressed {
                    any_pressed = true;
                    dev_idx = bs.dev_idx;
                }
            }
            b += 1;
        }
        if !any_bound_for_hand {
            ShmEval::Unbound
        } else if any_pressed && dev_idx != u32::MAX {
            ShmEval::Pressed(dev_idx)
        } else {
            ShmEval::Unpressed
        }
    }

    pub fn eval_hand_from_shm(&mut self, hand: Hand, now_ns: u64) -> Option<(bool, u32)> {
        match self.eval_hand_status(hand, now_ns) {
            ShmEval::Pressed(dev) => Some((true, dev)),
            ShmEval::Unpressed => Some((false, 0)),
            ShmEval::Unbound | ShmEval::Invalid => None,
        }
    }

    fn read_slot_value(
        sto: &MappedReadOnlyShm,
        off: usize,
        ctype: u32,
        layout: &SteamVrLayout,
    ) -> Option<u32> {
        if ctype == 1 {
            let b = sto.read_volatile_u8(off + layout.storage_value_off)?;
            Some(if b != 0 { 1 } else { 0 })
        } else {
            let raw = sto.read_volatile_u32(off + layout.storage_value_off)?;
            let f = f32::from_bits(raw);
            let bucket = if f.is_finite() {
                (f.clamp(-1.0, 1.0) * 20.0) as i32
            } else {
                0
            };
            Some(bucket as u32)
        }
    }

    fn read_all_slots_into(&mut self, into_prev: bool) -> bool {
        let mut idx = 0usize;
        let mut cidx = 0usize;
        while cidx < self.containers.len() {
            if let Some(ref c) = self.containers[cidx] {
                let mut s = 0usize;
                while s < c.slot_count {
                    let (off, ctype) = c.slots[s];
                    let Some(val) = Self::read_slot_value(&c.storage_shm, off, ctype, &self.layout)
                    else {
                        return false;
                    };
                    if into_prev {
                        self.prev_vals[idx] = val;
                    } else {
                        self.curr_vals[idx] = val;
                    }
                    idx += 1;
                    s += 1;
                }
            }
            cidx += 1;
        }
        idx == self.total_slots
    }

    #[inline]
    fn read_all_slots_into_prev(&mut self) -> bool {
        self.read_all_slots_into(true)
    }

    /// Checks SteamVR live controller property containers via SIMD with coherent before/after
    /// metadata snapshots.
    /// Returns `(should_sample_openvr, bindings_or_devices_changed)`.
    pub fn poll_should_sample(&mut self, simd: SimdMath, now_ns: u64) -> (bool, bool) {
        if !self.shm_valid {
            return (true, false);
        }

        let Some(before) = self.read_metadata_snapshot() else {
            self.shm_valid = false;
            return (true, false);
        };
        if !self.snapshot_matches_cached(&before) {
            self.grace_until_ns = now_ns.saturating_add(POST_EDGE_GRACE_NS);
            return (true, true);
        }

        if !self.read_all_slots_into(false) {
            self.shm_valid = false;
            return (true, false);
        }

        let Some(after) = self.read_metadata_snapshot() else {
            self.shm_valid = false;
            return (true, false);
        };
        if before != after {
            self.grace_until_ns = now_ns.saturating_add(POST_EDGE_GRACE_NS);
            return (true, true);
        }

        let n = self.total_slots;
        if simd.slices_equal_u32(&self.curr_vals[..n], &self.prev_vals[..n]) {
            return (now_ns < self.grace_until_ns, false);
        }

        self.prev_vals[..n].copy_from_slice(&self.curr_vals[..n]);
        self.grace_until_ns = now_ns.saturating_add(POST_EDGE_GRACE_NS);
        (true, false)
    }
}

/// Manages shared-memory fast-path lifecycle, bounded reprobe recovery (`5s -> 15s -> 60s`),
/// and 60-second idle health cross-checks against public OpenVR.
pub struct ShmMonitorManager {
    monitor: Option<BareMetalInputMonitor>,
    backoff_stage: u8,
    next_reprobe_ns: u64,
    next_health_check_ns: u64,
    next_controller_poll_ns: u64,
    warned_fallback: bool,
    last_layout_name: &'static str,
    last_trust: LayoutTrust,
}

impl ShmMonitorManager {
    pub fn new(session: &OpenVrSession, now_ns: u64) -> Self {
        let mut mgr = Self {
            monitor: None,
            backoff_stage: 0,
            next_reprobe_ns: 0,
            next_health_check_ns: now_ns.saturating_add(HEALTH_CHECK_INTERVAL_NS),
            next_controller_poll_ns: now_ns.saturating_add(NO_SLOTS_CONTROLLER_POLL_NS),
            warned_fallback: false,
            last_layout_name: "none",
            last_trust: LayoutTrust::Invalid,
        };
        mgr.try_open_fresh(session, now_ns);
        mgr
    }

    fn warn_fallback_once(&mut self, session: &OpenVrSession) {
        if !self.warned_fallback {
            self.warned_fallback = true;
            let mut buf = StackBuf::<256>::new();
            let _ = write!(
                buf,
                "nexashift[warn]: SHM fast path unavailable (runtime={}), using OpenVR fallback; Dashboard drag may be unavailable\n",
                session.runtime_version()
            );
            write_stderr(buf.as_bytes());
        }
    }

    fn enter_fallback(&mut self, session: &OpenVrSession, now_ns: u64, outcome: RebuildOutcome) {
        self.monitor = None;
        self.last_layout_name = "none";
        self.last_trust = LayoutTrust::Invalid;
        self.next_reprobe_ns = now_ns.saturating_add(reprobe_backoff_ns(self.backoff_stage));
        if self.backoff_stage < 2 {
            self.backoff_stage += 1;
        }
        if outcome == RebuildOutcome::LayoutInvalid {
            self.warn_fallback_once(session);
        }
    }

    fn record_ready(&mut self, now_ns: u64) {
        self.backoff_stage = 0;
        self.warned_fallback = false;
        self.next_health_check_ns = now_ns.saturating_add(HEALTH_CHECK_INTERVAL_NS);
        self.next_controller_poll_ns = now_ns.saturating_add(NO_SLOTS_CONTROLLER_POLL_NS);
        if let Some(ref m) = self.monitor {
            self.last_layout_name = m.layout_name();
            self.last_trust = m.trust();
        }
    }

    fn try_open_fresh(&mut self, session: &OpenVrSession, now_ns: u64) {
        match BareMetalInputMonitor::open_and_validate(session, now_ns) {
            OpenMonitorResult::Ready(m) => {
                self.monitor = Some(m);
                self.record_ready(now_ns);
            }
            OpenMonitorResult::BindingsNotReady => {
                self.enter_fallback(session, now_ns, RebuildOutcome::BindingsNotReady);
            }
            OpenMonitorResult::LayoutInvalid => {
                self.enter_fallback(session, now_ns, RebuildOutcome::LayoutInvalid);
            }
        }
    }

    /// Idle poll handler: checks bindings dirty, runs SHM SIMD diff, rebuilds on topology change,
    /// performs bounded recovery reprobes (`5s -> 15s -> 60s`), and executes 60s health cross-checks.
    /// Returns `true` when OpenVR should be sampled on this tick.
    pub fn poll_idle(&mut self, session: &OpenVrSession, simd: SimdMath, now_ns: u64) -> bool {
        let event_bindings_dirty = session.take_bindings_dirty();

        if self.monitor.is_none() {
            if event_bindings_dirty || now_ns >= self.next_reprobe_ns {
                self.try_open_fresh(session, now_ns);
            }
            return true;
        }

        if event_bindings_dirty {
            let outcome = match self.monitor.as_mut() {
                Some(m) => m.rebuild_and_validate(session, now_ns),
                None => RebuildOutcome::LayoutInvalid,
            };
            if outcome == RebuildOutcome::Ready {
                self.record_ready(now_ns);
            } else {
                self.enter_fallback(session, now_ns, outcome);
            }
            return true;
        }

        let (should_sample, topology_changed, shm_valid, has_bindings) = {
            let m = self.monitor.as_mut().expect("checked is_some above");
            let (s, topo) = m.poll_should_sample(simd, now_ns);
            (s, topo, m.is_shm_valid(), m.has_validated_bindings())
        };

        if !shm_valid {
            self.enter_fallback(session, now_ns, RebuildOutcome::LayoutInvalid);
            return true;
        }

        if topology_changed {
            let outcome = self
                .monitor
                .as_mut()
                .expect("checked is_some")
                .rebuild_and_validate(session, now_ns);
            if outcome == RebuildOutcome::Ready {
                self.record_ready(now_ns);
            } else {
                self.enter_fallback(session, now_ns, outcome);
            }
            return true;
        }

        // If 0 controllers were awake at startup, verify periodically whether a controller woke up
        // even if an unexpected SteamVR change prevented metadata deltas from firing.
        if !has_bindings && now_ns >= self.next_controller_poll_ns {
            self.next_controller_poll_ns = now_ns.saturating_add(NO_SLOTS_CONTROLLER_POLL_NS);
            let left_up =
                (session.controller_device_index(Hand::Left) as usize) < K_UN_MAX_TRACKED_DEVICE_COUNT;
            let right_up = (session.controller_device_index(Hand::Right) as usize)
                < K_UN_MAX_TRACKED_DEVICE_COUNT;
            if left_up || right_up {
                let outcome = self
                    .monitor
                    .as_mut()
                    .expect("checked is_some")
                    .rebuild_and_validate(session, now_ns);
                if outcome == RebuildOutcome::Ready {
                    self.record_ready(now_ns);
                } else {
                    self.enter_fallback(session, now_ns, outcome);
                }
                return true;
            }
        }

        // Periodic 60-second idle health check against public GetDigitalActionData().
        if !should_sample && has_bindings && now_ns >= self.next_health_check_ns {
            self.next_health_check_ns = now_ns.saturating_add(HEALTH_CHECK_INTERVAL_NS);
            let check = self
                .monitor
                .as_mut()
                .expect("checked is_some")
                .cross_check_with_openvr(session, now_ns);
            if check == CrossCheckOutcome::Mismatch {
                self.enter_fallback(session, now_ns, RebuildOutcome::LayoutInvalid);
                return true;
            }
        }

        should_sample
    }

    pub fn on_drag_finished(&mut self, session: &OpenVrSession, now_ns: u64) {
        let mut need_rebuild = false;
        if let Some(m) = self.monitor.as_mut() {
            m.extend_grace(now_ns);
            if !m.reseed_current_slots() {
                need_rebuild = true;
            }
        }
        if need_rebuild {
            let outcome = match self.monitor.as_mut() {
                Some(m) => m.rebuild_and_validate(session, now_ns),
                None => RebuildOutcome::LayoutInvalid,
            };
            if outcome == RebuildOutcome::Ready {
                self.record_ready(now_ns);
            } else {
                self.enter_fallback(session, now_ns, outcome);
            }
        }
    }

    #[inline]
    pub fn monitor_mut(&mut self) -> Option<&mut BareMetalInputMonitor> {
        self.monitor.as_mut()
    }

    #[inline]
    pub fn vrserver_wait_handle(&self) -> Option<isize> {
        self.monitor.as_ref().and_then(|m| m.vrserver_wait_handle())
    }

    #[inline]
    pub fn is_shm_active(&self) -> bool {
        self.monitor.as_ref().is_some_and(|m| m.is_shm_valid())
    }

    #[inline]
    pub fn watched_slot_count(&self) -> usize {
        self.monitor.as_ref().map_or(0, |m| m.watched_slot_count())
    }

    #[inline]
    pub fn watched_bound_count(&self) -> usize {
        self.monitor.as_ref().map_or(0, |m| m.watched_bound_count())
    }

    #[inline]
    pub fn has_validated_bindings(&self) -> bool {
        self.monitor
            .as_ref()
            .is_some_and(|m| m.has_validated_bindings())
    }

    #[inline]
    pub fn layout_name(&self) -> &'static str {
        self.last_layout_name
    }

    #[inline]
    pub fn trust(&self) -> LayoutTrust {
        self.last_trust
    }
}

impl Drop for OpenVrSession {
    fn drop(&mut self) {
        unsafe {
            (self.fn_shutdown)();
        }
    }
}

pub fn set_steamvr_autolaunch(enable: bool) -> Result<(), OpenVrError> {
    let mut vrmanifest_path = StackBuf::<520>::new();
    if !resolve_vrmanifest_path(&mut vrmanifest_path) {
        return Err(OpenVrError::ManifestNotFound);
    }

    let lib = crate::platform::load_openvr_library().ok_or(OpenVrError::LibraryNotFound)?;
    let fn_init: FnVrInitInternal = unsafe {
        let ptr = lib
            .get_symbol(b"VR_InitInternal\0")
            .ok_or(OpenVrError::MissingEntryPoint("VR_InitInternal"))?;
        core::mem::transmute(ptr)
    };
    let fn_shutdown: FnVrShutdownInternal = unsafe {
        let ptr = lib
            .get_symbol(b"VR_ShutdownInternal\0")
            .ok_or(OpenVrError::MissingEntryPoint("VR_ShutdownInternal"))?;
        core::mem::transmute(ptr)
    };
    let fn_get_interface: FnVrGetGenericInterface = unsafe {
        let ptr = lib
            .get_symbol(b"VR_GetGenericInterface\0")
            .ok_or(OpenVrError::MissingEntryPoint("VR_GetGenericInterface"))?;
        core::mem::transmute(ptr)
    };

    let mut init_err = VR_INIT_ERROR_NONE;
    let _token = unsafe { fn_init(&mut init_err, EVR_APPLICATION_BACKGROUND) };
    if init_err != VR_INIT_ERROR_NONE {
        return Err(OpenVrError::VrInitFailed(init_err));
    }

    let mut iface_err = VR_INIT_ERROR_NONE;
    let apps_ptr = unsafe { fn_get_interface(b"FnTable:IVRApplications_007\0".as_ptr(), &mut iface_err) };
    if apps_ptr == 0 || iface_err != VR_INIT_ERROR_NONE {
        unsafe { fn_shutdown() };
        return Err(OpenVrError::InterfaceFailed("IVRApplications_007", iface_err));
    }
    let vr_apps = apps_ptr as *const VR_IVRApplications_FnTable;
    let app_key = b"nexashift.spacedrag\0".as_ptr();

    let res = if enable {
        let add_err = unsafe {
            ((*vr_apps).add_application_manifest)(vrmanifest_path.as_bytes().as_ptr(), false)
        };
        if add_err != 0 {
            Err(OpenVrError::AppInstallFailed("AddApplicationManifest", add_err))
        } else {
            let auto_err = unsafe { ((*vr_apps).set_application_auto_launch)(app_key, true) };
            if auto_err != 0 {
                Err(OpenVrError::AppInstallFailed("SetApplicationAutoLaunch", auto_err))
            } else {
                Ok(())
            }
        }
    } else {
        let _ = unsafe { ((*vr_apps).set_application_auto_launch)(app_key, false) };
        let rem_err = unsafe {
            ((*vr_apps).remove_application_manifest)(vrmanifest_path.as_bytes().as_ptr())
        };
        if rem_err != 0 {
            Err(OpenVrError::AppInstallFailed("RemoveApplicationManifest", rem_err))
        } else {
            Ok(())
        }
    };

    unsafe { fn_shutdown() };
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openvr_abi_struct_sizes() {
        assert_eq!(size_of::<HmdMatrix34_t>(), 48);
        assert_eq!(size_of::<TrackedDevicePose_t>(), 80);
        assert_eq!(size_of::<InputDigitalActionData_t>(), 24);
        assert_eq!(size_of::<InputOriginInfo_t>(), 144);
        assert_eq!(size_of::<InputBindingInfo_t>(), 544);
        assert_eq!(size_of::<VRActiveActionSet_t>(), 32);
        #[cfg(target_os = "linux")]
        assert_eq!(size_of::<VREvent_t>(), 60);
        #[cfg(not(target_os = "linux"))]
        assert_eq!(size_of::<VREvent_t>(), 64);
        assert_eq!(
            size_of::<VR_IVROverlay_FnTable>(),
            82 * size_of::<*const c_void>()
        );
        assert_eq!(
            size_of::<VR_IVRSettings_FnTable>(),
            11 * size_of::<*const c_void>()
        );
        assert_eq!(
            core::mem::offset_of!(VR_IVROverlay_FnTable, wait_frame_sync),
            47 * size_of::<*const c_void>()
        );
        assert_eq!(
            core::mem::offset_of!(VR_IVROverlay_FnTable, is_dashboard_visible),
            68 * size_of::<*const c_void>()
        );
        assert_eq!(
            core::mem::offset_of!(VR_IVRApplications_FnTable, set_application_auto_launch),
            17 * size_of::<*const c_void>()
        );
        assert_eq!(
            core::mem::offset_of!(VR_IVRSystem_FnTable, get_runtime_version),
            45 * size_of::<*const c_void>()
        );
    }

    #[test]
    fn double_click_detector_latches_within_window_and_rejects_single_or_slow_clicks() {
        let mut d = DoubleClickDetector::default();
        // Single press + release should not activate.
        assert!(!d.step(true, 100_000_000));
        assert!(!d.step(false, 150_000_000));
        // Second press within 500 ms of first press latches until release.
        assert!(d.step(true, 300_000_000));
        assert!(d.step(true, 320_000_000));
        assert!(!d.step(false, 900_000_000));

        // Slow second press after window expires should NOT latch.
        assert!(!d.step(true, 2_000_000_000));
        assert!(!d.step(false, 2_100_000_000));
        assert!(!d.step(true, 2_700_000_000));
        assert!(!d.step(false, 2_800_000_000));
    }

    #[test]
    fn resolve_binding_component_path_maps_hand_mode_slot_and_avoids_duplicate_suffix() {
        let mut p = StackBuf::<256>::new();
        let mut fb = StackBuf::<256>::new();
        let res = resolve_binding_component_path(
            b"/user/hand/left",
            b"/input/joystick",
            b"click",
            &mut p,
            &mut fb,
        )
        .unwrap();
        assert_eq!(res, (Hand::Left, TriggerKind::Direct, true));
        assert_eq!(p.as_bytes(), b"/user/hand/left/input/joystick/click\0");
        assert_eq!(fb.as_bytes(), b"/user/hand/left/input/joystick/value\0");

        let mut p2 = StackBuf::<256>::new();
        let mut fb2 = StackBuf::<256>::new();
        let res2 = resolve_binding_component_path(
            b"/user/hand/right",
            b"/input/b/click",
            b"double",
            &mut p2,
            &mut fb2,
        )
        .unwrap();
        assert_eq!(res2, (Hand::Right, TriggerKind::Double, false));
        assert_eq!(p2.as_bytes(), b"/user/hand/right/input/b/click\0");
    }

    struct SliceShmView<'a>(&'a [u8]);

    impl ShmReadView for SliceShmView<'_> {
        #[inline]
        fn len(&self) -> usize {
            self.0.len()
        }

        #[inline]
        fn read_u8(&self, byte_offset: usize) -> Option<u8> {
            self.0.get(byte_offset).copied()
        }

        #[inline]
        fn read_u32(&self, byte_offset: usize) -> Option<u32> {
            if (byte_offset & 3) != 0 || byte_offset + 4 > self.0.len() {
                return None;
            }
            let bytes = self.0[byte_offset..byte_offset + 4].try_into().ok()?;
            Some(u32::from_le_bytes(bytes))
        }

        #[inline]
        fn read_u64(&self, byte_offset: usize) -> Option<u64> {
            if (byte_offset & 7) != 0 || byte_offset + 8 > self.0.len() {
                return None;
            }
            let bytes = self.0[byte_offset..byte_offset + 8].try_into().ok()?;
            Some(u64::from_le_bytes(bytes))
        }
    }

    struct FakeContainer<'a> {
        dev_idx: u32,
        hdr: &'a [u8],
        tbl: &'a [u8],
        sto: &'a [u8],
    }

    struct SliceContainerOpener<'a> {
        containers: &'a [FakeContainer<'a>],
    }

    impl<'a> ContainerOpener for SliceContainerOpener<'a> {
        type View = SliceShmView<'a>;

        fn open(&mut self, dev_idx: u32) -> Option<ContainerViews<Self::View>> {
            for c in self.containers {
                if c.dev_idx == dev_idx {
                    return Some(ContainerViews {
                        hdr: SliceShmView(c.hdr),
                        tbl: SliceShmView(c.tbl),
                        sto: SliceShmView(c.sto),
                    });
                }
            }
            None
        }
    }

    #[inline]
    fn write_u32_le(buf: &mut [u8], off: usize, v: u32) {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn write_u64_le(buf: &mut [u8], off: usize, v: u64) {
        buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }

    const TEST_HANDLE_L: u64 = 0x0001_2345_6789_ABC1;
    const TEST_HANDLE_R: u64 = 0x0001_2345_6789_ABC2;

    fn populate_synthetic_shm(
        layout: &SteamVrLayout,
        conn: &mut [u8],
        input: &mut [u8],
        hdr1: &mut [u8],
        tbl1: &mut [u8],
        sto1: &mut [u8],
        hdr2: &mut [u8],
        tbl2: &mut [u8],
        sto2: &mut [u8],
    ) {
        write_u64_le(conn, layout.conn_seq_off, 42);
        write_u32_le(input, layout.input_count_off, 2);
        write_u32_le(input, layout.active_set_count_off, 1);

        let entries = [
            (0usize, TEST_HANDLE_L, 1u32, 1u32, 5u32, 0usize, hdr1, tbl1, sto1),
            (1usize, TEST_HANDLE_R, 1u32, 2u32, 7u32, 64usize, hdr2, tbl2, sto2),
        ];
        for (idx, handle, ctype, dev, prop_lo, sto_off, hdr, tbl, sto) in entries {
            let e_off = layout.input_array_off + idx * layout.input_entry_bytes;
            write_u32_le(input, e_off + layout.entry_ctype_off, ctype);
            write_u64_le(input, e_off + layout.entry_handle_off, handle);
            write_u32_le(input, e_off + layout.entry_dev_idx_off, dev);
            write_u32_le(input, e_off + layout.entry_prop_id_off, prop_lo);

            let prop_count = 3u32;
            let storage_sz = (sto_off + layout.storage_component_bytes) as u32;
            write_u32_le(hdr, layout.hdr_prop_count_off, prop_count);
            write_u32_le(hdr, layout.hdr_storage_sz_off, storage_sz);

            // Entry 0: unrelated property so stride errors at index >= 1 are caught immediately.
            write_u32_le(tbl, layout.prop_entry_id_off, 0x2000_0001);
            write_u32_le(tbl, layout.prop_entry_type_off, 1);
            write_u32_le(tbl, layout.prop_entry_offset_off, 0);
            write_u32_le(tbl, layout.prop_entry_len_off, 4);

            // Entry 1: target input component property.
            let t1 = layout.prop_table_entry_bytes;
            write_u32_le(tbl, t1 + layout.prop_entry_id_off, 0x1000_0000 | prop_lo);
            write_u32_le(
                tbl,
                t1 + layout.prop_entry_type_off,
                layout.prop_type_input_component,
            );
            write_u32_le(tbl, t1 + layout.prop_entry_offset_off, sto_off as u32);
            write_u32_le(
                tbl,
                t1 + layout.prop_entry_len_off,
                layout.storage_component_bytes as u32,
            );

            // Entry 2: second unrelated property.
            let t2 = 2 * layout.prop_table_entry_bytes;
            write_u32_le(tbl, t2 + layout.prop_entry_id_off, 0x2000_0002);
            write_u32_le(tbl, t2 + layout.prop_entry_type_off, 1);
            write_u32_le(tbl, t2 + layout.prop_entry_offset_off, 4);
            write_u32_le(tbl, t2 + layout.prop_entry_len_off, 4);

            write_u32_le(sto, sto_off + layout.storage_ctype_off, ctype);
            sto[sto_off + layout.storage_value_off] = 0;
        }
    }

    #[test]
    fn validate_layout_accepts_steamvr_v2_synthetic_buffers() {
        let mut conn = [0u8; 0x500];
        let mut input = [0u8; 0xA000];
        let mut hdr1 = [0u8; 64];
        let mut tbl1 = [0u8; 256];
        let mut sto1 = [0u8; 512];
        let mut hdr2 = [0u8; 64];
        let mut tbl2 = [0u8; 256];
        let mut sto2 = [0u8; 512];

        populate_synthetic_shm(
            &STEAMVR_LAYOUT_V2,
            &mut conn,
            &mut input,
            &mut hdr1,
            &mut tbl1,
            &mut sto1,
            &mut hdr2,
            &mut tbl2,
            &mut sto2,
        );

        let containers = [
            FakeContainer {
                dev_idx: 1,
                hdr: &hdr1,
                tbl: &tbl1,
                sto: &sto1,
            },
            FakeContainer {
                dev_idx: 2,
                hdr: &hdr2,
                tbl: &tbl2,
                sto: &sto2,
            },
        ];
        let mut opener = SliceContainerOpener {
            containers: &containers,
        };
        let mut targets = [None; MAX_BOUND_SLOTS];
        targets[0] = Some(BindingTarget {
            hand: Hand::Left,
            kind: TriggerKind::Direct,
            primary_handle: TEST_HANDLE_L,
            fallback_handle: 0,
            expected_dev_idx: Some(1),
            ctype_hint: Some(1),
        });
        targets[1] = Some(BindingTarget {
            hand: Hand::Right,
            kind: TriggerKind::Direct,
            primary_handle: TEST_HANDLE_R,
            fallback_handle: 0,
            expected_dev_idx: Some(2),
            ctype_hint: Some(1),
        });

        let outcome = validate_layout_on_views(
            &STEAMVR_LAYOUT_V2,
            &SliceShmView(&conn),
            &SliceShmView(&input),
            &mut opener,
            &targets,
            2,
            (1, 2),
        );
        assert_eq!(outcome, RebuildOutcome::Ready);
    }

    #[test]
    fn discover_layout_finds_shifted_80byte_entry_and_32byte_table_strides() {
        let shifted = SteamVrLayout {
            name: "shifted_test",
            conn_seq_off: 0x408,
            input_array_off: 0,
            input_entry_bytes: 80,
            max_components: 512,
            input_count_off: 512 * 80,
            active_set_count_off: 512 * 80 + 0x808,
            entry_ctype_off: 0x00,
            entry_handle_off: 0x10,
            entry_dev_idx_off: 0x20,
            entry_prop_id_off: 0x30,
            hdr_prop_count_off: 4,
            hdr_storage_sz_off: 8,
            prop_table_entry_bytes: 32,
            prop_entry_id_off: 0,
            prop_entry_type_off: 8,
            prop_entry_offset_off: 16,
            prop_entry_len_off: 20,
            prop_type_input_component: PROP_TYPE_INPUT_COMPONENT,
            storage_component_bytes: 80,
            storage_ctype_off: 0x10,
            storage_value_off: 0x18,
        };

        let mut conn = [0u8; 0x500];
        let mut input = [0u8; 0xB000];
        let mut hdr1 = [0u8; 64];
        let mut tbl1 = [0u8; 256];
        let mut sto1 = [0u8; 512];
        let mut hdr2 = [0u8; 64];
        let mut tbl2 = [0u8; 256];
        let mut sto2 = [0u8; 512];

        populate_synthetic_shm(
            &shifted,
            &mut conn,
            &mut input,
            &mut hdr1,
            &mut tbl1,
            &mut sto1,
            &mut hdr2,
            &mut tbl2,
            &mut sto2,
        );

        let containers = [
            FakeContainer {
                dev_idx: 1,
                hdr: &hdr1,
                tbl: &tbl1,
                sto: &sto1,
            },
            FakeContainer {
                dev_idx: 2,
                hdr: &hdr2,
                tbl: &tbl2,
                sto: &sto2,
            },
        ];
        let mut opener = SliceContainerOpener {
            containers: &containers,
        };
        let mut targets = [None; MAX_BOUND_SLOTS];
        targets[0] = Some(BindingTarget {
            hand: Hand::Left,
            kind: TriggerKind::Direct,
            primary_handle: TEST_HANDLE_L,
            fallback_handle: 0,
            expected_dev_idx: Some(1),
            ctype_hint: Some(1),
        });
        targets[1] = Some(BindingTarget {
            hand: Hand::Right,
            kind: TriggerKind::Direct,
            primary_handle: TEST_HANDLE_R,
            fallback_handle: 0,
            expected_dev_idx: Some(2),
            ctype_hint: Some(1),
        });

        assert_eq!(
            validate_layout_on_views(
                &STEAMVR_LAYOUT_V2,
                &SliceShmView(&conn),
                &SliceShmView(&input),
                &mut opener,
                &targets,
                2,
                (1, 2),
            ),
            RebuildOutcome::LayoutInvalid
        );

        let mut anchors = [DiscoveryAnchor {
            handle: 0,
            expected_dev_idx: None,
            ctype_hint: None,
        }; 32];
        let n_anchors = build_discovery_anchors(&targets[..2], &mut anchors);
        let discovered = discover_layout_with_views(
            &SliceShmView(&conn),
            &SliceShmView(&input),
            &mut opener,
            &anchors[..n_anchors],
        )
        .expect("discovery should find 80-byte entry and 32-byte table layout");

        assert_eq!(discovered.input_entry_bytes, 80);
        assert_eq!(discovered.input_count_off, 512 * 80);
        assert_eq!(discovered.active_set_count_off, 512 * 80 + 0x808);
        assert_eq!(discovered.prop_table_entry_bytes, 32);
        assert_eq!(discovered.storage_component_bytes, 80);
        assert_eq!(
            validate_layout_on_views(
                &discovered,
                &SliceShmView(&conn),
                &SliceShmView(&input),
                &mut opener,
                &targets,
                2,
                (1, 2),
            ),
            RebuildOutcome::Ready
        );
    }

    #[test]
    fn validate_layout_rejects_undersized_or_corrupted_buffers() {
        let mut conn = [0u8; 0x500];
        let mut input = [0u8; 0xA000];
        let mut hdr1 = [0u8; 64];
        let mut tbl1 = [0u8; 256];
        let mut sto1 = [0u8; 512];
        let mut hdr2 = [0u8; 64];
        let mut tbl2 = [0u8; 256];
        let mut sto2 = [0u8; 512];

        populate_synthetic_shm(
            &STEAMVR_LAYOUT_V2,
            &mut conn,
            &mut input,
            &mut hdr1,
            &mut tbl1,
            &mut sto1,
            &mut hdr2,
            &mut tbl2,
            &mut sto2,
        );

        let containers = [
            FakeContainer {
                dev_idx: 1,
                hdr: &hdr1,
                tbl: &tbl1,
                sto: &sto1,
            },
            FakeContainer {
                dev_idx: 2,
                hdr: &hdr2,
                tbl: &tbl2,
                sto: &sto2,
            },
        ];
        let mut opener = SliceContainerOpener {
            containers: &containers,
        };
        let mut targets = [None; MAX_BOUND_SLOTS];
        targets[0] = Some(BindingTarget {
            hand: Hand::Left,
            kind: TriggerKind::Direct,
            primary_handle: TEST_HANDLE_L,
            fallback_handle: 0,
            expected_dev_idx: Some(1),
            ctype_hint: Some(1),
        });

        // 1. Undersized input_state region.
        assert_eq!(
            validate_layout_on_views(
                &STEAMVR_LAYOUT_V2,
                &SliceShmView(&conn),
                &SliceShmView(&input[..0x9000]),
                &mut opener,
                &targets,
                1,
                (1, 2),
            ),
            RebuildOutcome::LayoutInvalid
        );

        // 2. Unaligned offset inside layout descriptor.
        let unaligned_layout = SteamVrLayout {
            input_count_off: 0x9002,
            ..STEAMVR_LAYOUT_V2
        };
        assert!(!unaligned_layout.is_structurally_sane());
        assert_eq!(
            validate_layout_on_views(
                &unaligned_layout,
                &SliceShmView(&conn),
                &SliceShmView(&input),
                &mut opener,
                &targets,
                1,
                (1, 2),
            ),
            RebuildOutcome::LayoutInvalid
        );

        // 3. Mismatched expected controller role device index.
        let mut wrong_dev_targets = [None; MAX_BOUND_SLOTS];
        wrong_dev_targets[0] = Some(BindingTarget {
            hand: Hand::Left,
            kind: TriggerKind::Direct,
            primary_handle: TEST_HANDLE_L,
            fallback_handle: 0,
            expected_dev_idx: Some(9),
            ctype_hint: Some(1),
        });
        assert_eq!(
            validate_layout_on_views(
                &STEAMVR_LAYOUT_V2,
                &SliceShmView(&conn),
                &SliceShmView(&input),
                &mut opener,
                &wrong_dev_targets,
                1,
                (9, 2),
            ),
            RebuildOutcome::LayoutInvalid
        );

        // 4. Connected controllers with zero resolved bindings returns BindingsNotReady (never Ready).
        assert_eq!(
            validate_layout_on_views(
                &STEAMVR_LAYOUT_V2,
                &SliceShmView(&conn),
                &SliceShmView(&input),
                &mut opener,
                &targets,
                0,
                (1, 2),
            ),
            RebuildOutcome::BindingsNotReady
        );

        // 5. Out-of-bounds component count c1.
        write_u32_le(&mut input, STEAMVR_LAYOUT_V2.input_count_off, 9999);
        assert_eq!(
            validate_layout_on_views(
                &STEAMVR_LAYOUT_V2,
                &SliceShmView(&conn),
                &SliceShmView(&input),
                &mut opener,
                &targets,
                1,
                (1, 2),
            ),
            RebuildOutcome::LayoutInvalid
        );
    }

    #[test]
    fn reprobe_backoff_schedule_is_5s_15s_60s() {
        assert_eq!(reprobe_backoff_ns(0), 5_000_000_000);
        assert_eq!(reprobe_backoff_ns(1), 15_000_000_000);
        assert_eq!(reprobe_backoff_ns(2), 60_000_000_000);
        assert_eq!(reprobe_backoff_ns(3), 60_000_000_000);
        assert_eq!(reprobe_backoff_ns(255), 60_000_000_000);
    }
}
