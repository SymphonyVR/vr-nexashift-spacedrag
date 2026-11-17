use core::cell::Cell;
use core::ffi::c_void;
use core::mem::{MaybeUninit, size_of};

use crate::math::{HmdMatrix34_t, SimdMath, Vec3};
use crate::platform::{
    DynamicLibrary, MappedReadOnlyShm, ProcessWaitHandle, StackBuf, find_connected_vrserver_shm,
    open_property_container_shm, resolve_action_manifest_path,
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
    pub get_runtime_version: *const c_void,
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
    pub _after_wait_frame_sync: [*const c_void; 34],
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

        let mut err = VR_INIT_ERROR_NONE;
        let settings_ptr =
            unsafe { fn_get_interface(b"FnTable:IVRSettings_003\0".as_ptr(), &mut err) };
        if settings_ptr != 0 && err == VR_INIT_ERROR_NONE {
            let vr_settings = settings_ptr as *const VR_IVRSettings_FnTable;
            let mut s_err = 0i32;
            let enabled = unsafe {
                ((*vr_settings).get_bool)(
                    b"steamvr\0".as_ptr(),
                    b"globalActionSetPriority\0".as_ptr(),
                    &mut s_err,
                )
            };
            if s_err != 0 || !enabled {
                s_err = 0;
                unsafe {
                    ((*vr_settings).set_bool)(
                        b"steamvr\0".as_ptr(),
                        b"globalActionSetPriority\0".as_ptr(),
                        true,
                        &mut s_err,
                    );
                }
            }
        }

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
        })
    }

    #[inline]
    pub fn wait_frame_sync(&self, timeout_ms: u32) -> bool {
        unsafe { ((*self.vr_overlay).wait_frame_sync)(timeout_ms) == 0 }
    }

    #[inline]
    pub fn hmd_refresh_hz(&self) -> f32 {
        self.hmd_refresh_hz
    }

    #[inline]
    pub fn drag_interval_ns(&self) -> u64 {
        (1_000_000_000.0f32 / self.hmd_refresh_hz) as u64
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
                        ((*self.vr_system).acknowledge_quit_exiting)();
                        should_quit = true;
                    }
                    VR_EVENT_CHAPERONE_DATA_HAS_CHANGED
                    | VR_EVENT_CHAPERONE_UNIVERSE_HAS_CHANGED
                    | VR_EVENT_CHAPERONE_TEMP_DATA_HAS_CHANGED
                    | VR_EVENT_CHAPERONE_ROOM_SETUP_COMMITTED
                    | VR_EVENT_STANDING_ZERO_POSE_RESET => {
                        if !self.dirty_uncommitted.get() {
                            self.working_set_synced.set(false);
                        }
                    }
                    _ => {}
                }
            }
        }
        should_quit
    }

    pub fn sample_inputs(&self, active_device_hint: Option<(Hand, u32)>) -> InputFrame {
        let mut active_set = VRActiveActionSet_t {
            ul_action_set: self.action_set_main,
            ul_restricted_to_device: 0,
            ul_secondary_action_set: 0,
            un_padding: 0,
            n_priority: K_N_ACTION_SET_OVERLAY_GLOBAL_PRIORITY_MAX,
        };
        let err = unsafe {
            ((*self.vr_input).update_action_state)(
                &mut active_set,
                size_of::<VRActiveActionSet_t>() as u32,
                1,
            )
        };
        if err != VR_INPUT_ERROR_NONE {
            return InputFrame::IDLE;
        }

        let mut left_data = InputDigitalActionData_t::default();
        let mut right_data = InputDigitalActionData_t::default();
        let data_size = size_of::<InputDigitalActionData_t>() as u32;

        let left_err = unsafe {
            ((*self.vr_input).get_digital_action_data)(
                self.action_space_drag,
                &mut left_data,
                data_size,
                self.source_left,
            )
        };
        let right_err = unsafe {
            ((*self.vr_input).get_digital_action_data)(
                self.action_space_drag,
                &mut right_data,
                data_size,
                self.source_right,
            )
        };

        let left_pressed = left_err == VR_INPUT_ERROR_NONE && left_data.b_active && left_data.b_state;
        let right_pressed =
            right_err == VR_INPUT_ERROR_NONE && right_data.b_active && right_data.b_state;

        // Idle fast path: skip all device pose queries when neither trigger is held or changed.
        if !left_pressed
            && !right_pressed
            && !left_data.b_changed
            && !right_data.b_changed
            && active_device_hint.is_none()
        {
            return InputFrame::IDLE;
        }

        let mut poses = MaybeUninit::<[TrackedDevicePose_t; K_UN_MAX_TRACKED_DEVICE_COUNT]>::uninit();
        unsafe {
            ((*self.vr_system).get_device_to_absolute_tracking_pose)(
                E_TRACKING_UNIVERSE_RAW_AND_UNCALIBRATED,
                0.0,
                poses.as_mut_ptr().cast::<TrackedDevicePose_t>(),
                K_UN_MAX_TRACKED_DEVICE_COUNT as u32,
            );
        }
        let poses = unsafe { poses.assume_init() };

        let left = self.build_hand_input(
            Hand::Left,
            left_pressed,
            left_data.active_origin,
            active_device_hint,
            &poses,
        );
        let right = self.build_hand_input(
            Hand::Right,
            right_pressed,
            right_data.active_origin,
            active_device_hint,
            &poses,
        );

        InputFrame { left, right }
    }

    fn build_hand_input(
        &self,
        hand: Hand,
        pressed: bool,
        active_origin: u64,
        active_device_hint: Option<(Hand, u32)>,
        poses: &[TrackedDevicePose_t; K_UN_MAX_TRACKED_DEVICE_COUNT],
    ) -> HandInput {
        if !pressed {
            return HandInput::UNPRESSED;
        }

        let device_index = match active_device_hint {
            Some((active_hand, idx)) if active_hand == hand => idx,
            _ => self.resolve_device_index(hand, active_origin),
        };

        if (device_index as usize) >= K_UN_MAX_TRACKED_DEVICE_COUNT {
            return HandInput {
                pressed: true,
                device_index: INVALID_DEVICE_INDEX,
                pose_valid: false,
                position_raw: Vec3::ZERO,
            };
        }

        let pose = &poses[device_index as usize];
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

    #[inline]
    pub fn commit_working_standing_pose(&self, mat: &HmdMatrix34_t) -> bool {
        let ok = unsafe {
            ((*self.chaperone_setup).set_working_standing_zero_pose_to_raw_tracking_pose)(mat);
            ((*self.chaperone_setup).commit_working_copy)(E_CHAPERONE_CONFIG_FILE_LIVE)
        };
        self.working_set_synced.set(true);
        self.dirty_uncommitted.set(false);
        ok
    }

    pub fn flush_pending_commit(&self) {
        if self.dirty_uncommitted.get() {
            unsafe {
                let _ = ((*self.chaperone_setup).commit_working_copy)(E_CHAPERONE_CONFIG_FILE_LIVE);
            }
            self.dirty_uncommitted.set(false);
        }
    }

    pub fn space_drag_max_component_type(&self) -> u32 {
        if self.action_space_drag == 0 {
            return 3;
        }
        let mut infos = [InputBindingInfo_t::default(); 16];
        let mut count = 0u32;
        let err = unsafe {
            ((*self.vr_input).get_action_binding_info)(
                self.action_space_drag,
                infos.as_mut_ptr(),
                size_of::<InputBindingInfo_t>() as u32,
                infos.len() as u32,
                &mut count,
            )
        };
        if err != VR_INPUT_ERROR_NONE || count == 0 {
            // Safe conservative watch (digital + scalar + vec2) if binding info isn't available yet.
            return 3;
        }
        let n = (count as usize).min(infos.len());
        let mut max_type = 1u32;
        let mut i = 0;
        while i < n {
            let src_type = cstr_slice(&infos[i].rch_input_source_type);
            let mode = cstr_slice(&infos[i].rch_mode_name);
            let slot = cstr_slice(&infos[i].rch_slot_name);
            let digital_slot = matches!(
                slot,
                b"click" | b"double" | b"touch" | b"held" | b"long" | b"force"
            );
            let button_mode = mode == b"button" || mode == b"toggle_button";
            if !digital_slot && !button_mode {
                return 3;
            }
            if src_type == b"trigger" || src_type == b"grip" || slot == b"pull" || slot == b"force"
            {
                max_type = max_type.max(2);
            }
            i += 1;
        }
        max_type
    }

    pub fn open_bare_metal_monitor(&self) -> Option<BareMetalInputMonitor> {
        let (server_pid, server_wait_handle, input_state_shm, conn_state_shm) =
            find_connected_vrserver_shm()?;
        let max_watched_component_type = self.space_drag_max_component_type();
        let mut monitor = BareMetalInputMonitor {
            server_pid,
            server_wait_handle,
            input_state_shm,
            conn_state_shm,
            shm_valid: true,
            last_conn_seq: 0,
            last_c1: 0,
            last_c2: 0,
            max_watched_component_type,
            grace_until_ns: 0,
            containers: [const { None }; MAX_WATCHED_CONTAINERS],
            total_slots: 0,
            prev_vals: [0; MAX_TOTAL_SLOTS],
            curr_vals: [0; MAX_TOTAL_SLOTS],
        };
        if !monitor.rebuild_containers_and_seed() {
            return None;
        }
        Some(monitor)
    }
}

const SHM_MAX_COMPONENTS: usize = 512;
const SHM_COMPONENT_BYTES: usize = 72;
const SHM_COUNT_OFFSET: usize = 0x9000;
const SHM_ACTIVE_SET_COUNT_OFFSET: usize = 0x9808;
const SHM_CONN_SEQ_OFFSET: usize = 0x408;
const MAX_WATCHED_CONTAINERS: usize = 8;
const MAX_SLOTS_PER_CONTAINER: usize = 32;
const MAX_TOTAL_SLOTS: usize = 128;
const PROP_TYPE_INPUT_COMPONENT: u32 = 0x21;
const PROP_TABLE_ENTRY_BYTES: usize = 28;
const POST_EDGE_GRACE_NS: u64 = 750_000_000;

#[inline]
fn cstr_slice(buf: &[u8]) -> &[u8] {
    let mut len = 0usize;
    while len < buf.len() && buf[len] != 0 {
        len += 1;
    }
    &buf[..len]
}

struct WatchedContainer {
    hdr_shm: MappedReadOnlyShm,
    storage_shm: MappedReadOnlyShm,
    prop_count: u32,
    storage_sz: u32,
    slot_count: usize,
    slots: [(usize, u32); MAX_SLOTS_PER_CONTAINER],
}

/// Bare-metal shared-memory input watcher for SteamVR controller property containers.
///
/// Maps each connected controller's `PC_<container>_storage0` read-only and samples only the live
/// input component value fields (`+0x18`) with `fearless_simd`, avoiding `VR_InputMutex` syscalls
/// and `vrclient_x64.dll` per-frame heap allocations while idle.
pub struct BareMetalInputMonitor {
    server_pid: u32,
    server_wait_handle: Option<ProcessWaitHandle>,
    input_state_shm: MappedReadOnlyShm,
    conn_state_shm: MappedReadOnlyShm,
    shm_valid: bool,
    last_conn_seq: u64,
    last_c1: u32,
    last_c2: u32,
    max_watched_component_type: u32,
    grace_until_ns: u64,
    containers: [Option<WatchedContainer>; MAX_WATCHED_CONTAINERS],
    total_slots: usize,
    prev_vals: [u32; MAX_TOTAL_SLOTS],
    curr_vals: [u32; MAX_TOTAL_SLOTS],
}

impl BareMetalInputMonitor {
    #[inline]
    pub fn vrserver_wait_handle(&self) -> Option<isize> {
        self.server_wait_handle.as_ref().map(|h| h.raw())
    }

    #[inline]
    pub fn watched_slot_count(&self) -> usize {
        self.total_slots
    }

    #[inline]
    pub fn extend_grace(&mut self, now_ns: u64) {
        self.grace_until_ns = now_ns.saturating_add(POST_EDGE_GRACE_NS);
    }

    pub fn rebuild_containers_and_seed(&mut self) -> bool {
        let Some(conn_seq) = self.conn_state_shm.read_volatile_u64(SHM_CONN_SEQ_OFFSET) else {
            self.shm_valid = false;
            return false;
        };
        let Some(c1) = self.input_state_shm.read_volatile_u32(SHM_COUNT_OFFSET) else {
            self.shm_valid = false;
            return false;
        };
        let Some(c2) = self
            .input_state_shm
            .read_volatile_u32(SHM_ACTIVE_SET_COUNT_OFFSET)
        else {
            self.shm_valid = false;
            return false;
        };
        if (c1 as usize) > SHM_MAX_COMPONENTS || c2 > 128 {
            self.shm_valid = false;
            return false;
        }

        let mut devs = [u32::MAX; MAX_WATCHED_CONTAINERS];
        let mut dev_count = 0usize;
        let mut i = 0usize;
        while i < (c1 as usize) {
            let entry_off = i * SHM_COMPONENT_BYTES;
            let Some(comp_type) = self.input_state_shm.read_volatile_u32(entry_off) else {
                self.shm_valid = false;
                return false;
            };
            let Some(dev_idx) = self.input_state_shm.read_volatile_u32(entry_off + 0x20) else {
                self.shm_valid = false;
                return false;
            };
            if comp_type <= 2 && (1..64).contains(&dev_idx) {
                let mut seen = false;
                let mut k = 0;
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

        for slot in &mut self.containers {
            *slot = None;
        }
        self.total_slots = 0;

        let mut container_idx = 0usize;
        let mut d = 0usize;
        while d < dev_count {
            let cid = 0x1_0000_0000u64 + (devs[d] as u64);
            let Some((hdr_shm, tbl_shm, sto_shm)) =
                open_property_container_shm(self.server_pid, cid)
            else {
                // Controller device listed in VR_InputState without readable PC SHM: fall back safely.
                self.shm_valid = false;
                return false;
            };
            let Some(prop_count) = hdr_shm.read_volatile_u32(4) else {
                self.shm_valid = false;
                return false;
            };
            let Some(storage_sz) = hdr_shm.read_volatile_u32(8) else {
                self.shm_valid = false;
                return false;
            };

            let mut container = WatchedContainer {
                hdr_shm,
                storage_shm: sto_shm,
                prop_count,
                storage_sz,
                slot_count: 0,
                slots: [(0, 0); MAX_SLOTS_PER_CONTAINER],
            };

            let mut p = 0usize;
            while p < (prop_count as usize) {
                let toff = p * PROP_TABLE_ENTRY_BYTES;
                let Some(ptype) = tbl_shm.read_volatile_u32(toff + 8) else {
                    break;
                };
                let Some(soff) = tbl_shm.read_volatile_u32(toff + 16) else {
                    break;
                };
                let Some(slen) = tbl_shm.read_volatile_u32(toff + 20) else {
                    break;
                };
                if ptype == PROP_TYPE_INPUT_COMPONENT
                    && (slen as usize) == SHM_COMPONENT_BYTES
                    && (soff as usize) + SHM_COMPONENT_BYTES <= (storage_sz as usize)
                {
                    let Some(ctype) = container.storage_shm.read_volatile_u32(soff as usize + 0x10)
                    else {
                        break;
                    };
                    if ctype >= 1 && ctype <= self.max_watched_component_type.min(2) {
                        if container.slot_count < MAX_SLOTS_PER_CONTAINER
                            && self.total_slots < MAX_TOTAL_SLOTS
                        {
                            container.slots[container.slot_count] = (soff as usize, ctype);
                            container.slot_count += 1;
                            self.total_slots += 1;
                        }
                    }
                }
                p += 1;
            }

            if container.slot_count == 0 {
                self.shm_valid = false;
                return false;
            }
            self.containers[container_idx] = Some(container);
            container_idx += 1;
            d += 1;
        }

        self.last_conn_seq = conn_seq;
        self.last_c1 = c1;
        self.last_c2 = c2;
        if !self.read_all_slots_into_prev() {
            self.shm_valid = false;
            return false;
        }
        self.shm_valid = true;
        true
    }

    fn read_slot_value(sto: &MappedReadOnlyShm, off: usize, ctype: u32) -> Option<u32> {
        if ctype == 1 {
            let b = sto.read_volatile_u8(off + 0x18)?;
            Some(if b != 0 { 1 } else { 0 })
        } else {
            let raw = sto.read_volatile_u32(off + 0x18)?;
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
                    let Some(val) = Self::read_slot_value(&c.storage_shm, off, ctype) else {
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

    /// Checks SteamVR live controller property containers via SIMD.
    /// Returns `(should_sample_openvr, bindings_or_devices_changed)`.
    pub fn poll_should_sample(&mut self, simd: SimdMath, now_ns: u64) -> (bool, bool) {
        if !self.shm_valid {
            return (true, false);
        }

        let Some(conn_seq) = self.conn_state_shm.read_volatile_u64(SHM_CONN_SEQ_OFFSET) else {
            self.shm_valid = false;
            return (true, false);
        };
        let Some(c1) = self.input_state_shm.read_volatile_u32(SHM_COUNT_OFFSET) else {
            self.shm_valid = false;
            return (true, false);
        };
        let Some(c2) = self
            .input_state_shm
            .read_volatile_u32(SHM_ACTIVE_SET_COUNT_OFFSET)
        else {
            self.shm_valid = false;
            return (true, false);
        };
        if (c1 as usize) > SHM_MAX_COMPONENTS || c2 > 128 {
            self.shm_valid = false;
            return (true, false);
        }

        let conn_changed = conn_seq != self.last_conn_seq;
        let mut topology_changed = conn_changed || c1 != self.last_c1 || c2 != self.last_c2;
        if !topology_changed {
            let mut cidx = 0usize;
            while cidx < self.containers.len() {
                if let Some(ref c) = self.containers[cidx] {
                    let Some(cur_count) = c.hdr_shm.read_volatile_u32(4) else {
                        self.shm_valid = false;
                        return (true, false);
                    };
                    let Some(cur_sz) = c.hdr_shm.read_volatile_u32(8) else {
                        self.shm_valid = false;
                        return (true, false);
                    };
                    if cur_count != c.prop_count || cur_sz != c.storage_sz {
                        topology_changed = true;
                        break;
                    }
                }
                cidx += 1;
            }
        }

        if topology_changed {
            self.grace_until_ns = now_ns.saturating_add(POST_EDGE_GRACE_NS);
            return (true, true);
        }

        if !self.read_all_slots_into(false) {
            self.shm_valid = false;
            return (true, false);
        }

        let n = self.total_slots;
        if simd.slices_equal_u32(&self.curr_vals[..n], &self.prev_vals[..n]) {
            return (now_ns < self.grace_until_ns, false);
        }

        self.prev_vals[..n].copy_from_slice(&self.curr_vals[..n]);
        self.grace_until_ns = now_ns.saturating_add(POST_EDGE_GRACE_NS);
        (true, false)
    }

    #[inline]
    pub fn set_max_watched_component_type(&mut self, max_type: u32) {
        self.max_watched_component_type = max_type.clamp(1, 2);
    }
}

impl Drop for OpenVrSession {
    fn drop(&mut self) {
        self.flush_pending_commit();
        unsafe {
            (self.fn_shutdown)();
        }
    }
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
    }
}
