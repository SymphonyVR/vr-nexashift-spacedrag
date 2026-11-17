use core::ffi::c_void;
use core::mem::{MaybeUninit, size_of};

use crate::math::{HmdMatrix34_t, Vec3};
use crate::platform::{DynamicLibrary, StackBuf, resolve_action_manifest_path};
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
    pub commit_working_copy: *const c_void,
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
    pub get_action_binding_info: *const c_void,
    pub show_action_origins: *const c_void,
    pub show_bindings_for_action_set: *const c_void,
    pub get_component_state_for_binding: *const c_void,
    pub is_using_legacy_input: *const c_void,
    pub open_binding_ui: *const c_void,
    pub get_binding_variant: *const c_void,
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
    action_set_main: u64,
    action_space_drag: u64,
    source_left: u64,
    source_right: u64,
    hmd_refresh_hz: f32,
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

        // Synchronize the working set once on startup without touching disk.
        unsafe {
            ((*chaperone_setup).revert_working_copy)();
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
            action_set_main,
            action_space_drag,
            source_left,
            source_right,
            hmd_refresh_hz,
        })
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
                if ev.event_type == VR_EVENT_QUIT {
                    ((*self.vr_system).acknowledge_quit_exiting)();
                    should_quit = true;
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
            n_priority: 0,
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

        // Idle fast path: skip all device pose queries when neither trigger is held.
        if !left_pressed && !right_pressed && active_device_hint.is_none() {
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
        let ok = unsafe {
            ((*self.chaperone_setup).get_working_standing_zero_pose_to_raw_tracking_pose)(&mut mat)
        };
        if ok {
            return Some(mat);
        }

        unsafe {
            ((*self.chaperone_setup).revert_working_copy)();
        }
        let ok_retry = unsafe {
            ((*self.chaperone_setup).get_working_standing_zero_pose_to_raw_tracking_pose)(&mut mat)
        };
        if ok_retry { Some(mat) } else { None }
    }

    #[inline]
    pub fn apply_working_standing_pose(&self, mat: &HmdMatrix34_t) {
        unsafe {
            ((*self.chaperone_setup).set_working_standing_zero_pose_to_raw_tracking_pose)(mat);
            ((*self.chaperone_setup).show_working_set_preview)();
        }
    }
}

impl Drop for OpenVrSession {
    fn drop(&mut self) {
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
        assert_eq!(size_of::<VRActiveActionSet_t>(), 32);
        #[cfg(target_os = "linux")]
        assert_eq!(size_of::<VREvent_t>(), 60);
        #[cfg(not(target_os = "linux"))]
        assert_eq!(size_of::<VREvent_t>(), 64);
    }
}
