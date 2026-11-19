use crate::math::{HmdMatrix34_t, SimdMath, Vec3};

pub const INVALID_DEVICE_INDEX: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hand {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandInput {
    pub pressed: bool,
    pub device_index: u32,
    pub pose_valid: bool,
    pub position_raw: Vec3,
}

impl HandInput {
    pub const UNPRESSED: Self = Self {
        pressed: false,
        device_index: INVALID_DEVICE_INDEX,
        pose_valid: false,
        position_raw: Vec3::ZERO,
    };

    #[inline]
    pub fn has_valid_pose(&self) -> bool {
        self.pose_valid && self.device_index != INVALID_DEVICE_INDEX && self.position_raw.is_finite()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InputFrame {
    pub left: HandInput,
    pub right: HandInput,
}

impl InputFrame {
    pub const IDLE: Self = Self {
        left: HandInput::UNPRESSED,
        right: HandInput::UNPRESSED,
    };

    #[inline]
    pub const fn hand(&self, hand: Hand) -> HandInput {
        match hand {
            Hand::Left => self.left,
            Hand::Right => self.right,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DragPhase {
    Idle,
    Dragging {
        hand: Hand,
        device_index: u32,
        c0: Vec3,
        last_raw: Vec3,
        m0: HmdMatrix34_t,
        last_m: HmdMatrix34_t,
        moved: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepAction {
    Idle,
    UpdatePlayspace(HmdMatrix34_t),
    CommitPlayspace(HmdMatrix34_t),
    AbortDrag,
}

/// Deterministic zero-allocation 1:1 playspace drag state machine.
pub struct DragStateMachine {
    phase: DragPhase,
    left_latched_ignore: bool,
    right_latched_ignore: bool,
    suspended_hand: Option<Hand>,
    simd: SimdMath,
}

impl DragStateMachine {
    pub fn new() -> Self {
        Self::with_simd(SimdMath::new_detected())
    }

    pub const fn with_simd(simd: SimdMath) -> Self {
        Self {
            phase: DragPhase::Idle,
            left_latched_ignore: false,
            right_latched_ignore: false,
            suspended_hand: None,
            simd,
        }
    }

    #[inline]
    pub const fn is_dragging(&self) -> bool {
        matches!(self.phase, DragPhase::Dragging { .. })
    }

    #[inline]
    pub const fn active_device_hint(&self) -> Option<(Hand, u32)> {
        match self.phase {
            DragPhase::Dragging {
                hand, device_index, ..
            } => Some((hand, device_index)),
            DragPhase::Idle => None,
        }
    }

    #[inline]
    pub const fn phase(&self) -> DragPhase {
        self.phase
    }

    #[inline]
    fn set_latched_ignore(&mut self, hand: Hand, ignore: bool) {
        match hand {
            Hand::Left => self.left_latched_ignore = ignore,
            Hand::Right => self.right_latched_ignore = ignore,
        }
    }

    #[inline]
    fn other_hand(hand: Hand) -> Hand {
        match hand {
            Hand::Left => Hand::Right,
            Hand::Right => Hand::Left,
        }
    }

    pub fn step(&mut self, frame: InputFrame, current_working_m0: Option<HmdMatrix34_t>) -> StepAction {
        self.step_with(frame, || current_working_m0)
    }

    pub fn step_with<F>(&mut self, frame: InputFrame, get_m0: F) -> StepAction
    where
        F: FnOnce() -> Option<HmdMatrix34_t>,
    {
        if !frame.left.pressed {
            self.left_latched_ignore = false;
        }
        if !frame.right.pressed {
            self.right_latched_ignore = false;
        }
        if let Some(sh) = self.suspended_hand {
            if !frame.hand(sh).pressed {
                self.suspended_hand = None;
            }
        }

        match self.phase {
            DragPhase::Dragging {
                hand,
                device_index,
                c0,
                last_raw,
                m0,
                last_m,
                moved,
            } => {
                let other = Self::other_hand(hand);
                if frame.hand(other).pressed {
                    self.set_latched_ignore(other, true);
                }

                let active = frame.hand(hand);
                if !active.pressed {
                    self.phase = DragPhase::Idle;
                    return if moved {
                        StepAction::CommitPlayspace(last_m)
                    } else {
                        StepAction::Idle
                    };
                }

                if !active.has_valid_pose() || active.device_index != device_index {
                    self.phase = DragPhase::Idle;
                    self.suspended_hand = Some(hand);
                    return StepAction::AbortDrag;
                }

                if active.position_raw == last_raw {
                    return StepAction::Idle;
                }

                let delta = self.simd.sub_vec3(active.position_raw, c0);
                // Ignore sub-2mm tremor before the drag actually starts so a quick tap
                // never attaches a live working-set preview to SteamVR.
                if !moved {
                    let delta_sq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
                    if delta_sq < (0.002 * 0.002) {
                        return StepAction::Idle;
                    }
                }
                let m_new = self.simd.apply_raw_translation(&m0, delta);
                self.phase = DragPhase::Dragging {
                    hand,
                    device_index,
                    c0,
                    last_raw: active.position_raw,
                    m0,
                    last_m: m_new,
                    moved: true,
                };
                StepAction::UpdatePlayspace(m_new)
            }
            DragPhase::Idle => {
                let selected = if let Some(sh) = self.suspended_hand {
                    let other = Self::other_hand(sh);
                    if frame.hand(other).pressed {
                        self.set_latched_ignore(other, true);
                    }
                    Some(sh)
                } else {
                    let left_eligible = frame.left.pressed && !self.left_latched_ignore;
                    let right_eligible = frame.right.pressed && !self.right_latched_ignore;
                    if left_eligible && right_eligible {
                        // Deterministic simultaneous priority: Left wins; Right stays ignored until released.
                        self.right_latched_ignore = true;
                        Some(Hand::Left)
                    } else if left_eligible {
                        Some(Hand::Left)
                    } else if right_eligible {
                        Some(Hand::Right)
                    } else {
                        None
                    }
                };

                let Some(hand) = selected else {
                    return StepAction::Idle;
                };

                let input = frame.hand(hand);
                if !input.has_valid_pose() {
                    self.suspended_hand = Some(hand);
                    return StepAction::Idle;
                }

                let Some(m0) = get_m0() else {
                    self.suspended_hand = Some(hand);
                    return StepAction::Idle;
                };
                if !m0.is_finite() {
                    self.suspended_hand = Some(hand);
                    return StepAction::Idle;
                }

                self.suspended_hand = None;
                self.phase = DragPhase::Dragging {
                    hand,
                    device_index: input.device_index,
                    c0: input.position_raw,
                    last_raw: input.position_raw,
                    m0,
                    last_m: m0,
                    moved: false,
                };
                StepAction::Idle
            }
        }
    }
}

impl Default for DragStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::apply_raw_translation;

    fn hand(pressed: bool, id: u32, pos: (f32, f32, f32)) -> HandInput {
        HandInput {
            pressed,
            device_index: id,
            pose_valid: true,
            position_raw: Vec3::new(pos.0, pos.1, pos.2),
        }
    }

    #[test]
    fn left_hand_drag_and_release_preserves_offset() {
        let mut sm = DragStateMachine::new();
        let m0 = HmdMatrix34_t::IDENTITY;

        let f1 = InputFrame {
            left: hand(true, 1, (0.0, 1.0, 0.0)),
            right: hand(false, 2, (0.5, 1.0, 0.0)),
        };
        assert_eq!(sm.step(f1, Some(m0)), StepAction::Idle);
        assert!(sm.is_dragging());

        // Move left by (0.3, -0.2, 0.5), move right wildly (ignored).
        let f2 = InputFrame {
            left: hand(true, 1, (0.3, 0.8, 0.5)),
            right: hand(false, 2, (9.0, 9.0, 9.0)),
        };
        let delta1 = Vec3::new(0.3f32 - 0.0f32, 0.8f32 - 1.0f32, 0.5f32 - 0.0f32);
        let expected_m1 = apply_raw_translation(&m0, delta1);
        assert_eq!(sm.step(f2, None), StepAction::UpdatePlayspace(expected_m1));

        // Release left -> Idle and commit final pose.
        let f3 = InputFrame {
            left: hand(false, 1, (0.3, 0.8, 0.5)),
            right: hand(false, 2, (9.0, 9.0, 9.0)),
        };
        assert_eq!(sm.step(f3, None), StepAction::CommitPlayspace(expected_m1));
        assert!(!sm.is_dragging());

        // Re-drag with right hand starting from preserved expected_m1.
        let f4 = InputFrame {
            left: hand(false, 1, (0.3, 0.8, 0.5)),
            right: hand(true, 2, (1.0, 1.0, 1.0)),
        };
        assert_eq!(sm.step(f4, Some(expected_m1)), StepAction::Idle);
        let f5 = InputFrame {
            left: hand(false, 1, (0.3, 0.8, 0.5)),
            right: hand(true, 2, (1.2, 1.1, 0.75)),
        };
        let delta2 = Vec3::new(1.2f32 - 1.0f32, 1.1f32 - 1.0f32, 0.75f32 - 1.0f32);
        let expected_m2 = apply_raw_translation(&expected_m1, delta2);
        assert_eq!(sm.step(f5, None), StepAction::UpdatePlayspace(expected_m2));
        let t = expected_m2.translation();
        assert!((t.x - 0.5).abs() < 1e-6);
        assert!((t.y - (-0.1)).abs() < 1e-6);
        assert!((t.z - 0.25).abs() < 1e-6);
    }

    #[test]
    fn simultaneous_activation_gives_deterministic_priority_and_never_switches() {
        let mut sm = DragStateMachine::new();
        let m0 = HmdMatrix34_t::IDENTITY;

        // Both pressed at once -> Left wins deterministically.
        let both = InputFrame {
            left: hand(true, 1, (0.0, 1.0, 0.0)),
            right: hand(true, 2, (0.5, 1.0, 0.0)),
        };
        assert_eq!(sm.step(both, Some(m0)), StepAction::Idle);
        assert_eq!(sm.active_device_hint(), Some((Hand::Left, 1)));

        // Right controller moves while held: must not affect output.
        let right_moves = InputFrame {
            left: hand(true, 1, (0.2, 1.0, 0.0)),
            right: hand(true, 2, (5.0, 5.0, 5.0)),
        };
        let expected_m = apply_raw_translation(&m0, Vec3::new(0.2, 0.0, 0.0));
        assert_eq!(
            sm.step(right_moves, None),
            StepAction::UpdatePlayspace(expected_m)
        );

        // Left releases while Right is STILL held -> must end drag cleanly without switching to Right!
        let left_released = InputFrame {
            left: hand(false, 1, (0.2, 1.0, 0.0)),
            right: hand(true, 2, (5.0, 5.0, 5.0)),
        };
        assert_eq!(
            sm.step(left_released, None),
            StepAction::CommitPlayspace(expected_m)
        );
        assert!(!sm.is_dragging());
        // Still holding Right on next frame remains Idle.
        assert_eq!(sm.step(left_released, Some(m0)), StepAction::Idle);

        // Right releases and presses again -> now Right activates normally.
        let right_up = InputFrame {
            left: hand(false, 1, (0.2, 1.0, 0.0)),
            right: hand(false, 2, (5.0, 5.0, 5.0)),
        };
        assert_eq!(sm.step(right_up, None), StepAction::Idle);
        let right_down = InputFrame {
            left: hand(false, 1, (0.2, 1.0, 0.0)),
            right: hand(true, 2, (1.0, 1.0, 1.0)),
        };
        assert_eq!(sm.step(right_down, Some(m0)), StepAction::Idle);
        assert_eq!(sm.active_device_hint(), Some((Hand::Right, 2)));
    }

    #[test]
    fn non_active_press_during_drag_does_not_steal_or_chain() {
        let mut sm = DragStateMachine::new();
        let m0 = HmdMatrix34_t::IDENTITY;

        // Start with Right.
        let r_down = InputFrame {
            left: hand(false, 1, (0.0, 0.0, 0.0)),
            right: hand(true, 2, (0.0, 1.0, 0.0)),
        };
        assert_eq!(sm.step(r_down, Some(m0)), StepAction::Idle);

        // Left presses during active Right drag -> Right remains active.
        let both = InputFrame {
            left: hand(true, 1, (10.0, 10.0, 10.0)),
            right: hand(true, 2, (0.0, 1.4, 0.0)),
        };
        let m1 = apply_raw_translation(&m0, Vec3::new(0.0, 1.4f32 - 1.0f32, 0.0));
        assert_eq!(sm.step(both, None), StepAction::UpdatePlayspace(m1));
        assert_eq!(sm.active_device_hint(), Some((Hand::Right, 2)));

        // Right releases while Left still held -> goes Idle, commits m1, does NOT switch to Left.
        let r_up = InputFrame {
            left: hand(true, 1, (10.0, 10.0, 10.0)),
            right: hand(false, 2, (0.0, 1.4, 0.0)),
        };
        assert_eq!(sm.step(r_up, None), StepAction::CommitPlayspace(m1));
        assert!(!sm.is_dragging());
        assert_eq!(sm.step(r_up, Some(m1)), StepAction::Idle);
    }

    #[test]
    fn tracking_loss_aborts_and_requires_fresh_capture_without_jump() {
        let mut sm = DragStateMachine::new();
        let m0 = HmdMatrix34_t::IDENTITY;

        // Capture at (0, 1, 0).
        let f1 = InputFrame {
            left: hand(true, 1, (0.0, 1.0, 0.0)),
            right: HandInput::UNPRESSED,
        };
        assert_eq!(sm.step(f1, Some(m0)), StepAction::Idle);

        // Drag to (0.5, 1, 0).
        let f2 = InputFrame {
            left: hand(true, 1, (0.5, 1.0, 0.0)),
            right: HandInput::UNPRESSED,
        };
        let preserved_m = apply_raw_translation(&m0, Vec3::new(0.5, 0.0, 0.0));
        assert_eq!(sm.step(f2, None), StepAction::UpdatePlayspace(preserved_m));

        // Tracking lost (pose_valid = false) while held -> AbortDrag, back to Idle.
        let f_lost = InputFrame {
            left: HandInput {
                pressed: true,
                device_index: 1,
                pose_valid: false,
                position_raw: Vec3::new(99.0, 99.0, 99.0),
            },
            right: hand(true, 2, (0.0, 0.0, 0.0)),
        };
        assert_eq!(sm.step(f_lost, None), StepAction::AbortDrag);
        assert!(!sm.is_dragging());

        // Right hand still cannot hijack while suspended left is held & invalid.
        assert_eq!(sm.step(f_lost, Some(preserved_m)), StepAction::Idle);

        // Tracking recovers at completely different raw position (2.0, 1.0, 0.0):
        // Must perform a fresh capture with preserved_m and c0=(2,1,0), producing ZERO jump!
        let f_recover = InputFrame {
            left: hand(true, 1, (2.0, 1.0, 0.0)),
            right: hand(true, 2, (0.0, 0.0, 0.0)),
        };
        assert_eq!(sm.step(f_recover, Some(preserved_m)), StepAction::Idle);

        // Moving +0.1 in Z from new capture adds +0.1 to preserved_m.
        let f_after = InputFrame {
            left: hand(true, 1, (2.0, 1.0, 0.1)),
            right: hand(true, 2, (0.0, 0.0, 0.0)),
        };
        let expected_final = apply_raw_translation(&preserved_m, Vec3::new(0.0, 0.0, 0.1));
        assert_eq!(sm.step(f_after, None), StepAction::UpdatePlayspace(expected_final));
        assert_eq!(expected_final.translation(), Vec3::new(0.5, 0.0, 0.1));
    }

    #[test]
    fn no_incremental_accumulation_drift_across_many_frames() {
        let mut sm = DragStateMachine::new();
        let m0 = HmdMatrix34_t::IDENTITY;
        let c0 = (0.1234, 1.2345, -0.5678);

        assert_eq!(
            sm.step(
                InputFrame {
                    left: hand(true, 1, c0),
                    right: HandInput::UNPRESSED
                },
                Some(m0)
            ),
            StepAction::Idle
        );

        // Oscillate 10,000 times and return to c0 -> transform must return to exact m0.
        for i in 1..10_000 {
            let offset = (i as f32) * 0.00013;
            let pos = (c0.0 + offset, c0.1 - offset * 0.5, c0.2 + offset * 0.25);
            let _ = sm.step(
                InputFrame {
                    left: hand(true, 1, pos),
                    right: HandInput::UNPRESSED,
                },
                None,
            );
        }

        let final_step = sm.step(
            InputFrame {
                left: hand(true, 1, c0),
                right: HandInput::UNPRESSED,
            },
            None,
        );
        assert_eq!(final_step, StepAction::UpdatePlayspace(m0));
    }

    #[test]
    fn stationary_held_controller_skips_redundant_updates() {
        let mut sm = DragStateMachine::new();
        let m0 = HmdMatrix34_t::IDENTITY;
        let f = InputFrame {
            left: hand(true, 1, (0.2, 1.0, -0.3)),
            right: HandInput::UNPRESSED,
        };
        assert_eq!(sm.step(f, Some(m0)), StepAction::Idle);
        assert!(sm.is_dragging());
        assert_eq!(sm.step(f, None), StepAction::Idle);
        assert!(sm.is_dragging());
        let released = InputFrame {
            left: hand(false, 1, (0.2, 1.0, -0.3)),
            right: HandInput::UNPRESSED,
        };
        assert_eq!(sm.step(released, None), StepAction::Idle);
        assert!(!sm.is_dragging());
    }

    #[test]
    fn sub_2mm_button_tap_skips_preview_until_threshold_crossed() {
        let mut sm = DragStateMachine::new();
        let m0 = HmdMatrix34_t::IDENTITY;
        let down = InputFrame {
            left: hand(true, 1, (0.0, 1.0, 0.0)),
            right: HandInput::UNPRESSED,
        };
        assert_eq!(sm.step(down, Some(m0)), StepAction::Idle);

        // 1mm tremor stays Idle without emitting UpdatePlayspace.
        let tiny = InputFrame {
            left: hand(true, 1, (0.001, 1.0005, 0.0)),
            right: HandInput::UNPRESSED,
        };
        assert_eq!(sm.step(tiny, None), StepAction::Idle);

        // Releasing after tiny tremor also stays Idle without emitting CommitPlayspace.
        let up = InputFrame {
            left: hand(false, 1, (0.001, 1.0005, 0.0)),
            right: HandInput::UNPRESSED,
        };
        assert_eq!(sm.step(up, None), StepAction::Idle);
    }
}
