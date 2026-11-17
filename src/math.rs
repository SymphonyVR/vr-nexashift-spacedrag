/// 3D Cartesian vector in meters (OpenVR right-handed coordinate system:
/// +X right, +Y up, -Z forward).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

/// Row-major 3x4 rigid transformation matrix matching OpenVR `HmdMatrix34_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HmdMatrix34_t {
    pub m: [[f32; 4]; 3],
}

impl HmdMatrix34_t {
    pub const IDENTITY: Self = Self {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };

    #[inline]
    pub const fn translation(self) -> Vec3 {
        Vec3::new(self.m[0][3], self.m[1][3], self.m[2][3])
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        let mut r = 0;
        while r < 3 {
            let mut c = 0;
            while c < 4 {
                if !self.m[r][c].is_finite() {
                    return false;
                }
                c += 1;
            }
            r += 1;
        }
        true
    }
}

/// Computes `M = Translation(delta) * m0` in raw tracking coordinates.
///
/// Since `Translation(delta)` has identity rotation and translation `delta`,
/// left-multiplying `m0 = [R0 | t0]` preserves `R0` and sets `t = t0 + delta`.
#[inline]
pub fn apply_raw_translation(m0: &HmdMatrix34_t, delta: Vec3) -> HmdMatrix34_t {
    let mut out = *m0;
    out.m[0][3] += delta.x;
    out.m[1][3] += delta.y;
    out.m[2][3] += delta.z;
    out
}

/// Converts a point in raw tracking coordinates into standing playspace coordinates
/// given the standing-to-raw rigid transform `stand_to_raw = [R | t]`:
/// `P_stand = R^T * (P_raw - t)`.
#[inline]
pub fn raw_to_standing_point(stand_to_raw: &HmdMatrix34_t, p_raw: Vec3) -> Vec3 {
    let dx = p_raw.x - stand_to_raw.m[0][3];
    let dy = p_raw.y - stand_to_raw.m[1][3];
    let dz = p_raw.z - stand_to_raw.m[2][3];
    Vec3 {
        x: stand_to_raw.m[0][0] * dx + stand_to_raw.m[1][0] * dy + stand_to_raw.m[2][0] * dz,
        y: stand_to_raw.m[0][1] * dx + stand_to_raw.m[1][1] * dy + stand_to_raw.m[2][1] * dz,
        z: stand_to_raw.m[0][2] * dx + stand_to_raw.m[1][2] * dy + stand_to_raw.m[2][2] * dz,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Vec3, b: Vec3, eps: f32) -> bool {
        (a.x - b.x).abs() <= eps && (a.y - b.y).abs() <= eps && (a.z - b.z).abs() <= eps
    }

    fn yaw_matrix(radians: f32, tx: f32, ty: f32, tz: f32) -> HmdMatrix34_t {
        let c = radians.cos();
        let s = radians.sin();
        HmdMatrix34_t {
            m: [
                [c, 0.0, s, tx],
                [0.0, 1.0, 0.0, ty],
                [-s, 0.0, c, tz],
            ],
        }
    }

    #[test]
    fn matrix_size_is_48_bytes() {
        assert_eq!(core::mem::size_of::<HmdMatrix34_t>(), 48);
        assert_eq!(core::mem::size_of::<Vec3>(), 12);
    }

    #[test]
    fn six_directions_produce_exact_1_to_1_inverse_virtual_displacement() {
        let m0 = HmdMatrix34_t::IDENTITY;
        let hmd_raw = Vec3::new(0.1, 1.65, -0.2);
        let hmd_stand_initial = raw_to_standing_point(&m0, hmd_raw);
        let c0 = Vec3::new(0.25, 1.1, -0.4);

        let axes = [
            Vec3::new(0.5, 0.0, 0.0),   // +X right
            Vec3::new(-0.5, 0.0, 0.0),  // -X left
            Vec3::new(0.0, 0.5, 0.0),   // +Y up
            Vec3::new(0.0, -0.5, 0.0),  // -Y down
            Vec3::new(0.0, 0.0, 0.5),   // +Z backward
            Vec3::new(0.0, 0.0, -0.5),  // -Z forward
        ];

        for d in axes {
            let c = c0.add(d);
            let delta = c.sub(c0);
            let m = apply_raw_translation(&m0, delta);
            let hmd_stand_after = raw_to_standing_point(&m, hmd_raw);
            let virtual_shift = hmd_stand_after.sub(hmd_stand_initial);
            let expected = Vec3::new(-d.x, -d.y, -d.z);
            assert!(
                approx_eq(virtual_shift, expected, 1e-6),
                "direction {:?}: got {:?}, expected {:?}",
                d,
                virtual_shift,
                expected
            );
        }
    }

    #[test]
    fn rotated_playspace_preserves_rotation_and_1_to_1_magnitude() {
        let m0 = yaw_matrix(0.73, 1.25, -0.4, 2.1);
        let hmd_raw = Vec3::new(0.2, 1.7, -0.3);
        let hmd_stand_initial = raw_to_standing_point(&m0, hmd_raw);
        let c0 = Vec3::new(-0.15, 1.0, -0.5);
        let move_raw = Vec3::new(0.3, -0.4, 0.5);
        let c_now = c0.add(move_raw);

        let m = apply_raw_translation(&m0, c_now.sub(c0));
        // Rotation block must remain untouched.
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(m.m[r][c], m0.m[r][c]);
            }
        }

        let hmd_stand_after = raw_to_standing_point(&m, hmd_raw);
        let virtual_shift = hmd_stand_after.sub(hmd_stand_initial);
        let mag_before = (move_raw.x * move_raw.x + move_raw.y * move_raw.y + move_raw.z * move_raw.z).sqrt();
        let mag_after = (virtual_shift.x * virtual_shift.x
            + virtual_shift.y * virtual_shift.y
            + virtual_shift.z * virtual_shift.z)
            .sqrt();
        assert!((mag_before - mag_after).abs() < 1e-5);
    }

    #[test]
    fn walking_and_crouching_with_controller_fixed_to_body_cancels_completely() {
        // Test with arbitrary initial rotated playspace transform M0.
        let m0 = yaw_matrix(-1.12, -0.8, 0.35, 1.4);
        let hmd_0 = Vec3::new(0.0, 1.72, 0.0);
        let c0 = Vec3::new(-0.2, 1.25, -0.35);
        let hmd_stand_0 = raw_to_standing_point(&m0, hmd_0);

        let body_motions = [
            Vec3::new(0.0, 0.0, -1.0), // Walk forward 1 meter
            Vec3::new(0.3, 0.0, 0.0),  // Side-step 30 cm
            Vec3::new(0.0, -0.55, 0.0), // Crouch 55 cm
            Vec3::new(0.45, 0.2, 0.7), // Combined 3D body displacement
        ];

        for body_move in body_motions {
            let hmd_now = hmd_0.add(body_move);
            let c_now = c0.add(body_move);
            let delta = c_now.sub(c0);
            let m = apply_raw_translation(&m0, delta);
            let hmd_stand_now = raw_to_standing_point(&m, hmd_now);
            assert!(
                approx_eq(hmd_stand_now, hmd_stand_0, 1e-5),
                "body move {:?} failed to cancel: start={:?}, now={:?}",
                body_move,
                hmd_stand_0,
                hmd_stand_now
            );
        }
    }

    #[test]
    fn walking_plus_relative_controller_motion_produces_only_relative_virtual_motion() {
        let m0 = HmdMatrix34_t::IDENTITY;
        let hmd_0 = Vec3::new(0.0, 1.7, 0.0);
        let c0 = Vec3::new(0.2, 1.1, -0.3);
        let hmd_stand_0 = raw_to_standing_point(&m0, hmd_0);

        let body_walk = Vec3::new(0.0, 0.0, -1.0);
        let relative_hand = Vec3::new(0.25, -0.15, 0.4);

        let hmd_now = hmd_0.add(body_walk);
        let c_now = c0.add(body_walk).add(relative_hand);
        let m = apply_raw_translation(&m0, c_now.sub(c0));
        let hmd_stand_now = raw_to_standing_point(&m, hmd_now);
        let virtual_shift = hmd_stand_now.sub(hmd_stand_0);
        let expected = Vec3::new(-relative_hand.x, -relative_hand.y, -relative_hand.z);
        assert!(approx_eq(virtual_shift, expected, 1e-5));
    }

    #[test]
    fn non_finite_detection() {
        let mut m = HmdMatrix34_t::IDENTITY;
        assert!(m.is_finite());
        m.m[2][3] = f32::NAN;
        assert!(!m.is_finite());
        assert!(!Vec3::new(0.0, f32::INFINITY, 0.0).is_finite());
    }
}
