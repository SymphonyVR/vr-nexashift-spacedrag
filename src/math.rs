use fearless_simd::{Level, Simd, dispatch, f32x4, prelude::*};

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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "xsave")]
unsafe fn read_xcr0() -> u64 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::_xgetbv;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::_xgetbv;
    unsafe { _xgetbv(0) }
}

/// Detects the CPU's highest supported `fearless_simd::Level` directly in `#![no_std]`
/// using CPUID/XGETBV once at startup (no `std` `LazyLock` cache required).
#[cold]
pub fn detect_simd_level() -> Level {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{__cpuid, __cpuid_count};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{__cpuid, __cpuid_count};
        use fearless_simd::x86::{Avx2, Avx512, Sse2, Sse4_2};

        let leaf0 = __cpuid(0);
        let max_leaf = leaf0.eax;
        if max_leaf < 1 {
            return Level::baseline();
        }
        let max_ext = __cpuid(0x8000_0000).eax;

        let l1 = __cpuid(1);
        let has_fxsr = (l1.edx & (1 << 24)) != 0;
        let has_sse2 = (l1.edx & (1 << 26)) != 0;
        let has_pclmulqdq = (l1.ecx & (1 << 1)) != 0;
        let has_fma = (l1.ecx & (1 << 12)) != 0;
        let has_cmpxchg16b = (l1.ecx & (1 << 13)) != 0;
        let has_sse4_2 = (l1.ecx & (1 << 20)) != 0;
        let has_movbe = (l1.ecx & (1 << 22)) != 0;
        let has_popcnt = (l1.ecx & (1 << 23)) != 0;
        let has_aes = (l1.ecx & (1 << 25)) != 0;
        let has_xsave = (l1.ecx & (1 << 26)) != 0;
        let has_osxsave = (l1.ecx & (1 << 27)) != 0;
        let has_avx = (l1.ecx & (1 << 28)) != 0;
        let has_f16c = (l1.ecx & (1 << 29)) != 0;
        let has_rdrand = (l1.ecx & (1 << 30)) != 0;

        let has_lzcnt = if max_ext >= 0x8000_0001 {
            (__cpuid(0x8000_0001).ecx & (1 << 5)) != 0
        } else {
            false
        };

        let xcr0 = if has_xsave && has_osxsave {
            // SAFETY: CPUID leaf 1 confirmed both XSAVE and OSXSAVE are enabled.
            unsafe { read_xcr0() }
        } else {
            0
        };
        let os_avx = (xcr0 & 0b110) == 0b110;
        let os_avx512 = (xcr0 & 0b1110_0110) == 0b1110_0110;

        if max_leaf >= 7 && has_avx && os_avx {
            let l7 = __cpuid_count(7, 0);
            let has_bmi1 = (l7.ebx & (1 << 3)) != 0;
            let has_avx2 = (l7.ebx & (1 << 5)) != 0;
            let has_bmi2 = (l7.ebx & (1 << 8)) != 0;
            let has_avx512f = (l7.ebx & (1 << 16)) != 0;
            let has_avx512dq = (l7.ebx & (1 << 17)) != 0;
            let has_rdseed = (l7.ebx & (1 << 18)) != 0;
            let has_adx = (l7.ebx & (1 << 19)) != 0;
            let has_avx512ifma = (l7.ebx & (1 << 21)) != 0;
            let has_avx512cd = (l7.ebx & (1 << 28)) != 0;
            let has_sha = (l7.ebx & (1 << 29)) != 0;
            let has_avx512bw = (l7.ebx & (1 << 30)) != 0;
            let has_avx512vl = (l7.ebx & (1 << 31)) != 0;

            let has_avx512vbmi = (l7.ecx & (1 << 1)) != 0;
            let has_avx512vbmi2 = (l7.ecx & (1 << 6)) != 0;
            let has_gfni = (l7.ecx & (1 << 8)) != 0;
            let has_vaes = (l7.ecx & (1 << 9)) != 0;
            let has_vpclmulqdq = (l7.ecx & (1 << 10)) != 0;
            let has_avx512vnni = (l7.ecx & (1 << 11)) != 0;
            let has_avx512bitalg = (l7.ecx & (1 << 12)) != 0;
            let has_avx512vpopcntdq = (l7.ecx & (1 << 14)) != 0;

            let (has_xsaveopt, has_xsavec, has_xsaves) = if max_leaf >= 0xD {
                let ld = __cpuid_count(0xD, 1);
                (
                    (ld.eax & (1 << 0)) != 0,
                    (ld.eax & (1 << 1)) != 0,
                    (ld.eax & (1 << 3)) != 0,
                )
            } else {
                (false, false, false)
            };

            if os_avx512
                && has_adx
                && has_aes
                && has_avx512bitalg
                && has_avx512bw
                && has_avx512cd
                && has_avx512dq
                && has_avx512f
                && has_avx512ifma
                && has_avx512vbmi
                && has_avx512vbmi2
                && has_avx512vl
                && has_avx512vnni
                && has_avx512vpopcntdq
                && has_bmi1
                && has_bmi2
                && has_cmpxchg16b
                && has_fma
                && has_fxsr
                && has_gfni
                && has_lzcnt
                && has_movbe
                && has_pclmulqdq
                && has_popcnt
                && has_rdrand
                && has_rdseed
                && has_sha
                && has_vaes
                && has_vpclmulqdq
                && has_xsave
                && has_xsavec
                && has_xsaveopt
                && has_xsaves
            {
                // SAFETY: Every feature required by `Avx512::assume_supported` was verified via CPUID/XGETBV.
                return unsafe { Level::Avx512(Avx512::assume_supported()) };
            }

            if has_avx2
                && has_bmi1
                && has_bmi2
                && has_cmpxchg16b
                && has_f16c
                && has_fma
                && has_fxsr
                && has_lzcnt
                && has_movbe
                && has_popcnt
                && has_xsave
            {
                // SAFETY: Every feature required by `Avx2::assume_supported` was verified via CPUID/XGETBV.
                return unsafe { Level::Avx2(Avx2::assume_supported()) };
            }
        }

        if has_fxsr && has_sse4_2 && has_cmpxchg16b && has_popcnt {
            // SAFETY: Every feature required by `Sse4_2::assume_supported` was verified via CPUID.
            return unsafe { Level::Sse4_2(Sse4_2::assume_supported()) };
        }

        if has_sse2 && has_fxsr {
            // SAFETY: Every feature required by `Sse2::assume_supported` was verified via CPUID.
            return unsafe { Level::Sse2(Sse2::assume_supported()) };
        }
    }

    Level::baseline()
}

#[inline(always)]
fn sub_vec3_simd<S: Simd>(simd: S, a: Vec3, b: Vec3) -> Vec3 {
    let va = f32x4::from_slice(simd, &[a.x, a.y, a.z, 0.0]);
    let vb = f32x4::from_slice(simd, &[b.x, b.y, b.z, 0.0]);
    let d = va - vb;
    Vec3::new(d[0], d[1], d[2])
}

#[inline(always)]
fn add_vec3_simd<S: Simd>(simd: S, a: Vec3, b: Vec3) -> Vec3 {
    let va = f32x4::from_slice(simd, &[a.x, a.y, a.z, 0.0]);
    let vb = f32x4::from_slice(simd, &[b.x, b.y, b.z, 0.0]);
    let s = va + vb;
    Vec3::new(s[0], s[1], s[2])
}

#[inline(always)]
fn apply_raw_translation_simd<S: Simd>(
    simd: S,
    m0: &HmdMatrix34_t,
    delta: Vec3,
) -> HmdMatrix34_t {
    let r0 = f32x4::from_slice(simd, &m0.m[0]) + f32x4::from_slice(simd, &[0.0, 0.0, 0.0, delta.x]);
    let r1 = f32x4::from_slice(simd, &m0.m[1]) + f32x4::from_slice(simd, &[0.0, 0.0, 0.0, delta.y]);
    let r2 = f32x4::from_slice(simd, &m0.m[2]) + f32x4::from_slice(simd, &[0.0, 0.0, 0.0, delta.z]);
    let mut out = *m0;
    r0.store_slice(&mut out.m[0]);
    r1.store_slice(&mut out.m[1]);
    r2.store_slice(&mut out.m[2]);
    out
}

#[inline(always)]
fn raw_to_standing_point_simd<S: Simd>(
    simd: S,
    stand_to_raw: &HmdMatrix34_t,
    p_raw: Vec3,
) -> Vec3 {
    let dx = f32x4::splat(simd, p_raw.x - stand_to_raw.m[0][3]);
    let dy = f32x4::splat(simd, p_raw.y - stand_to_raw.m[1][3]);
    let dz = f32x4::splat(simd, p_raw.z - stand_to_raw.m[2][3]);

    let row0 = f32x4::from_slice(simd, &stand_to_raw.m[0]);
    let row1 = f32x4::from_slice(simd, &stand_to_raw.m[1]);
    let row2 = f32x4::from_slice(simd, &stand_to_raw.m[2]);

    let acc = row0 * dx;
    let acc = row1.mul_add(dy, acc);
    let acc = row2.mul_add(dz, acc);
    Vec3::new(acc[0], acc[1], acc[2])
}

/// Startup-resolved SIMD math table using `fearless_simd` with zero runtime static cache.
#[derive(Clone, Copy)]
pub struct SimdMath {
    level: Level,
    apply_raw_translation_fn: fn(&HmdMatrix34_t, Vec3) -> HmdMatrix34_t,
    raw_to_standing_point_fn: fn(&HmdMatrix34_t, Vec3) -> Vec3,
    sub_vec3_fn: fn(Vec3, Vec3) -> Vec3,
    add_vec3_fn: fn(Vec3, Vec3) -> Vec3,
}

impl SimdMath {
    #[cold]
    pub fn new_detected() -> Self {
        Self::with_level(detect_simd_level())
    }

    pub fn with_level(level: Level) -> Self {
        dispatch!(level, simd => Self::for_simd(simd))
    }

    pub fn for_simd<S: Simd>(simd: S) -> Self {
        const {
            assert!(core::mem::size_of::<S>() == 0);
            assert!(core::mem::align_of::<S>() == 1);
        }

        fn apply_impl<S: Simd>(m0: &HmdMatrix34_t, delta: Vec3) -> HmdMatrix34_t {
            // SAFETY: `S` is a zero-sized, 1-aligned proof token and holding `simd: S` proves
            // that all target features for `S` are supported on this CPU.
            let s: S = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
            s.vectorize(#[inline(always)] || apply_raw_translation_simd(s, m0, delta))
        }

        fn raw_to_standing_impl<S: Simd>(stand_to_raw: &HmdMatrix34_t, p_raw: Vec3) -> Vec3 {
            let s: S = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
            s.vectorize(#[inline(always)] || raw_to_standing_point_simd(s, stand_to_raw, p_raw))
        }

        fn sub_impl<S: Simd>(a: Vec3, b: Vec3) -> Vec3 {
            let s: S = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
            s.vectorize(#[inline(always)] || sub_vec3_simd(s, a, b))
        }

        fn add_impl<S: Simd>(a: Vec3, b: Vec3) -> Vec3 {
            let s: S = unsafe { core::mem::MaybeUninit::uninit().assume_init() };
            s.vectorize(#[inline(always)] || add_vec3_simd(s, a, b))
        }

        Self {
            level: simd.level(),
            apply_raw_translation_fn: apply_impl::<S>,
            raw_to_standing_point_fn: raw_to_standing_impl::<S>,
            sub_vec3_fn: sub_impl::<S>,
            add_vec3_fn: add_impl::<S>,
        }
    }

    #[inline]
    pub const fn level(&self) -> Level {
        self.level
    }

    pub const fn level_name(&self) -> &'static str {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        match self.level {
            Level::Avx512(_) => return "AVX-512",
            Level::Avx2(_) => return "AVX2",
            Level::Sse4_2(_) => return "SSE4.2",
            Level::Sse2(_) => return "SSE2",
            _ => {}
        }
        #[cfg(target_arch = "aarch64")]
        if matches!(self.level, Level::Neon(_)) {
            return "NEON";
        }
        "Fallback"
    }

    #[inline]
    pub fn apply_raw_translation(&self, m0: &HmdMatrix34_t, delta: Vec3) -> HmdMatrix34_t {
        (self.apply_raw_translation_fn)(m0, delta)
    }

    #[inline]
    pub fn raw_to_standing_point(&self, stand_to_raw: &HmdMatrix34_t, p_raw: Vec3) -> Vec3 {
        (self.raw_to_standing_point_fn)(stand_to_raw, p_raw)
    }

    #[inline]
    pub fn sub_vec3(&self, a: Vec3, b: Vec3) -> Vec3 {
        (self.sub_vec3_fn)(a, b)
    }

    #[inline]
    pub fn add_vec3(&self, a: Vec3, b: Vec3) -> Vec3 {
        (self.add_vec3_fn)(a, b)
    }
}

impl Default for SimdMath {
    fn default() -> Self {
        Self::with_level(Level::baseline())
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

    #[test]
    fn simd_math_detects_at_startup_and_matches_scalar_reference() {
        let simd = SimdMath::new_detected();
        let baseline = SimdMath::with_level(Level::baseline());
        assert!(!simd.level_name().is_empty());

        let m0 = yaw_matrix(0.61, 1.5, -0.75, 2.25);
        let c0 = Vec3::new(-0.3, 1.15, -0.45);
        let c1 = Vec3::new(0.45, 0.85, 0.25);

        let delta_scalar = c1.sub(c0);
        let delta_simd = simd.sub_vec3(c1, c0);
        let delta_base = baseline.sub_vec3(c1, c0);
        assert!(approx_eq(delta_simd, delta_scalar, 1e-6));
        assert!(approx_eq(delta_base, delta_scalar, 1e-6));

        let sum_scalar = c0.add(c1);
        assert!(approx_eq(simd.add_vec3(c0, c1), sum_scalar, 1e-6));

        let m_scalar = apply_raw_translation(&m0, delta_scalar);
        let m_simd = simd.apply_raw_translation(&m0, delta_simd);
        let m_base = baseline.apply_raw_translation(&m0, delta_base);
        assert_eq!(m_simd, m_scalar);
        assert_eq!(m_base, m_scalar);

        let p_raw = Vec3::new(1.8, 1.4, -0.9);
        let p_scalar = raw_to_standing_point(&m_scalar, p_raw);
        let p_simd = simd.raw_to_standing_point(&m_simd, p_raw);
        let p_base = baseline.raw_to_standing_point(&m_base, p_raw);
        assert!(approx_eq(p_simd, p_scalar, 1e-5));
        assert!(approx_eq(p_base, p_scalar, 1e-5));
    }
}
