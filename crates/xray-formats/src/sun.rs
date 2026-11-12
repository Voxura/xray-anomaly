//! Pure-Rust Cascaded Sun Shadow Maps (CSM) & Frustum-Light Culling Volumes
//!
//! Grounds X-Ray's directional sun shadow projection and frustum-light hull builders directly from:
//! - `src/Layers/xrRenderPC_R3/r2_R_sun.cpp` (`init_cacades`, `render_sun_cascade`, texel snapping)
//! - `src/Layers/xrRenderPC_R3/r3_R_sun_support.h` (`FixedConvexVolume`, `DumbConvexVolume`)
//! - `src/xrCore/_matrix.h` (4x4 multiplication, `mulA_44` / `mulB_44`, affine/full inverses)

use crate::camera::{matrix_build_camera_dir, matrix_build_projection};
use crate::hom::{ClipFrustum, Plane};
use crate::ogf::{
    matrix_mul_44, matrix_scale, matrix_transform_point, matrix_transform_projected, XRayMatrix,
    XRAY_IDENTITY_MATRIX,
};
use crate::skinning::matrix_transform_dir;
use crate::weather::dir_from_hp;

const EPS_L: f32 = 1e-3;
const EPS_S: f32 = 1e-7;
const TWEAK_COP_INITIAL_OFFS: f32 = 1200.0;

/// 8 NDC frustum corners in D3D `[0, 1]` depth range (`r2_R_sun.cpp:21-26`).
pub const SUN_FRUSTUM_CORNERS: [[f32; 3]; 8] = [
    [-1.0, -1.0, 0.0],
    [-1.0, -1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [-1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, -1.0, 1.0],
    [1.0, -1.0, 0.0],
];

/// 6 quads of the frustum corner cube (sides `0..4`, near `4`, far `5`, `r2_R_sun.cpp:27-32`).
pub const SUN_FRUSTUM_FACETABLE: [[usize; 4]; 6] = [
    [6, 7, 5, 4],
    [1, 0, 7, 6],
    [1, 2, 3, 0],
    [3, 2, 4, 5],
    [0, 3, 5, 7],
    [1, 6, 4, 2],
];

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize3_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len_sq = dot3(v, v);
    if len_sq > 1e-12 {
        let inv = 1.0 / len_sq.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        fallback
    }
}

/// Ray used by X-Ray cascaded sun shadow placement (`sun::ray`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunRay {
    pub origin: [f32; 3],
    pub dir: [f32; 3],
}

impl Default for SunRay {
    fn default() -> Self {
        Self {
            origin: [0.0; 3],
            dir: [0.0, 0.0, 1.0],
        }
    }
}

/// General 4x4 cofactor matrix inverse (handles perspective & orthographic matrices,
/// matching `D3DXMatrixInverse`). Returns `None` if singular or non-finite.
pub fn matrix_invert_44(m: &XRayMatrix) -> Option<XRayMatrix> {
    let s0 = m[0][0] * m[1][1] - m[1][0] * m[0][1];
    let s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
    let s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3];
    let s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
    let s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
    let s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];

    let c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    let c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    let c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    let c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    let c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    let c0 = m[2][0] * m[3][1] - m[3][0] * m[2][1];

    let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if det.abs() <= 1e-12 || !det.is_finite() {
        return None;
    }
    let inv_det = 1.0 / det;

    Some([
        [
            (m[1][1] * c5 - m[1][2] * c4 + m[1][3] * c3) * inv_det,
            (-m[0][1] * c5 + m[0][2] * c4 - m[0][3] * c3) * inv_det,
            (m[3][1] * s5 - m[3][2] * s4 + m[3][3] * s3) * inv_det,
            (-m[2][1] * s5 + m[2][2] * s4 - m[2][3] * s3) * inv_det,
        ],
        [
            (-m[1][0] * c5 + m[1][2] * c2 - m[1][3] * c1) * inv_det,
            (m[0][0] * c5 - m[0][2] * c2 + m[0][3] * c1) * inv_det,
            (-m[3][0] * s5 + m[3][2] * s2 - m[3][3] * s1) * inv_det,
            (m[2][0] * s5 - m[2][2] * s2 + m[2][3] * s1) * inv_det,
        ],
        [
            (m[1][0] * c4 - m[1][1] * c2 + m[1][3] * c0) * inv_det,
            (-m[0][0] * c4 + m[0][1] * c2 - m[0][3] * c0) * inv_det,
            (m[3][0] * s4 - m[3][1] * s2 + m[3][3] * s0) * inv_det,
            (-m[2][0] * s4 + m[2][1] * s2 - m[2][3] * s0) * inv_det,
        ],
        [
            (-m[1][0] * c3 + m[1][1] * c1 - m[1][2] * c0) * inv_det,
            (m[0][0] * c3 - m[0][1] * c1 + m[0][2] * c0) * inv_det,
            (-m[3][0] * s3 + m[3][1] * s1 - m[3][2] * s0) * inv_det,
            (m[2][0] * s3 - m[2][1] * s1 + m[2][2] * s0) * inv_det,
        ],
    ])
}

/// Build an off-center left-handed orthographic projection matrix for `[0, 1]` D3D depth
/// (`D3DXMatrixOrthoOffCenterLH`) using X-Ray row-vector layout.
pub fn matrix_ortho_off_center_lh(
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    min_z: f32,
    max_z: f32,
) -> XRayMatrix {
    let w = max_x - min_x;
    let h = max_y - min_y;
    let d = max_z - min_z;
    let sx = if w.abs() > 1e-6 { 2.0 / w } else { 1.0 };
    let sy = if h.abs() > 1e-6 { 2.0 / h } else { 1.0 };
    let sz = if d.abs() > 1e-6 { 1.0 / d } else { 1.0 };
    let tx = if w.abs() > 1e-6 {
        (min_x + max_x) / (min_x - max_x)
    } else {
        0.0
    };
    let ty = if h.abs() > 1e-6 {
        (min_y + max_y) / (min_y - max_y)
    } else {
        0.0
    };
    let tz = if d.abs() > 1e-6 {
        min_z / (min_z - max_z)
    } else {
        0.0
    };

    [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [tx, ty, tz, 1.0],
    ]
}

/// Construct a pure translation `XRayMatrix` (`Fmatrix::translate`).
#[inline]
pub fn matrix_translate(offset: [f32; 3]) -> XRayMatrix {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [offset[0], offset[1], offset[2], 1.0],
    ]
}

/// Fixed-size directional light cuboid clipper matching `FixedConvexVolume` (`r3_R_sun_support.h:9-279`).
#[derive(Debug, Clone)]
pub struct FixedConvexVolume {
    pub view_frustum_rays: Vec<SunRay>,
    pub view_ray: SunRay,
    pub light_ray: SunRay,
    pub light_cuboid_points: [[f32; 3]; 8],
    pub light_cuboid_polys: [([usize; 4], Plane); 4],
}

impl FixedConvexVolume {
    pub fn from_cull_xform_inv(
        cull_xform_inv: &XRayMatrix,
        view_frustum_rays: Vec<SunRay>,
        view_ray: SunRay,
        light_ray: SunRay,
    ) -> Self {
        let mut points = [[0.0f32; 3]; 8];
        for (i, corner) in SUN_FRUSTUM_CORNERS.iter().enumerate() {
            points[i] =
                matrix_transform_projected(cull_xform_inv, *corner).unwrap_or([0.0, 0.0, 0.0]);
        }
        let default_plane = Plane {
            n: [0.0, 1.0, 0.0],
            d: 0.0,
        };
        let mut polys = [([0usize; 4], default_plane); 4];
        for plane_idx in 0..4 {
            polys[plane_idx].0 = SUN_FRUSTUM_FACETABLE[plane_idx];
        }

        let mut volume = Self {
            view_frustum_rays,
            view_ray,
            light_ray,
            light_cuboid_points: points,
            light_cuboid_polys: polys,
        };
        volume.compute_planes();
        volume
    }

    pub fn compute_planes(&mut self) {
        for poly in &mut self.light_cuboid_polys {
            poly.1 = Plane::from_points(
                self.light_cuboid_points[poly.0[0]],
                self.light_cuboid_points[poly.0[2]],
                self.light_cuboid_points[poly.0[1]],
            );
        }
    }

    fn translate_light_model(&mut self, translate: [f32; 3]) {
        for poly in &mut self.light_cuboid_polys {
            poly.1.d -= dot3(translate, poly.1.n);
        }
    }

    fn check_cull_plane_valid(&self, plane: &Plane, mad_factor: f32) -> Option<f32> {
        let mut valid = false;
        let mut orient = 0.0f32;
        let mut oriented = false;

        for ray in &self.view_frustum_rays {
            let tmp_pt = [
                ray.origin[0] + ray.dir[0] * mad_factor,
                ray.origin[1] + ray.dir[1] * mad_factor,
                ray.origin[2] + ray.dir[2] * mad_factor,
            ];
            let tmp_dist = plane.classify(tmp_pt);
            if tmp_dist.abs() < EPS_L {
                continue;
            }
            if !oriented {
                orient = if tmp_dist > 0.0 { 1.0 } else { -1.0 };
                valid = true;
                oriented = true;
                continue;
            }
            if (tmp_dist < 0.0 && orient < 0.0) || (tmp_dist > 0.0 && orient > 0.0) {
                continue;
            }
            valid = false;
            break;
        }

        if valid {
            Some(orient)
        } else {
            None
        }
    }

    /// Compute shadow-caster culling planes, aligned light shift, and advance `view_frustum_rays`
    /// to the exit boundary of this cascade (`r3_R_sun_support.h:56-237`).
    pub fn compute_caster_model_fixed(
        &mut self,
        map_size: f32,
        clip_by_view_near: bool,
    ) -> (Vec<Plane>, [f32; 3]) {
        let mut dest = Vec::new();
        let mut translation = [0.0f32; 3];

        if (1.0 - dot3(self.view_ray.dir, self.light_ray.dir).abs()).abs() < EPS_S {
            return (dest, translation);
        }

        self.compute_planes();

        let mut align_planes = [0usize; 2];
        let mut align_count = 0usize;
        for (i, (_, plane)) in self.light_cuboid_polys.iter().enumerate() {
            let tmp_dot = dot3(self.view_ray.dir, plane.n);
            if tmp_dot <= EPS_L {
                continue;
            }
            align_planes[align_count] = i;
            align_count += 1;
            if align_count == 2 {
                break;
            }
        }

        let mut align_vector = [0.0f32; 3];
        for &p in &align_planes[..align_count] {
            let mut min_dist = 10_000.0f32;
            for ray in &self.view_frustum_rays {
                let d = self.light_cuboid_polys[p].1.classify(ray.origin);
                min_dist = min_dist.min(d);
            }
            let n = self.light_cuboid_polys[p].1.n;
            align_vector[0] += n[0] * min_dist;
            align_vector[1] += n[1] * min_dist;
            align_vector[2] += n[2] * min_dist;
        }

        for k in 0..3 {
            translation[k] += align_vector[k];
            self.light_ray.origin[k] += align_vector[k];
        }

        align_vector = [0.0; 3];
        for &p in &align_planes[..align_count] {
            let plane_n = self.light_cuboid_polys[p].1.n;
            let mut max_mag = 0.0f32;
            for ray in &self.view_frustum_rays {
                let plane_dot_ray = dot3(ray.dir, plane_n);
                if plane_dot_ray < 0.0 {
                    let per_plane_view = cross3(plane_n, self.view_ray.dir);
                    let per_view_to_plane = cross3(per_plane_view, self.view_ray.dir);
                    let denom = dot3(ray.dir, per_view_to_plane);
                    if denom.abs() > EPS_S {
                        let tmp_mag = -plane_dot_ray / denom;
                        if tmp_mag > max_mag {
                            max_mag = tmp_mag;
                        }
                    }
                }
            }
            if max_mag.abs() < EPS_L {
                continue;
            }
            let dist = -dot3(plane_n, translation);
            let scale = dist * max_mag;
            align_vector[0] += plane_n[0] * scale;
            align_vector[1] += plane_n[1] * scale;
            align_vector[2] += plane_n[2] * scale;
        }

        for k in 0..3 {
            translation[k] += align_vector[k];
            self.light_ray.origin[k] += align_vector[k];
        }
        self.translate_light_model(translation);

        // Side culling planes from view-frustum ray edges extruded by light direction.
        for ray in &self.view_frustum_rays {
            let tmp_vec = cross3(ray.dir, self.light_ray.dir);
            if dot3(tmp_vec, tmp_vec) < 1e-12 {
                continue;
            }
            let mut plane = Plane::from_point_normal(ray.origin, tmp_vec);
            if let Some(sign) = self.check_cull_plane_valid(&plane, 5.0) {
                plane.n[0] *= -sign;
                plane.n[1] *= -sign;
                plane.n[2] *= -sign;
                plane.d *= -sign;
                dest.push(plane);
            }
        }

        // Near view cap plane.
        if clip_by_view_near && dot3(self.view_ray.dir, self.light_ray.dir).abs() < 0.8 {
            let perp_light_view = cross3(self.view_ray.dir, self.light_ray.dir);
            let perp_light_to_view = cross3(perp_light_view, self.light_ray.dir);
            if dot3(perp_light_to_view, perp_light_to_view) > 1e-12 {
                let mut plane = Plane::from_point_normal(self.view_ray.origin, perp_light_to_view);
                let mut max_dist = -1000.0f32;
                for ray in &self.view_frustum_rays {
                    max_dist = max_dist.max(plane.classify(ray.origin));
                }
                for ray in &self.view_frustum_rays {
                    let p = [
                        ray.origin[0] + ray.dir[0] * 5.0,
                        ray.origin[1] + ray.dir[1] * 5.0,
                        ray.origin[2] + ray.dir[2] * 5.0,
                    ];
                    if plane.classify(p) > max_dist {
                        max_dist = 0.0;
                        break;
                    }
                }
                if max_dist > -1000.0 {
                    plane.d += max_dist;
                    dest.push(plane);
                }
            }
        }

        // Outward-facing cuboid side planes.
        for (_, poly_plane) in &self.light_cuboid_polys {
            dest.push(Plane {
                n: [-poly_plane.n[0], -poly_plane.n[1], -poly_plane.n[2]],
                d: -poly_plane.d,
            });
        }

        // Advance frustum rays to cascade exit boundary for the subsequent cascade.
        for ray in &mut self.view_frustum_rays {
            let mut min_dist = 2.0 * map_size;
            for (_, poly_plane) in &self.light_cuboid_polys {
                let d_dot_n = dot3(poly_plane.n, ray.dir);
                let dist = if d_dot_n > -0.1 {
                    map_size
                } else {
                    let numer = -poly_plane.classify(ray.origin);
                    let denom = d_dot_n;
                    if denom.abs() > 1e-6 {
                        numer / denom
                    } else {
                        map_size
                    }
                };
                if dist > EPS_L && dist < min_dist {
                    min_dist = dist;
                }
            }
            ray.origin[0] += ray.dir[0] * min_dist;
            ray.origin[1] += ray.dir[1] * min_dist;
            ray.origin[2] += ray.dir[2] * min_dist;
        }

        (dest, translation)
    }
}

#[derive(Debug, Clone)]
struct DumbPoly {
    points: Vec<usize>,
    plane_n: [f32; 3],
    plane_d: f32,
}

impl DumbPoly {
    #[inline]
    fn classify(&self, p: [f32; 3]) -> f32 {
        dot3(self.plane_n, p) + self.plane_d
    }
}

#[derive(Debug, Clone, Copy)]
struct DumbEdge {
    p0: usize,
    p1: usize,
    counter: i32,
}

impl DumbEdge {
    fn new(a: usize, b: usize, counter: i32) -> Self {
        if a > b {
            Self {
                p0: b,
                p1: a,
                counter,
            }
        } else {
            Self {
                p0: a,
                p1: b,
                counter,
            }
        }
    }
}

/// Directional infinite shadow caster hull extruded from a camera frustum
/// (`DumbConvexVolume`, `r3_R_sun_support.h:287-490`).
#[derive(Debug, Clone)]
pub struct DumbConvexVolume {
    pub points: Vec<[f32; 3]>,
    polys: Vec<DumbPoly>,
}

impl DumbConvexVolume {
    /// Build base frustum polyhedron from an inverse view-projection matrix (`fullxform_inv`).
    pub fn from_frustum_inv(full_xform_inv: &XRayMatrix) -> Self {
        let mut points = Vec::with_capacity(9);
        for &corner in &SUN_FRUSTUM_CORNERS {
            points.push(matrix_transform_projected(full_xform_inv, corner).unwrap_or([0.0; 3]));
        }
        let mut polys = Vec::with_capacity(6);
        for face in &SUN_FRUSTUM_FACETABLE {
            polys.push(DumbPoly {
                points: face.to_vec(),
                plane_n: [0.0, 1.0, 0.0],
                plane_d: 0.0,
            });
        }
        Self { points, polys }
    }

    fn compute_planes(&mut self) {
        let mut i = 0usize;
        while i < self.polys.len() {
            let (p0, p1, p2) = (
                self.points[self.polys[i].points[0]],
                self.points[self.polys[i].points[1]],
                self.points[self.polys[i].points[2]],
            );
            let t1 = [p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]];
            let t2 = [p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2]];
            let mut n = cross3(t1, t2);
            let mut len = dot3(n, n).sqrt();
            if len <= f32::MIN_POSITIVE {
                if self.polys[i].points.len() > 3 {
                    let p3 = self.points[self.polys[i].points[3]];
                    let t2_alt = [p0[0] - p3[0], p0[1] - p3[1], p0[2] - p3[2]];
                    n = cross3(t1, t2_alt);
                    len = dot3(n, n).sqrt();
                }
                if len <= f32::MIN_POSITIVE {
                    self.polys.remove(i);
                    continue;
                }
            }
            let inv = 1.0 / len;
            n[0] *= inv;
            n[1] *= inv;
            n[2] *= inv;
            let d = -dot3(n, p0);
            self.polys[i].plane_n = n;
            self.polys[i].plane_d = d;
            i += 1;
        }
    }

    /// Extrude base convex hull against `light_direction` and return outward culling planes.
    pub fn compute_caster_model(&mut self, light_direction: [f32; 3]) -> Vec<Plane> {
        if self.points.is_empty() {
            return Vec::new();
        }
        let mut cog = [0.0f32; 3];
        for p in &self.points {
            cog[0] += p[0];
            cog[1] += p[1];
            cog[2] += p[2];
        }
        let n_pts = self.points.len() as f32;
        cog[0] /= n_pts;
        cog[1] /= n_pts;
        cog[2] /= n_pts;

        self.compute_planes();
        for poly in &mut self.polys {
            if poly.classify(cog) > 0.0 {
                poly.points.reverse();
            }
        }

        self.compute_planes();
        let mut edges: Vec<DumbEdge> = Vec::new();
        let mut it = 0usize;
        while it < self.polys.len() {
            let marker = if dot3(self.polys[it].plane_n, light_direction) <= 0.0 {
                -1
            } else {
                1
            };
            let pcount = self.polys[it].points.len();
            for p in 0..pcount {
                let e = DumbEdge::new(
                    self.polys[it].points[p],
                    self.polys[it].points[(p + 1) % pcount],
                    marker,
                );
                if let Some(existing) = edges.iter_mut().find(|x| x.p0 == e.p0 && x.p1 == e.p1) {
                    existing.counter += marker;
                } else {
                    edges.push(e);
                }
            }
            if marker < 0 {
                self.polys.remove(it);
            } else {
                it += 1;
            }
        }

        for edge in edges {
            if edge.counter != 0 {
                continue;
            }
            let a = self.points[edge.p0];
            let b = self.points[edge.p1];
            self.points.push([
                a[0] - light_direction[0],
                a[1] - light_direction[1],
                a[2] - light_direction[2],
            ]);
            self.points.push([
                b[0] - light_direction[0],
                b[1] - light_direction[1],
                b[2] - light_direction[2],
            ]);
            let pend = self.points.len();
            self.polys.push(DumbPoly {
                points: vec![edge.p0, edge.p1, pend - 1, pend - 2],
                plane_n: [0.0, 1.0, 0.0],
                plane_d: 0.0,
            });
        }

        self.compute_planes();
        for poly in &mut self.polys {
            if poly.classify(cog) > 0.0 {
                poly.points.reverse();
            }
        }
        self.compute_planes();

        self.polys
            .iter()
            .map(|p| Plane {
                n: p.plane_n,
                d: p.plane_d,
            })
            .collect()
    }
}

/// Single directional sun cascade specification (`sun::cascade`).
#[derive(Debug, Clone, PartialEq)]
pub struct SunCascadeSpec {
    pub size: f32,
    pub bias: f32,
    pub reset_chain: bool,
}

/// 3-cascade CSM configuration matching `CRender::init_cacades` (`r2_R_sun.cpp:1088-1111`).
#[derive(Debug, Clone, PartialEq)]
pub struct SunCascadeConfig {
    pub cascades: Vec<SunCascadeSpec>,
}

impl Default for SunCascadeConfig {
    fn default() -> Self {
        Self::from_sizes([20.0, 40.0, 160.0], -0.0000025)
    }
}

impl SunCascadeConfig {
    pub fn from_sizes(sizes: [f32; 3], bias_per_meter: f32) -> Self {
        Self {
            cascades: vec![
                SunCascadeSpec {
                    size: sizes[0],
                    bias: sizes[0] * bias_per_meter,
                    reset_chain: true,
                },
                SunCascadeSpec {
                    size: sizes[1],
                    bias: sizes[1] * bias_per_meter,
                    reset_chain: false,
                },
                SunCascadeSpec {
                    size: sizes[2],
                    bias: sizes[2] * bias_per_meter,
                    reset_chain: false,
                },
            ],
        }
    }
}

/// Computed per-cascade shadow transform & spatial culling volume.
#[derive(Debug, Clone, PartialEq)]
pub struct SunCascadeResult {
    pub view_proj: XRayMatrix,
    pub cull_frustum: ClipFrustum,
    pub cull_cop: [f32; 3],
    pub bias: f32,
    pub size: f32,
}

/// Snap an orthographic shadow view-projection matrix onto the shadow-map texel grid to prevent
/// cascade swimming (`r2_R_sun.cpp:1310-1344`).
pub fn snap_sun_cascade_to_texels(
    cull_xform: &XRayMatrix,
    cam_pos: [f32; 3],
    light_xz_shift: [f32; 3],
    smap_size: u32,
) -> XRayMatrix {
    let view_dim = smap_size.max(1) as f32;
    let m_viewport: XRayMatrix = [
        [view_dim * 0.5, 0.0, 0.0, 0.0],
        [0.0, -view_dim * 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [view_dim * 0.5, view_dim * 0.5, 0.0, 1.0],
    ];
    let m_viewport_inv = matrix_invert_44(&m_viewport).unwrap_or(XRAY_IDENTITY_MATRIX);
    let cull_xform_inv = matrix_invert_44(cull_xform).unwrap_or(XRAY_IDENTITY_MATRIX);

    const ALIGN_AIM_STEP: f32 = 4.0;
    const ALIGN_GRANULARITY: f32 = 4.0;
    let snapped_cam = [
        ((cam_pos[0] / ALIGN_AIM_STEP).floor() + ALIGN_AIM_STEP * 0.5) * ALIGN_AIM_STEP,
        ((cam_pos[1] / ALIGN_AIM_STEP).floor() + ALIGN_AIM_STEP * 0.5) * ALIGN_AIM_STEP,
        ((cam_pos[2] / ALIGN_AIM_STEP).floor() + ALIGN_AIM_STEP * 0.5) * ALIGN_AIM_STEP,
    ];

    let cam_ndc = matrix_transform_projected(cull_xform, snapped_cam).unwrap_or([0.0; 3]);
    let mut cam_pixel = matrix_transform_projected(&m_viewport, cam_ndc).unwrap_or([0.0; 3]);

    let shift_view = matrix_transform_dir(cull_xform, light_xz_shift);
    let shift_px = matrix_transform_dir(&m_viewport, shift_view);
    let shift_snap = [
        if shift_px[0] > 0.0 {
            ALIGN_GRANULARITY
        } else {
            -ALIGN_GRANULARITY
        },
        if shift_px[1] > 0.0 {
            ALIGN_GRANULARITY
        } else {
            -ALIGN_GRANULARITY
        },
        0.0,
    ];

    cam_pixel[0] = (cam_pixel[0] / ALIGN_GRANULARITY - (cam_pixel[0] / ALIGN_GRANULARITY).floor())
        * ALIGN_GRANULARITY
        - shift_snap[0];
    cam_pixel[1] = (cam_pixel[1] / ALIGN_GRANULARITY - (cam_pixel[1] / ALIGN_GRANULARITY).floor())
        * ALIGN_GRANULARITY
        - shift_snap[1];
    cam_pixel[2] = 0.0;

    let inv_vp = matrix_transform_dir(&m_viewport_inv, cam_pixel);
    let inv_world = matrix_transform_dir(&cull_xform_inv, inv_vp);
    let diff = [-inv_world[0], -inv_world[1], -inv_world[2]];
    let adjust = matrix_translate(diff);
    matrix_mul_44(cull_xform, &adjust)
}

/// Compute cascaded directional sun shadow matrices & frustum culling planes for all cascades
/// (`CRender::render_sun_cascades`, `r2_R_sun.cpp:1113-1356`).
#[allow(clippy::too_many_arguments)]
pub fn compute_sun_cascades(
    config: &SunCascadeConfig,
    cam_pos: [f32; 3],
    cam_dir: [f32; 3],
    view: &XRayMatrix,
    proj: &XRayMatrix,
    sun_pos: [f32; 3],
    sun_dir: [f32; 3],
    smap_size: u32,
    sunshafts_active: bool,
) -> Vec<SunCascadeResult> {
    if config.cascades.is_empty() {
        return Vec::new();
    }
    let full = matrix_mul_44(proj, view);
    let full_inv = match matrix_invert_44(&full) {
        Some(inv) => inv,
        None => return Vec::new(),
    };

    let l_dir = normalize3_or(sun_dir, [0.0, -1.0, 0.0]);
    let mut l_right = [1.0f32, 0.0, 0.0];
    if dot3(l_right, l_dir).abs() > 0.99 {
        l_right = [0.0, 0.0, 1.0];
    }
    let l_up = normalize3_or(cross3(l_dir, l_right), [0.0, 1.0, 0.0]);
    let view_ray = SunRay {
        origin: cam_pos,
        dir: normalize3_or(cam_dir, [0.0, 0.0, 1.0]),
    };

    let build_initial_rays = || -> Vec<SunRay> {
        let mut rays = Vec::with_capacity(4);
        for p in 0..4 {
            let near_p =
                matrix_transform_projected(&full_inv, SUN_FRUSTUM_CORNERS[SUN_FRUSTUM_FACETABLE[4][p]])
                    .unwrap_or(cam_pos);
            let far_p =
                matrix_transform_projected(&full_inv, SUN_FRUSTUM_CORNERS[SUN_FRUSTUM_FACETABLE[5][p]])
                    .unwrap_or(near_p);
            let edge = normalize3_or(
                [far_p[0] - near_p[0], far_p[1] - near_p[1], far_p[2] - near_p[2]],
                view_ray.dir,
            );
            rays.push(SunRay {
                origin: near_p,
                dir: edge,
            });
        }
        rays
    };

    let light_top_plane = Plane::from_point_normal(sun_pos, l_dir);
    let dist_top = light_top_plane.classify(cam_pos);

    let mut current_rays = build_initial_rays();
    let mut results = Vec::with_capacity(config.cascades.len());
    let last_idx = config.cascades.len() - 1;

    for (idx, cascade) in config.cascades.iter().enumerate() {
        let reset = cascade.reset_chain || (sunshafts_active && idx == last_idx);
        let rays = if idx == 0 || reset {
            build_initial_rays()
        } else {
            current_rays.clone()
        };

        let mdir_view = matrix_build_camera_dir(sun_pos, l_dir, l_up);
        let half = cascade.size * 0.5;
        let mdir_project =
            matrix_ortho_off_center_lh(-half, half, -half, half, 0.1, (dist_top + cascade.size).max(1.0));
        let base_cull = matrix_mul_44(&mdir_project, &mdir_view);
        let base_inv = matrix_invert_44(&base_cull).unwrap_or(XRAY_IDENTITY_MATRIX);

        let mut cuboid = FixedConvexVolume::from_cull_xform_inv(
            &base_inv,
            rays,
            view_ray,
            SunRay {
                origin: sun_pos,
                dir: l_dir,
            },
        );
        let (cull_planes, light_xz_shift) =
            cuboid.compute_caster_model_fixed(cascade.size, reset);
        current_rays = cuboid.view_frustum_rays;

        let cam_shifted = [
            sun_pos[0] + light_xz_shift[0],
            sun_pos[1] + light_xz_shift[1],
            sun_pos[2] + light_xz_shift[2],
        ];
        let shifted_view = matrix_build_camera_dir(cam_shifted, l_dir, l_up);
        let shifted_xform = matrix_mul_44(&mdir_project, &shifted_view);
        let snapped = snap_sun_cascade_to_texels(&shifted_xform, cam_pos, light_xz_shift, smap_size);

        let cull_cop = [
            cam_pos[0] - l_dir[0] * TWEAK_COP_INITIAL_OFFS,
            cam_pos[1] - l_dir[1] * TWEAK_COP_INITIAL_OFFS,
            cam_pos[2] - l_dir[2] * TWEAK_COP_INITIAL_OFFS,
        ];

        results.push(SunCascadeResult {
            view_proj: snapped,
            cull_frustum: ClipFrustum {
                planes: cull_planes,
            },
            cull_cop,
            bias: cascade.bias,
            size: cascade.size,
        });
    }

    results
}

/// Depth range/bias & TSM parameters for `accum_direct` shadow projection
/// (`xrRender_console.cpp:229-239`, `r4_rendertarget_accum_direct.cpp:165-196`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunCascadeBiasConfig {
    pub near_scale: f32,
    pub near_bias: f32,
    pub far_scale: f32,
    pub far_bias: f32,
    pub tsm_bias: f32,
}

impl Default for SunCascadeBiasConfig {
    fn default() -> Self {
        Self {
            near_scale: 1.0,
            near_bias: 0.00001,
            far_scale: 1.0,
            far_bias: -0.00002,
            tsm_bias: -0.01,
        }
    }
}

/// Build the view-space -> sun shadow-map UV+depth transform (`m_shadow`) for `accum_direct`
/// (`r4_rendertarget_accum_direct.cpp:165-196`).
pub fn compute_sun_shadow_matrix(
    is_near_cascade: bool,
    apply_tsm_bias: bool,
    sun_combine: &XRayMatrix,
    inv_view: &XRayMatrix,
    sun_dir_view: [f32; 3],
    cfg: &SunCascadeBiasConfig,
) -> XRayMatrix {
    let f_range = if is_near_cascade {
        cfg.near_scale
    } else {
        cfg.far_scale
    };
    let f_bias = if is_near_cascade {
        -cfg.near_bias
    } else {
        cfg.far_bias
    };
    let m_texel_adjust: XRayMatrix = [
        [0.5, 0.0, 0.0, 0.0],
        [0.0, -0.5, 0.0, 0.0],
        [0.0, 0.0, f_range, 0.0],
        [0.5, 0.5, f_bias, 1.0],
    ];
    let xf_project = matrix_mul_44(&m_texel_adjust, sun_combine);
    let mut m_shadow = matrix_mul_44(&xf_project, inv_view);
    if apply_tsm_bias {
        let dir = normalize3_or(sun_dir_view, [0.0, 0.0, 1.0]);
        let bias_t = matrix_translate([
            dir[0] * cfg.tsm_bias,
            dir[1] * cfg.tsm_bias,
            dir[2] * cfg.tsm_bias,
        ]);
        m_shadow = matrix_mul_44(&m_shadow, &bias_t);
    }
    m_shadow
}

/// Persistent scrolling UV transform generator for directional cloud shadows (`m_sunmask`,
/// `r4_rendertarget_accum_direct.cpp:200-222`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CloudsShadowState {
    pub w_shift: f32,
}

impl CloudsShadowState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance cloud shadow scrolling by `dt` seconds and return the view-space `m_sunmask` matrix.
    pub fn step_and_matrix(
        &mut self,
        dt: f32,
        sun_dir_world: [f32; 3],
        wind_direction_rad: f32,
        wind_velocity: f32,
        inv_view: &XRayMatrix,
    ) -> XRayMatrix {
        let w_speed = (wind_velocity * 0.001).clamp(0.1, 1.0);
        self.w_shift -= 0.005 * w_speed * dt.max(0.0);

        let normal = dir_from_hp(-wind_direction_rad, 0.0);
        let dir = normalize3_or(sun_dir_world, [0.0, -1.0, 0.0]);
        let m_xform = matrix_build_camera_dir([0.0, 0.0, 0.0], dir, normal);
        let local_normal = normalize3_or(matrix_transform_dir(&m_xform, normal), [0.0, 1.0, 0.0]);

        let mut m_clouds_shadow = matrix_mul_44(&m_xform, inv_view);
        let m_scale = matrix_scale(0.002, 0.002, 1.0);
        m_clouds_shadow = matrix_mul_44(&m_scale, &m_clouds_shadow);
        let m_trans = matrix_translate([
            local_normal[0] * self.w_shift,
            local_normal[1] * self.w_shift,
            local_normal[2] * self.w_shift,
        ]);
        matrix_mul_44(&m_trans, &m_clouds_shadow)
    }
}

/// Dynamic wet-surface rain shadow settings (`xrRender_console.cpp:466, 496-498`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RainWetSurfConfig {
    pub near: f32,
    pub far: f32,
    pub sm_res: u32,
    pub ssfx_gloss_method: i32,
    pub smap_size: u32,
}

impl Default for RainWetSurfConfig {
    fn default() -> Self {
        Self {
            near: 10.0,
            far: 30.0,
            sm_res: 256,
            ssfx_gloss_method: 0,
            smap_size: 2048,
        }
    }
}

/// SSFX rain streak & splash parameters (`ps_ssfx_rain_*`, `xrRender_console.cpp:476-479`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsfxRainConfig {
    pub rain_1: [f32; 4],
    pub rain_2: [f32; 4],
    pub rain_3: [f32; 4],
    pub drops_setup: [f32; 4],
}

impl Default for SsfxRainConfig {
    fn default() -> Self {
        Self {
            rain_1: [2.0, 0.1, 0.6, 2.0],
            rain_2: [0.5, 0.1, 1.0, 0.5],
            rain_3: [0.5, 1.0, 0.0, 0.0],
            drops_setup: [2500.0, 15.0, 0.0, 0.0],
        }
    }
}

/// Viewport dimensions `[w / 8.0, h / 8.0]` for `phase_ssfx_rain` (`r4_rendertarget_phase_rain.cpp:22`).
#[inline]
pub fn ssfx_rain_pass_viewport(screen_w: u32, screen_h: u32) -> [f32; 2] {
    [screen_w.max(1) as f32 / 8.0, screen_h.max(1) as f32 / 8.0]
}

/// Dynamic rain shadow frustum, pixel-snapped SMAP transform, and `draw_rain` GPU parameters
/// (`r4_R_rain.cpp:30-245`, `r4_rendertarget_draw_rain.cpp:18-170, 253-259`).
#[derive(Debug, Clone, PartialEq)]
pub struct RainShadowResult {
    pub rain_factor: f32,
    pub bounding_sphere_radius: f32,
    pub cull_frustum: ClipFrustum,
    pub cull_cop: [f32; 3],
    pub cull_xform: XRayMatrix,
    pub smap_viewport_limit: u32,
    pub m_shadow: XRayMatrix,
    pub m_sunmask: XRayMatrix,
    pub l_dynamic_dir: [f32; 4],
    pub world_x: [f32; 4],
    pub world_z: [f32; 4],
    pub rain_falloff: [f32; 4],
    pub clip_d_z: f32,
}

/// Compute dynamic wet-surface rain shadow setup & accumulation shader inputs matching
/// `CRender::render_rain` and `CRenderTarget::draw_rain`. Returns `None` if `rain_factor < EPS_L`.
#[allow(clippy::too_many_arguments)]
pub fn compute_rain_shadow_setup(
    rain_density: f32,
    wetness_factor: f32,
    cam_pos: [f32; 3],
    cam_dir: [f32; 3],
    fov_deg: f32,
    aspect: f32,
    viewport_near: f32,
    view: &XRayMatrix,
    inv_view: &XRayMatrix,
    full_transform: &XRayMatrix,
    cfg: &RainWetSurfConfig,
) -> Option<RainShadowResult> {
    const SOURCE_OFFSET: f32 = 10_000.0;
    const TWEAK_RAIN_COP_OFFS: f32 = 1200.0;
    const TWEAK_RAIN_ORTHO_OFFS: f32 = 1000.0;

    let rain_factor = if cfg.ssfx_gloss_method == 0 {
        rain_density
    } else {
        wetness_factor
    };
    if !rain_factor.is_finite() || rain_factor < EPS_L {
        return None;
    }

    let rain_far = cfg.far.max(viewport_near + 0.1);
    let cam_dir_norm = normalize3_or(cam_dir, [0.0, 0.0, 1.0]);
    let rain_dir = [0.0f32, -1.0, 0.0];
    let rain_pos = [cam_pos[0], cam_pos[1] + SOURCE_OFFSET, cam_pos[2]];

    let ex_project = matrix_build_projection(fov_deg.to_radians(), aspect, viewport_near, rain_far);
    let ex_full = matrix_mul_44(&ex_project, view);
    let ex_full_inv = matrix_invert_44(&ex_full)?;

    let a = (fov_deg.to_radians() * 0.5).tan();
    let c = ((fov_deg * aspect).to_radians() * 0.5).tan();
    let b_2 = rain_far * rain_far * (1.0 + a * a + c * c);
    let bounding_sphere_radius = b_2 / (2.0 * rain_far);

    let mut hull = DumbConvexVolume::from_frustum_inv(&ex_full_inv);
    let base_points: [[f32; 3]; 8] = std::array::from_fn(|i| hull.points[i]);
    let cull_planes = hull.compute_caster_model(rain_dir);
    let cull_frustum = ClipFrustum {
        planes: cull_planes,
    };

    let cull_cop = [
        cam_pos[0] - rain_dir[0] * TWEAK_RAIN_COP_OFFS + bounding_sphere_radius * cam_dir_norm[0],
        cam_pos[1] - rain_dir[1] * TWEAK_RAIN_COP_OFFS,
        cam_pos[2] - rain_dir[2] * TWEAK_RAIN_COP_OFFS + bounding_sphere_radius * cam_dir_norm[2],
    ];

    let l_right = [1.0f32, 0.0, 0.0];
    let l_up = normalize3_or(cross3(rain_dir, l_right), [0.0, 0.0, 1.0]);
    let mdir_view = matrix_build_camera_dir(rain_pos, rain_dir, l_up);

    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for &p in &base_points {
        let xf = matrix_transform_point(&mdir_view, p);
        min_z = min_z.min(xf[2]);
        max_z = max_z.max(xf[2]);
    }
    min_z -= EPS_L;
    let _ = max_z + EPS_L;

    let v_rect_x = bounding_sphere_radius * cam_dir_norm[0];
    let v_rect_z = bounding_sphere_radius * cam_dir_norm[2];
    let min_x = -bounding_sphere_radius + v_rect_x;
    let max_x = bounding_sphere_radius + v_rect_x;
    let min_y = -bounding_sphere_radius + v_rect_z;
    let max_y = bounding_sphere_radius + v_rect_z;

    let mdir_project = matrix_ortho_off_center_lh(
        min_x,
        max_x,
        min_y,
        max_y,
        min_z - TWEAK_RAIN_ORTHO_OFFS,
        min_z + 2.0 * TWEAK_RAIN_ORTHO_OFFS,
    );
    let mut cull_xform = matrix_mul_44(&mdir_project, &mdir_view);

    let limit = cfg.smap_size.min(cfg.sm_res).max(1);
    let smap_size_f = cfg.smap_size.max(1) as f32;
    let view_dim = limit as f32;
    let f_texel_offs = 0.5 / smap_size_f;
    let m_viewport: XRayMatrix = [
        [view_dim * 0.5, 0.0, 0.0, 0.0],
        [0.0, -view_dim * 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [
            view_dim * 0.5 + f_texel_offs,
            view_dim * 0.5 + f_texel_offs,
            0.0,
            1.0,
        ],
    ];
    let m_viewport_inv = matrix_invert_44(&m_viewport).unwrap_or(XRAY_IDENTITY_MATRIX);

    let cam_proj = matrix_transform_projected(&cull_xform, [0.0, 0.0, 0.0]).unwrap_or([0.0; 3]);
    let mut cam_pixel =
        matrix_transform_projected(&m_viewport, cam_proj).unwrap_or([0.0, 0.0, 0.0]);
    cam_pixel[0] = cam_pixel[0].floor();
    cam_pixel[1] = cam_pixel[1].floor();
    let cam_snapped = matrix_transform_projected(&m_viewport_inv, cam_pixel).unwrap_or(cam_proj);
    let diff = [
        cam_snapped[0] - cam_proj[0],
        cam_snapped[1] - cam_proj[1],
        cam_snapped[2] - cam_proj[2],
    ];
    let adjust = matrix_translate(diff);
    cull_xform = matrix_mul_44(&adjust, &cull_xform);

    // draw_rain shadow transform
    let f_bias = -0.0001f32;
    let view_dim_norm = limit as f32 / smap_size_f;
    let m_texel_adjust: XRayMatrix = [
        [view_dim_norm * 0.5, 0.0, 0.0, 0.0],
        [0.0, -view_dim_norm * 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [
            view_dim_norm * 0.5 + f_texel_offs,
            view_dim_norm * 0.5 + f_texel_offs,
            f_bias,
            1.0,
        ],
    ];
    let xf_project = matrix_mul_44(&m_texel_adjust, &cull_xform);
    let m_shadow = matrix_mul_44(&xf_project, inv_view);

    // draw_rain clouds shadow matrix (`m_xform = identity`, `w_shift = 0`, `scale(1,1,1)`)
    let normal = dir_from_hp(1.0, 0.0);
    let m_xform = XRAY_IDENTITY_MATRIX;
    let local_normal = normalize3_or(matrix_transform_dir(&m_xform, normal), [0.0, 1.0, 0.0]);
    let mut m_sunmask = matrix_mul_44(&m_xform, inv_view);
    let m_s = matrix_scale(1.0, 1.0, 1.0);
    m_sunmask = matrix_mul_44(&m_s, &m_sunmask);
    let m_t = matrix_translate([
        local_normal[0] * 0.0,
        local_normal[1] * 0.0,
        local_normal[2] * 0.0,
    ]);
    m_sunmask = matrix_mul_44(&m_t, &m_sunmask);

    let draw_rain_far = if cfg.ssfx_gloss_method == 0 {
        cfg.far
    } else {
        250.0
    };
    let center_pt = [
        cam_pos[0] + cam_dir_norm[0] * draw_rain_far,
        cam_pos[1] + cam_dir_norm[1] * draw_rain_far,
        cam_pos[2] + cam_dir_norm[2] * draw_rain_far,
    ];
    let clip_d_z = matrix_transform_projected(full_transform, center_pt)
        .map(|p| p[2])
        .unwrap_or(1.0);

    let l_dir_v = normalize3_or(matrix_transform_dir(view, rain_dir), [0.0, -1.0, 0.0]);
    let wx_v = normalize3_or(matrix_transform_dir(view, [1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
    let wz_v = normalize3_or(matrix_transform_dir(view, [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]);

    Some(RainShadowResult {
        rain_factor,
        bounding_sphere_radius,
        cull_frustum,
        cull_cop,
        cull_xform,
        smap_viewport_limit: limit,
        m_shadow,
        m_sunmask,
        l_dynamic_dir: [l_dir_v[0], l_dir_v[1], l_dir_v[2], 0.0],
        world_x: [wx_v[0], wx_v[1], wx_v[2], 0.0],
        world_z: [wz_v[0], wz_v[1], wz_v[2], 0.0],
        rain_falloff: [cfg.near, cfg.far, 0.0, 0.0],
        clip_d_z,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{matrix_build_camera_dir, matrix_build_projection};

    #[test]
    fn matrix_invert_44_roundtrips_perspective_and_ortho() {
        let view = matrix_build_camera_dir([5.0, 2.0, -3.0], [0.2, -0.1, 1.0], [0.0, 1.0, 0.0]);
        let proj = matrix_build_projection(67.5f32.to_radians(), 9.0 / 16.0, 0.2, 200.0);
        let vp = matrix_mul_44(&proj, &view);
        let vp_inv = matrix_invert_44(&vp).expect("perspective vp invertible");
        let prod = matrix_mul_44(&vp, &vp_inv);
        for (r, row) in prod.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-3,
                    "prod[{r}][{c}] = {val}",
                );
            }
        }

        let ortho = matrix_ortho_off_center_lh(-10.0, 10.0, -10.0, 10.0, 0.1, 200.0);
        let o_inv = matrix_invert_44(&ortho).expect("ortho invertible");
        let corner = matrix_transform_projected(&o_inv, [-1.0, 1.0, 1.0]).unwrap();
        assert!((corner[0] - (-10.0)).abs() < 1e-4);
        assert!((corner[1] - 10.0).abs() < 1e-4);
        assert!((corner[2] - 200.0).abs() < 1e-3);
    }

    #[test]
    fn dumb_and_fixed_convex_volumes_and_cascades_stay_finite() {
        let view = matrix_build_camera_dir([0.0, 1.7, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        let proj = matrix_build_projection(75.0f32.to_radians(), 9.0 / 16.0, 0.2, 300.0);
        let vp = matrix_mul_44(&proj, &view);
        let vp_inv = matrix_invert_44(&vp).unwrap();

        let mut dumb = DumbConvexVolume::from_frustum_inv(&vp_inv);
        let sun_dir = normalize3_or([0.4, -0.7, 0.5], [0.0, -1.0, 0.0]);
        let planes = dumb.compute_caster_model(sun_dir);
        assert!(planes.len() >= 4);
        assert!(planes.iter().all(|p| p.classify([0.0, 1.7, 10.0]) <= 0.1));

        let cfg = SunCascadeConfig::default();
        let cascades = compute_sun_cascades(
            &cfg,
            [0.0, 1.7, 0.0],
            [0.0, 0.0, 1.0],
            &view,
            &proj,
            [0.0, 100.0, 0.0],
            sun_dir,
            2048,
            false,
        );
        assert_eq!(cascades.len(), 3);
        assert_eq!(cascades[0].size, 20.0);
        assert_eq!(cascades[1].size, 40.0);
        assert_eq!(cascades[2].size, 160.0);
        for c in &cascades {
            assert!(!c.cull_frustum.planes.is_empty());
            for row in &c.view_proj {
                assert!(row.iter().all(|v| v.is_finite()));
            }
        }

        let inv_view = matrix_invert_44(&view).unwrap();
        let sun_dir_view = matrix_transform_dir(&view, sun_dir);
        let bias_cfg = SunCascadeBiasConfig::default();
        let near_mat = compute_sun_shadow_matrix(
            true,
            false,
            &cascades[0].view_proj,
            &inv_view,
            sun_dir_view,
            &bias_cfg,
        );
        let far_mat = compute_sun_shadow_matrix(
            false,
            true,
            &cascades[2].view_proj,
            &inv_view,
            sun_dir_view,
            &bias_cfg,
        );
        assert!(near_mat.iter().flatten().all(|v| v.is_finite()));
        assert!(far_mat.iter().flatten().all(|v| v.is_finite()));
        assert_ne!(near_mat, far_mat);

        let mut clouds = CloudsShadowState::new();
        let c0 = clouds.step_and_matrix(1.0, sun_dir, 0.5, 50.0, &inv_view);
        assert!((clouds.w_shift - (-0.005 * 0.1 * 1.0)).abs() < 1e-6);
        let c1 = clouds.step_and_matrix(2.0, sun_dir, 0.5, 500.0, &inv_view);
        assert!((clouds.w_shift - (-0.0005 - 0.005 * 0.5 * 2.0)).abs() < 1e-6);
        assert_ne!(c0[3][1], c1[3][1]);
    }

    #[test]
    fn rain_shadow_setup_and_ssfx_rain_defaults() {
        let view = matrix_build_camera_dir([10.0, 2.0, -5.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        let inv_view = matrix_invert_44(&view).unwrap();
        let proj = matrix_build_projection(67.5f32.to_radians(), 9.0 / 16.0, 0.2, 200.0);
        let full = matrix_mul_44(&proj, &view);

        let cfg = RainWetSurfConfig::default();
        assert!(compute_rain_shadow_setup(
            0.0,
            0.8,
            [10.0, 2.0, -5.0],
            [0.0, 0.0, 1.0],
            67.5,
            9.0 / 16.0,
            0.2,
            &view,
            &inv_view,
            &full,
            &cfg,
        )
        .is_none());

        let rain = compute_rain_shadow_setup(
            0.65,
            0.2,
            [10.0, 2.0, -5.0],
            [0.0, 0.0, 1.0],
            67.5,
            9.0 / 16.0,
            0.2,
            &view,
            &inv_view,
            &full,
            &cfg,
        )
        .expect("rain density active");
        assert!((rain.rain_factor - 0.65).abs() < 1e-6);
        assert_eq!(rain.smap_viewport_limit, 256);
        assert_eq!(rain.rain_falloff, [10.0, 30.0, 0.0, 0.0]);
        assert!(rain.bounding_sphere_radius > 15.0);
        assert!(!rain.cull_frustum.planes.is_empty());
        assert!(rain.m_shadow.iter().flatten().all(|v| v.is_finite()));
        assert!(rain.m_sunmask.iter().flatten().all(|v| v.is_finite()));

        // Verify origin pixel snap lands cleanly on integer texel boundaries.
        let view_dim = rain.smap_viewport_limit as f32;
        let texel_offs = 0.5 / cfg.smap_size as f32;
        let m_viewport: XRayMatrix = [
            [view_dim * 0.5, 0.0, 0.0, 0.0],
            [0.0, -view_dim * 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [
                view_dim * 0.5 + texel_offs,
                view_dim * 0.5 + texel_offs,
                0.0,
                1.0,
            ],
        ];
        let origin_proj = matrix_transform_projected(&rain.cull_xform, [0.0, 0.0, 0.0]).unwrap();
        let origin_px = matrix_transform_projected(&m_viewport, origin_proj).unwrap();
        assert!((origin_px[0] - origin_px[0].round()).abs() < 1e-3);
        assert!((origin_px[1] - origin_px[1].round()).abs() < 1e-3);

        // SSFX gloss method switches activation signal to `wetness_factor` and `draw_rain` clip to 250m.
        let ssfx_cfg = RainWetSurfConfig {
            ssfx_gloss_method: 1,
            ..cfg
        };
        let wet = compute_rain_shadow_setup(
            0.0,
            0.75,
            [10.0, 2.0, -5.0],
            [0.0, 0.0, 1.0],
            67.5,
            9.0 / 16.0,
            0.2,
            &view,
            &inv_view,
            &full,
            &ssfx_cfg,
        )
        .expect("wetness factor active");
        assert!((wet.rain_factor - 0.75).abs() < 1e-6);
        assert!(wet.clip_d_z > rain.clip_d_z);

        let ssfx_rain = SsfxRainConfig::default();
        assert_eq!(ssfx_rain.rain_1, [2.0, 0.1, 0.6, 2.0]);
        assert_eq!(ssfx_rain.drops_setup, [2500.0, 15.0, 0.0, 0.0]);
        assert_eq!(ssfx_rain_pass_viewport(1920, 1080), [240.0, 135.0]);
    }
}
