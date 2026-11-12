//! Pure Rust X-Ray `level.hom` Hierarchical Occlusion Mapping, Frustum & Software Rasterizer
//!
//! Replaces legacy C++ `CHOM`, `occRasterizer`, and `CFrustum` (`src/Layers/xrRender/HOM.cpp`,
//! `occRasterizer.cpp`, `occRasterizer_core.cpp`, `src/xrCDB/Frustum.cpp`):
//! - Parses `level.hom` Chunk 1 (`HOM_poly`: 40-byte vertex triple + two-sided flags)
//! - Welds vertices (`0.01m`) and computes triangle edge adjacency (`occTri::adjacent`)
//! - Implements `ClipFrustum` (`CreateFromMatrix`, AABB/sphere test, polygon clipping)
//! - Implements `OccRasterizer` (64x64 viewport with 2px guard border + 64->32->16->8 depth pyramid)

use std::collections::HashMap;

use crate::chunk::{ByteCursor, ChunkReader, FormatError};
use crate::ogf::{OgfBBox, XRayMatrix};

/// On-disk size of `HOM_poly` (`#pragma pack(push,4)`: `3 * Fvector + u32 flags`).
pub const HOM_POLY_SIZE: usize = 40;

pub const OCC_DIM_0: usize = 64;
pub const OCC_DIM_1: usize = OCC_DIM_0 / 2;
pub const OCC_DIM_2: usize = OCC_DIM_1 / 2;
pub const OCC_DIM_3: usize = OCC_DIM_2 / 2;
pub const OCC_BORDER: usize = 2;
pub const OCC_DIM: usize = OCC_DIM_0 + OCC_BORDER * 2;
pub const OCC_Q_S32: f32 = 0x4000_0000 as f32;

const EPS: f32 = 1e-5;

/// Frustum plane mask flags matching `FRUSTUM_P_*` (`src/xrCDB/Frustum.h`).
pub mod frustum_planes {
    pub const LEFT: u32 = 1 << 0;
    pub const RIGHT: u32 = 1 << 1;
    pub const TOP: u32 = 1 << 2;
    pub const BOTTOM: u32 = 1 << 3;
    pub const NEAR: u32 = 1 << 4;
    pub const FAR: u32 = 1 << 5;
    pub const LRTB: u32 = LEFT | RIGHT | TOP | BOTTOM;
    pub const ALL: u32 = LRTB | NEAR | FAR;
}

/// 3D plane `dot(n, p) + d = 0` (`Fplane` from `src/xrCore/_plane.h`).
///
/// For `ClipFrustum`, normals point outward so `classify(p) <= 0.0` is inside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub n: [f32; 3],
    pub d: f32,
}

impl Plane {
    /// Build a unit plane from 3 points matching `Fplane::build(v0, v1, v2)`.
    #[inline]
    pub fn from_points(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> Self {
        let e1 = [v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]];
        let e2 = [v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2]];
        let cx = e1[1] * e2[2] - e1[2] * e2[1];
        let cy = e1[2] * e2[0] - e1[0] * e2[2];
        let cz = e1[0] * e2[1] - e1[1] * e2[0];
        let len = (cx * cx + cy * cy + cz * cz).sqrt();
        if len <= 1e-12 {
            return Self {
                n: [0.0, 1.0, 0.0],
                d: -v0[1],
            };
        }
        let inv = 1.0 / len;
        let n = [cx * inv, cy * inv, cz * inv];
        let d = -(n[0] * v0[0] + n[1] * v0[1] + n[2] * v0[2]);
        Self { n, d }
    }

    #[inline]
    pub fn from_point_normal(point: [f32; 3], normal: [f32; 3]) -> Self {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let n = if len > 1e-12 {
            let inv = 1.0 / len;
            [normal[0] * inv, normal[1] * inv, normal[2] * inv]
        } else {
            [0.0, 1.0, 0.0]
        };
        let d = -(n[0] * point[0] + n[1] * point[1] + n[2] * point[2]);
        Self { n, d }
    }

    #[inline]
    pub fn classify(&self, p: [f32; 3]) -> f32 {
        self.n[0] * p[0] + self.n[1] * p[1] + self.n[2] * p[2] + self.d
    }

    #[inline]
    pub fn normalize(mut self) -> Self {
        let len = (self.n[0] * self.n[0] + self.n[1] * self.n[1] + self.n[2] * self.n[2]).sqrt();
        if len > 1e-12 {
            let inv = 1.0 / len;
            self.n[0] *= inv;
            self.n[1] *= inv;
            self.n[2] *= inv;
            self.d *= inv;
        }
        self
    }
}

/// Three-state frustum visibility result (`EFC_Visible`, `src/xrCDB/Frustum.h:16-22`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FrustumVisibility {
    None = 0,
    Partial = 1,
    Fully = 2,
}

/// View frustum with outward-facing unit planes (`CFrustum`).
#[derive(Debug, Clone, PartialEq)]
pub struct ClipFrustum {
    pub planes: Vec<Plane>,
}

impl ClipFrustum {
    /// Extract frustum planes from an X-Ray row-vector `mFullTransform` (`CFrustum::CreateFromMatrix`).
    pub fn from_view_proj(m: &XRayMatrix, mask: u32) -> Self {
        let mut planes = Vec::with_capacity(6);
        if (mask & frustum_planes::LEFT) != 0 {
            planes.push(
                Plane {
                    n: [-(m[0][3] + m[0][0]), -(m[1][3] + m[1][0]), -(m[2][3] + m[2][0])],
                    d: -(m[3][3] + m[3][0]),
                }
                .normalize(),
            );
        }
        if (mask & frustum_planes::RIGHT) != 0 {
            planes.push(
                Plane {
                    n: [-(m[0][3] - m[0][0]), -(m[1][3] - m[1][0]), -(m[2][3] - m[2][0])],
                    d: -(m[3][3] - m[3][0]),
                }
                .normalize(),
            );
        }
        if (mask & frustum_planes::TOP) != 0 {
            planes.push(
                Plane {
                    n: [-(m[0][3] - m[0][1]), -(m[1][3] - m[1][1]), -(m[2][3] - m[2][1])],
                    d: -(m[3][3] - m[3][1]),
                }
                .normalize(),
            );
        }
        if (mask & frustum_planes::BOTTOM) != 0 {
            planes.push(
                Plane {
                    n: [-(m[0][3] + m[0][1]), -(m[1][3] + m[1][1]), -(m[2][3] + m[2][1])],
                    d: -(m[3][3] + m[3][1]),
                }
                .normalize(),
            );
        }
        if (mask & frustum_planes::FAR) != 0 {
            planes.push(
                Plane {
                    n: [-(m[0][3] - m[0][2]), -(m[1][3] - m[1][2]), -(m[2][3] - m[2][2])],
                    d: -(m[3][3] - m[3][2]),
                }
                .normalize(),
            );
        }
        if (mask & frustum_planes::NEAR) != 0 {
            planes.push(
                Plane {
                    n: [-(m[0][3] + m[0][2]), -(m[1][3] + m[1][2]), -(m[2][3] + m[2][2])],
                    d: -(m[3][3] + m[3][2]),
                }
                .normalize(),
            );
        }
        Self { planes }
    }

    /// Variant using DirectX `[0..1]` near plane (`z >= 0` -> `-col2 <= 0`) when requested.
    pub fn from_view_proj_dx(m: &XRayMatrix, mask: u32) -> Self {
        let mut f = Self::from_view_proj(m, mask & !frustum_planes::NEAR);
        if (mask & frustum_planes::NEAR) != 0 {
            f.planes.push(
                Plane {
                    n: [-m[0][2], -m[1][2], -m[2][2]],
                    d: -m[3][2],
                }
                .normalize(),
            );
        }
        f
    }

    /// Returns `true` if `aabb` intersects or lies inside the frustum (`testAABB`).
    pub fn test_aabb(&self, aabb: &OgfBBox) -> bool {
        for plane in &self.planes {
            // Pick the box corner with the smallest signed distance (n-vertex toward inside).
            let px = if plane.n[0] >= 0.0 {
                aabb.min[0]
            } else {
                aabb.max[0]
            };
            let py = if plane.n[1] >= 0.0 {
                aabb.min[1]
            } else {
                aabb.max[1]
            };
            let pz = if plane.n[2] >= 0.0 {
                aabb.min[2]
            } else {
                aabb.max[2]
            };
            if plane.classify([px, py, pz]) > 0.0 {
                return false;
            }
        }
        true
    }

    /// Returns `true` if sphere intersects or lies inside the frustum (`testSphere_dirty`).
    pub fn test_sphere(&self, center: [f32; 3], radius: f32) -> bool {
        for plane in &self.planes {
            if plane.classify(center) > radius {
                return false;
            }
        }
        true
    }

    /// Three-state sphere visibility with hierarchical plane mask propagation
    /// matching `CFrustum::testSphere` (`src/xrCDB/Frustum.cpp:73-90`).
    ///
    /// Clears any bit in `test_mask` whose plane completely contains the sphere (`abs(cls) >= radius`)
    /// and zeroes `*test_mask` when returning [`FrustumVisibility::None`].
    pub fn test_sphere_masked(
        &self,
        center: [f32; 3],
        radius: f32,
        test_mask: &mut u32,
    ) -> FrustumVisibility {
        let mut bit = 1u32;
        for plane in self.planes.iter().take(32) {
            if (*test_mask & bit) != 0 {
                let cls = plane.classify(center);
                if cls > radius {
                    *test_mask = 0;
                    return FrustumVisibility::None;
                }
                if cls.abs() >= radius {
                    *test_mask &= !bit;
                }
            }
            bit = bit.wrapping_shl(1);
        }
        if *test_mask != 0 {
            FrustumVisibility::Partial
        } else {
            FrustumVisibility::Fully
        }
    }

    /// Three-state sphere visibility across all planes in the frustum
    /// (`FrustumVisibility::None`, `Partial`, or `Fully`).
    pub fn test_sphere_visibility(&self, center: [f32; 3], radius: f32) -> FrustumVisibility {
        let mut partial = false;
        for plane in &self.planes {
            let cls = plane.classify(center);
            if cls > radius {
                return FrustumVisibility::None;
            }
            if cls.abs() < radius {
                partial = true;
            }
        }
        if partial {
            FrustumVisibility::Partial
        } else {
            FrustumVisibility::Fully
        }
    }

    /// Three-state AABB visibility matching `CFrustum::testAABB` / `AABB_OverlapPlane`
    /// (`src/xrCDB/Frustum.h:53-67`, `Frustum.cpp:114-133`).
    pub fn test_aabb_visibility(&self, aabb: &OgfBBox) -> FrustumVisibility {
        let mut partial = false;
        for plane in &self.planes {
            let neg = [
                if plane.n[0] >= 0.0 { aabb.min[0] } else { aabb.max[0] },
                if plane.n[1] >= 0.0 { aabb.min[1] } else { aabb.max[1] },
                if plane.n[2] >= 0.0 { aabb.min[2] } else { aabb.max[2] },
            ];
            if plane.classify(neg) > 0.0 {
                return FrustumVisibility::None;
            }
            let pos = [
                if plane.n[0] >= 0.0 { aabb.max[0] } else { aabb.min[0] },
                if plane.n[1] >= 0.0 { aabb.max[1] } else { aabb.min[1] },
                if plane.n[2] >= 0.0 { aabb.max[2] } else { aabb.min[2] },
            ];
            if plane.classify(pos) > 0.0 {
                partial = true;
            }
        }
        if partial {
            FrustumVisibility::Partial
        } else {
            FrustumVisibility::Fully
        }
    }

    /// Clip a convex polygon against all frustum planes (`CFrustum::ClipPoly`).
    pub fn clip_poly(&self, poly: &[[f32; 3]]) -> Option<Vec<[f32; 3]>> {
        if poly.len() < 3 {
            return None;
        }
        let mut current = poly.to_vec();
        let mut next = Vec::with_capacity(poly.len() + 2);

        for plane in &self.planes {
            if current.len() < 3 {
                return None;
            }
            next.clear();
            let len = current.len();
            let mut prev = current[len - 1];
            let mut prev_dist = plane.classify(prev);
            let mut prev_in = prev_dist <= 0.0;

            for &curr in &current {
                let curr_dist = plane.classify(curr);
                let curr_in = curr_dist <= 0.0;

                if curr_in != prev_in {
                    let denom = prev_dist - curr_dist;
                    let t = if denom.abs() > 1e-12 {
                        (prev_dist / denom).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    next.push([
                        prev[0] + (curr[0] - prev[0]) * t,
                        prev[1] + (curr[1] - prev[1]) * t,
                        prev[2] + (curr[2] - prev[2]) * t,
                    ]);
                }
                if curr_in {
                    next.push(curr);
                }
                prev = curr;
                prev_dist = curr_dist;
                prev_in = curr_in;
            }

            std::mem::swap(&mut current, &mut next);
        }

        if current.len() >= 3 {
            Some(current)
        } else {
            None
        }
    }
}

/// Precomputed occluder triangle (`occTri` + source `HOM_poly` data).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomTriangle {
    pub verts: [[f32; 3]; 3],
    pub flags: u32,
    pub plane: Plane,
    pub center: [f32; 3],
    pub area: f32,
    pub adjacent: [Option<u32>; 3],
}

impl HomTriangle {
    #[inline]
    pub fn is_two_sided(&self) -> bool {
        self.flags != 0
    }
}

/// Raw unpacked triangle + flags from `HOM_poly` (`(v0, v1, v2, flags)`).
pub type RawHomTriangle = ([f32; 3], [f32; 3], [f32; 3], u32);

/// Parsed `level.hom` hierarchical occlusion mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct HomMesh {
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<[u32; 3]>,
    pub triangles: Vec<HomTriangle>,
    pub aabb: OgfBBox,
}

fn tri_area(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> f32 {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let cx = e1[1] * e2[2] - e1[2] * e2[1];
    let cy = e1[2] * e2[0] - e1[0] * e2[2];
    let cz = e1[0] * e2[1] - e1[1] * e2[0];
    0.5 * (cx * cx + cy * cy + cz * cz).sqrt()
}

fn quantize_vertex_01(v: [f32; 3]) -> (i32, i32, i32) {
    (
        (v[0] * 100.0).round() as i32,
        (v[1] * 100.0).round() as i32,
        (v[2] * 100.0).round() as i32,
    )
}

impl HomMesh {
    /// Parse a binary `level.hom` buffer (`CHOM::Load`, Chunk 1).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let chunk = reader.require_chunk(1)?;
        if !chunk.len().is_multiple_of(HOM_POLY_SIZE) {
            return Err(FormatError::InvalidValue(
                "level.hom chunk 1 size is not a multiple of 40-byte HOM_poly",
            ));
        }

        let count = chunk.len() / HOM_POLY_SIZE;
        let mut raw = Vec::with_capacity(count);
        let mut cur = ByteCursor::new(&chunk);
        for _ in 0..count {
            let v0 = cur.read_f32_3()?;
            let v1 = cur.read_f32_3()?;
            let v2 = cur.read_f32_3()?;
            let flags = cur.read_u32()?;
            raw.push((v0, v1, v2, flags));
        }

        Ok(Self::from_raw_triangles(&raw))
    }

    /// Build a `HomMesh` with `0.01m` vertex welding and edge adjacency calculation.
    pub fn from_raw_triangles(raw: &[RawHomTriangle]) -> Self {
        let mut vertices: Vec<[f32; 3]> = Vec::new();
        let mut vert_map: HashMap<(i32, i32, i32), u32> = HashMap::new();
        let mut indices: Vec<[u32; 3]> = Vec::with_capacity(raw.len());

        let mut weld_or_push = |v: [f32; 3]| -> u32 {
            let key = quantize_vertex_01(v);
            if let Some(&idx) = vert_map.get(&key) {
                return idx;
            }
            let idx = vertices.len() as u32;
            vertices.push(v);
            vert_map.insert(key, idx);
            idx
        };

        for &(v0, v1, v2, _) in raw {
            let i0 = weld_or_push(v0);
            let i1 = weld_or_push(v1);
            let i2 = weld_or_push(v2);
            indices.push([i0, i1, i2]);
        }

        // Compute directed-edge lookup for adjacency (matching CDB::Collector::calc_adjacency).
        let mut edge_to_tris: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (tri_idx, &tri) in indices.iter().enumerate() {
            let edges = [
                (tri[0].min(tri[1]), tri[0].max(tri[1])),
                (tri[1].min(tri[2]), tri[1].max(tri[2])),
                (tri[2].min(tri[0]), tri[2].max(tri[0])),
            ];
            for e in edges {
                edge_to_tris.entry(e).or_default().push(tri_idx as u32);
            }
        }

        let mut triangles = Vec::with_capacity(raw.len());
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];

        for (tri_idx, &(v0, v1, v2, flags)) in raw.iter().enumerate() {
            let idx = indices[tri_idx];
            let p0 = vertices[idx[0] as usize];
            let p1 = vertices[idx[1] as usize];
            let p2 = vertices[idx[2] as usize];

            for p in [v0, v1, v2] {
                for axis in 0..3 {
                    min[axis] = min[axis].min(p[axis]);
                    max[axis] = max[axis].max(p[axis]);
                }
            }

            let edges = [
                (idx[0].min(idx[1]), idx[0].max(idx[1])),
                (idx[1].min(idx[2]), idx[1].max(idx[2])),
                (idx[2].min(idx[0]), idx[2].max(idx[0])),
            ];
            let mut adjacent = [None; 3];
            for (ei, e) in edges.into_iter().enumerate() {
                if let Some(list) = edge_to_tris.get(&e) {
                    adjacent[ei] = list.iter().copied().find(|&other| other != tri_idx as u32);
                }
            }

            triangles.push(HomTriangle {
                verts: [p0, p1, p2],
                flags,
                plane: Plane::from_points(p0, p1, p2),
                center: [
                    (p0[0] + p1[0] + p2[0]) / 3.0,
                    (p0[1] + p1[1] + p2[1]) / 3.0,
                    (p0[2] + p1[2] + p2[2]) / 3.0,
                ],
                area: tri_area(p0, p1, p2),
                adjacent,
            });
        }

        let aabb = if raw.is_empty() {
            OgfBBox {
                min: [0.0; 3],
                max: [0.0; 3],
            }
        } else {
            OgfBBox { min, max }
        };

        Self {
            vertices,
            indices,
            triangles,
            aabb,
        }
    }

    /// Return the AABB bounding box of a single occluder triangle.
    pub fn triangle_aabb(&self, tri_index: usize) -> Option<OgfBBox> {
        let tri = self.triangles.get(tri_index)?;
        let [v0, v1, v2] = tri.verts;
        Some(OgfBBox {
            min: [
                v0[0].min(v1[0]).min(v2[0]),
                v0[1].min(v1[1]).min(v2[1]),
                v0[2].min(v1[2]).min(v2[2]),
            ],
            max: [
                v0[0].max(v1[0]).max(v2[0]),
                v0[1].max(v1[1]).max(v2[1]),
                v0[2].max(v1[2]).max(v2[2]),
            ],
        })
    }
}

/// Software 64x64 (+2px guard band) hierarchical depth occluder matching `occRasterizer`.
#[derive(Debug, Clone)]
pub struct OccRasterizer {
    buf_frame: Vec<Option<u32>>,
    buf_depth: Vec<f32>,
    depth_0: Vec<i32>,
    depth_1: Vec<i32>,
    depth_2: Vec<i32>,
    depth_3: Vec<i32>,
}

impl Default for OccRasterizer {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn xform_point_w(p: [f32; 3], m: &XRayMatrix) -> [f32; 3] {
    let w = p[0] * m[0][3] + p[1] * m[1][3] + p[2] * m[2][3] + m[3][3];
    let iw = if w.abs() > 1e-9 { 1.0 / w } else { 0.0 };
    [
        (p[0] * m[0][0] + p[1] * m[1][0] + p[2] * m[2][0] + m[3][0]) * iw,
        (p[0] * m[0][1] + p[1] * m[1][1] + p[2] * m[2][1] + m[3][1]) * iw,
        (p[0] * m[0][2] + p[1] * m[1][2] + p[2] * m[2][2] + m[3][2]) * iw,
    ]
}

#[inline]
fn mat_mul_row(b: &XRayMatrix, a: &XRayMatrix) -> XRayMatrix {
    let mut out = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] =
                b[i][0] * a[0][j] + b[i][1] * a[1][j] + b[i][2] * a[2][j] + b[i][3] * a[3][j];
        }
    }
    out
}

#[inline]
pub fn viewport_matrix_64() -> XRayMatrix {
    let half = OCC_DIM_0 as f32 * 0.5;
    [
        [half, 0.0, 0.0, 0.0],
        [0.0, -half, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [half, half, 0.0, 1.0],
    ]
}

#[inline]
pub fn viewport_matrix_01() -> XRayMatrix {
    [
        [0.5, 0.0, 0.0, 0.0],
        [0.0, -0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0, 1.0],
    ]
}

#[inline]
fn df_2_s32(d: f32) -> i32 {
    (d * OCC_Q_S32).floor() as i32
}

#[inline]
fn df_2_s32_up(d: f32) -> i32 {
    (d * OCC_Q_S32).ceil() as i32
}

#[inline]
fn v_clamp(v: i32, a: i32, b: i32) -> i32 {
    if v < a {
        a
    } else if v >= b {
        b - 1
    } else {
        v
    }
}

#[inline]
fn shared_tri(tris: &[HomTriangle], a: u32, b: Option<u32>) -> bool {
    let Some(tb) = b else {
        return false;
    };
    if a == tb {
        return true;
    }
    tris[a as usize].adjacent.contains(&Some(tb))
}

#[inline]
fn shared_opt(tris: &[HomTriangle], a: Option<u32>, b: Option<u32>) -> bool {
    let Some(ta) = a else {
        return false;
    };
    shared_tri(tris, ta, b)
}

fn propagate_level(src: &[i32], src_dim: usize, dst: &mut [i32], dst_dim: usize) {
    for y in 0..dst_dim {
        for x in 0..dst_dim {
            let row0 = (y * 2) * src_dim + x * 2;
            let row1 = (y * 2 + 1) * src_dim + x * 2;
            let m = src[row0]
                .max(src[row0 + 1])
                .max(src[row1])
                .max(src[row1 + 1]);
            dst[y * dst_dim + x] = m;
        }
    }
}

impl OccRasterizer {
    pub fn new() -> Self {
        Self {
            buf_frame: vec![None; OCC_DIM * OCC_DIM],
            buf_depth: vec![1.0; OCC_DIM * OCC_DIM],
            depth_0: vec![df_2_s32(1.0); OCC_DIM_0 * OCC_DIM_0],
            depth_1: vec![df_2_s32(1.0); OCC_DIM_1 * OCC_DIM_1],
            depth_2: vec![df_2_s32(1.0); OCC_DIM_2 * OCC_DIM_2],
            depth_3: vec![df_2_s32(1.0); OCC_DIM_3 * OCC_DIM_3],
        }
    }

    pub fn clear(&mut self) {
        self.buf_frame.fill(None);
        self.buf_depth.fill(1.0);
    }

    /// Clear buffers and rasterize `mesh` using `full_transform` (`View * Proj`) and `camera_pos`.
    pub fn render_mesh(
        &mut self,
        mesh: &HomMesh,
        full_transform: &XRayMatrix,
        camera_pos: [f32; 3],
    ) {
        self.clear();
        let frustum =
            ClipFrustum::from_view_proj(full_transform, frustum_planes::LRTB | frustum_planes::FAR);
        let clip_near = ClipFrustum::from_view_proj(full_transform, frustum_planes::NEAR);
        let xform = mat_mul_row(full_transform, &viewport_matrix_64());

        // Collect potentially visible occluder triangles inside frustum & sort front-to-back.
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for (idx, tri) in mesh.triangles.iter().enumerate() {
            if !tri.is_two_sided() && tri.plane.classify(camera_pos) <= 0.0 {
                continue;
            }
            if let Some(aabb) = mesh.triangle_aabb(idx)
                && !frustum.test_aabb(&aabb)
            {
                continue;
            }
            let dx = tri.center[0] - camera_pos[0];
            let dy = tri.center[1] - camera_pos[1];
            let dz = tri.center[2] - camera_pos[2];
            candidates.push((idx, dx * dx + dy * dy + dz * dz));
        }

        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));

        for (tri_idx, _) in candidates {
            let tri = &mesh.triangles[tri_idx];
            let Some(clipped) = clip_near.clip_poly(&tri.verts) else {
                continue;
            };
            let p0 = xform_point_w(clipped[0], &xform);
            for v in 1..(clipped.len() - 1) {
                let p1 = xform_point_w(clipped[v], &xform);
                let p2 = xform_point_w(clipped[v + 1], &xform);
                self.rasterize_triangle(tri_idx as u32, [p0, p1, p2], &mesh.triangles);
            }
        }

        self.propagate(&mesh.triangles);
    }

    /// Rasterize a single already viewport-projected triangle in 64x64 coordinates.
    pub fn rasterize_triangle(
        &mut self,
        tri_idx: u32,
        raster: [[f32; 3]; 3],
        tris: &[HomTriangle],
    ) -> u32 {
        let [a, b, c] = raster;
        let (min_v, mid_v, max_v) = if a[1] <= b[1] {
            if b[1] <= c[1] {
                (a, b, c)
            } else if a[1] <= c[1] {
                (a, c, b)
            } else {
                (c, a, b)
            }
        } else if a[1] <= c[1] {
            (b, a, c)
        } else if b[1] <= c[1] {
            (b, c, a)
        } else {
            (c, b, a)
        };

        let ca = [min_v[0] + 2.0, min_v[1] + 2.0, min_v[2]];
        let cb = [mid_v[0] + 2.0, mid_v[1] + 2.0, mid_v[2]];
        let cc = [max_v[0] + 2.0, max_v[1] + 2.0, max_v[2]];

        let mut pixels = 0u32;
        let b_mid = (cb[1] - cb[1].floor()) > 0.5;

        pixels += self.rasterize_section(true, b_mid, ca, cb, cc, tri_idx, tris);
        pixels += self.rasterize_section(false, !b_mid, ca, cb, cc, tri_idx, tris);
        pixels
    }

    #[allow(clippy::too_many_arguments)]
    fn rasterize_section(
        &mut self,
        is_bottom: bool,
        b_middle: bool,
        ca: [f32; 3],
        cb: [f32; 3],
        cc: [f32; 3],
        tri_idx: u32,
        tris: &[HomTriangle],
    ) -> u32 {
        let (mut start_y, mut end_y, startp1, startp2, e1, e2) = if is_bottom {
            let sy = ca[1].ceil() as i32;
            let mut ey = cb[1].floor() as i32 - 1;
            if b_middle {
                ey += 1;
            }
            let test = cc[1].floor() as i32;
            if ey >= test {
                ey -= 1;
            }
            (
                sy,
                ey,
                ca,
                ca,
                [cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]],
                [cc[0] - ca[0], cc[1] - ca[1], cc[2] - ca[2]],
            )
        } else {
            let mut sy = cb[1].ceil() as i32;
            let ey = cc[1].floor() as i32;
            if b_middle {
                sy -= 1;
            }
            let test = ca[1].ceil() as i32;
            if sy < test {
                sy += 1;
            }
            (
                sy,
                ey,
                ca,
                cb,
                [cc[0] - ca[0], cc[1] - ca[1], cc[2] - ca[2]],
                [cc[0] - cb[0], cc[1] - cb[1], cc[2] - cb[2]],
            )
        };

        start_y = v_clamp(start_y, 0, OCC_DIM as i32);
        end_y = v_clamp(end_y, 0, OCC_DIM as i32);
        if start_y >= end_y || e1[1].abs() <= 1e-7 || e2[1].abs() <= 1e-7 {
            return 0;
        }

        let me1 = e1[0] / e1[1];
        let me2 = e2[0] / e2[1];
        let e1_dy = start_y as f32 - startp1[1];
        let e2_dy = start_y as f32 - startp2[1];

        let e1_is_left = (me1 < me2 && is_bottom) || (me1 > me2 && !is_bottom);
        let (mut left_x, left_dx, mut left_z, left_dz, mut right_x, right_dx, mut right_z, right_dz) =
            if e1_is_left {
                let t1 = e1_dy / e1[1];
                let t2 = e2_dy / e2[1];
                (
                    startp1[0] + e1[0] * t1,
                    me1,
                    startp1[2] + e1[2] * t1,
                    e1[2] / e1[1],
                    startp2[0] + e2[0] * t2,
                    me2,
                    startp2[2] + e2[2] * t2,
                    e2[2] / e2[1],
                )
            } else {
                let t2 = e2_dy / e2[1];
                let t1 = e1_dy / e1[1];
                (
                    startp2[0] + e2[0] * t2,
                    me2,
                    startp2[2] + e2[2] * t2,
                    e2[2] / e2[1],
                    startp1[0] + e1[0] * t1,
                    me1,
                    startp1[2] + e1[2] * t1,
                    e1[2] / e1[1],
                )
            };

        let lhx = left_dx * 0.5;
        left_x += lhx;
        let rhx = right_dx * 0.5;
        right_x += rhx;

        let mut pixels = 0u32;
        for cur_y in start_y..=end_y {
            pixels += self.scanline(
                cur_y as usize,
                left_x,
                lhx,
                right_x,
                rhx,
                left_z,
                right_z,
                tri_idx,
                tris,
            );
            left_x += left_dx;
            right_x += right_dx;
            left_z += left_dz;
            right_z += right_dz;
        }
        pixels
    }

    #[allow(clippy::too_many_arguments)]
    fn scanline(
        &mut self,
        cur_y: usize,
        left_x: f32,
        lhx: f32,
        right_x: f32,
        rhx: f32,
        start_z: f32,
        end_z: f32,
        tri_idx: u32,
        tris: &[HomTriangle],
    ) -> u32 {
        let start_c = left_x + lhx;
        let end_c = right_x + rhx;
        let start_r = left_x - lhx;
        let end_r = right_x - rhx;

        let (start_t, start_x) = if start_c < start_r {
            (start_c, start_r)
        } else {
            (start_r, start_c)
        };
        let (end_t, end_x) = if end_c < end_r {
            (end_r, end_c)
        } else {
            (end_c, end_r)
        };

        let min_t = v_clamp(start_t.floor() as i32 - 1, 1, OCC_DIM as i32 - 1) as usize;
        let max_t = v_clamp(end_t.ceil() as i32 + 1, 1, OCC_DIM as i32 - 1) as usize;
        if min_t >= max_t {
            return 0;
        }

        let min_x = v_clamp(start_x.ceil() as i32, 0, OCC_DIM as i32) as usize;
        let max_x = v_clamp(end_x.floor() as i32, 0, OCC_DIM as i32) as usize;
        let (lim_left, lim_right) = if min_x > max_x {
            (max_x, min_x)
        } else {
            (min_x, max_x)
        };

        let len_r = end_r - start_r;
        let (mut z, z_end, dz) = if len_r.abs() > 1e-7 {
            let z_len = end_z - start_z;
            let z0 = start_z + ((min_t as f32 - start_r) / len_r) * z_len;
            let z1 = start_z + ((max_t as f32 - start_r) / len_r) * z_len;
            let step = (z1 - z0) / (max_t - min_t) as f32;
            (z0 + 0.5 * step.abs(), z1, step)
        } else {
            (start_z.min(end_z), start_z.max(end_z), 0.0)
        };

        let i_base = cur_y * OCC_DIM;
        let mut pixels = 0u32;

        // Left guard-band connector
        let mut i = i_base + min_t;
        let limit_left = i_base + lim_left;
        while i < limit_left {
            if shared_tri(tris, tri_idx, self.buf_frame[i - 1]) && z < self.buf_depth[i] {
                self.buf_frame[i] = Some(tri_idx);
                self.buf_depth[i] = z.max(self.buf_depth[i - 1]);
                pixels += 1;
            }
            i += 1;
            z += dz;
        }

        // Core interior span
        let limit_core = i_base + max_x;
        while i < limit_core {
            if z < self.buf_depth[i] {
                self.buf_frame[i] = Some(tri_idx);
                self.buf_depth[i] = z;
                pixels += 1;
            }
            i += 1;
            z += dz;
        }

        // Right guard-band connector
        if max_t > lim_right {
            let mut ri = (i_base + max_t - 1) as isize;
            let limit_right = (i_base + lim_right) as isize;
            let mut rz = z_end - dz;
            while ri >= limit_right {
                let u = ri as usize;
                if shared_tri(tris, tri_idx, self.buf_frame[u + 1]) && rz < self.buf_depth[u] {
                    self.buf_frame[u] = Some(tri_idx);
                    self.buf_depth[u] = rz.max(self.buf_depth[u + 1]);
                    pixels += 1;
                }
                ri -= 1;
                rz -= dz;
            }
        }

        pixels
    }

    /// Quantize depth and downsample the 64->32->16->8 hierarchical depth pyramid.
    pub fn propagate(&mut self, tris: &[HomTriangle]) {
        let total = OCC_DIM * OCC_DIM;
        for y in 0..OCC_DIM_0 {
            for x in 0..OCC_DIM_0 {
                let ox = x + OCC_BORDER;
                let oy = y + OCC_BORDER;
                let pos = oy * OCC_DIM + ox;
                let pos_up = pos.saturating_sub(OCC_DIM);
                let pos_down = if pos + OCC_DIM < total {
                    pos + OCC_DIM
                } else {
                    pos
                };
                let pos_down2 = if pos_down + OCC_DIM < total {
                    pos_down + OCC_DIM
                } else {
                    pos_down
                };

                if let Some(tu1) = self.buf_frame[pos_up] {
                    if shared_opt(tris, Some(tu1), self.buf_frame[pos_down]) {
                        let zr = 0.5 * (self.buf_depth[pos_up] + self.buf_depth[pos_down]);
                        if zr < self.buf_depth[pos] {
                            self.buf_frame[pos] = Some(tu1);
                            self.buf_depth[pos] = zr;
                        }
                    } else if shared_opt(tris, Some(tu1), self.buf_frame[pos_down2]) {
                        let zr = 0.5 * (self.buf_depth[pos_up] + self.buf_depth[pos_down2]);
                        if zr < self.buf_depth[pos] {
                            self.buf_frame[pos] = Some(tu1);
                            self.buf_depth[pos] = zr;
                        }
                    }
                }

                let d = self.buf_depth[pos].clamp(-1.99, 1.99);
                self.depth_0[y * OCC_DIM_0 + x] = df_2_s32(d);
            }
        }

        propagate_level(&self.depth_0, OCC_DIM_0, &mut self.depth_1, OCC_DIM_1);
        propagate_level(&self.depth_1, OCC_DIM_1, &mut self.depth_2, OCC_DIM_2);
        propagate_level(&self.depth_2, OCC_DIM_2, &mut self.depth_3, OCC_DIM_3);
    }

    /// Test a normalized `[0..1]` screen rect against `depth_0` (`occRasterizer::test`).
    pub fn test_rect(&self, _x0: f32, _y0: f32, _x1: f32, _y1: f32, _z: f32) -> bool {
        let z = df_2_s32_up(_z) + 1;
        let dim = OCC_DIM_0 as i32;
        let x0 = ((_x0 * OCC_DIM_0 as f32 + 0.5).floor() as i32).clamp(0, dim - 1) as usize;
        let x1 = ((_x1 * OCC_DIM_0 as f32 + 0.5).floor() as i32).clamp(x0 as i32, dim - 1) as usize;
        let y0 = ((_y0 * OCC_DIM_0 as f32 + 0.5).floor() as i32).clamp(0, dim - 1) as usize;
        let y1 = ((_y1 * OCC_DIM_0 as f32 + 0.5).floor() as i32).clamp(y0 as i32, dim - 1) as usize;

        for y in y0..=y1 {
            let row = y * OCC_DIM_0;
            for x in x0..=x1 {
                if z < self.depth_0[row + x] {
                    return true;
                }
            }
        }
        false
    }

    /// Conservative AABB visibility test (`CHOM::visible(Fbox3)`): returns `false` iff fully occluded.
    pub fn test_aabb(
        &self,
        aabb: &OgfBBox,
        full_transform: &XRayMatrix,
        camera_pos: [f32; 3],
    ) -> bool {
        if camera_pos[0] >= aabb.min[0]
            && camera_pos[0] <= aabb.max[0]
            && camera_pos[1] >= aabb.min[1]
            && camera_pos[1] <= aabb.max[1]
            && camera_pos[2] >= aabb.min[2]
            && camera_pos[2] <= aabb.max[2]
        {
            return true;
        }

        let xform01 = mat_mul_row(full_transform, &viewport_matrix_01());
        let corners = [
            [aabb.min[0], aabb.min[1], aabb.min[2]],
            [aabb.min[0], aabb.min[1], aabb.max[2]],
            [aabb.max[0], aabb.min[1], aabb.max[2]],
            [aabb.max[0], aabb.min[1], aabb.min[2]],
            [aabb.min[0], aabb.max[1], aabb.min[2]],
            [aabb.min[0], aabb.max[1], aabb.max[2]],
            [aabb.max[0], aabb.max[1], aabb.max[2]],
            [aabb.max[0], aabb.max[1], aabb.min[2]],
        ];

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;

        for c in corners {
            let cz = c[0] * xform01[0][2]
                + c[1] * xform01[1][2]
                + c[2] * xform01[2][2]
                + xform01[3][2];
            if cz < EPS {
                return true;
            }
            let cw = c[0] * xform01[0][3]
                + c[1] * xform01[1][3]
                + c[2] * xform01[2][3]
                + xform01[3][3];
            if cw <= 1e-6 {
                return true;
            }
            let iw = 1.0 / cw;
            let sx = (c[0] * xform01[0][0]
                + c[1] * xform01[1][0]
                + c[2] * xform01[2][0]
                + xform01[3][0])
                * iw;
            let sy = (c[0] * xform01[0][1]
                + c[1] * xform01[1][1]
                + c[2] * xform01[2][1]
                + xform01[3][1])
                * iw;
            let sz = cz * iw;
            min_x = min_x.min(sx);
            max_x = max_x.max(sx);
            min_y = min_y.min(sy);
            max_y = max_y.max(sy);
            min_z = min_z.min(sz);
        }

        self.test_rect(min_x, min_y, max_x, max_y, min_z)
    }

    /// Return quantized depth buffer slice for pyramid level `0` (64x64), `1` (32x32), `2` (16x16), or `3` (8x8).
    pub fn depth_level(&self, level: usize) -> Option<&[i32]> {
        match level {
            0 => Some(&self.depth_0),
            1 => Some(&self.depth_1),
            2 => Some(&self.depth_2),
            3 => Some(&self.depth_3),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;

    fn build_perspective_lh(fov_y_rad: f32, aspect: f32, zn: f32, zf: f32) -> XRayMatrix {
        let y_scale = 1.0 / (fov_y_rad * 0.5).tan();
        let x_scale = y_scale / aspect;
        let q = zf / (zf - zn);
        [
            [x_scale, 0.0, 0.0, 0.0],
            [0.0, y_scale, 0.0, 0.0],
            [0.0, 0.0, q, 1.0],
            [0.0, 0.0, -zn * q, 0.0],
        ]
    }

    fn build_synthetic_hom_wall() -> Vec<u8> {
        let mut payload = Vec::new();
        // Two triangles forming a large two-sided occluder quad at z = 5.0 covering x,y in [-4, 4].
        // Winding order matches front-face toward origin camera looking along +Z.
        let tris = [
            ([-4.0f32, -4.0, 5.0], [-4.0f32, 4.0, 5.0], [4.0f32, 4.0, 5.0], 1u32),
            ([-4.0f32, -4.0, 5.0], [4.0f32, 4.0, 5.0], [4.0f32, -4.0, 5.0], 1u32),
        ];
        for (a, b, c, flags) in tris {
            for p in [a, b, c] {
                for f in p {
                    payload.extend_from_slice(&f.to_le_bytes());
                }
            }
            payload.extend_from_slice(&flags.to_le_bytes());
        }
        let mut cw = ChunkWriter::new();
        cw.write_chunk(1, &payload);
        cw.into_bytes()
    }

    #[test]
    fn parse_hom_chunk_and_adjacency() {
        let data = build_synthetic_hom_wall();
        let mesh = HomMesh::parse(&data).unwrap();
        assert_eq!(mesh.triangles.len(), 2);
        assert_eq!(mesh.vertices.len(), 4);
        assert!(mesh.triangles[0].is_two_sided());
        assert!((mesh.triangles[0].area - 32.0).abs() < 1e-4);
        // Diagonal edge is shared between tri 0 and tri 1
        assert!(mesh.triangles[0].adjacent.contains(&Some(1)));
        assert!(mesh.triangles[1].adjacent.contains(&Some(0)));
    }

    #[test]
    fn occ_rasterizer_culls_occluded_aabb_behind_wall() {
        let data = build_synthetic_hom_wall();
        let mesh = HomMesh::parse(&data).unwrap();
        let proj = build_perspective_lh(std::f32::consts::FRAC_PI_2, 1.0, 0.2, 100.0);

        let mut raster = OccRasterizer::new();
        raster.render_mesh(&mesh, &proj, [0.0, 0.0, 0.0]);

        // Small box strictly behind the z=5 wall at z=[10, 12] -> occluded!
        let behind_wall = OgfBBox {
            min: [-1.0, -1.0, 10.0],
            max: [1.0, 1.0, 12.0],
        };
        assert!(!raster.test_aabb(&behind_wall, &proj, [0.0, 0.0, 0.0]));

        // Box strictly in front of the wall at z=[2, 3] -> visible!
        let in_front = OgfBBox {
            min: [-0.5, -0.5, 2.0],
            max: [0.5, 0.5, 3.0],
        };
        assert!(raster.test_aabb(&in_front, &proj, [0.0, 0.0, 0.0]));

        // Coarse pyramid levels propagated properly
        assert_eq!(raster.depth_level(0).unwrap().len(), 64 * 64);
        assert_eq!(raster.depth_level(3).unwrap().len(), 8 * 8);
    }

    #[test]
    fn frustum_culling_and_single_sided_backface() {
        let proj = build_perspective_lh(std::f32::consts::FRAC_PI_2, 1.0, 0.5, 50.0);
        let frustum = ClipFrustum::from_view_proj(&proj, frustum_planes::ALL);

        assert!(frustum.test_aabb(&OgfBBox {
            min: [-1.0, -1.0, 5.0],
            max: [1.0, 1.0, 6.0],
        }));
        assert!(!frustum.test_aabb(&OgfBBox {
            min: [50.0, 50.0, 5.0],
            max: [55.0, 55.0, 6.0],
        }));
        assert!(!frustum.test_sphere([0.0, 0.0, -10.0], 1.0));
        assert_eq!(
            frustum.test_sphere_visibility([0.0, 0.0, 10.0], 1.0),
            FrustumVisibility::Fully
        );
        assert_eq!(
            frustum.test_sphere_visibility([0.0, 0.0, 0.5], 0.5),
            FrustumVisibility::Partial
        );
        assert_eq!(
            frustum.test_sphere_visibility([0.0, 0.0, -10.0], 1.0),
            FrustumVisibility::None
        );

        let mut mask = frustum_planes::ALL;
        assert_eq!(
            frustum.test_sphere_masked([0.0, 0.0, 10.0], 1.0, &mut mask),
            FrustumVisibility::Fully
        );
        assert_eq!(mask, 0);

        let mut mask_part = frustum_planes::ALL;
        assert_eq!(
            frustum.test_sphere_masked([0.0, 0.0, 0.5], 0.5, &mut mask_part),
            FrustumVisibility::Partial
        );
        assert_ne!(mask_part, 0);
        assert_eq!(
            frustum.test_aabb_visibility(&OgfBBox {
                min: [-0.5, -0.5, 10.0],
                max: [0.5, 0.5, 11.0],
            }),
            FrustumVisibility::Fully
        );
        assert_eq!(
            frustum.test_aabb_visibility(&OgfBBox {
                min: [-0.5, -0.5, 0.2],
                max: [0.5, 0.5, 1.0],
            }),
            FrustumVisibility::Partial
        );
        assert_eq!(
            frustum.test_aabb_visibility(&OgfBBox {
                min: [-0.5, -0.5, -5.0],
                max: [0.5, 0.5, -2.0],
            }),
            FrustumVisibility::None
        );

        // Single-sided triangle facing away from camera should not occlude.
        let backface = HomMesh::from_raw_triangles(&[(
            [-4.0, -4.0, 5.0],
            [4.0, 4.0, 5.0],
            [-4.0, 4.0, 5.0],
            0u32,
        )]);
        let mut raster = OccRasterizer::new();
        raster.render_mesh(&backface, &proj, [0.0, 0.0, 0.0]);
        assert!(raster.test_aabb(
            &OgfBBox {
                min: [-0.5, -0.5, 10.0],
                max: [0.5, 0.5, 11.0],
            },
            &proj,
            [0.0, 0.0, 0.0]
        ));
    }
}
