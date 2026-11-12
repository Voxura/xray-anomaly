//! Pure Rust X-Ray `level.cform` Collision Database & BVH Raycast Accelerator
//!
//! Replaces legacy C++ `xrCDB` / `Opcode` (`src/xrEngine/xrLevel.h`, `src/xrCDB/xrCDB.h`,
//! `src/xrCDB/xrCDB_ray.cpp`) with a deterministic pure-Rust AABB BVH, Möller–Trumbore
//! ray-triangle queries (`OPT_CULL`, `OPT_ONLYFIRST`, `OPT_ONLYNEAREST`), SAT triangle-box
//! overlap queries (`box_query`), and direct `GameMaterialLibrary` hit material resolution.

use crate::chunk::{ByteCursor, FormatError};
use crate::gamemtl::{GameMaterial, GameMaterialLibrary};
use crate::ogf::OgfBBox;

pub const CFORM_CURRENT_VERSION: u32 = 4;

/// Ray & box query flags matching `CDB::OPT_*` (`src/xrCDB/xrCDB.h:133-139`).
pub mod ray_options {
    pub const OPT_CULL: u32 = 1 << 0;
    pub const OPT_ONLYFIRST: u32 = 1 << 1;
    pub const OPT_ONLYNEAREST: u32 = 1 << 2;
    pub const OPT_FULL_TEST: u32 = 1 << 3;
}

/// 36-byte header at offset 0 of `level.cform` (`struct hdrCFORM` in `xrLevel.h:94-100`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CformHeader {
    pub version: u32,
    pub vert_count: u32,
    pub face_count: u32,
    pub aabb: OgfBBox,
}

/// 16-byte `CDB::TRI` collision triangle unpacked into clean Rust fields (`xrCDB.h:37-55`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CformTriangle {
    pub verts: [u32; 3],
    pub material: u16,
    pub suppress_shadows: bool,
    pub suppress_wm: bool,
    pub sector: u16,
}

impl CformTriangle {
    #[inline]
    pub fn from_raw(verts: [u32; 3], dummy: u32) -> Self {
        Self {
            verts,
            material: (dummy & 0x3FFF) as u16,
            suppress_shadows: (dummy & (1 << 14)) != 0,
            suppress_wm: (dummy & (1 << 15)) != 0,
            sector: (dummy >> 16) as u16,
        }
    }

    #[inline]
    pub fn to_raw_dummy(self) -> u32 {
        (u32::from(self.material) & 0x3FFF)
            | if self.suppress_shadows { 1 << 14 } else { 0 }
            | if self.suppress_wm { 1 << 15 } else { 0 }
            | (u32::from(self.sector) << 16)
    }
}

/// Parsed `level.cform` static world collision mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct CformModel {
    pub header: CformHeader,
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<CformTriangle>,
}

impl CformModel {
    /// Parse a binary `level.cform` stream (`CObjectSpace::Load`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);
        let version = cur.read_u32()?;
        if version != CFORM_CURRENT_VERSION {
            return Err(FormatError::InvalidValue(
                "unsupported level.cform version (expected 4)",
            ));
        }

        let vert_count = cur.read_u32()?;
        let face_count = cur.read_u32()?;
        let aabb = OgfBBox {
            min: cur.read_f32_3()?,
            max: cur.read_f32_3()?,
        };

        let vert_bytes = (vert_count as usize)
            .checked_mul(12)
            .ok_or(FormatError::UnexpectedEof)?;
        let tri_bytes = (face_count as usize)
            .checked_mul(16)
            .ok_or(FormatError::UnexpectedEof)?;
        let raw_verts = cur.read_bytes(vert_bytes)?;
        let raw_tris = cur.read_bytes(tri_bytes)?;

        let mut vertices = Vec::with_capacity(vert_count as usize);
        for rec in raw_verts.as_chunks::<12>().0 {
            let mut vc = ByteCursor::new(rec);
            vertices.push(vc.read_f32_3()?);
        }

        let mut triangles = Vec::with_capacity(face_count as usize);
        for rec in raw_tris.as_chunks::<16>().0 {
            let mut tc = ByteCursor::new(rec);
            let verts = [tc.read_u32()?, tc.read_u32()?, tc.read_u32()?];
            for idx in verts {
                if idx >= vert_count {
                    return Err(FormatError::InvalidValue(
                        "level.cform triangle references out-of-range vertex index",
                    ));
                }
            }
            let dummy = tc.read_u32()?;
            triangles.push(CformTriangle::from_raw(verts, dummy));
        }

        Ok(Self {
            header: CformHeader {
                version,
                vert_count,
                face_count,
                aabb,
            },
            vertices,
            triangles,
        })
    }

    /// Construct a `CformModel` directly from vertices and triangles with computed AABB.
    pub fn from_parts(
        vertices: Vec<[f32; 3]>,
        triangles: Vec<CformTriangle>,
    ) -> Result<Self, FormatError> {
        let vcount = vertices.len() as u32;
        for tri in &triangles {
            if tri.verts.iter().any(|&v| v >= vcount) {
                return Err(FormatError::InvalidValue(
                    "triangle references out-of-range vertex index",
                ));
            }
        }

        let aabb = if let Some(first) = vertices.first() {
            let mut min = *first;
            let mut max = *first;
            for v in &vertices[1..] {
                for axis in 0..3 {
                    min[axis] = min[axis].min(v[axis]);
                    max[axis] = max[axis].max(v[axis]);
                }
            }
            OgfBBox { min, max }
        } else {
            OgfBBox {
                min: [0.0; 3],
                max: [0.0; 3],
            }
        };

        Ok(Self {
            header: CformHeader {
                version: CFORM_CURRENT_VERSION,
                vert_count: vcount,
                face_count: triangles.len() as u32,
                aabb,
            },
            vertices,
            triangles,
        })
    }

    /// Build a cache-friendly pure-Rust AABB BVH acceleration structure replacing `Opcode`.
    pub fn build_bvh(&self) -> CformBvh<'_> {
        CformBvh::build(self)
    }

    /// Resolve triangle material against a loaded `GameMaterialLibrary` (`GetMaterialByIdx`).
    #[inline]
    pub fn triangle_material<'a>(
        &self,
        tri_index: usize,
        mtl_lib: &'a GameMaterialLibrary,
    ) -> Option<&'a GameMaterial> {
        let tri = self.triangles.get(tri_index)?;
        mtl_lib.material_by_idx(tri.material as usize)
    }
}

/// Raycast query configuration matching `CDB::COLLIDER::ray_options`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RaycastOptions {
    pub cull_backfaces: bool,
    pub only_first: bool,
    pub only_nearest: bool,
}

impl RaycastOptions {
    #[inline]
    pub fn from_flags(flags: u32) -> Self {
        Self {
            cull_backfaces: (flags & ray_options::OPT_CULL) != 0,
            only_first: (flags & ray_options::OPT_ONLYFIRST) != 0,
            only_nearest: (flags & ray_options::OPT_ONLYNEAREST) != 0,
        }
    }
}

/// Raycast intersection result matching `CDB::RESULT`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaycastHit {
    pub tri_index: usize,
    pub range: f32,
    pub u: f32,
    pub v: f32,
    pub point: [f32; 3],
    pub verts: [[f32; 3]; 3],
    pub material: u16,
    pub suppress_shadows: bool,
    pub suppress_wm: bool,
    pub sector: u16,
}

/// Box overlap hit matching `CDB::COLLIDER::box_query`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxQueryHit {
    pub tri_index: usize,
    pub verts: [[f32; 3]; 3],
    pub material: u16,
    pub sector: u16,
}

#[derive(Debug, Clone, Copy)]
struct BvhNode {
    min: [f32; 3],
    max: [f32; 3],
    first: u32,
    right: u32,
    count: u32,
}

/// Pure-Rust AABB BVH spatial index over a `CformModel` (zero external collision dependencies).
#[derive(Debug, Clone)]
pub struct CformBvh<'a> {
    model: &'a CformModel,
    nodes: Vec<BvhNode>,
    tri_indices: Vec<u32>,
}

const MAX_LEAF_TRIS: usize = 4;
const MAX_BVH_DEPTH: usize = 64;

impl<'a> CformBvh<'a> {
    pub fn build(model: &'a CformModel) -> Self {
        let count = model.triangles.len();
        if count == 0 {
            return Self {
                model,
                nodes: Vec::new(),
                tri_indices: Vec::new(),
            };
        }

        let mut tri_indices: Vec<u32> = (0..(count as u32)).collect();
        let mut tri_mins = Vec::with_capacity(count);
        let mut tri_maxs = Vec::with_capacity(count);
        let mut tri_centroids = Vec::with_capacity(count);

        for tri in &model.triangles {
            let p0 = model.vertices[tri.verts[0] as usize];
            let p1 = model.vertices[tri.verts[1] as usize];
            let p2 = model.vertices[tri.verts[2] as usize];
            tri_mins.push([
                p0[0].min(p1[0]).min(p2[0]),
                p0[1].min(p1[1]).min(p2[1]),
                p0[2].min(p1[2]).min(p2[2]),
            ]);
            tri_maxs.push([
                p0[0].max(p1[0]).max(p2[0]),
                p0[1].max(p1[1]).max(p2[1]),
                p0[2].max(p1[2]).max(p2[2]),
            ]);
            tri_centroids.push([
                (p0[0] + p1[0] + p2[0]) * (1.0 / 3.0),
                (p0[1] + p1[1] + p2[1]) * (1.0 / 3.0),
                (p0[2] + p1[2] + p2[2]) * (1.0 / 3.0),
            ]);
        }

        let mut nodes = Vec::with_capacity(count * 2);

        #[allow(clippy::too_many_arguments)]
        fn build_node(
            start: usize,
            end: usize,
            depth: usize,
            tri_indices: &mut [u32],
            tri_mins: &[[f32; 3]],
            tri_maxs: &[[f32; 3]],
            tri_centroids: &[[f32; 3]],
            nodes: &mut Vec<BvhNode>,
        ) -> u32 {
            let mut bmin = [f32::INFINITY; 3];
            let mut bmax = [f32::NEG_INFINITY; 3];
            let mut cmin = [f32::INFINITY; 3];
            let mut cmax = [f32::NEG_INFINITY; 3];

            for &idx in &tri_indices[start..end] {
                let i = idx as usize;
                for axis in 0..3 {
                    bmin[axis] = bmin[axis].min(tri_mins[i][axis]);
                    bmax[axis] = bmax[axis].max(tri_maxs[i][axis]);
                    cmin[axis] = cmin[axis].min(tri_centroids[i][axis]);
                    cmax[axis] = cmax[axis].max(tri_centroids[i][axis]);
                }
            }

            let node_idx = nodes.len() as u32;
            let tri_count = end - start;
            if tri_count <= MAX_LEAF_TRIS || depth >= MAX_BVH_DEPTH {
                nodes.push(BvhNode {
                    min: bmin,
                    max: bmax,
                    first: start as u32,
                    right: 0,
                    count: tri_count as u32,
                });
                return node_idx;
            }

            let extent = [cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2]];
            let axis = if extent[1] > extent[0] && extent[1] >= extent[2] {
                1
            } else if extent[2] > extent[0] && extent[2] >= extent[1] {
                2
            } else {
                0
            };

            let split_val = 0.5 * (cmin[axis] + cmax[axis]);
            let slice = &mut tri_indices[start..end];
            let mut mid = partition_slice(slice, |&idx| tri_centroids[idx as usize][axis] < split_val);

            if mid == 0 || mid == tri_count {
                // Degenerate centroid cluster along this axis: force balanced median split
                mid = tri_count / 2;
                slice.select_nth_unstable_by(mid, |&a, &b| {
                    tri_centroids[a as usize][axis].total_cmp(&tri_centroids[b as usize][axis])
                });
            }

            // Placeholder internal node
            nodes.push(BvhNode {
                min: bmin,
                max: bmax,
                first: 0,
                right: 0,
                count: 0,
            });

            let split = start + mid;
            let left = build_node(
                start,
                split,
                depth + 1,
                tri_indices,
                tri_mins,
                tri_maxs,
                tri_centroids,
                nodes,
            );
            let right = build_node(
                split,
                end,
                depth + 1,
                tri_indices,
                tri_mins,
                tri_maxs,
                tri_centroids,
                nodes,
            );

            nodes[node_idx as usize].first = left;
            nodes[node_idx as usize].right = right;
            node_idx
        }

        build_node(
            0,
            count,
            0,
            &mut tri_indices,
            &tri_mins,
            &tri_maxs,
            &tri_centroids,
            &mut nodes,
        );

        Self {
            model,
            nodes,
            tri_indices,
        }
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Execute a raycast query matching `CDB::COLLIDER::ray_query`.
    ///
    /// - `origin`: world-space ray origin
    /// - `dir`: ray direction (range is expressed in multiples of `dir`, normally unit length)
    /// - `max_range`: maximum parametric distance along `dir`
    pub fn raycast(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        max_range: f32,
        options: RaycastOptions,
    ) -> Vec<RaycastHit> {
        let mut results = Vec::new();
        if self.nodes.is_empty() || max_range <= 0.0 {
            return results;
        }

        let mut limit = max_range;
        let mut stack = Vec::with_capacity(64);
        stack.push(0u32);

        while let Some(node_idx) = stack.pop() {
            let node = self.nodes[node_idx as usize];
            if !ray_aabb_hit(origin, dir, node.min, node.max, limit) {
                continue;
            }

            if node.count > 0 {
                let first = node.first as usize;
                let end = first + node.count as usize;
                for &tri_u32 in &self.tri_indices[first..end] {
                    let tri_idx = tri_u32 as usize;
                    let tri = self.model.triangles[tri_idx];
                    let p0 = self.model.vertices[tri.verts[0] as usize];
                    let p1 = self.model.vertices[tri.verts[1] as usize];
                    let p2 = self.model.vertices[tri.verts[2] as usize];

                    if let Some((range, u, v)) =
                        ray_triangle_xray(origin, dir, p0, p1, p2, options.cull_backfaces)
                        && range > 0.0
                        && range <= limit
                    {
                        let point = [
                            origin[0] + dir[0] * range,
                            origin[1] + dir[1] * range,
                            origin[2] + dir[2] * range,
                        ];
                        let hit = RaycastHit {
                            tri_index: tri_idx,
                            range,
                            u,
                            v,
                            point,
                            verts: [p0, p1, p2],
                            material: tri.material,
                            suppress_shadows: tri.suppress_shadows,
                            suppress_wm: tri.suppress_wm,
                            sector: tri.sector,
                        };

                        if options.only_nearest {
                            limit = range;
                            if results.is_empty() {
                                results.push(hit);
                            } else {
                                results[0] = hit;
                            }
                        } else {
                            results.push(hit);
                            if options.only_first {
                                return results;
                            }
                        }
                    }
                }
            } else {
                // Push farther child first or just explore both
                stack.push(node.right);
                stack.push(node.first);
            }
        }

        results
    }

    /// Convenience helper returning the nearest raycast hit if any.
    pub fn raycast_nearest(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        max_range: f32,
        cull_backfaces: bool,
    ) -> Option<RaycastHit> {
        self.raycast(
            origin,
            dir,
            max_range,
            RaycastOptions {
                cull_backfaces,
                only_first: false,
                only_nearest: true,
            },
        )
        .into_iter()
        .next()
    }

    /// Look up the `GameMaterial` for a `RaycastHit` (`CGameMtlLibrary::GetMaterialByIdx`).
    #[inline]
    pub fn hit_material<'m>(
        &self,
        hit: &RaycastHit,
        mtl_lib: &'m GameMaterialLibrary,
    ) -> Option<&'m GameMaterial> {
        mtl_lib.material_by_idx(hit.material as usize)
    }

    /// Execute an axis-aligned box overlap query matching `CDB::COLLIDER::box_query`.
    ///
    /// When `full_test == true` (`OPT_FULL_TEST`), tests exact triangle-AABB intersection using the
    /// 13-axis separating axis theorem instead of triangle AABB vs query AABB.
    pub fn box_query(
        &self,
        center: [f32; 3],
        half_extents: [f32; 3],
        full_test: bool,
        only_first: bool,
    ) -> Vec<BoxQueryHit> {
        let mut results = Vec::new();
        if self.nodes.is_empty() {
            return results;
        }

        let qmin = [
            center[0] - half_extents[0],
            center[1] - half_extents[1],
            center[2] - half_extents[2],
        ];
        let qmax = [
            center[0] + half_extents[0],
            center[1] + half_extents[1],
            center[2] + half_extents[2],
        ];

        let mut stack = Vec::with_capacity(64);
        stack.push(0u32);

        while let Some(node_idx) = stack.pop() {
            let node = self.nodes[node_idx as usize];
            if !aabb_aabb_overlap(node.min, node.max, qmin, qmax) {
                continue;
            }

            if node.count > 0 {
                let first = node.first as usize;
                let end = first + node.count as usize;
                for &tri_u32 in &self.tri_indices[first..end] {
                    let tri_idx = tri_u32 as usize;
                    let tri = self.model.triangles[tri_idx];
                    let p0 = self.model.vertices[tri.verts[0] as usize];
                    let p1 = self.model.vertices[tri.verts[1] as usize];
                    let p2 = self.model.vertices[tri.verts[2] as usize];

                    let tmin = [
                        p0[0].min(p1[0]).min(p2[0]),
                        p0[1].min(p1[1]).min(p2[1]),
                        p0[2].min(p1[2]).min(p2[2]),
                    ];
                    let tmax = [
                        p0[0].max(p1[0]).max(p2[0]),
                        p0[1].max(p1[1]).max(p2[1]),
                        p0[2].max(p1[2]).max(p2[2]),
                    ];
                    if !aabb_aabb_overlap(tmin, tmax, qmin, qmax) {
                        continue;
                    }

                    if full_test && !tri_aabb_sat(center, half_extents, p0, p1, p2) {
                        continue;
                    }

                    results.push(BoxQueryHit {
                        tri_index: tri_idx,
                        verts: [p0, p1, p2],
                        material: tri.material,
                        sector: tri.sector,
                    });
                    if only_first {
                        return results;
                    }
                }
            } else {
                stack.push(node.right);
                stack.push(node.first);
            }
        }

        results
    }
}

fn partition_slice<T, F>(slice: &mut [T], mut pred: F) -> usize
where
    F: FnMut(&T) -> bool,
{
    let mut left = 0usize;
    let mut right = slice.len();
    while left < right {
        if pred(&slice[left]) {
            left += 1;
        } else {
            right -= 1;
            slice.swap(left, right);
        }
    }
    left
}

#[inline]
fn ray_aabb_hit(origin: [f32; 3], dir: [f32; 3], bmin: [f32; 3], bmax: [f32; 3], max_t: f32) -> bool {
    let mut tmin = 0.0f32;
    let mut tmax = max_t;
    for axis in 0..3 {
        let d = dir[axis];
        if d.abs() < 1e-12 {
            if origin[axis] < bmin[axis] || origin[axis] > bmax[axis] {
                return false;
            }
        } else {
            let inv = 1.0 / d;
            let mut t0 = (bmin[axis] - origin[axis]) * inv;
            let mut t1 = (bmax[axis] - origin[axis]) * inv;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            tmin = tmin.max(t0);
            tmax = tmax.min(t1);
            if tmin > tmax {
                return false;
            }
        }
    }
    true
}

/// Public Möller–Trumbore ray-triangle test matching `CDB::TestRayTri` (`xrCDB_ray.cpp:275-318`).
#[inline]
pub fn ray_triangle_test(
    origin: [f32; 3],
    dir: [f32; 3],
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    cull_backfaces: bool,
) -> Option<(f32, f32, f32)> {
    ray_triangle_xray(origin, dir, p0, p1, p2, cull_backfaces)
}

/// Möller–Trumbore ray-triangle test matching `xrCDB_ray.cpp:275-318`.
#[inline]
fn ray_triangle_xray(
    origin: [f32; 3],
    dir: [f32; 3],
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    cull: bool,
) -> Option<(f32, f32, f32)> {
    const EPS: f32 = 1e-6;
    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let pvec = cross(dir, e2);
    let det = dot(e1, pvec);

    if cull {
        if det < EPS {
            return None;
        }
        let tvec = [origin[0] - p0[0], origin[1] - p0[1], origin[2] - p0[2]];
        let u = dot(tvec, pvec);
        if u < 0.0 || u > det {
            return None;
        }
        let qvec = cross(tvec, e1);
        let v = dot(dir, qvec);
        if v < 0.0 || (u + v) > det {
            return None;
        }
        let inv_det = 1.0 / det;
        Some((dot(e2, qvec) * inv_det, u * inv_det, v * inv_det))
    } else {
        if det.abs() < EPS {
            return None;
        }
        let inv_det = 1.0 / det;
        let tvec = [origin[0] - p0[0], origin[1] - p0[1], origin[2] - p0[2]];
        let u = dot(tvec, pvec) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let qvec = cross(tvec, e1);
        let v = dot(dir, qvec) * inv_det;
        if v < 0.0 || (u + v) > 1.0 {
            return None;
        }
        Some((dot(e2, qvec) * inv_det, u, v))
    }
}

#[inline]
fn aabb_aabb_overlap(amin: [f32; 3], amax: [f32; 3], bmin: [f32; 3], bmax: [f32; 3]) -> bool {
    amin[0] <= bmax[0]
        && amax[0] >= bmin[0]
        && amin[1] <= bmax[1]
        && amax[1] >= bmin[1]
        && amin[2] <= bmax[2]
        && amax[2] >= bmin[2]
}

/// Exact Akenine-Möller separating axis theorem triangle-AABB overlap test (`OPT_FULL_TEST`).
fn tri_aabb_sat(
    box_center: [f32; 3],
    half: [f32; 3],
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
) -> bool {
    let v0 = sub(p0, box_center);
    let v1 = sub(p1, box_center);
    let v2 = sub(p2, box_center);

    let f0 = sub(v1, v0);
    let f1 = sub(v2, v1);
    let f2 = sub(v0, v2);

    // Test 9 cross-product separating axes e_i x f_j
    let edges = [f0, f1, f2];
    for e in &edges {
        // Axis (0, -e.z, e.y)
        if !axis_test_x01(v0, v1, v2, e[2], e[1], half) {
            return false;
        }
        // Axis (e.z, 0, -e.x)
        if !axis_test_y02(v0, v1, v2, e[2], e[0], half) {
            return false;
        }
        // Axis (-e.y, e.x, 0)
        if !axis_test_z12(v0, v1, v2, e[1], e[0], half) {
            return false;
        }
    }

    // 3 box face normals already covered by `aabb_aabb_overlap` caller; test triangle plane normal:
    let normal = cross(f0, f1);
    let r = half[0] * normal[0].abs() + half[1] * normal[1].abs() + half[2] * normal[2].abs();
    let d = dot(normal, v0).abs();
    d <= r
}

#[inline]
fn axis_test_x01(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], ez: f32, ey: f32, half: [f32; 3]) -> bool {
    let p0 = -ez * v0[1] + ey * v0[2];
    let p1 = -ez * v1[1] + ey * v1[2];
    let p2 = -ez * v2[1] + ey * v2[2];
    let r = half[1] * ez.abs() + half[2] * ey.abs();
    let min = p0.min(p1).min(p2);
    let max = p0.max(p1).max(p2);
    min <= r && max >= -r
}

#[inline]
fn axis_test_y02(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], ez: f32, ex: f32, half: [f32; 3]) -> bool {
    let p0 = ez * v0[0] - ex * v0[2];
    let p1 = ez * v1[0] - ex * v1[2];
    let p2 = ez * v2[0] - ex * v2[2];
    let r = half[0] * ez.abs() + half[2] * ex.abs();
    let min = p0.min(p1).min(p2);
    let max = p0.max(p1).max(p2);
    min <= r && max >= -r
}

#[inline]
fn axis_test_z12(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], ey: f32, ex: f32, half: [f32; 3]) -> bool {
    let p0 = -ey * v0[0] + ex * v0[1];
    let p1 = -ey * v1[0] + ex * v1[1];
    let p2 = -ey * v2[0] + ex * v2[1];
    let r = half[0] * ey.abs() + half[1] * ex.abs();
    let min = p0.min(p1).min(p2);
    let max = p0.max(p1).max(p2);
    min <= r && max >= -r
}

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;
    use crate::gamemtl::{gamemtl_chunks, gamemtl_flags, gamemtl_root_chunks};

    fn serialize_cform(verts: &[[f32; 3]], tris: &[CformTriangle], min: [f32; 3], max: [f32; 3]) -> Vec<u8> {
        let mut out = Vec::with_capacity(36 + verts.len() * 12 + tris.len() * 16);
        out.extend_from_slice(&CFORM_CURRENT_VERSION.to_le_bytes());
        out.extend_from_slice(&(verts.len() as u32).to_le_bytes());
        out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
        for f in min.into_iter().chain(max) {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for v in verts {
            for f in v {
                out.extend_from_slice(&f.to_le_bytes());
            }
        }
        for t in tris {
            for idx in t.verts {
                out.extend_from_slice(&idx.to_le_bytes());
            }
            out.extend_from_slice(&t.to_raw_dummy().to_le_bytes());
        }
        out
    }

    fn build_two_material_gamemtl() -> GameMaterialLibrary {
        let make_mtl = |id: u32, name: &str| {
            let mut mw = ChunkWriter::new();
            let mut main = id.to_le_bytes().to_vec();
            main.extend_from_slice(name.as_bytes());
            main.push(0);
            mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_MAIN, &main);
            mw.write_chunk(
                gamemtl_chunks::GAMEMTL_CHUNK_FLAGS,
                &gamemtl_flags::FL_SHOOTABLE.to_le_bytes(),
            );
            mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_PHYSICS, &[0u8; 20]);
            mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_FACTORS, &[0u8; 16]);
            mw.into_bytes()
        };
        let mut mtls = ChunkWriter::new();
        mtls.write_chunk(0, &make_mtl(0, "default"));
        mtls.write_chunk(1, &make_mtl(100, "objects\\wood"));
        let mut root = ChunkWriter::new();
        root.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_VERSION, &1u16.to_le_bytes());
        root.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_AUTOINC, &[0u8; 8]);
        root.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_MTLS, &mtls.into_bytes());
        GameMaterialLibrary::parse(&root.into_bytes()).unwrap()
    }

    #[test]
    fn test_cform_binary_parse_bvh_raycast_and_gamemtl_hit() {
        // Two walls facing +Z (wait: e1 x e2 along (1,0,0)x(0,1,0) = (0,0,1), so ray (0,0,-1) has det > 0)
        // Wall A at z = 2.0 (material idx 0, sector 7, suppress_wm = true)
        // Wall B at z = 5.0 (material idx 1, sector 12, suppress_shadows = true)
        let verts = vec![
            // Wall A (front)
            [-1.0, -1.0, 2.0],
            [-1.0, 1.0, 2.0],
            [1.0, -1.0, 2.0],
            // Wall B (back)
            [-1.0, -1.0, 5.0],
            [-1.0, 1.0, 5.0],
            [1.0, -1.0, 5.0],
            // Flipped triangle on z = 1.0 (culled when OPT_CULL is enabled)
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        let tris = vec![
            CformTriangle {
                verts: [0, 1, 2],
                material: 0,
                suppress_shadows: false,
                suppress_wm: true,
                sector: 7,
            },
            CformTriangle {
                verts: [3, 4, 5],
                material: 1,
                suppress_shadows: true,
                suppress_wm: false,
                sector: 12,
            },
            CformTriangle {
                verts: [6, 7, 8],
                material: 16383,
                suppress_shadows: true,
                suppress_wm: true,
                sector: 65535,
            },
        ];

        let raw = serialize_cform(&verts, &tris, [-1.0, -1.0, 1.0], [1.0, 1.0, 5.0]);
        let model = CformModel::parse(&raw).unwrap();
        assert_eq!(model.triangles[0].material, 0);
        assert!(model.triangles[0].suppress_wm);
        assert!(!model.triangles[0].suppress_shadows);
        assert_eq!(model.triangles[0].sector, 7);
        assert_eq!(model.triangles[2].material, 16383);
        assert_eq!(model.triangles[2].sector, 65535);

        let bvh = model.build_bvh();
        let gmlib = build_two_material_gamemtl();

        // Nearest culled ray skips flipped tri at z=1 and hits Wall A at z=2
        let hit = bvh
            .raycast_nearest([-0.25, -0.25, 0.0], [0.0, 0.0, 1.0], 100.0, true)
            .unwrap();
        assert_eq!(hit.tri_index, 0);
        assert!((hit.range - 2.0).abs() < 1e-5);
        assert_eq!(hit.sector, 7);
        assert_eq!(bvh.hit_material(&hit, &gmlib).unwrap().name, "default");

        // Starting between Wall A and Wall B hits Wall B (`objects\\wood`)
        let hit_b = bvh
            .raycast_nearest([-0.25, -0.25, 3.0], [0.0, 0.0, 1.0], 100.0, true)
            .unwrap();
        assert_eq!(hit_b.tri_index, 1);
        assert!((hit_b.range - 2.0).abs() < 1e-5);
        assert_eq!(bvh.hit_material(&hit_b, &gmlib).unwrap().name, "objects\\wood");

        // Unculled query sees all 3 triangles (z=1, z=2, z=5)
        let all_hits = bvh.raycast(
            [-0.25, -0.25, 0.0],
            [0.0, 0.0, 1.0],
            100.0,
            RaycastOptions::default(),
        );
        assert_eq!(all_hits.len(), 3);

        // OPT_ONLYFIRST returns immediately with 1 hit
        let first_only = bvh.raycast(
            [-0.25, -0.25, 0.0],
            [0.0, 0.0, 1.0],
            100.0,
            RaycastOptions::from_flags(ray_options::OPT_ONLYFIRST),
        );
        assert_eq!(first_only.len(), 1);
    }

    #[test]
    fn test_cform_box_query_sat_distinguishes_aabb_from_triangle() {
        // Large diagonal triangle spanning (0,0,0), (10,0,0), (0,10,0).
        // Its triangle AABB is [0..10, 0..10, 0..0], so a box at (9, 9, 0) half=(0.5, 0.5, 0.5)
        // overlaps the triangle's AABB, yet sits well outside the triangle's diagonal edge (x+y<=10)!
        let model = CformModel::from_parts(
            vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            vec![CformTriangle {
                verts: [0, 1, 2],
                material: 0,
                suppress_shadows: false,
                suppress_wm: false,
                sector: 0,
            }],
        )
        .unwrap();
        let bvh = model.build_bvh();

        // Coarse AABB overlap finds it:
        assert_eq!(bvh.box_query([9.0, 9.0, 0.0], [0.5, 0.5, 0.5], false, false).len(), 1);
        // Exact SAT (`OPT_FULL_TEST`) rejects the non-intersecting corner box:
        assert_eq!(bvh.box_query([9.0, 9.0, 0.0], [0.5, 0.5, 0.5], true, false).len(), 0);
        // Box overlapping the triangle center (2, 2, 0) succeeds under full SAT:
        assert_eq!(bvh.box_query([2.0, 2.0, 0.0], [0.5, 0.5, 0.5], true, false).len(), 1);
    }

    #[test]
    fn test_cform_empty_and_validation_errors() {
        let empty = CformModel::from_parts(vec![], vec![]).unwrap();
        let bvh = empty.build_bvh();
        assert!(bvh.raycast_nearest([0.0; 3], [0.0, 0.0, 1.0], 10.0, true).is_none());
        assert!(bvh.box_query([0.0; 3], [1.0; 3], true, false).is_empty());

        let mut bad_ver = serialize_cform(&[], &[], [0.0; 3], [0.0; 3]);
        bad_ver[0..4].copy_from_slice(&3u32.to_le_bytes());
        assert!(matches!(CformModel::parse(&bad_ver), Err(FormatError::InvalidValue(_))));

        assert!(matches!(
            CformModel::from_parts(
                vec![[0.0; 3]],
                vec![CformTriangle {
                    verts: [0, 1, 0],
                    material: 0,
                    suppress_shadows: false,
                    suppress_wm: false,
                    sector: 0,
                }],
            ),
            Err(FormatError::InvalidValue(_))
        ));
    }
}
