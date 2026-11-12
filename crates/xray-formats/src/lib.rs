//! # xray-formats
//!
//! Pure Rust X-Ray Monolith binary asset & model format parsers:
//! - `ai_graph`: `level.ai` (v10/v13 + A* pathfinder + `pv_compress_normal`/`pv_decompress_normal`), `level.gct` cross-tables & global `game.graph`
//! - `camera`: X-Ray view/projection matrices, `CamEffectorInfo`, `.anm`/`.anms` & `.ppe` players + `CameraManagerState`
//! - `cform`: `level.cform` collision database, pure-Rust AABB BVH raycast (`OPT_CULL`/`OPT_ONLYFIRST`/`OPT_ONLYNEAREST`) & SAT box queries
//! - `chunk`: Zero-copy X-Ray Chunked File Format reader (`ID + size + payload` + `CFS_CompressMark` LZHuf)
//! - `details`: `.dm` & `level.details` (`DETAIL_VERSION_3` / `DETAIL_VERSION_4`) flora/grass slots + SSFX interactive grass benders
//! - `envelopes`: LightWave `CEnvelope` splines (`TCB`/`Hermite`/`Bezier`/`Linear`/`Step`/`Bezier2D`), `.anm` `ObjectMotion` & `.ppe` `PostprocessEffect`
//! - `fonts`: `CGameFont` resolution rung selector (`texture_h<N>` + `[string_table] font_prefix`), UI `.ini` glyph atlases, word-wrap & DX12 glyph quad emitter
//! - `gamemtl`: X-Ray `gamemtl.xr` physical/surface material & interaction pair library parser
//! - `hom`: `level.hom` Hierarchical Occlusion Mapping parser, `ClipFrustum` (frustum/portal/occluder/poly) & 64x64->8 `OccRasterizer`
//! - `lanims`: `$game_data$/lanims.xr` keyframed `LightAnimItem` / `LightAnimLibrary` color anims
//! - `level`: `level` & `level.geom`/`level.geomx` scene, visuals, sectors, portals, portal/HOM traversal, lights & `D3DVERTEXELEMENT9` VB/IB/SWIS decoders
//! - `lighting`: Pure-Rust dynamic lights, `SMAP_Allocator`, `ComputeXFS`, spot/point/GI accumulation, `light_vis`/`smapvis`/grass shadows, BRDF/jitter/normal LUTs, Phase 15 Henyey-Greenstein phase, 16-sample raymarching & Void-and-Cluster blue noise, `LightMarkerState` & `LightTrack` cache
//! - `ogf`: X-Ray OGF v4 static, progressive, hierarchy, LOD (`ScreenLodThresholds`/`LodImposterHwVertex`/batcher/dsgraph sort & pass schedule), tree (`TreeWindState`), and skeletal mesh + bone/IK parser
//! - `omf`: X-Ray `.omf` / embedded `OGF_S_MOTIONS` skeletal animation clips, partitions, channels, `CBlend` & key math parser
//! - `particle_sim`: Pure-Rust PAPI runtime simulator, 3D Perlin noise, `ParticleGroupInstance` timeline & billboard emitter
//! - `particles`: `particles.xr` / loose `.pe` & `.pg` PAPI bytecode systems + LTX blood wallmark configs (no binary `level.wallmarks` in Anomaly/CoP)
//! - `postprocess`: Phase-13/15 PPE noise/quads, bloom/luminance adaptation + bloom/combine quads, R4 RT catalog & phase specs, SSAO/HDAO/SMAA/SSFX TAA & volumetric shafts plans, screen TL/TL2uv quads, tonemap, gamma LUT, HDR10 params & schedules, DOF, gasmask, NVG/heatvision, SSFX blur/SSS helpers
//! - `savegames`: Anomaly `.scop` LZO1X ALife savegame reader/writer (`GameDateTime`, spawn header, dynamic objects & 7 binary registries)
//! - `scoc`: CoC / Anomaly / xrRazom `.scoc` `lmarshal.c` binary Lua marshal codec & typed peer state (`ScocInventoryItem`, `item_remaining_uses`, `visited_levels`, `skills_levels`, `psy_table`, `head_tbl`, `visited_campfires`)
//! - `shaders_xr`: `shaders_xrlc.xr`, `shaders.xr`, `uber_deffer`, R4/internal blender compilers, `level.env_mod`/`.ltx`, `level.ps_static` & `level.fog_vol`
//! - `skinning`: Pure-Rust OGF `m2b_transform` bone palette, multi-channel `AnimatedBlendMixer`, OMF pose evaluation & `1W`–`4W` CPU skinning
//! - `sound_env`: `senvironment.xr` v4/v5, `level.som` BVH acoustic occlusion, `level.snd_env` & `level.snd_static`
//! - `spawn_overlays`: Anomaly `spawn_overlays.ltx` / `mod_spawn_overlays_*.ltx` DLTX graph links/unlinks, patches & `level_cut` applicators
//! - `spawns`: `level.game` (`RPoint`/`PatrolPath`), `patrol_paths.ltx`, `level.spawn`, `all.spawn` (`AlifeSpawnArchive`) & ALife `NET_Packet` entity/subclass codecs
//! - `sun`: Pure-Rust Cascaded Sun Shadow Maps (CSM), dynamic rain wet-surface shadows, `DumbConvexVolume`/`FixedConvexVolume` frustum-light hulls & texel snapping
//! - `thm`: X-Ray `.thm` / `.ltx` texture descriptor parser (`_bump` / `_bump#` / detail / material links)
//! - `weather`: Weather cycles (`weathers\*.ltx`), weather effects, sun positions, ambients, lens flares, thunderbolts & atmospheric mixer
//! - `world_maps`: `game_levels.ltx` (`GameLevelsConfig`), `game_maps_single.ltx` + `level.ltx` PDA/minimap bounds (`WorldMapsConfig`) & coordinate transforms
//! - `xml_ui`: Pure-Rust Windows-1251/UTF-8 UI XML DOM (`#include` & `a:b:c` nav), virtual `1024x768` viewport scaling (`UiViewport`), localization tables, UI atlases & color defs
//! - `vfs_bridge`: Direct `xray-vfs` streaming integration across all binary/LTX/XML/font/savegame assets

pub mod ai_graph;
pub mod camera;
pub mod cform;
pub mod chunk;
pub mod details;
pub mod envelopes;
pub mod fonts;
pub mod gamemtl;
pub mod hom;
pub mod lanims;
pub mod level;
pub mod lighting;
pub mod ogf;
pub mod omf;
pub mod particle_sim;
pub mod particles;
pub mod postprocess;
pub mod savegames;
pub mod scoc;
pub mod shaders_xr;
pub mod skinning;
pub mod sound_env;
pub mod spawn_overlays;
pub mod spawns;
pub mod sun;
pub mod thm;
pub mod vfs_bridge;
pub mod weather;
pub mod world_maps;
pub mod xml_ui;

pub use ai_graph::{
    cover_in_direction_nibbles, pv_compress_normal, pv_decompress_normal,
    vertex_type_matches_mask, AiGraphHeader, AiSearchOutcome, AiSearchParams, AiSearchWorkspace,
    AiVertex, BucketPriorityQueue, CrossTableCell, CrossTableHeader, FixedVertexStorage,
    GameDeathPoint, GameEdge, GameGraph, GameGraphHeader, GameLevelInfo, GameVertex,
    HashVertexStorage, IndexedBinaryHeap, LevelAccessMask, LevelCrossTable, LevelGraph,
    DEFAULT_BUCKET_COUNT, DEFAULT_MAX_VISITED_NODES, INVALID_AI_LINK_V10, INVALID_AI_LINK_V13,
    INVALID_AI_VERTEX, INVALID_GAME_VERTEX, XRAI_CURRENT_VERSION, XRAI_LARGE_VERSION,
};
pub use camera::{
    decode_bc5_normal_xy, decode_tangent_space_normal, f32_to_unorm8, hud_projection_matrix,
    linearize_depth_perspective_lh, linearize_depth_xray_proj, matrix_build_camera,
    matrix_build_camera_dir, matrix_build_projection, matrix_build_projection_hat,
    matrix_build_projection_ortho, matrix_transpose_44, mix_postprocess_effectors,
    ndc_to_uv_dx12, pack_gbuffer_albedo_ao, pack_gbuffer_material,
    pack_gbuffer_normal_roughness, reconstruct_view_pos_from_depth_xray, reconstruct_view_pos_r4,
    reconstruct_world_pos_from_depth_xray, reconstruct_world_pos_from_dx12_camera,
    ui_3d_icon_atlas_projection, ui_3d_icon_projection, unorm8_to_f32, unpack_gbuffer_albedo_ao,
    unpack_gbuffer_material, unpack_gbuffer_normal_roughness, uv_to_ndc_dx12, CamEffectorInfo,
    CameraManagerState, CameraMotionEffector, CameraMotionPlayer, Dx12CameraConstants,
    Dx12DepthDecompressParams, GbufferMrtEncoded, GbufferMrtPixel, PosDecompressParams,
    PostprocessPlayer, RenderXforms, DEFAULT_HUD_FOV_FACTOR, DEFAULT_PARALLAX_H, HUD_BASE_FOV_DEG,
    UI_3D_ICON_DISTANCE, UI_3D_ICON_FAR_PLANE, UI_3D_ICON_NEAR_PLANE, UI_3D_ICON_ORTHO_HEIGHT,
};
pub use cform::{
    classify_sphere_triangle, ray_options, ray_triangle_test, sphere_triangle_test,
    triangle_aabb_overlap, BoxQueryHit, CformBvh, CformHeader, CformModel, CformTriangle,
    RaycastHit, RaycastOptions, SphereTriResult, CFORM_CURRENT_VERSION,
};
pub use chunk::{ByteCursor, ChunkEntry, ChunkIter, ChunkReader, ChunkWriter, FormatError};
pub use details::{
    compute_r_ssa_discard, detail_bwdithermap, detail_hash2, detail_interpolate,
    detail_interpolate_and_dither, quantize_detail_coord, DetailCacheGrid, DetailCacheLevel1,
    DetailCacheSlotState, DetailDecompressConfig, DetailGpuInstanceData, DetailHeader,
    DetailHwBatchMesh, DetailHwObjectSpan, DetailHwVertex, DetailMesh, DetailMeshVertexOut,
    DetailPalette, DetailRandom, DetailSlot, DetailSlotItem, DetailSlotPart, DetailSurfaceHit,
    DetailSwingConfig, DetailSwingState, DetailSwingValue, DetailVertex, DetailVisibilityConfig,
    DetailVisibleBatches, DetailWaveConstants, GrassBenderAnim, GrassBenderHandle, GrassBenderPool,
    GrassBenderSlot, GrassBendersConfig, GrassBendersGpuData, LevelDetails, DETAIL_CACHE1_COUNT,
    DETAIL_DEFAULT_DENSITY, DETAIL_DEFAULT_HEIGHT, DETAIL_DEFAULT_L_AMBIENT,
    DETAIL_DEFAULT_L_ANISO, DETAIL_DEFAULT_PS_SSA_DISCARD, DETAIL_DEFAULT_RADIUS,
    DETAIL_DEFAULT_SSA_DISCARD, DETAIL_HW_QUANT, DETAIL_ID_EMPTY, DETAIL_ID_EMPTY_V3,
    DETAIL_MAX_OBJECTS, DETAIL_OBJ_IN_SLOT, DETAIL_SLOT_HALF_SIZE, DETAIL_SLOT_SIZE,
    DETAIL_VERSION_3, DETAIL_VERSION_4, DO_NO_WAVING, GRASS_BENDERS_MAX_SLOTS,
};
pub use envelopes::{
    decode_q16, encode_q16, Envelope, EnvelopeBehavior, EnvelopeKey, EnvelopeShape, ObjectMotion,
    ObjectMotionLibrary, PostprocessColorCurve, PostprocessEffect, PostprocessGpuParams,
    PostprocessState, EOBJ_OMOTION, EOBJ_OMOTION_VERSION, EOBJ_OMOTION_VERSION_V3,
    EOBJ_OMOTION_VERSION_V4, POSTPROCESS_FILE_VERSION, POSTPROCESS_FILE_VERSION_V1,
};
pub use fonts::{
    font_atlas_ini_vpath, font_flags, is_alpha_character, is_bad_end_character,
    is_bad_start_character, is_need_space_character, mbh_multi2wide, parse_colored_text_lines,
    rank_font_candidates, rank_font_candidates_owned, resolve_font_section,
    resolve_font_section_owned, resolve_font_texture_name, resolve_ui_font_section,
    FontAlignment, FontAtlas, FontAtlasKind, FontDrawParams, FontGlyphCoord, FontGlyphQuad,
    FontRungCandidate, FontSectionConfig, UiColoredSegment, UiMultilineTextOptions,
    UiTextVerticalAlign, STANDARD_FONT_SECTIONS,
};
pub use gamemtl::{
    gamemtl_chunks, gamemtl_flags, gamemtl_pair_chunks, gamemtl_pair_flags,
    gamemtl_root_chunks, GameMaterial, GameMaterialLibrary, GameMaterialPair,
    GAMEMTL_CURRENT_VERSION, GAMEMTL_NONE_ID,
};
pub use hom::{
    frustum_planes, simplify_poly_aabb, viewport_matrix_01, viewport_matrix_64, ClipFrustum,
    FrustumVisibility, HomMesh, HomTriangle, OccRasterizer, Plane, RawHomTriangle, HOM_POLY_SIZE,
    OCC_BORDER, OCC_DIM, OCC_DIM_0, OCC_DIM_1, OCC_DIM_2, OCC_DIM_3, OCC_Q_S32,
};
pub use lanims::{
    lanim_chunks, lanim_item_chunks, pack_argb, pack_rgba_f32, swap_rb_keep_ag, unpack_argb_f32,
    LightAnimItem, LightAnimLibrary, LANIM_VERSION,
};
pub use level::{
    build_light_type, d3d_decl_type, d3d_decl_usage, d3d_light_type, level_chunks,
    portal_traversal_flags, sector_chunks, BuildLightRecord, BuildLights, LevelData, LevelGeometry,
    LevelGlow, LevelLight, LevelPortal, LevelSector, LevelShaderRef, LevelVertexBuffer,
    PortalFadeEntry, PortalScissorRect, PortalSectorVisibility, PortalTraversalOptions,
    PortalTraversalResult, ResolvedLevelMesh, TorchLightConfig, VertexDeclarator, VertexElement9,
    BUILD_LIGHT_RECORD_SIZE, LEVEL_GLOW_RECORD_SIZE,
};
pub use lighting::{
    accum_hemi_cube, blue_noise_jitter_at_pixel, build_brdf_material_lut, build_reflected_lights,
    build_ssfx_volumetric_temporal_jitter_table_16, build_volumetric_slices_mesh,
    check_grass_shadow_light, compute_light_vis_safe_area, compute_point_accum_constants,
    compute_r_ssa_glod_bounds, compute_reflected_accum_constants, compute_spot_accum_constants,
    compute_sun_light_rig, compute_texgen_jitter, compute_texgen_screen, compute_u_diffuse2s,
    compute_volumetric_raymarch_16, compute_volumetric_raymarch_16_dir,
    compute_volumetric_spot_params, compute_volumetric_spot_params_with_fade,
    dual_henyey_greenstein_phase, encode_hemi_material_id, fpack_normal_coord, fpack_normal_z,
    generate_blue_noise_64x64_f32, generate_blue_noise_offset_table_16x16,
    generate_hdao_jitter_texture, generate_indirect_photons, generate_indirect_photons_cform,
    generate_jitter_cell, generate_jitter_textures, generate_ssfx_blue_noise_rgba8,
    hemi_cube_faces, henyey_greenstein_phase, henyey_greenstein_phase_isotropic_normalized,
    henyey_greenstein_phase_rays, henyey_greenstein_phase_to_light,
    integrate_volumetric_raymarch_16, interleaved_gradient_noise, near_plane_intersects_sphere,
    need_to_render_sunshafts, pack_smap_requests, project_volumetric_raymarch_16_to_csm,
    r2_low_discrepancy_offset, sort_light_package_by_key, ssfx_volumetric_distance_fade,
    transform_volumetric_clip_planes, use_minmax_sm_this_frame, volumetric_raymarch_16_step_offsets,
    vpack_normal, vpack_normal_with_radius, vunpack_normal, AccumMaskElement, AccumShaderElement,
    CrngLcg, DynamicLight, GiConfig, GlossConfig, GrassShadowConfig, HemiMaterialGpuParams,
    LightIndirectPhoton, LightMarkerState, LightPackageOrderKey, LightSmapConfig, LightSmapXform,
    LightTrackConfig, LightVisConfig, LightVisDecision, LightVisState, MinMaxSmMode,
    ObjectLightTracker, OmnipartFaceLight, PointAccumConstants, ReflectedAccumConstants,
    ReflectedLightInstance, RenderLightType, RenderVolumeDrawDesc, RenderVolumeGeometryKind,
    SmapAllocator, SmapPackAssignment, SmapRect, SmapVisSolver, SmapVisStage, SpotAccumConfig,
    SpotAccumConstants, SsfxVolumetricDistanceResult, SunLightPose, SunLightRig, SunLightRigConfig,
    TrackedLightEntry, VolumetricCsmSample, VolumetricLightConfig, VolumetricRaymarch16,
    VolumetricRaymarchConfig, VolumetricRaymarchSample, VolumetricSlicesMesh, VolumetricSpotParams,
    DU_BOX_INDICES, DU_BOX_NUMFACES, DU_BOX_NUMVERTEX, DU_BOX_NUMVERTEX2, DU_BOX_VERTICES,
    DU_BOX_VERTICES2, DU_CONE_INDICES, DU_CONE_NUMFACES, DU_CONE_NUMVERTEX, DU_CONE_VERTICES,
    DU_CYLINDER_INDICES, DU_CYLINDER_LINES, DU_CYLINDER_NUMFACES, DU_CYLINDER_NUMLINES,
    DU_CYLINDER_NUMVERTEX, DU_CYLINDER_VERTICES, DU_SPHERE_INDICES, DU_SPHERE_NUMFACES,
    DU_SPHERE_NUMVERTEX, DU_SPHERE_PART_INDICES, DU_SPHERE_PART_NUMFACES,
    DU_SPHERE_PART_NUMVERTEX, DU_SPHERE_PART_VERTICES, DU_SPHERE_VERTICES, GI_DEFAULT_CLIP,
    GI_DEFAULT_PHOTONS, GI_DEFAULT_REFL, GI_SEED, GOLDEN_RATIO_CONJUGATE, HEMI_SAMPLE_DIRS,
    HENYEY_GREENSTEIN_MAX_G, LIGHT_LS_SQUALITY_DEFAULT, LIGHT_MARKER_ID_INIT,
    LIGHT_SLEEP_FRAMES_DEFAULT, LIGHT_SLIGHT_FADE_DEFAULT, LIGHT_SSA_GLOD_END_DEFAULT,
    LIGHT_SSA_GLOD_START_DEFAULT, LIGHT_VIS_CULL_FRAGMENTS, LIGHT_VIS_DELAY_LARGE_MAX,
    LIGHT_VIS_DELAY_LARGE_MIN, LIGHT_VIS_DELAY_SMALL_MAX, LIGHT_VIS_DELAY_SMALL_MIN, LT_DEC,
    LT_HEMISAMPLES, LT_INC, OMNIPART_CUBE_DIRS, OMNIPART_CUBE_UP_HINTS, R2_QUASI_RANDOM_ALPHA,
    SMAP_ADAPT_MAX, SMAP_ADAPT_MIN, SMAP_ADAPT_OPTIMAL, SMAP_NEAR_PLANE,
    SSFX_BLUE_NOISE_OFFSET_TABLE_SIZE, SSFX_BLUE_NOISE_SIZE, SSFX_VOLUMETRIC_DEFAULT_G,
    SSFX_VOLUMETRIC_JITTER_FRAMES, SSFX_VOLUMETRIC_RAYMARCH_SAMPLES, SUN_ADAPTED_DOWN_Y,
    SUN_LIGHT_DISTANCE, SUN_LIGHT_RANGE, TEX_JITTER_COUNT, TEX_JITTER_SIZE, TEX_MATERIAL_COUNT,
    TEX_MATERIAL_LDOTH, TEX_MATERIAL_LDOTN, VOLUMETRIC_SLICES,
};
pub use ogf::{
    adjusted_distance_from_camera, bone_shape_flags, build_custom_fov_projection, calc_ssa,
    compute_screen_lod_thresholds, deffer_shader_elements, dsgraph_pass_schedule, fvf_flags,
    is_valuable_to_render_dynamic, is_valuable_to_render_static, lod_shader_element_id,
    lod_shader_elements, matrix_forward_dir, matrix_invert_43, matrix_mul_43, matrix_mul_44,
    matrix_rotate_x, matrix_rotate_y, matrix_rotate_z, matrix_scale, matrix_set_hpb,
    matrix_set_xyz_translate, matrix_set_xyzi_translate, matrix_transform_point,
    matrix_transform_projected, ogf_chunks, pack_lod_imposter_batches, route_dynamic_visual,
    route_static_visual, select_dynamic_shader_element, select_static_shader_element,
    skel_vert_type, sort_dsgraph_texture_nodes, sphere_volume, vector_generate_orthonormal_basis,
    BoneIkData, BoneIkSection, BonePoseMatrices, BoneShape, BoneShapeType, DsgraphPassGroup,
    DsgraphProjectionKind, DsgraphQueueBucket, DsgraphQueueStep, DsgraphRt3Slot,
    DsgraphTextureNode, DsgraphTraverseOrder, LodBatchChunk, LodDrawGroup, LodImposterHwVertex,
    LodImposterSample, LodRenderInput, OgfBBox, OgfBSphere, OgfBone, OgfBufferRef, OgfDesc,
    OgfGContainerRef, OgfHeader, OgfLodDef, OgfLodFacet, OgfLodVertex, OgfModel, OgfObb,
    OgfSkeleton, OgfSlideWindows, OgfTextureLink, OgfTreeDef, OgfVertices, OgfVisualType,
    RenderPhase, ScreenLodConfig, ScreenLodThresholds, SlideWindow, StaticVertex,
    StaticVertexLayout, TreeInstanceGpuConstants, TreeLightingColor, TreeWindConfig,
    TreeWindFrame, TreeWindState, UnifiedSkinnedVertex, VertBoned1W, VertBoned2W, VertBoned3W,
    VertBoned4W, VisualOptConfig, VisualPrimaryBucket, VisualRoutingDecision, VisualRoutingFlags,
    XRayMatrix, FTREE_VISUAL_QUANT, FTREE_VISUAL_TILE, LOD_IMPOSTER_QUAD_ORDER, OGF_LODDEF2_SIZE,
    OGF_LOD_VERTEX_SIZE, OGF_TREEDEF2_SIZE, VISUAL_OPT_BASE_FOV, VISUAL_OPT_DYNAMIC_DIST,
    VISUAL_OPT_DYNAMIC_VOL, VISUAL_OPT_STATIC_DIST, VISUAL_OPT_STATIC_VOL, XRAY_IDENTITY_MATRIX,
};
pub use omf::{
    motion_def_flags, motion_key_flags, slerp_quat_xyzw, AnimationBlend, AnimationChannels,
    BlendCurvature, BoneKeyframe, BoneMotionTrack, BonePartitionDef, BonePartitionSet,
    BoneTranslationKeys, ChannelDef, ChannelMixType, ChannelRule, MotionClip, MotionDef,
    MotionID, MotionMark, MotionPartition, MotionPartitionBone, OmfMotions, QuantizedRotKey,
    CHANNEL_RULES, KEY_QUANT, KEY_QUANT_INV, MAX_ANIM_CHANNELS, MAX_ANIM_SLOTS,
    MAX_BLENDED_PER_CHANNEL, MAX_BLENDED_POOL, MAX_PARTS, MOTION_QUANTIZER_RANGE_EXT, SAMPLE_FPS,
    SAMPLE_SPF, XR_OGF_SM_PARAMS_VERSION,
};
pub use particle_sim::{
    intersect_segment_domain, DomainDeflectorHit, ParticleBillboardDrawBatch,
    ParticleBillboardInstance, ParticleCollisionConfig, ParticleCollisionHit,
    ParticleCollisionReport, ParticleDynamicHit, ParticleEffectInstance, ParticleGroupInstance,
    ParticleLifeEvent, ParticleQuadVertex, ParticleRng, ParticleState, ParticleStepEvents,
    PerlinNoise3, DEFAULT_COLLISION_PARAMS,
};
pub use particles::{
    build_wallmark_view_matrix, calc_triangle_adjacency, encode_compiled_action_list,
    pack_triangle_vertices, papi_flags, parse_compiled_action_list, ped_chunks, ped_flags,
    pg_effect_flags, pgd_chunks, project_skeleton_wallmark, project_static_wallmark_from_tris,
    project_static_wallmark_on_cform, ps_chunks, BloodWallmarksConfig, CompiledParticleAction,
    GroupEffectItem, ParticleActionKind, ParticleCollisionParams, ParticleDomain,
    ParticleDomainType, ParticleEffectDef, ParticleGroupDef, ParticleSpriteFrame, ParticlesLibrary,
    SkeletonWallmarkConfig, SkeletonWallmarkFace, SkeletonWallmarkPool, SkeletonWallmarkRecord,
    StaticWallmarkBatch, StaticWallmarkConfig, StaticWallmarkPool, StaticWallmarkRecord,
    WallmarkSlotConfig, WallmarkVertex, PARTICLE_DOMAIN_BINARY_SIZE, PED_VERSION, PGD_VERSION,
    PS_VERSION, WALLMARK_COS_88_DEG, WALLMARK_DEFAULT_TTL, WALLMARK_DUPLICATE_TOLERANCE,
    WALLMARK_RANGE_SKELETON, WALLMARK_RANGE_STATIC, WALLMARK_SHIFT_PP, WALLMARK_SHIFT_V,
};
pub use postprocess::{
    apply_st2084_pq_rgb, apply_vignette_rgb, bloom_build_4tap_uvs, bloom_filter_taps_horizontal,
    bloom_filter_taps_vertical, build_bloom_build_quad, build_bloom_fast_filter_quad,
    build_bloom_filter_quad_horizontal, build_bloom_filter_quad_vertical, build_combine2_aa_quad,
    build_combine_volumetric_quad, build_create_minmax_sm_quad, build_draw_rain_light_quad,
    build_draw_rain_mask_quad, build_mark_msaa_edges_quad, build_phase_ssfx_rain_quad,
    build_postprocess_pass_plan, build_r4_render_target_catalog, build_ssfx_bloom_quad,
    build_ssfx_volumetric_shafts_plan, build_stencil_optimize_quad,
    build_ui_3d_icons_postprocess_quad, calc_gauss_k7, calc_gauss_wave,
    combine2_aa_quad_uvs, combine2_shader_element, compute_combine1_params, compute_hdao_dispatch,
    compute_ssao_params, effects_sun_hdr_blend_factor, luminance_filter_quad_uvs,
    luminance_filter_taps, luminance_pass0_uvs, lut_pass_desc, phase_blur_schedule, r4_phase_spec,
    smaa_schedule, ssfx_ao_blur_schedule, ssfx_bloom_lens_blur_schedule, ssfx_bloom_schedule,
    ssfx_il_blur_schedule, ssfx_taa_schedule, ssfx_volumetric_blur_schedule, sunshafts_schedule,
    sunshafts_uv_bounds, uncharted2_tonemap_curve, BloomAdaptState, BloomBuildVertex, BloomConfig,
    BloomFilterGpuWeights, BloomFilterVertex, Combine1Config, Combine1GpuParams,
    Combine2AaCornerUvs, Combine2AaVertex, Combine2Config, Combine2GpuParams, Combine2MotionState,
    DofConfig, DofGpuParams, GammaControlConfig, GasmaskConfig, GasmaskGpuParams, HdaoDispatch,
    Hdr10Colorspace, Hdr10Config, Hdr10GpuParams, Hdr10PassKind, Hdr10PingPongPass,
    Hdr10TonemapMode, Hdr10Tonemapper, HeatVisionConfig, HeatVisionGpuParams,
    ImageCorrectionConfig, ImageCorrectionGpuParams, LuminanceAdaptState, LuminanceConfig,
    LutPassDesc, NightVisionGen, NightVisionProfile, OcclusionQueryAllocation, OcclusionQueryPool,
    PhasePpVertex, PostprocessNoiseState, PostprocessPassPlan, R4DepthSlot, R4PhaseClearSpec,
    R4PhaseStateOptions, R4RenderPhaseKind, R4RenderPhaseSpec, R4RenderTargetConfig,
    R4RenderTargetSpec, R4RtSlot, R4TargetFormat, ScreenTl2uvVertex, ScreenTlVertex, SmaaPassDesc,
    SmaaPassKind, SsaoParams, SsfxBloomGpuParams, SsfxBloomPassDesc, SsfxBloomPassKind,
    SsfxBlurPassDesc, SsfxLightingConfig, SsfxLutConfig, SsfxScreenConfig, SsfxSssCandidate,
    SsfxSssExtSlots, SsfxTaaConfig, SsfxTaaPassDesc, SsfxTaaPassKind, SsfxVolumetricPassPlan,
    SsfxVolumetricShaftsGpuParams, SunshaftsConfig,
    SunshaftsPassDesc, SunshaftsPassKind, TonemapGpuParams, TonemapParams, BLOOM_SIZE_X,
    BLOOM_SIZE_Y, HDAO_GROUP_TEXEL_DIMENSION, HDAO_GROUP_TEXEL_OVERLAP, HDAO_GROUP_TEXEL_STEP,
    LUT_ATLAS_TEXTURE, LUT_NOISE_TEXTURE, LUT_SHADER_NAME, OCCQ_DEFAULT_LIMIT,
    OCCQ_INVALID_HANDLE, SSFX_SSS_EXT_MAX_SLOTS,
};
pub use savegames::{
    compress_scop, decompress_scop, validate_scop_header, ActorStatDetail, ActorStatSection,
    GameDateTime, GameNewsEntry, GameTaskRegistryData, RelationData, SaveGameTime,
    SaveObjectRecord, SaveRegistries, SaveSpawnHeader, SavedGameTask, SavedMapLocation,
    SavedTaskScriptHelper, SavegameFile, ALIFE_CHUNK_DATA, ALIFE_VERSION, GAME_TIME_CHUNK_DATA,
    OBJECT_CHUNK_DATA, REGISTRY_CHUNK_DATA, SAVE_EXTENSION, SAVE_MAGIC, SPAWN_CHUNK_DATA,
};
pub use scoc::{
    decode_marshal_value, encode_marshal_value, validate_scoc_header, ScocCampfireEntry,
    ScocCampfireKey, ScocFile, ScocHeadTable, ScocHeader, ScocInventoryItem, ScocPeerState,
    ScocPsyTable, ScocSkillLevel, ScocTable, ScocValue, LUA_TBOOLEAN, LUA_TFUNCTION,
    LUA_TLIGHTUSERDATA, LUA_TNIL, LUA_TNUMBER, LUA_TSTRING, LUA_TTABLE, LUA_TTHREAD,
    LUA_TUSERDATA, MAR_TREF, MAR_TUSR, MAR_TVAL, SCOC_EXTENSION, SCOC_MAGIC,
    SCOC_MAX_RECURSION_DEPTH,
};
pub use shaders_xr::{
    clsid_to_string, env_mod_flags, fix_texture_name, matrix_mode, plan_r4_blender_compile,
    plan_r4_internal_blender, plan_uber_deffer, plan_uber_shadow, resolve_ui_shader_name,
    shader_xrlc_flags, tcm_flags, transform_uv, ui_primitive_count, BlendFactor, BlendState,
    BlenderClsid, BlenderDesc, BlenderProperty, BlenderPropertyValue, BlenderShader, CmpFunc,
    CompiledPassPlan, CompiledShaderPlan, CullMode, DefferElement, Dx10MsaaAlphaTest,
    EngineShaderLibrary, EnvModAccum, EnvModifier, FogVolume, LevelEnvModifiers, LevelFogVolumes,
    LevelStaticParticles, R4BlenderSpec, R4CompileConfig, R4InternalBlenderKind,
    R4InternalBlenderOptions, SamplerBindingSpec, SamplerFilterMode, ShaderConstant, ShaderMatrix,
    ShaderXrlcEntry, ShaderXrlcLibrary, StaticParticle, StencilOp, StencilState, TessMethod,
    TexAddressMode, TextureBindingSpec, UberDefferInput, UberShadowInput, UiPointType,
    UiPrimitiveType, WaveForm, WaveFunction, CBLENDER_DESC_SIZE, SHADER_XRLC_ENTRY_SIZE,
};
pub use skinning::{
    compute_skinned_aabb, matrix_transform_dir, normalize_vec3, pick_ogf_bone,
    pick_ogf_single_bone, skin_ogf_hierarchy, skin_position_with_links,
    skin_position_with_wm_sentinel, skin_vertex_pos_and_influences, skin_vertices,
    AnimatedBlendInstance, AnimatedBlendMixer, AnimatedPoseResult, BoneMask, BoneMatrix4x3,
    BonePalette, BonePickHit, SkeletonCollisionIndex, SkeletonLeafInfo, SkinnedMeshPart,
    SkinnedVertex, SkinnedVertexInfluence,
};
pub use sound_env::{
    LevelSoundEnv, LevelStaticSounds, OggSoundMetadata, OggXRayParams, SomPolygon,
    SoundEnvironment, SoundEnvironmentLibrary, SoundOcclusionMesh, StaticSound,
    OGG_COMMENT_VERSION, SDEF_ENV_VERSION, SNDENV_VER_EXTENDED, SOM_POLY_SIZE,
};
pub use spawn_overlays::{
    parse_endpoint, unescape_custom_data, OverlayArchiveReport, OverlayEndpoint,
    OverlayGraphLink, OverlayGraphReport, OverlayGraphUnlink, OverlayObjectSelector,
    OverlaySpawnPatch, SpawnOverlaysConfig, DEFAULT_LINK_TOLERANCE, MATCH_TOLERANCE,
};
pub use spawns::{
    alife_object_flags, cse_spawn_flags, hanging_lamp_flags, m_spawn_flags, ph_skeleton_flags,
    wayobject_chunks, AlifeAnomalousZoneState, AlifeCarState, AlifeCreatureAbstractState,
    AlifeCreatureActorState, AlifeCustomZoneState, AlifeDynamicVisualState, AlifeHelicopterState,
    AlifeHumanAbstractState, AlifeHumanStalkerState, AlifeInventoryBoxState,
    AlifeInventoryItemState, AlifeItemAmmoState, AlifeItemArtefactState, AlifeItemBoltState,
    AlifeItemCustomOutfitState, AlifeItemDetectorState, AlifeItemDocumentState,
    AlifeItemExplosiveState, AlifeItemGrenadeState, AlifeItemHelmetState, AlifeItemPdaState,
    AlifeItemTorchState, AlifeItemWeaponAutoShotgunState, AlifeItemWeaponMagazinedState,
    AlifeItemWeaponMagazinedWglState, AlifeItemWeaponShotgunState, AlifeItemWeaponState,
    AlifeLevelChangerState, AlifeMonsterAbstractState, AlifeMonsterBaseState,
    AlifeMountedWeaponState, AlifeObjectBreakableState, AlifeObjectClimableState,
    AlifeObjectHangingLampState, AlifeObjectPhysicState, AlifeObjectState,
    AlifeOnlineOfflineGroupState, AlifeSmartCoverState, AlifeSmartZoneState,
    AlifeSpaceRestrictorState, AlifeSpawnArchive, AlifeStationaryMgunState,
    AlifeTeamBaseZoneState, AlifeTorridZoneState, AlifeTraderAbstractState, AlifeTraderState,
    AlifeZoneVisualState, GraphPointState, LevelGameData, LevelSpawnData, PatrolEdge, PatrolPath,
    PatrolPoint, RPoint, RPointType, ServerMotionState, ServerPhBoneState, ServerPhBonesData,
    ServerPhSkeletonState, ServerShapeDef, ServerVisualState, SpawnEdge, SpawnHeader, SpawnPacket,
    SpawnVertex, INVALID_OBJECT_ID, INVALID_SPAWN_STORY_ID, INVALID_STORY_ID, M_SPAWN, M_UPDATE,
    POINT_BASE, RPOINT_CHUNK, SPAWN_VERSION, WAYOBJECT_VERSION, WAY_BASE, WAY_PATROLPATH_CHUNK,
};
pub use sun::{
    build_3split_csm_atlas, build_accum_direct_lum_quad, compute_accum_direct_cuboid_vertices,
    compute_accum_direct_depth_bounds, compute_accum_direct_volumetric_range,
    compute_csm_cascade_blend, compute_direct_mask_dir, compute_rain_shadow_setup,
    compute_sun_cascades, compute_sun_shadow_matrix, compute_view_shadow_proj, matrix_invert_44,
    matrix_ortho_off_center_lh, matrix_translate, project_world_to_csm_atlas,
    select_csm_cascade, snap_sun_cascade_to_texels, ssfx_rain_pass_viewport, sun_pass_elements,
    AccumDirectLumVertex, CloudsShadowState, CsmAtlasPacking, CsmAtlasPlan, CsmCascadeAtlasSlot,
    CsmCascadeBlend, DumbConvexVolume, FixedConvexVolume, GpuCsmSplitUniforms, RainShadowResult,
    RainWetSurfConfig, SsfxRainConfig, SunCascadeBiasConfig, SunCascadeConfig, SunCascadeResult,
    SunCascadeSpec, SunRay, ACCUM_DIRECT_CUBOID_CORNERS, ACCUM_DIRECT_CUBOID_INDICES,
    ACCUM_DIRECT_LUM_SMOOTH, SUN_FAR_LIMIT_DEFAULT, SUN_FRUSTUM_CORNERS, SUN_FRUSTUM_FACETABLE,
};
pub use thm::{
    dds_caps2, detail_usage, strip_texture_extension, texture_flags, texture_stem_to_dds_vpath,
    texture_stem_to_thm_vpath, thm_chunks, BumpTextureLinks, CubemapFace, DdsCubemapLayout,
    DdsPixelFormatKind, DdsSubresourceSlice, DetailScaler, ResolvedTextureLinks, TextureAssoc,
    TextureBumpMode, TextureDescrEntry, TextureDescrTable, TextureFormat, TextureMaterial,
    TextureParams, TextureSpec, TextureType, DDS_DX10_HEADER_BYTES, DDS_HEADER_BYTES, DDS_MAGIC,
    DTEX_RANGE_DEFAULT,
};
pub use vfs_bridge::{
    discover_spawn_overlay_mods_from_vfs, load_all_spawn_from_vfs, load_anm_from_vfs,
    load_anms_from_vfs, load_blood_wallmarks_from_vfs, load_bone_parts_ltx_from_vfs,
    load_build_lights_from_vfs, load_cform_from_vfs, load_color_defs_from_vfs,
    load_cubemap_dds_from_vfs, load_details_from_vfs, load_dm_from_vfs, load_env_mods_from_vfs,
    load_fog_volumes_from_vfs, load_font_atlas_from_vfs, load_font_from_vfs,
    load_game_graph_from_vfs, load_game_levels_from_vfs, load_gamemtl_from_vfs,
    load_hom_from_vfs, load_lanims_from_vfs, load_level_ai_from_vfs, load_level_from_vfs,
    load_level_game_from_vfs, load_level_gct_from_vfs, load_level_geometry_from_vfs,
    load_level_sound_env_from_vfs, load_level_spawn_from_vfs, load_localization_table_from_vfs,
    load_ltx_tree_from_vfs, load_ogg_sound_metadata_from_vfs, load_ogf_from_vfs, load_omf_from_vfs,
    load_particles_from_vfs, load_patrol_paths_ltx_from_vfs, load_ppe_from_vfs,
    load_ps_static_from_vfs, load_scoc_from_vfs, load_scop_from_vfs, load_shaders_xr_from_vfs,
    load_shaders_xrlc_from_vfs, load_som_from_vfs, load_sound_environments_from_vfs,
    load_spawn_overlays_from_vfs, load_static_sounds_from_vfs, load_thm_from_vfs,
    load_torch_light_config_from_vfs, load_ui_texture_atlases_from_vfs,
    load_ui_xml_for_viewport_from_vfs, load_weather_library_from_vfs, load_world_maps_from_vfs,
    load_xml_from_vfs, resolve_font_config_from_vfs, resolve_ogf_texture_links, AssetLoadError,
    VfsLtxProvider, VfsXmlProvider,
};
pub use weather::{
    append_quad_indices, calculate_dynamic_sun_dir, clouds_hemi_mesh, dir_from_hp, hp_from_dir,
    lerp_descriptors, normalize_time, parse_hhmmss_seconds, parse_ltx_str, skybox_hbox_mesh,
    time_diff, time_weight, ActorShaderData, CloudsDomeVertex, ColorMapCacheState, DevDebugParams,
    EnvAmbientDef, EnvAmbientEffectDef, EnvBoost, EnvDescriptor, EnvGlobalParams, EnvLitVertex,
    EnvSoundChannelDef, FakeScopeConfig, FakeScopeUniforms, GlobalShaderConstants, LensFlareDef,
    LensFlareDrawQuad, LensFlareElement, LensFlareFrameOutput, LensFlarePlayState,
    LensFlareQuadKind, LensFlareState, MixedEnvironment, PdaShaderParams, PerlinNoise1D,
    RainDropItem, RainEffectConfig, RainEffectState, RainFrameOutput, RainSplashInstance,
    RainSplashParticle, S3dsScopeUniforms, ScopeEyeLerpState, SkyboxVertex, SsfxSurfaceConfig,
    SsfxSurfaceUniforms, StandardFrameUniforms, SunPositionsConfig, ThunderboltBoltDraw,
    ThunderboltCollectionDef, ThunderboltDescDef, ThunderboltEffectState, ThunderboltFlareDef,
    ThunderboltFrameOutput, ThunderboltGradientQuad, ThunderboltPlayState, WeatherCycle,
    WeatherLibrary, WetnessState, WindAnimState, CLOUDS_HEMI_INDICES, CLOUDS_HEMI_INDEX_COUNT,
    CLOUDS_HEMI_POSITIONS, CLOUDS_HEMI_TRIANGLE_COUNT, CLOUDS_HEMI_VERTEX_COUNT, DAY_LENGTH,
    DEFAULT_SCOPE_FOG_INTERP, DEFAULT_VIEWPORT_NEAR, DEFAULT_WETNESS_MULTIPLIERS,
    LENS_FLARE_BLEND_DEC_SPEED, LENS_FLARE_BLEND_INC_SPEED, LENS_FLARE_MAX_RAYS,
    LENS_FLARE_RAY_DELTAS, LENS_FLARE_RAY_MAX_DIST, LENS_FLARE_RAY_SCALE, QUAD_INDEX_PATTERN,
    RAIN_CPU_SOURCE_RADIUS, RAIN_DROP_ANGLE_RAD, RAIN_DROP_LENGTH, RAIN_DROP_MAX_ANGLE_RAD,
    RAIN_DROP_SPEED_MAX, RAIN_DROP_SPEED_MIN, RAIN_DROP_WIDTH, RAIN_MAX_DESIRED_ITEMS,
    RAIN_MAX_DISTANCE, RAIN_MAX_PARTICLES, RAIN_SINK_OFFSET, RAIN_SOURCE_OFFSET,
    RAIN_SOURCE_RADIUS, RAIN_SPLASH_TIME_SEC, RAIN_STREAK_UVS, R_VIEWPORT_NEAR,
    SKYBOX_HBOX_INDEX_COUNT, SKYBOX_HBOX_INDICES, SKYBOX_HBOX_POSITIONS, SKYBOX_HBOX_TEXCOORDS,
    SKYBOX_HBOX_TRIANGLE_COUNT, SKYBOX_HBOX_VERTEX_COUNT, WETNESS_CHANGE_TIME, WFX_TRANS_TIME,
};
pub use world_maps::{
    parse_ltx_color_argb, DefaultMapDef, GameLevelsConfig, GlobalMapDef, LevelIdSpec,
    LevelListEntry, LevelMapDef, WorldMapsConfig, DEFAULT_MAP_LABEL_COLOR_ARGB,
};
pub use xml_ui::{
    argb_to_rgba_f32, color_argb, decode_cp1251, decode_xray_text, expand_action_placeholders,
    expand_xml_includes, subst_alpha, unescape_xray_newlines, LocalizationTable, UiColorDefs,
    UiTextureAtlasMap, UiTextureRef, UiTextureRegion, UiTextDef, UiViewport, UiWindowDef,
    XmlDocument, XmlFileProvider, XmlNode, UI_BASE_HEIGHT, UI_BASE_WIDTH,
};
