//! # xray-formats
//!
//! Pure Rust X-Ray Monolith binary asset & model format parsers:
//! - `ai_graph`: `level.ai` (v10/v13 + A* pathfinder + `pv_compress_normal`/`pv_decompress_normal`), `level.gct` cross-tables & global `game.graph`
//! - `camera`: X-Ray view/projection matrices, `CamEffectorInfo`, `.anm`/`.anms` & `.ppe` players + `CameraManagerState`
//! - `cform`: `level.cform` collision database, pure-Rust AABB BVH raycast (`OPT_CULL`/`OPT_ONLYFIRST`/`OPT_ONLYNEAREST`) & SAT box queries
//! - `chunk`: Zero-copy X-Ray Chunked File Format reader (`ID + size + payload` + `CFS_CompressMark` LZHuf)
//! - `details`: `.dm` & `level.details` (`DETAIL_VERSION_3` / `DETAIL_VERSION_4`) flora/grass slots
//! - `envelopes`: LightWave `CEnvelope` splines (`TCB`/`Hermite`/`Bezier`/`Linear`/`Step`/`Bezier2D`), `.anm` `ObjectMotion` & `.ppe` `PostprocessEffect`
//! - `fonts`: `CGameFont` resolution rung selector (`texture_h<N>` + `[string_table] font_prefix`), UI `.ini` glyph atlases, word-wrap & DX12 glyph quad emitter
//! - `gamemtl`: X-Ray `gamemtl.xr` physical/surface material & interaction pair library parser
//! - `hom`: `level.hom` Hierarchical Occlusion Mapping parser, `ClipFrustum` & 64x64->8 `OccRasterizer`
//! - `lanims`: `$game_data$/lanims.xr` keyframed `LightAnimItem` / `LightAnimLibrary` color anims
//! - `level`: `level` & `level.geom`/`level.geomx` scene, visuals, sectors, portals, lights & `D3DVERTEXELEMENT9` VB/IB/SWIS decoders
//! - `ogf`: X-Ray OGF v4 static, progressive, hierarchy, LOD, tree, and skeletal mesh + bone/IK parser
//! - `omf`: X-Ray `.omf` / embedded `OGF_S_MOTIONS` skeletal animation clips, partitions, blending & marks parser
//! - `particle_sim`: Pure-Rust PAPI runtime simulator, 3D Perlin noise, `ParticleGroupInstance` timeline & billboard emitter
//! - `particles`: `particles.xr` / loose `.pe` & `.pg` PAPI bytecode systems + LTX blood wallmark configs (no binary `level.wallmarks` in Anomaly/CoP)
//! - `savegames`: Anomaly `.scop` LZO1X ALife savegame reader/writer (`GameDateTime`, spawn header, dynamic objects & 7 binary registries)
//! - `shaders_xr`: `shaders_xrlc.xr`, `shaders.xr`, `level.env_mod`/`.ltx`, `level.ps_static` & `level.fog_vol`
//! - `skinning`: Pure-Rust OGF `m2b_transform` bone palette, OMF pose evaluation & `1W`–`4W` CPU skinning
//! - `sound_env`: `senvironment.xr` v4/v5, `level.som` BVH acoustic occlusion, `level.snd_env` & `level.snd_static`
//! - `spawn_overlays`: Anomaly `spawn_overlays.ltx` / `mod_spawn_overlays_*.ltx` DLTX graph links/unlinks, patches & `level_cut` applicators
//! - `spawns`: `level.game` (`RPoint`/`PatrolPath`), `patrol_paths.ltx`, `level.spawn`, `all.spawn` (`AlifeSpawnArchive`) & ALife `NET_Packet` entity/subclass codecs
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
pub mod ogf;
pub mod omf;
pub mod particle_sim;
pub mod particles;
pub mod savegames;
pub mod shaders_xr;
pub mod skinning;
pub mod sound_env;
pub mod spawn_overlays;
pub mod spawns;
pub mod thm;
pub mod vfs_bridge;
pub mod weather;
pub mod world_maps;
pub mod xml_ui;

pub use ai_graph::{
    pv_compress_normal, pv_decompress_normal, AiGraphHeader, AiVertex, CrossTableCell,
    CrossTableHeader, GameDeathPoint, GameEdge, GameGraph, GameGraphHeader, GameLevelInfo,
    GameVertex, LevelCrossTable, LevelGraph, INVALID_AI_LINK_V10, INVALID_AI_LINK_V13,
    INVALID_AI_VERTEX, INVALID_GAME_VERTEX, XRAI_CURRENT_VERSION, XRAI_LARGE_VERSION,
};
pub use camera::{
    matrix_build_camera, matrix_build_camera_dir, matrix_build_projection,
    matrix_build_projection_hat, matrix_build_projection_ortho, mix_postprocess_effectors,
    CamEffectorInfo, CameraManagerState, CameraMotionEffector, CameraMotionPlayer,
    PostprocessPlayer,
};
pub use cform::{
    ray_options, ray_triangle_test, BoxQueryHit, CformBvh, CformHeader, CformModel, CformTriangle,
    RaycastHit, RaycastOptions, CFORM_CURRENT_VERSION,
};
pub use chunk::{ByteCursor, ChunkEntry, ChunkIter, ChunkReader, ChunkWriter, FormatError};
pub use details::{
    compute_r_ssa_discard, detail_bwdithermap, detail_hash2, detail_interpolate,
    detail_interpolate_and_dither, quantize_detail_coord, DetailCacheGrid, DetailCacheLevel1,
    DetailCacheSlotState, DetailDecompressConfig, DetailGpuInstanceData, DetailHeader,
    DetailHwBatchMesh, DetailHwObjectSpan, DetailHwVertex, DetailMesh, DetailMeshVertexOut,
    DetailPalette, DetailRandom, DetailSlot, DetailSlotItem, DetailSlotPart, DetailSurfaceHit,
    DetailSwingConfig, DetailSwingState, DetailSwingValue, DetailVertex, DetailVisibilityConfig,
    DetailVisibleBatches, DetailWaveConstants, LevelDetails, DETAIL_CACHE1_COUNT,
    DETAIL_DEFAULT_DENSITY, DETAIL_DEFAULT_HEIGHT, DETAIL_DEFAULT_L_AMBIENT,
    DETAIL_DEFAULT_L_ANISO, DETAIL_DEFAULT_PS_SSA_DISCARD, DETAIL_DEFAULT_RADIUS,
    DETAIL_DEFAULT_SSA_DISCARD, DETAIL_HW_QUANT, DETAIL_ID_EMPTY, DETAIL_ID_EMPTY_V3,
    DETAIL_MAX_OBJECTS, DETAIL_OBJ_IN_SLOT, DETAIL_SLOT_HALF_SIZE, DETAIL_SLOT_SIZE,
    DETAIL_VERSION_3, DETAIL_VERSION_4, DO_NO_WAVING,
};
pub use envelopes::{
    decode_q16, encode_q16, Envelope, EnvelopeBehavior, EnvelopeKey, EnvelopeShape, ObjectMotion,
    ObjectMotionLibrary, PostprocessColorCurve, PostprocessEffect, PostprocessState, EOBJ_OMOTION,
    EOBJ_OMOTION_VERSION, EOBJ_OMOTION_VERSION_V3, EOBJ_OMOTION_VERSION_V4,
    POSTPROCESS_FILE_VERSION, POSTPROCESS_FILE_VERSION_V1,
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
    frustum_planes, viewport_matrix_01, viewport_matrix_64, ClipFrustum, HomMesh, HomTriangle,
    OccRasterizer, Plane, RawHomTriangle, HOM_POLY_SIZE, OCC_BORDER, OCC_DIM, OCC_DIM_0,
    OCC_DIM_1, OCC_DIM_2, OCC_DIM_3, OCC_Q_S32,
};
pub use lanims::{
    lanim_chunks, lanim_item_chunks, pack_argb, pack_rgba_f32, swap_rb_keep_ag, unpack_argb_f32,
    LightAnimItem, LightAnimLibrary, LANIM_VERSION,
};
pub use level::{
    build_light_type, d3d_decl_type, d3d_decl_usage, d3d_light_type, level_chunks, sector_chunks,
    BuildLightRecord, BuildLights, LevelData, LevelGeometry, LevelGlow, LevelLight, LevelPortal,
    LevelSector, LevelShaderRef, LevelVertexBuffer, ResolvedLevelMesh, TorchLightConfig,
    VertexDeclarator, VertexElement9, BUILD_LIGHT_RECORD_SIZE, LEVEL_GLOW_RECORD_SIZE,
};
pub use ogf::{
    fvf_flags, matrix_forward_dir, matrix_invert_43, matrix_mul_43, matrix_mul_44, matrix_rotate_x,
    matrix_rotate_y, matrix_rotate_z, matrix_scale, matrix_set_hpb, matrix_set_xyz_translate,
    matrix_set_xyzi_translate, matrix_transform_point, matrix_transform_projected, ogf_chunks,
    skel_vert_type, vector_generate_orthonormal_basis, BoneIkData, BoneIkSection,
    BonePoseMatrices, BoneShape, BoneShapeType, LodImposterSample, OgfBBox, OgfBSphere, OgfBone,
    OgfBufferRef, OgfDesc, OgfGContainerRef, OgfHeader, OgfLodDef, OgfLodFacet, OgfLodVertex,
    OgfModel, OgfObb, OgfSkeleton, OgfSlideWindows, OgfTextureLink, OgfTreeDef, OgfVertices,
    OgfVisualType, SlideWindow, StaticVertex, StaticVertexLayout, TreeLightingColor,
    UnifiedSkinnedVertex, VertBoned1W, VertBoned2W, VertBoned3W, VertBoned4W, XRayMatrix,
    LOD_IMPOSTER_QUAD_ORDER, OGF_LODDEF2_SIZE, OGF_LOD_VERTEX_SIZE, OGF_TREEDEF2_SIZE,
    XRAY_IDENTITY_MATRIX,
};
pub use omf::{
    motion_def_flags, motion_key_flags, slerp_quat_xyzw, BoneKeyframe, BoneMotionTrack,
    BonePartitionDef, BonePartitionSet, BoneTranslationKeys, MotionClip, MotionDef, MotionMark,
    MotionPartition, MotionPartitionBone, OmfMotions, QuantizedRotKey, KEY_QUANT, KEY_QUANT_INV,
    MAX_PARTS, MOTION_QUANTIZER_RANGE_EXT, SAMPLE_FPS, SAMPLE_SPF, XR_OGF_SM_PARAMS_VERSION,
};
pub use particle_sim::{
    ParticleBillboardDrawBatch, ParticleBillboardInstance, ParticleEffectInstance,
    ParticleGroupInstance, ParticleLifeEvent, ParticleQuadVertex, ParticleRng, ParticleState,
    ParticleStepEvents, PerlinNoise3,
};
pub use particles::{
    build_wallmark_view_matrix, calc_triangle_adjacency, encode_compiled_action_list,
    pack_triangle_vertices, papi_flags, parse_compiled_action_list, ped_chunks, ped_flags,
    pg_effect_flags, pgd_chunks, project_static_wallmark_from_tris,
    project_static_wallmark_on_cform, ps_chunks, BloodWallmarksConfig, CompiledParticleAction,
    GroupEffectItem, ParticleActionKind, ParticleCollisionParams, ParticleDomain,
    ParticleDomainType, ParticleEffectDef, ParticleGroupDef, ParticleSpriteFrame, ParticlesLibrary,
    StaticWallmarkBatch, StaticWallmarkConfig, StaticWallmarkPool, StaticWallmarkRecord,
    WallmarkSlotConfig, WallmarkVertex, PARTICLE_DOMAIN_BINARY_SIZE, PED_VERSION, PGD_VERSION,
    PS_VERSION, WALLMARK_COS_88_DEG, WALLMARK_DEFAULT_TTL, WALLMARK_DUPLICATE_TOLERANCE,
    WALLMARK_RANGE_SKELETON, WALLMARK_RANGE_STATIC, WALLMARK_SHIFT_PP, WALLMARK_SHIFT_V,
};
pub use savegames::{
    compress_scop, decompress_scop, validate_scop_header, ActorStatDetail, ActorStatSection,
    GameDateTime, GameNewsEntry, GameTaskRegistryData, RelationData, SaveGameTime,
    SaveObjectRecord, SaveRegistries, SaveSpawnHeader, SavedGameTask, SavedMapLocation,
    SavedTaskScriptHelper, SavegameFile, ALIFE_CHUNK_DATA, ALIFE_VERSION, GAME_TIME_CHUNK_DATA,
    OBJECT_CHUNK_DATA, REGISTRY_CHUNK_DATA, SAVE_EXTENSION, SAVE_MAGIC, SPAWN_CHUNK_DATA,
};
pub use shaders_xr::{
    clsid_to_string, env_mod_flags, matrix_mode, shader_xrlc_flags, tcm_flags, transform_uv,
    BlenderDesc, BlenderProperty, BlenderPropertyValue, BlenderShader, EngineShaderLibrary,
    EnvModAccum, EnvModifier, FogVolume, LevelEnvModifiers, LevelFogVolumes, LevelStaticParticles,
    ShaderConstant, ShaderMatrix, ShaderXrlcEntry, ShaderXrlcLibrary, StaticParticle, WaveForm,
    WaveFunction, CBLENDER_DESC_SIZE, SHADER_XRLC_ENTRY_SIZE,
};
pub use skinning::{
    compute_skinned_aabb, matrix_transform_dir, normalize_vec3, skin_ogf_hierarchy,
    skin_vertices, BoneMatrix4x3, BonePalette, SkinnedMeshPart, SkinnedVertex,
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
pub use thm::{
    dds_caps2, strip_texture_extension, texture_flags, texture_stem_to_dds_vpath,
    texture_stem_to_thm_vpath, thm_chunks, BumpTextureLinks, CubemapFace, DdsCubemapLayout,
    DdsPixelFormatKind, DdsSubresourceSlice, ResolvedTextureLinks, TextureBumpMode, TextureFormat,
    TextureMaterial, TextureParams, TextureType, DDS_DX10_HEADER_BYTES, DDS_HEADER_BYTES,
    DDS_MAGIC,
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
    load_ps_static_from_vfs, load_scop_from_vfs, load_shaders_xr_from_vfs,
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
    time_diff, time_weight, CloudsDomeVertex, EnvAmbientDef, EnvAmbientEffectDef, EnvBoost,
    EnvDescriptor, EnvGlobalParams, EnvLitVertex, EnvSoundChannelDef, LensFlareDef,
    LensFlareDrawQuad, LensFlareElement, LensFlareFrameOutput, LensFlarePlayState,
    LensFlareQuadKind, LensFlareState, MixedEnvironment, PerlinNoise1D, RainDropItem,
    RainEffectConfig, RainEffectState, RainFrameOutput, RainSplashInstance, RainSplashParticle,
    SkyboxVertex, SunPositionsConfig, ThunderboltBoltDraw, ThunderboltCollectionDef,
    ThunderboltDescDef, ThunderboltEffectState, ThunderboltFlareDef, ThunderboltFrameOutput,
    ThunderboltGradientQuad, ThunderboltPlayState, WeatherCycle, WeatherLibrary, WindAnimState,
    CLOUDS_HEMI_INDICES, CLOUDS_HEMI_INDEX_COUNT, CLOUDS_HEMI_POSITIONS,
    CLOUDS_HEMI_TRIANGLE_COUNT, CLOUDS_HEMI_VERTEX_COUNT, DAY_LENGTH, LENS_FLARE_BLEND_DEC_SPEED,
    LENS_FLARE_BLEND_INC_SPEED, LENS_FLARE_MAX_RAYS, LENS_FLARE_RAY_DELTAS,
    LENS_FLARE_RAY_MAX_DIST, LENS_FLARE_RAY_SCALE, QUAD_INDEX_PATTERN, RAIN_CPU_SOURCE_RADIUS,
    RAIN_DROP_ANGLE_RAD, RAIN_DROP_LENGTH, RAIN_DROP_MAX_ANGLE_RAD, RAIN_DROP_SPEED_MAX,
    RAIN_DROP_SPEED_MIN, RAIN_DROP_WIDTH, RAIN_MAX_DESIRED_ITEMS, RAIN_MAX_DISTANCE,
    RAIN_MAX_PARTICLES, RAIN_SINK_OFFSET, RAIN_SOURCE_OFFSET, RAIN_SOURCE_RADIUS,
    RAIN_SPLASH_TIME_SEC, RAIN_STREAK_UVS, SKYBOX_HBOX_INDEX_COUNT, SKYBOX_HBOX_INDICES,
    SKYBOX_HBOX_POSITIONS, SKYBOX_HBOX_TEXCOORDS, SKYBOX_HBOX_TRIANGLE_COUNT,
    SKYBOX_HBOX_VERTEX_COUNT, WFX_TRANS_TIME,
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
