// Phase 14: Modern Deferred Combine & Lighting Pass Shader
// Renders via procedural full-screen triangle from SV_VertexID (0..2)
// Reconstructs world position from depth, evaluates direct sunlight with shadow mapping,
// dynamic point/spot lights from structured buffer, and ambient lighting into an HDR intermediate target.

struct PushCombineConstants
{
    uint gbuffer_albedo_idx;
    uint gbuffer_normal_idx;
    uint gbuffer_material_idx;
    uint gbuffer_depth_idx;
    uint shadow_map_idx;
    uint num_lights;
    float2 screen_res_inv;
    uint csm_active;
    uint volumetric_active;
    uint2 pad;
};

ConstantBuffer<PushCombineConstants> cb_combine : register(b0);

cbuffer CameraConstants : register(b1)
{
    column_major float4x4 view;
    column_major float4x4 proj;
    column_major float4x4 view_proj;
    float4 cam_pos;
    float4 padding[3];
};

cbuffer LightParams : register(b2)
{
    float4 SunDirection;
    float4 SunColor;
    float4 AmbientColor;
    column_major float4x4 SunViewProj;
    float4 FogColor;
    float4 FogRange; // [near, far, 0, 0]
    float4 HudProjection; // Perspective projection terms [x scale, y scale, z scale, z offset]
    float4 HudEnvironmentColor; // [RGB tint, weather blend]
    float4 HudAmbientHemi; // [RGB ambient, scalar hemisphere fallback]
    uint4 HudEnvironmentIndices;
    float4 HudHemiPositive;
    float4 HudHemiNegative;
};

cbuffer CsmSplitUniforms : register(b3)
{
    column_major float4x4 csm_cascade_view_proj[3];
    float4 csm_atlas_uv_scale_offset[3];
    float4 csm_cascade_uv_clamp[3];
    float4 csm_cascade_bias_texel[3];
    float4 csm_split_distances_and_blend; // [split0, split1, split2, blend_fraction]
    float4 csm_atlas_dims_inv;            // [width, height, 1/w, 1/h]
    float4 csm_padding[9];
};

Texture2D g_textures[] : register(t0, space0);
TextureCube g_environment_cubes[] : register(t0, space2);
SamplerState g_point_sampler : register(s0);
SamplerComparisonState g_shadow_sampler : register(s1);
SamplerState g_environment_sampler : register(s2);

ByteAddressBuffer g_lights : register(t0, space1);

struct GpuLightData
{
    float3 position;
    float range;
    float4 color;
    float3 direction;
    uint light_type;
    float2 spot_params; // cos_inner, cos_outer
    float2 attenuation;
};

GpuLightData load_light(uint index)
{
    uint offset = index * 64;
    uint4 w0 = g_lights.Load4(offset);
    uint4 w1 = g_lights.Load4(offset + 16);
    uint4 w2 = g_lights.Load4(offset + 32);
    uint4 w3 = g_lights.Load4(offset + 48);

    GpuLightData l;
    l.position = asfloat(w0.xyz);
    l.range = asfloat(w0.w);
    l.color = asfloat(w1);
    l.direction = asfloat(w2.xyz);
    l.light_type = w2.w;
    l.spot_params = asfloat(w3.xy);
    l.attenuation = asfloat(w3.zw);
    return l;
}

struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

VSOutput vs_deferred_combine(uint vertex_id : SV_VertexID)
{
    VSOutput output;
    float2 pos = float2((vertex_id == 2) ? 3.0 : -1.0, (vertex_id == 1) ? 3.0 : -1.0);
    output.position = float4(pos, 0.0, 1.0);
    output.uv = float2(pos.x * 0.5 + 0.5, 1.0 - (pos.y * 0.5 + 0.5));
    return output;
}

float sample_csm_cascade(Texture2D shadow_tex, uint cascade_idx, float3 world_pos)
{
    float4 shadow_clip = mul(csm_cascade_view_proj[cascade_idx], float4(world_pos, 1.0));
    float3 shadow_ndc = shadow_clip.xyz / max(shadow_clip.w, 1e-6);
    float2 local_uv = float2(shadow_ndc.x * 0.5 + 0.5, 1.0 - (shadow_ndc.y * 0.5 + 0.5));
    float2 atlas_uv = local_uv * csm_atlas_uv_scale_offset[cascade_idx].xy + csm_atlas_uv_scale_offset[cascade_idx].zw;
    atlas_uv = clamp(atlas_uv, csm_cascade_uv_clamp[cascade_idx].xy, csm_cascade_uv_clamp[cascade_idx].zw);

    float depth_bias = csm_cascade_bias_texel[cascade_idx].x;
    // The cascade stores a signed depth translation, like R4's m_TexelAdjust.
    float shadow_depth = saturate(shadow_ndc.z + depth_bias);

    // 4-tap PCF filtering using atlas dimensions
    float2 texel_size = csm_atlas_dims_inv.zw;
    float s0 = shadow_tex.SampleCmpLevelZero(g_shadow_sampler, atlas_uv + float2(-0.5, -0.5) * texel_size, shadow_depth);
    float s1 = shadow_tex.SampleCmpLevelZero(g_shadow_sampler, atlas_uv + float2( 0.5, -0.5) * texel_size, shadow_depth);
    float s2 = shadow_tex.SampleCmpLevelZero(g_shadow_sampler, atlas_uv + float2(-0.5,  0.5) * texel_size, shadow_depth);
    float s3 = shadow_tex.SampleCmpLevelZero(g_shadow_sampler, atlas_uv + float2( 0.5,  0.5) * texel_size, shadow_depth);
    return (s0 + s1 + s2 + s3) * 0.25;
}

float evaluate_csm_shadow(Texture2D shadow_tex, float3 world_pos)
{
    float dist = length(world_pos - cam_pos.xyz);
    float split0 = csm_split_distances_and_blend.x;
    float split1 = csm_split_distances_and_blend.y;
    float split2 = csm_split_distances_and_blend.z;
    float blend_fraction = csm_split_distances_and_blend.w;

    if (dist < split0)
    {
        float s0 = sample_csm_cascade(shadow_tex, 0, world_pos);
        float blend_range = split0 * blend_fraction;
        if (dist > (split0 - blend_range))
        {
            float s1 = sample_csm_cascade(shadow_tex, 1, world_pos);
            float t = (dist - (split0 - blend_range)) / max(blend_range, 1e-4);
            return lerp(s0, s1, t);
        }
        return s0;
    }
    else if (dist < split1)
    {
        float s1 = sample_csm_cascade(shadow_tex, 1, world_pos);
        float blend_range = (split1 - split0) * blend_fraction;
        if (dist > (split1 - blend_range))
        {
            float s2 = sample_csm_cascade(shadow_tex, 2, world_pos);
            float t = (dist - (split1 - blend_range)) / max(blend_range, 1e-4);
            return lerp(s1, s2, t);
        }
        return s1;
    }
    else if (dist < split2)
    {
        return sample_csm_cascade(shadow_tex, 2, world_pos);
    }
    return 1.0;
}

float4 ps_deferred_combine(VSOutput input) : SV_Target0
{
    Texture2D albedo_tex = g_textures[NonUniformResourceIndex(cb_combine.gbuffer_albedo_idx)];
    Texture2D normal_tex = g_textures[NonUniformResourceIndex(cb_combine.gbuffer_normal_idx)];
    Texture2D material_tex = g_textures[NonUniformResourceIndex(cb_combine.gbuffer_material_idx)];
    Texture2D depth_tex = g_textures[NonUniformResourceIndex(cb_combine.gbuffer_depth_idx)];

    float4 albedo_sample = albedo_tex.Sample(g_point_sampler, input.uv);
    float4 normal_sample = normal_tex.Sample(g_point_sampler, input.uv);
    float4 material_sample = material_tex.Sample(g_point_sampler, input.uv);
    float depth = depth_tex.Sample(g_point_sampler, input.uv).r;

    // Sky / far plane early-out
    if (depth >= 0.999999)
    {
        return float4(albedo_sample.rgb, 1.0);
    }

    // Decode world normal and material properties
    float3 normal = normalize(normal_sample.xyz * 2.0 - 1.0);
    float roughness = normal_sample.w;
    float specular_intensity = material_sample.b;

    // HUD geometry uses its own projection and Monolith's rmNear viewport [0, 0.02].
    bool is_hud = material_sample.a < 0.5 && HudProjection.x > 0.0;
    float metallic = is_hud ? 0.0 : material_sample.r;
    float4 projection = is_hud ? HudProjection : float4(proj[0][0], proj[1][1], proj[2][2], proj[2][3]);
    float depth_ndc = is_hud ? depth / 0.02 : depth;
    float2 ndc_xy = float2(input.uv.x * 2.0 - 1.0, (1.0 - input.uv.y) * 2.0 - 1.0);

    float view_x = ndc_xy.x / max(projection.x, 1e-6);
    float view_y = ndc_xy.y / max(projection.y, 1e-6);

    float3x3 inv_view_rot = transpose((float3x3)view);
    float3 world_pos;
    if (abs(projection.w) > 1e-6) // Perspective projection
    {
        float view_z = (-projection.w) / max(projection.z - depth_ndc, 1e-6);
        float3 pos_vs = float3(view_x * view_z, view_y * view_z, view_z);
        world_pos = mul(inv_view_rot, pos_vs) + cam_pos.xyz;
    }
    else // Orthographic / identity fallback
    {
        float view_z = depth / max(proj[2][2], 1e-6);
        float3 pos_vs = float3(view_x, view_y, view_z);
        world_pos = mul(inv_view_rot, pos_vs) + cam_pos.xyz;
    }

    // Directional Sunlight
    float3 L = normalize(SunDirection.xyz);
    float n_dot_l = saturate(dot(normal, L));

    // Shadow factor evaluation
    float shadow_factor = 1.0;
    if (cb_combine.shadow_map_idx != 0xFFFFFFFF)
    {
        Texture2D shadow_tex = g_textures[NonUniformResourceIndex(cb_combine.shadow_map_idx)];
        if (cb_combine.csm_active != 0)
        {
            shadow_factor = evaluate_csm_shadow(shadow_tex, world_pos);
        }
        else
        {
            float4 shadow_clip = mul(SunViewProj, float4(world_pos, 1.0));
            float3 shadow_ndc = shadow_clip.xyz / max(shadow_clip.w, 1e-6);
            float2 shadow_uv = float2(shadow_ndc.x * 0.5 + 0.5, 1.0 - (shadow_ndc.y * 0.5 + 0.5));
            float shadow_depth = shadow_ndc.z - 0.002;

            if (shadow_uv.x >= 0.0 && shadow_uv.x <= 1.0 && shadow_uv.y >= 0.0 && shadow_uv.y <= 1.0 && shadow_depth <= 1.0)
            {
                shadow_factor = shadow_tex.SampleCmpLevelZero(g_shadow_sampler, shadow_uv, shadow_depth);
            }
        }
    }

    // Direct sun diffuse and Blinn-Phong specular
    float3 V = normalize(cam_pos.xyz - world_pos);
    float3 H = normalize(L + V);
    float n_dot_h = saturate(dot(normal, H));
    float spec_power = (1.0 - roughness) * 64.0 + 1.0;
    float gloss_factor = is_hud ? specular_intensity : 1.0;
    float spec_factor = pow(n_dot_h, spec_power) * n_dot_l * gloss_factor;

    float3 sun_diffuse = albedo_sample.rgb * SunColor.rgb * (n_dot_l * shadow_factor);
    float3 sun_specular = SunColor.rgb * (spec_factor * shadow_factor * (0.04 * (1.0 - metallic) + albedo_sample.rgb * metallic));

    // Dynamic lights evaluation from structured buffer
    float3 dynamic_light_accum = float3(0.0, 0.0, 0.0);
    uint num_lights = min(cb_combine.num_lights, 64);
    for (uint i = 0; i < num_lights; ++i)
    {
        GpuLightData light = load_light(i);
        float3 to_light = light.position - world_pos;
        float dist = length(to_light);
        if (dist < light.range && dist > 0.0001)
        {
            float3 light_dir = to_light / dist;
            float light_ndotl = saturate(dot(normal, light_dir));
            if (light_ndotl > 0.0)
            {
                float att_dist = saturate(1.0 - dist / light.range);
                float att = att_dist * att_dist;

                if (light.light_type == 1) // SPOT
                {
                    float spot_cos = dot(-light_dir, normalize(light.direction));
                    float spot_factor = saturate((spot_cos - light.spot_params.y) / max(0.0001, light.spot_params.x - light.spot_params.y));
                    att *= spot_factor;
                }

                float3 light_h = normalize(light_dir + V);
                float light_spec = pow(saturate(dot(normal, light_h)), spec_power) * gloss_factor;
                dynamic_light_accum += (light.color.rgb * light.color.a) * (light_ndotl * att + light_spec * att * 0.5);
            }
        }
    }

    // Ambient Lighting
    float3 irradiance = AmbientColor.rgb;
    if (is_hud && HudEnvironmentIndices.x != 0xFFFFFFFF)
    {
        // Monolith hmodel: tint and blend the two weather irradiance maps,
        // square the diffuse color, then apply the material LUT's diffuse response.
        float3 sky_a = g_environment_cubes[NonUniformResourceIndex(HudEnvironmentIndices.x)]
            .SampleLevel(g_environment_sampler, normal, 0).rgb;
        float3 sky_b = g_environment_cubes[NonUniformResourceIndex(HudEnvironmentIndices.y)]
            .SampleLevel(g_environment_sampler, normal, 0).rgb;
        float3 environment = lerp(sky_a, sky_b, HudEnvironmentColor.w) * HudEnvironmentColor.rgb;
        float3 hemisphere = float3(
            normal.x >= 0.0 ? HudHemiPositive.x : HudHemiNegative.x,
            normal.y >= 0.0 ? HudHemiPositive.y : HudHemiNegative.y,
            normal.z >= 0.0 ? HudHemiPositive.z : HudHemiNegative.z
        );
        float h = saturate(max(HudAmbientHemi.w, dot(hemisphere, abs(normal))));
        float4 diffuse_response = float4(pow(h, 0.75), pow(h, 0.9), h, h);
        float slice = clamp(material_sample.r * 4.0 - 0.5, 0.0, 3.0);
        uint lo = (uint)slice;
        float weight = lerp(diffuse_response[lo], diffuse_response[min(lo + 1, 3)], frac(slice));
        irradiance = HudAmbientHemi.rgb + environment * environment * weight;
    }
    float3 ambient = irradiance * albedo_sample.rgb;

    // SSFX Volumetric Light Shafts
    float3 volumetric_shafts = float3(0.0, 0.0, 0.0);
    if (cb_combine.volumetric_active != 0 && cb_combine.shadow_map_idx != 0xFFFFFFFF && cb_combine.csm_active != 0)
    {
        Texture2D shadow_tex = g_textures[NonUniformResourceIndex(cb_combine.shadow_map_idx)];
        float3 ray_dir = world_pos - cam_pos.xyz;
        float ray_len = min(length(ray_dir), csm_split_distances_and_blend.y); // Cap to mid cascade distance
        ray_dir = normalize(ray_dir);

        uint num_steps = 16;
        float step_size = ray_len / (float)num_steps;
        float cos_theta = dot(ray_dir, L);
        float g = 0.8;
        float phase = (1.0 - g * g) / (4.0 * 3.14159265 * pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1e-4), 1.5));

        float density = 0.03;
        float extinction = 0.04;
        float transmittance = 1.0;

        for (uint s = 0; s < num_steps; s++)
        {
            float t = ((float)s + 0.5) * step_size;
            float3 sample_pos = cam_pos.xyz + ray_dir * t;
            float s_shadow = evaluate_csm_shadow(shadow_tex, sample_pos);
            float step_transmittance = exp(-extinction * step_size);
            volumetric_shafts += SunColor.rgb * s_shadow * phase * density * transmittance * step_size;
            transmittance *= step_transmittance;
        }
    }

    float3 final_hdr = sun_diffuse + sun_specular + albedo_sample.rgb * dynamic_light_accum + ambient + volumetric_shafts;
    // Monolith's combine applies weather fog to the lit scene before tonemapping.
    if (FogRange.y > FogRange.x)
    {
        float distance_to_eye = length(world_pos - cam_pos.xyz);
        float fog = saturate((distance_to_eye - FogRange.x) / (FogRange.y - FogRange.x));
        final_hdr = lerp(final_hdr, FogColor.rgb, fog);
    }
    return float4(final_hdr, 1.0);
}
