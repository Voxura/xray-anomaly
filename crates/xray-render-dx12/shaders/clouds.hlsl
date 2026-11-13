// Phase 11: Dual-Layer Scrolling Clouds Atmosphere Shader
// Procedural sky cube rasterized directly from SV_VertexID (0..36) with ZERO vertex buffer allocations.
// Features dual-texture weather cycle blending, dynamic wind UV velocity scrolling,
// atmospheric sun forward-scattering, and far-plane depth clamping (Z_NDC = 1.0) with Early-Z occlusion.

cbuffer CameraConstants : register(b0)
{
    column_major float4x4 view;
    column_major float4x4 proj;
    column_major float4x4 view_proj;
    float4 cam_pos;
    float4 padding[3];
};

cbuffer CloudsPushConstants : register(b1)
{
    uint clouds_texture0;
    uint clouds_texture1;
    float blend_factor;
    float clouds_rotation;
    float4 clouds_color;
    float4 sun_color;
    float4 sun_dir;
    float2 uv_offset0;
    float2 uv_offset1;
    float2 clouds_scale;
    float2 push_padding;
};

Texture2D g_textures[] : register(t0);
SamplerState g_sampler : register(s0);

static const float3 CUBE_VERTICES[8] = {
    float3(-1.0, -1.0, -1.0),
    float3( 1.0, -1.0, -1.0),
    float3( 1.0,  1.0, -1.0),
    float3(-1.0,  1.0, -1.0),
    float3(-1.0, -1.0,  1.0),
    float3( 1.0, -1.0,  1.0),
    float3( 1.0,  1.0,  1.0),
    float3(-1.0,  1.0,  1.0),
};

static const uint CUBE_INDICES[36] = {
    0, 2, 1, 0, 3, 2,
    5, 6, 4, 4, 6, 7,
    4, 7, 0, 0, 7, 3,
    1, 2, 5, 5, 2, 6,
    3, 7, 2, 2, 7, 6,
    4, 0, 5, 5, 0, 1,
};

struct CloudsVSOutput
{
    float4 position : SV_Position;
    float3 ray_dir : TEXCOORD0;
};

CloudsVSOutput vs_main(uint vertex_id : SV_VertexID)
{
    CloudsVSOutput output;
    float3 local_pos = CUBE_VERTICES[CUBE_INDICES[vertex_id]];
    output.ray_dir = local_pos;

    // Apply rotation around Y axis
    if (abs(clouds_rotation) > 0.0001)
    {
        float s = sin(clouds_rotation);
        float c = cos(clouds_rotation);
        output.ray_dir = float3(local_pos.x * c - local_pos.z * s, local_pos.y, local_pos.x * s + local_pos.z * c);
    }

    float3x3 rot = (float3x3)view;
    float4 clip_pos = mul(proj, float4(mul(rot, local_pos), 1.0));
    output.position = clip_pos.xyww;
    return output;
}

float4 sample_or_fallback(uint texture_index, float2 uv)
{
    if (texture_index != 0xFFFFFFFF)
    {
        return g_textures[NonUniformResourceIndex(texture_index)].Sample(g_sampler, uv);
    }
    // Procedural soft cloud noise fallback
    float n1 = sin(uv.x * 6.283) * cos(uv.y * 6.283) * 0.3 + 0.5;
    float n2 = sin(uv.x * 12.566 + 0.7) * cos(uv.y * 12.566 + 1.2) * 0.2 + 0.3;
    float a = saturate(n1 + n2);
    return float4(1.0, 1.0, 1.0, a);
}

float4 ps_main(CloudsVSOutput input) : SV_Target0
{
    float3 dir = normalize(input.ray_dir);

    // Planar cloud canopy UV projection
    float2 planar_uv = (dir.y > 0.01) ? (dir.xz / dir.y * 0.3 + 0.5) : (dir.xz * 0.5 + 0.5);
    float2 uv0 = planar_uv + uv_offset0;
    float2 uv1 = planar_uv * 1.5 + uv_offset1;

    float4 col0 = sample_or_fallback(clouds_texture0, uv0);
    float4 col1 = sample_or_fallback(clouds_texture1, uv1);

    float4 blended = lerp(col0, col1, saturate(blend_factor));

    // Atmospheric sun forward scattering
    float sun_dot = saturate(dot(dir, normalize(sun_dir.xyz)));
    float sun_scatter = pow(sun_dot, 8.0) * sun_color.a;

    float3 final_rgb = blended.rgb * clouds_color.rgb + sun_color.rgb * sun_scatter;
    float final_a = saturate(blended.a * clouds_color.a);

    return float4(final_rgb, final_a);
}
