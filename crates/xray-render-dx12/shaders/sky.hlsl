// Phase 10: Infinite Skybox & Atmosphere Environment Shader
cbuffer CameraConstants : register(b0)
{
    column_major float4x4 view;
    column_major float4x4 proj;
    column_major float4x4 view_proj;
    float4 cam_pos;
    float4 padding[3];
};

cbuffer SkyPushConstants : register(b1)
{
    uint sky_texture0;
    uint sky_texture1;
    float blend_factor;
    uint is_cubemap;
    float4 sky_color;
    float4 sun_color;
    float4 sun_dir;
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

struct SkyVSOutput
{
    float4 position : SV_Position;
    float3 ray_dir : TEXCOORD0;
};

SkyVSOutput vs_main(uint vertex_id : SV_VertexID)
{
    SkyVSOutput output;
    float3 local_pos = CUBE_VERTICES[CUBE_INDICES[vertex_id]];
    output.ray_dir = local_pos;

    float3x3 rot = (float3x3)view;
    float4 clip_pos = mul(proj, float4(mul(rot, local_pos), 1.0));
    output.position = clip_pos.xyww;
    return output;
}

float4 ps_main(SkyVSOutput input) : SV_Target0
{
    float3 dir = normalize(input.ray_dir);
    float4 color0 = float4(0.2, 0.3, 0.5, 1.0);
    float4 color1 = float4(0.1, 0.15, 0.3, 1.0);

    // Equirectangular lat-long UV mapping
    float2 uv = float2(atan2(dir.z, dir.x) * 0.15915494 + 0.5, asin(clamp(dir.y, -1.0, 1.0)) * 0.31830988 + 0.5);
    if (sky_texture0 != 0xFFFFFFFF)
    {
        color0 = g_textures[NonUniformResourceIndex(sky_texture0)].Sample(g_sampler, uv);
    }
    if (sky_texture1 != 0xFFFFFFFF)
    {
        color1 = g_textures[NonUniformResourceIndex(sky_texture1)].Sample(g_sampler, uv);
    }

    float4 blended = lerp(color0, color1, saturate(blend_factor)) * sky_color;

    // Atmospheric sun glow / phase flare
    if (sun_color.a > 0.0)
    {
        float sun_cos = saturate(dot(dir, normalize(sun_dir.xyz)));
        float sun_flare = pow(sun_cos, 128.0) * sun_color.a;
        blended.rgb += sun_color.rgb * sun_flare;
    }

    return blended;
}
