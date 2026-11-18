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
    float sky_rotation;
    float3 sky_padding;
};

Texture2D g_textures[] : register(t0);
TextureCube g_cubemaps[] : register(t0, space1);
SamplerState g_sampler : register(s0);

struct SkyVSOutput
{
    float4 position : SV_Position;
    float3 local_pos : TEXCOORD0;
    float3 cube_dir : TEXCOORD1;
    float3 world_dir : TEXCOORD2;
};

SkyVSOutput vs_main(uint vertex_id : SV_VertexID)
{
    SkyVSOutput output;
    uint index = SKY_INDICES[vertex_id];
    float3 local_pos = SKY_POSITIONS[index];
    output.local_pos = local_pos;
    output.cube_dir = SKY_TEXCOORDS[index];

    float sine, cosine;
    sincos(sky_rotation, sine, cosine);
    output.world_dir = float3(
        cosine * local_pos.x + sine * local_pos.z,
        local_pos.y,
        cosine * local_pos.z - sine * local_pos.x
    );

    float3x3 rot = (float3x3)view;
    float4 clip_pos = mul(proj, float4(mul(rot, output.world_dir), 1.0));
    output.position = clip_pos.xyww;
    return output;
}

float4 sample_sky(uint index, bool cube, float3 dir, float2 uv)
{
    if (cube)
        return g_cubemaps[NonUniformResourceIndex(index)].Sample(g_sampler, dir);
    return g_textures[NonUniformResourceIndex(index)].Sample(g_sampler, uv);
}

float4 ps_main(SkyVSOutput input) : SV_Target0
{
    float3 dir = normalize(input.local_pos);
    float4 color0 = float4(0.2, 0.3, 0.5, 1.0);
    float4 color1 = float4(0.1, 0.15, 0.3, 1.0);

    // Equirectangular lat-long UV mapping
    float2 uv = float2(atan2(dir.z, dir.x) * 0.15915494 + 0.5, asin(clamp(dir.y, -1.0, 1.0)) * 0.31830988 + 0.5);
    if (sky_texture0 != 0xFFFFFFFF)
    {
        color0 = sample_sky(sky_texture0, (is_cubemap & 1) != 0, input.cube_dir, uv);
    }
    if (sky_texture1 != 0xFFFFFFFF)
    {
        color1 = sample_sky(sky_texture1, (is_cubemap & 2) != 0, input.cube_dir, uv);
    }

    float4 blended = lerp(color0, color1, saturate(blend_factor)) * sky_color;

    // Atmospheric sun glow / phase flare
    if (sun_color.a > 0.0)
    {
        float sun_cos = saturate(dot(normalize(input.world_dir), normalize(sun_dir.xyz)));
        float sun_flare = pow(sun_cos, 128.0) * sun_color.a;
        blended.rgb += sun_color.rgb * sun_flare;
    }

    return blended;
}
