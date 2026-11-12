// Phase 11: Instanced Rain Streaks Shader
// Billboard rain streaks generated procedurally from SV_VertexID (0..5) and SV_InstanceID.
// Streaks stretch along the fall/wind direction vector and expand perpendicularly to the camera view ray.
// Drops stream via ByteAddressBuffer with bindless textures and alpha blending.

cbuffer CameraConstants : register(b0)
{
    column_major float4x4 view;
    column_major float4x4 proj;
    column_major float4x4 view_proj;
    float4 cam_pos;
    float4 padding[3];
};

cbuffer RainPushConstants : register(b1)
{
    uint rain_texture;
    float rain_density;
    float streak_length_scale;
    float streak_width_scale;
    float4 rain_color;
};

Texture2D g_textures[] : register(t0);
SamplerState g_sampler : register(s0);
ByteAddressBuffer g_raindrops : register(t0, space1);

struct RainDrop
{
    float3 position;
    float length;
    float3 direction;
    float alpha;
    float width;
    uint uv_set;
};

RainDrop load_raindrop(uint instance_id)
{
    uint offset = instance_id * 48;
    float4 r0 = asfloat(g_raindrops.Load4(offset));
    float4 r1 = asfloat(g_raindrops.Load4(offset + 16));
    float4 r2 = asfloat(g_raindrops.Load4(offset + 32));

    RainDrop d;
    d.position  = r0.xyz;
    d.length    = r0.w;
    d.direction = r1.xyz;
    d.alpha     = r1.w;
    d.width     = r2.x;
    d.uv_set    = asuint(r2.y);
    return d;
}

struct RainVSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float alpha : TEXCOORD1;
    nointerpolation uint uv_set : BLENDINDICES0;
};

RainVSOutput vs_rain(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    RainDrop drop = load_raindrop(instance_id);

    float len = drop.length * ((streak_length_scale > 0.001) ? streak_length_scale : 1.0);
    float wid = drop.width  * ((streak_width_scale  > 0.001) ? streak_width_scale  : 1.0);

    float3 pos_head  = drop.position;
    float3 pos_trail = drop.position - drop.direction * len;

    float3 center = (pos_head + pos_trail) * 0.5;
    float3 cam_dir = normalize(center - cam_pos.xyz);

    // Cross product between camera view ray and streak direction creates camera-facing billboard side vector
    float3 side = normalize(cross(cam_dir, drop.direction)) * (wid * 0.5);

    float3 quad_positions[6] = {
        pos_trail - side,
        pos_trail + side,
        pos_head  - side,
        pos_trail + side,
        pos_head  + side,
        pos_head  - side
    };

    float2 quad_uvs[6] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 0.0),
        float2(1.0, 1.0),
        float2(0.0, 1.0)
    };

    float3 world_pos = quad_positions[vertex_id];

    RainVSOutput output;
    output.position = mul(view_proj, float4(world_pos, 1.0));
    output.uv = quad_uvs[vertex_id];
    output.alpha = drop.alpha;
    output.uv_set = drop.uv_set;
    return output;
}

float4 ps_rain(RainVSOutput input) : SV_Target0
{
    // Needle-like streak profile with quadratic horizontal falloff and linear vertical fade
    float x_dist = abs(input.uv.x - 0.5) * 2.0;
    float needle = pow(saturate(1.0 - x_dist), 2.5) * input.uv.y;

    float3 rgb = rain_color.rgb;
    float a = needle * input.alpha * rain_density * rain_color.a;

    if (rain_texture != 0xFFFFFFFF)
    {
        float4 tex = g_textures[NonUniformResourceIndex(rain_texture)].Sample(g_sampler, input.uv);
        rgb *= tex.rgb;
        a *= tex.a;
    }

    return float4(rgb, saturate(a));
}
