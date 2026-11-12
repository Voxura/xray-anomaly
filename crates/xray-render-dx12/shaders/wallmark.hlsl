// Phase 12: Wallmark Decal Projection Shader
// Projects static decals (bullet impacts, blood splatters, scorch marks) onto level geometry.
// Decal vertices stream via ByteAddressBuffer with bindless textures, depth bias, and alpha blending.

cbuffer CameraConstants : register(b0)
{
    column_major float4x4 view;
    column_major float4x4 proj;
    column_major float4x4 view_proj;
    float4 cam_pos;
    float4 padding[3];
};

cbuffer WallmarkPushConstants : register(b1)
{
    uint texture_index;
    float depth_bias;
    float alpha_multiplier;
    uint flags;
};

Texture2D g_textures[] : register(t0);
SamplerState g_sampler : register(s0);
ByteAddressBuffer g_wallmark_vertices : register(t0, space1);

struct WallmarkVSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
};

float4 unpack_argb(uint c)
{
    float a = float((c >> 24) & 0xFF) / 255.0;
    float r = float((c >> 16) & 0xFF) / 255.0;
    float g = float((c >> 8)  & 0xFF) / 255.0;
    float b = float(c & 0xFF)        / 255.0;
    return float4(r, g, b, a);
}

WallmarkVSOutput vs_wallmark(uint vertex_id : SV_VertexID)
{
    // 24-byte vertex record: float3 position (12B) + uint color (4B) + float2 uv (8B)
    uint offset = vertex_id * 24;
    float4 r0 = asfloat(g_wallmark_vertices.Load4(offset));
    float3 pos = r0.xyz;
    uint packed_color = asuint(r0.w);
    float2 uv = asfloat(g_wallmark_vertices.Load2(offset + 16));

    float4 clip_pos = mul(view_proj, float4(pos, 1.0));

    // Apply depth bias to completely eliminate Z-fighting against underlying wall polygons
    clip_pos.z -= depth_bias * clip_pos.w;

    WallmarkVSOutput output;
    output.position = clip_pos;
    output.uv = uv;
    output.color = unpack_argb(packed_color);
    return output;
}

float4 ps_wallmark(WallmarkVSOutput input) : SV_Target
{
    float4 tex_color = float4(1.0, 1.0, 1.0, 1.0);
    if (texture_index != 0xFFFFFFFF)
    {
        tex_color = g_textures[NonUniformResourceIndex(texture_index)].Sample(g_sampler, input.uv);
    }

    // Vertex color carries base tint RGB and TTL age fade in alpha
    float4 final_color = tex_color * input.color;
    final_color.a *= alpha_multiplier;
    return final_color;
}
