// Phase 4 baseline raster shader & Phase 5 bindless textured shader.
cbuffer CameraConstants : register(b0)
{
    column_major float4x4 view;
    column_major float4x4 proj;
    column_major float4x4 view_proj;
    float4 cam_pos;
    float4 padding[3];
};

struct PushDrawConstants
{
    uint albedo_index;
    uint normal_index;
    uint material_index;
    uint transform_index;
    uint custom_param0;
    uint custom_param1;
    uint custom_param2;
    uint custom_param3;
};
ConstantBuffer<PushDrawConstants> cb_draw : register(b1);

Texture2D g_textures[] : register(t0);
SamplerState g_sampler : register(s0);

struct VSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    float3 tangent : TANGENT;
};

struct PSInput
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

PSInput vs_main(VSInput input)
{
    PSInput output;
    output.position = mul(view_proj, float4(input.position, 1.0));
    output.normal = input.normal;
    output.uv = input.uv;
    return output;
}

float4 ps_main(PSInput input) : SV_Target0
{
    if (cb_draw.albedo_index != 0xFFFFFFFF)
    {
        return g_textures[NonUniformResourceIndex(cb_draw.albedo_index)].Sample(g_sampler, input.uv);
    }
    // Normal visualization fallback when no texture is bound
    return float4(abs(input.normal), 1.0);
}
