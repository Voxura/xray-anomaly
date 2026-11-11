// Phase 4 baseline raster shader. Matrices use column-major storage.
cbuffer CameraConstants : register(b0)
{
    column_major float4x4 view;
    column_major float4x4 proj;
    column_major float4x4 view_proj;
    float4 cam_pos;
    float4 padding[3];
};

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
};

PSInput vs_main(VSInput input)
{
    PSInput output;
    output.position = mul(view_proj, float4(input.position, 1.0));
    output.normal = input.normal;
    return output;
}

float4 ps_main(PSInput input) : SV_Target0
{
    // Normal visualization until the material/texture API is defined.
    return float4(abs(input.normal), 1.0);
}
