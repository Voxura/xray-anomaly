// X-Ray Monolith Engine - SM 6.6 DirectX 12 GBuffer Shader
// Outputs: SV_Target0 (Albedo RGB, AO A), SV_Target1 (World Normal XYZ, Roughness W), SV_Target2 (Specular/Material)

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

ConstantBuffer<PushDrawConstants> cb_draw : register(b0);

cbuffer FrameConstants : register(b1)
{
    float4x4 ViewProjection;
    float4x4 InvViewProjection;
    float4 CameraPosition;
    float4 SunDirection;
    float4 SunColor;
    float4 AmbientColor;
};

struct VSInput
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float4 tangent  : TANGENT;
    float2 uv       : TEXCOORD0;
};

struct PSInput
{
    float4 position_cs : SV_Position;
    float3 position_ws : POSITION;
    float3 normal_ws   : NORMAL;
    float4 tangent_ws  : TANGENT;
    float2 uv          : TEXCOORD0;
};

struct PSOutput
{
    float4 albedo   : SV_Target0; // RGB: Diffuse Albedo, A: Occlusion
    float4 normal   : SV_Target1; // XYZ: Encoded World Normal, W: Roughness
    float4 material : SV_Target2; // R: Metallic, G: Roughness, B: Specular, A: MaterialID
};

PSInput vs_main(VSInput input)
{
    PSInput output;
    output.position_ws = input.position;
    output.position_cs = mul(ViewProjection, float4(input.position, 1.0f));
    output.normal_ws = normalize(input.normal);
    output.tangent_ws = float4(normalize(input.tangent.xyz), input.tangent.w);
    output.uv = input.uv;
    return output;
}

SamplerState g_sampler : register(s0);

PSOutput ps_main(PSInput input)
{
    PSOutput output;

    // Sample diffuse texture dynamically using bindless SM 6.6 descriptor heap
    Texture2D diffuse_tex = ResourceDescriptorHeap[cb_draw.albedo_index];
    float4 diffuse_sample = diffuse_tex.Sample(g_sampler, input.uv);

    // Sample normal map if present
    float3 normal = input.normal_ws;
    if (cb_draw.normal_index != 0xFFFFFFFF)
    {
        Texture2D normal_tex = ResourceDescriptorHeap[cb_draw.normal_index];
        float3 normal_map = normal_tex.Sample(g_sampler, input.uv).xyz * 2.0f - 1.0f;
        float3 bitangent = cross(input.normal_ws, input.tangent_ws.xyz) * input.tangent_ws.w;
        float3x3 tbn = float3x3(input.tangent_ws.xyz, bitangent, input.normal_ws);
        normal = normalize(mul(normal_map, tbn));
    }

    // Material properties
    float metallic = 0.0f;
    float roughness = 0.5f;
    float specular = 0.04f;
    if (cb_draw.material_index != 0xFFFFFFFF)
    {
        Texture2D mat_tex = ResourceDescriptorHeap[cb_draw.material_index];
        float4 mat_sample = mat_tex.Sample(g_sampler, input.uv);
        roughness = mat_sample.g;
        metallic = mat_sample.r;
    }

    output.albedo = float4(diffuse_sample.rgb, 1.0f);
    output.normal = float4(normal * 0.5f + 0.5f, roughness);
    output.material = float4(metallic, roughness, specular, 1.0f);

    return output;
}
