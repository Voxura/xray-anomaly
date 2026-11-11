// Phase 4 baseline raster shader & Phase 5 bindless textured shader.
cbuffer CameraConstants : register(b0)
{
    column_major float4x4 view;
    column_major float4x4 proj;
    column_major float4x4 view_proj;
    float4 cam_pos;
    float4 padding[3];
};

cbuffer cb_draw : register(b1)
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

Texture2D g_textures[] : register(t0);
SamplerState g_sampler : register(s0);
ByteAddressBuffer g_transforms : register(t0, space1);

float4x4 load_transform(uint index)
{
    uint offset = index * 64;
    float4 r0 = asfloat(g_transforms.Load4(offset));
    float4 r1 = asfloat(g_transforms.Load4(offset + 16));
    float4 r2 = asfloat(g_transforms.Load4(offset + 32));
    float4 r3 = asfloat(g_transforms.Load4(offset + 48));
    return float4x4(r0, r1, r2, r3);
}

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
    float3 tangent : TANGENT;
    float3 bitangent : BINORMAL;
};

PSInput vs_main(VSInput input)
{
    PSInput output;
    float4 local_pos = float4(input.position, 1.0);
    float4 world_pos;
    float3 world_normal;
    float3 world_tangent;

    if (transform_index != 0xFFFFFFFF)
    {
        float4x4 world = load_transform(transform_index);
        world_pos = mul(local_pos, world);
        world_normal = normalize(mul(input.normal, (float3x3)world));
        world_tangent = normalize(mul(input.tangent, (float3x3)world));
    }
    else
    {
        world_pos = local_pos;
        world_normal = input.normal;
        world_tangent = input.tangent;
    }

    output.position = mul(view_proj, world_pos);
    output.normal = world_normal;
    output.uv = input.uv;
    output.tangent = world_tangent;
    output.bitangent = cross(world_normal, world_tangent);
    return output;
}

float4 ps_main(PSInput input) : SV_Target0
{
    // Decode perturbed normal if normal_index is bound
    float3 world_normal = input.normal;
    if (normal_index != 0xFFFFFFFF)
    {
        float4 n_tex = g_textures[NonUniformResourceIndex(normal_index)].Sample(g_sampler, input.uv);
        // Supports BC5/ATI2 (RG), DXT5nm (Alpha/Green), and standard 3-channel RGB normal maps
        float2 xy;
        if (n_tex.a > 0.0 && n_tex.b < 0.1)
        {
            xy = float2(n_tex.a, n_tex.g) * 2.0 - 1.0;
        }
        else
        {
            xy = n_tex.rg * 2.0 - 1.0;
        }
        float z = sqrt(max(0.0, 1.0 - dot(xy, xy)));
        float3 tangent_normal = float3(xy, z);
        if (n_tex.b >= 0.1)
        {
            tangent_normal = n_tex.rgb * 2.0 - 1.0;
        }

        float3 N = normalize(input.normal);
        float3 T = normalize(input.tangent - N * dot(input.tangent, N));
        float3 B = cross(N, T);
        float3x3 TBN = float3x3(T, B, N);
        world_normal = normalize(mul(tangent_normal, TBN));
    }

    if (albedo_index != 0xFFFFFFFF)
    {
        float4 albedo = g_textures[NonUniformResourceIndex(albedo_index)].Sample(g_sampler, input.uv);
        if (normal_index != 0xFFFFFFFF)
        {
            float3 light_dir = normalize(float3(0.577, 0.577, 0.577));
            float diff = saturate(dot(world_normal, light_dir));
            float3 lit = albedo.rgb * (0.3 + 0.7 * diff);
            return float4(lit, albedo.a);
        }
        return albedo;
    }

    // Normal visualization fallback (perturbed if normal map bound, geometric otherwise)
    return float4(abs(world_normal), 1.0);
}
