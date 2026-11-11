// Phase 4 baseline raster shader & Phase 5/6/7 bindless textured, depth-prepass & shadow shader.
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
    uint custom_param0; // shadow_map_index
    uint custom_param1; // shadow_transform_index
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
    float4 world_pos : TEXCOORD1;
};

// Fast depth-only prepass & shadow map rasterization vertex shader
float4 vs_depth_main(VSInput input) : SV_Position
{
    float4 local_pos = float4(input.position, 1.0);
    float4 world_pos;
    if (transform_index != 0xFFFFFFFF)
    {
        float4x4 world = load_transform(transform_index);
        world_pos = mul(local_pos, world);
    }
    else
    {
        world_pos = local_pos;
    }
    return mul(view_proj, world_pos);
}

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
    output.world_pos = world_pos;
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

    // Directional sunlight and shadow evaluation
    float3 sun_dir = normalize(float3(0.577, 0.577, 0.577));
    float shadow_factor = 1.0;

    // custom_param0 binds shadow map bindless index
    if (custom_param0 != 0xFFFFFFFF)
    {
        float4x4 light_mat = (custom_param1 != 0xFFFFFFFF) ? load_transform(custom_param1) : view_proj;
        float4 light_clip = mul(light_mat, input.world_pos);
        if (light_clip.w > 0.0001)
        {
            float3 light_ndc = light_clip.xyz / light_clip.w;
            // Direct3D NDC: X in [-1, 1], Y in [-1, 1], Z in [0, 1]
            float2 shadow_uv = float2(light_ndc.x * 0.5 + 0.5, -light_ndc.y * 0.5 + 0.5);
            if (shadow_uv.x >= 0.0 && shadow_uv.x <= 1.0 && shadow_uv.y >= 0.0 && shadow_uv.y <= 1.0)
            {
                float shadow_depth = g_textures[NonUniformResourceIndex(custom_param0)].SampleLevel(g_sampler, shadow_uv, 0).r;
                float current_depth = light_ndc.z;
                float bias = 0.003;
                if (current_depth - bias > shadow_depth)
                {
                    shadow_factor = 0.25;
                }
            }
        }
    }

    float diff = saturate(dot(world_normal, sun_dir));

    if (albedo_index != 0xFFFFFFFF)
    {
        float4 albedo = g_textures[NonUniformResourceIndex(albedo_index)].Sample(g_sampler, input.uv);
        if (normal_index != 0xFFFFFFFF || custom_param0 != 0xFFFFFFFF)
        {
            float3 lit = albedo.rgb * (0.2 + 0.8 * diff * shadow_factor);
            return float4(lit, albedo.a);
        }
        return albedo;
    }

    // Normal visualization fallback (perturbed if normal map bound, geometric otherwise)
    float3 base_color = abs(world_normal);
    if (custom_param0 != 0xFFFFFFFF)
    {
        return float4(base_color * (0.25 + 0.75 * diff * shadow_factor), 1.0);
    }
    return float4(base_color, 1.0);
}
