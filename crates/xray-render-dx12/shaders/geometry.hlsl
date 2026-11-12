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
ByteAddressBuffer g_lights : register(t1, space1);

struct GpuLightData
{
    float3 position;
    float range;
    float4 color;
    float3 direction;
    uint light_type;
    float2 spot_params; // cos_inner, cos_outer
    float2 attenuation;
};

GpuLightData load_light(uint index)
{
    uint offset = index * 64;
    uint4 w0 = g_lights.Load4(offset);
    uint4 w1 = g_lights.Load4(offset + 16);
    uint4 w2 = g_lights.Load4(offset + 32);
    uint4 w3 = g_lights.Load4(offset + 48);

    GpuLightData l;
    l.position = asfloat(w0.xyz);
    l.range = asfloat(w0.w);
    l.color = asfloat(w1);
    l.direction = asfloat(w2.xyz);
    l.light_type = w2.w;
    l.spot_params = asfloat(w3.xy);
    l.attenuation = asfloat(w3.zw);
    return l;
}

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

    // Dynamic point and spot light accumulation
    float3 dynamic_light_accum = float3(0.0, 0.0, 0.0);
    uint num_lights = (custom_param2 != 0xFFFFFFFF) ? min(custom_param2, 32u) : 0u;
    for (uint li = 0; li < num_lights; ++li)
    {
        GpuLightData light = load_light(li);
        float3 to_light = light.position - input.world_pos.xyz;
        float dist = length(to_light);
        if (dist < light.range && dist > 0.0001)
        {
            float3 L = to_light / dist;
            float n_dot_l = saturate(dot(world_normal, L));
            if (n_dot_l > 0.0)
            {
                float att_dist = saturate(1.0 - dist / light.range);
                float att = att_dist * att_dist;

                if (light.light_type == 1) // SPOT
                {
                    float spot_cos = dot(-L, normalize(light.direction));
                    float spot_factor = saturate((spot_cos - light.spot_params.y) / max(0.0001, light.spot_params.x - light.spot_params.y));
                    att *= spot_factor;
                }

                float3 V = normalize(cam_pos.xyz - input.world_pos.xyz);
                float3 H = normalize(L + V);
                float spec = pow(saturate(dot(world_normal, H)), 32.0);

                dynamic_light_accum += (light.color.rgb * light.color.a) * (n_dot_l * att + spec * att * 0.5);
            }
        }
    }

    if (albedo_index != 0xFFFFFFFF)
    {
        float4 albedo = g_textures[NonUniformResourceIndex(albedo_index)].Sample(g_sampler, input.uv);
        if (normal_index != 0xFFFFFFFF || custom_param0 != 0xFFFFFFFF || num_lights > 0)
        {
            float3 sun_lit = albedo.rgb * (0.2 + 0.8 * diff * shadow_factor);
            float3 lit = sun_lit + albedo.rgb * dynamic_light_accum;
            return float4(lit, albedo.a);
        }
        return albedo;
    }

    // Normal visualization fallback (perturbed if normal map bound, geometric otherwise)
    float3 base_color = abs(world_normal);
    if (custom_param0 != 0xFFFFFFFF || num_lights > 0)
    {
        float3 sun_lit = base_color * (0.25 + 0.75 * diff * shadow_factor);
        return float4(sun_lit + base_color * dynamic_light_accum, 1.0);
    }
    return float4(base_color, 1.0);
}
