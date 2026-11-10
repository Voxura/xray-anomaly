// X-Ray Monolith Engine - SM 6.6 DirectX 12 Deferred Lighting Compute Shader
// Reads GBuffer descriptors via ResourceDescriptorHeap, computes direct & ambient lighting,
// and writes to HDR UAV output.

struct PushLightingConstants
{
    uint gbuffer_albedo_idx;
    uint gbuffer_normal_idx;
    uint gbuffer_material_idx;
    uint gbuffer_depth_idx;
    uint output_uav_idx;
    uint screen_width;
    uint screen_height;
    uint flags;
};

ConstantBuffer<PushLightingConstants> cb_lighting : register(b0);

cbuffer FrameConstants : register(b1)
{
    float4x4 ViewProjection;
    float4x4 InvViewProjection;
    float4 CameraPosition;
    float4 SunDirection;
    float4 SunColor;
    float4 AmbientColor;
};

SamplerState g_point_sampler : register(s0);

[numthreads(8, 8, 1)]
void cs_main(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    if (dispatch_thread_id.x >= cb_lighting.screen_width || dispatch_thread_id.y >= cb_lighting.screen_height)
    {
        return;
    }

    float2 uv = (float2(dispatch_thread_id.xy) + 0.5f) / float2(cb_lighting.screen_width, cb_lighting.screen_height);

    Texture2D albedo_tex = ResourceDescriptorHeap[cb_lighting.gbuffer_albedo_idx];
    Texture2D normal_tex = ResourceDescriptorHeap[cb_lighting.gbuffer_normal_idx];
    Texture2D material_tex = ResourceDescriptorHeap[cb_lighting.gbuffer_material_idx];
    Texture2D<float> depth_tex = ResourceDescriptorHeap[cb_lighting.gbuffer_depth_idx];
    RWTexture2D<float4> output_tex = ResourceDescriptorHeap[cb_lighting.output_uav_idx];

    float4 albedo_val = albedo_tex.SampleLevel(g_point_sampler, uv, 0);
    float4 normal_val = normal_tex.SampleLevel(g_point_sampler, uv, 0);
    float4 material_val = material_tex.SampleLevel(g_point_sampler, uv, 0);
    float depth = depth_tex.SampleLevel(g_point_sampler, uv, 0);

    // Unpack normal
    float3 normal = normalize(normal_val.xyz * 2.0f - 1.0f);
    float roughness = normal_val.w;
    float metallic = material_val.r;

    // World position reconstruction from depth
    float4 clip_pos = float4(uv.x * 2.0f - 1.0f, (1.0f - uv.y) * 2.0f - 1.0f, depth, 1.0f);
    float4 world_pos_h = mul(InvViewProjection, clip_pos);
    float3 world_pos = world_pos_h.xyz / world_pos_h.w;

    // Direct Sun Lighting (Lambertian diffuse + Blinn-Phong specular)
    float3 light_dir = normalize(-SunDirection.xyz);
    float n_dot_l = max(dot(normal, light_dir), 0.0f);
    float3 diffuse_light = albedo_val.rgb * SunColor.rgb * n_dot_l;

    // Specular
    float3 view_dir = normalize(CameraPosition.xyz - world_pos);
    float3 half_vec = normalize(light_dir + view_dir);
    float n_dot_h = max(dot(normal, half_vec), 0.0f);
    float spec_power = (1.0f - roughness) * 128.0f + 1.0f;
    float spec_factor = pow(n_dot_h, spec_power) * n_dot_l;
    float3 spec_light = SunColor.rgb * spec_factor * (0.04f * (1.0f - metallic) + albedo_val.rgb * metallic);

    // Ambient Lighting
    float3 ambient = AmbientColor.rgb * albedo_val.rgb;

    float3 final_color = diffuse_light + spec_light + ambient;

    output_tex[dispatch_thread_id.xy] = float4(final_color, 1.0f);
}
