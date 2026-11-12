// Phase 12: Instanced Detail Foliage & Vegetation Shader
// Renders instanced detail objects (grass clumps, flowers, rocks, shrubs) with
// SSFX wind waving animation, distance dissolve, and alpha-tested cut-out rasterization.

cbuffer CameraConstants : register(b0)
{
    column_major float4x4 view;
    column_major float4x4 proj;
    column_major float4x4 view_proj;
    float4 cam_pos;
    float4 padding[3];
};

cbuffer DetailPushConstants : register(b1)
{
    float2 wind_direction;
    float wind_speed;
    float wind_strength;
    float time_seconds;
    uint texture_index;
    float alpha_threshold;
    float custom_param;
};

Texture2D g_textures[] : register(t0);
SamplerState g_sampler : register(s0);
ByteAddressBuffer g_detail_vertices : register(t0, space1);
ByteAddressBuffer g_detail_instances : register(t0, space2);

struct DetailVSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 lighting : COLOR0;
    float alpha : TEXCOORD1;
};

struct DetailInstanceData
{
    float3 position;
    float scale;
    float rotation_y;
    float alpha;
    float hemi_light;
    float sun_light;
    float4 color_tint;
};

DetailInstanceData load_instance(uint instance_id)
{
    // 48 bytes per instance (3 float4s)
    uint offset = instance_id * 48;
    float4 r0 = asfloat(g_detail_instances.Load4(offset));
    float4 r1 = asfloat(g_detail_instances.Load4(offset + 16));
    float4 r2 = asfloat(g_detail_instances.Load4(offset + 32));

    DetailInstanceData inst;
    inst.position   = r0.xyz;
    inst.scale      = r0.w;
    inst.rotation_y = r1.x;
    inst.alpha      = r1.y;
    inst.hemi_light = r1.z;
    inst.sun_light  = r1.w;
    inst.color_tint = r2;
    return inst;
}

DetailVSOutput vs_detail(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    DetailInstanceData inst = load_instance(instance_id);

    // 24 bytes per mesh vertex: float3 pos (12B), float2 uv (8B), float height_fraction (4B)
    uint v_offset = vertex_id * 24;
    float4 v0 = asfloat(g_detail_vertices.Load4(v_offset));
    float2 uv = asfloat(g_detail_vertices.Load2(v_offset + 16));

    float3 local_pos = v0.xyz * inst.scale;
    float height_frac = v0.w;

    // Rotate around Y axis
    float s = sin(inst.rotation_y);
    float c = cos(inst.rotation_y);
    float3 rotated_pos = float3(
        local_pos.x * c + local_pos.z * s,
        local_pos.y,
        -local_pos.x * s + local_pos.z * c
    );

    // Procedural wind wave animation (Monolith / SSFX wind tree formula)
    // Displaces vertex horizontally proportional to height from ground
    float wave_phase = time_seconds * wind_speed + dot(inst.position.xz, float2(0.12, 0.12));
    float wave = sin(wave_phase) * cos(wave_phase * 0.73);
    float3 wind_disp = float3(wind_direction.x, 0.0, wind_direction.y) * (wave * wind_strength * height_frac);

    float3 world_pos = inst.position + rotated_pos + wind_disp;
    float4 clip_pos = mul(view_proj, float4(world_pos, 1.0));

    // Combine hemispheric ambient and directional sunlight
    float light_factor = saturate(inst.hemi_light * 0.6 + inst.sun_light * 0.4);
    float4 lit_color = inst.color_tint * float4(light_factor, light_factor, light_factor, 1.0);

    DetailVSOutput output;
    output.position = clip_pos;
    output.uv = uv;
    output.lighting = lit_color;
    output.alpha = inst.alpha;
    return output;
}

float4 ps_detail(DetailVSOutput input) : SV_Target
{
    float4 tex_color = float4(1.0, 1.0, 1.0, 1.0);
    if (texture_index != 0xFFFFFFFF)
    {
        tex_color = g_textures[NonUniformResourceIndex(texture_index)].Sample(g_sampler, input.uv);
    }

    // Alpha test cut-out with distance dissolve factor
    float effective_alpha = tex_color.a * input.alpha;
    clip(effective_alpha - alpha_threshold);

    float3 final_rgb = tex_color.rgb * input.lighting.rgb;
    return float4(final_rgb, effective_alpha);
}
