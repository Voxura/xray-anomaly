// Phase 7 Particle Billboard Instancing Shader
// Generates quad vertices purely procedurally via SV_VertexID (0..5) without any vertex buffer upload,
// reading particle instance data directly from ByteAddressBuffer.

cbuffer CameraConstants : register(b0)
{
    column_major float4x4 view;
    column_major float4x4 proj;
    column_major float4x4 view_proj;
    float4 cam_pos;
    float4 padding[3];
};

Texture2D g_textures[] : register(t0);
SamplerState g_sampler : register(s0);
ByteAddressBuffer g_particles : register(t0, space1);

struct ParticleData
{
    float3 position;
    float rotation;
    float2 size;
    float2 uv_min;
    float2 uv_max;
    float4 color;
    uint texture_index;
    uint blend_mode;
};

ParticleData load_particle(uint instance_id)
{
    uint offset = instance_id * 80;
    float4 r0 = asfloat(g_particles.Load4(offset));
    float4 r1 = asfloat(g_particles.Load4(offset + 16));
    float4 r2 = asfloat(g_particles.Load4(offset + 32));
    float4 r3 = asfloat(g_particles.Load4(offset + 48));
    uint4  r4 = g_particles.Load4(offset + 64);

    ParticleData p;
    p.position = r0.xyz;
    p.rotation = r0.w;
    p.size = r1.xy;
    p.uv_min = r1.zw;
    p.uv_max = r2.xy;
    p.color = r3;
    p.texture_index = r4.x;
    p.blend_mode = r4.y;
    return p;
}

struct PSParticleInput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
    nointerpolation uint texture_index : BLENDINDICES0;
};

PSParticleInput vs_particle(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    ParticleData inst = load_particle(instance_id);

    // Quad geometry: 2 triangles (6 vertices)
    // 0: bottom-left  (-0.5, -0.5)
    // 1: top-left     (-0.5,  0.5)
    // 2: bottom-right ( 0.5, -0.5)
    // 3: bottom-right ( 0.5, -0.5)
    // 4: top-left     (-0.5,  0.5)
    // 5: top-right    ( 0.5,  0.5)
    float2 quad_pos[6] = {
        float2(-0.5, -0.5),
        float2(-0.5,  0.5),
        float2( 0.5, -0.5),
        float2( 0.5, -0.5),
        float2(-0.5,  0.5),
        float2( 0.5,  0.5)
    };

    float2 quad_uv[6] = {
        float2(inst.uv_min.x, inst.uv_max.y),
        float2(inst.uv_min.x, inst.uv_min.y),
        float2(inst.uv_max.x, inst.uv_max.y),
        float2(inst.uv_max.x, inst.uv_max.y),
        float2(inst.uv_min.x, inst.uv_min.y),
        float2(inst.uv_max.x, inst.uv_min.y)
    };

    float2 corner = quad_pos[vertex_id] * inst.size;

    // Apply 2D in-plane rotation
    if (abs(inst.rotation) > 0.0001)
    {
        float s = sin(inst.rotation);
        float c = cos(inst.rotation);
        corner = float2(corner.x * c - corner.y * s, corner.x * s + corner.y * c);
    }

    // Camera billboard vectors extracted from view matrix (transposed view rows)
    float3 cam_right = float3(view[0][0], view[1][0], view[2][0]);
    float3 cam_up    = float3(view[0][1], view[1][1], view[2][1]);

    float3 world_pos = inst.position + (cam_right * corner.x + cam_up * corner.y);

    PSParticleInput output;
    output.position = mul(view_proj, float4(world_pos, 1.0));
    output.uv = quad_uv[vertex_id];
    output.color = inst.color;
    output.texture_index = inst.texture_index;
    return output;
}

float4 ps_particle(PSParticleInput input) : SV_Target0
{
    float4 tex_col = float4(1.0, 1.0, 1.0, 1.0);
    if (input.texture_index != 0xFFFFFFFF)
    {
        tex_col = g_textures[NonUniformResourceIndex(input.texture_index)].Sample(g_sampler, input.uv);
    }
    return tex_col * input.color;
}
