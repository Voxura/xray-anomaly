// Phase 8 Direct3D 12 2D UI Canvas & HUD Immediate-Mode Shader
// Generates procedural quads from SV_VertexID without any dynamic vertex or index buffers.

cbuffer UiPushConstants : register(b0)
{
    uint screen_width;
    uint screen_height;
    uint virtual_width;   // e.g. 1024 (X-Ray standard virtual resolution) or 0 for direct pixels
    uint virtual_height;  // e.g. 768 or 0
};

struct GpuUiQuad
{
    float4 rect;        // [x, y, width, height] in pixels or virtual units
    float4 uv;          // [u1, v1, u2, v2] normalized texture coordinates
    float4 color;       // [r, g, b, a] tint / vertex color multiplier
    uint texture_index; // Bindless SRV descriptor index in g_textures, or 0xFFFFFFFF for solid color
    uint pad0;
    uint pad1;
    uint pad2;
};

Texture2D g_textures[] : register(t0);
SamplerState g_sampler : register(s0);
StructuredBuffer<GpuUiQuad> g_quads : register(t0, space1);

static const float2 QUAD_CORNERS[6] = {
    float2(0.0, 0.0), // 0: Top-left
    float2(1.0, 0.0), // 1: Top-right
    float2(1.0, 1.0), // 2: Bottom-right
    float2(0.0, 0.0), // 3: Top-left
    float2(1.0, 1.0), // 4: Bottom-right
    float2(0.0, 1.0), // 5: Bottom-left
};

struct PSInput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
    nointerpolation uint texture_index : TEXCOORD1;
};

PSInput vs_main(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    PSInput output;
    GpuUiQuad quad = g_quads[instance_id];
    float2 corner = QUAD_CORNERS[vertex_id];

    // Compute pixel position from rect [x, y, w, h]
    float2 pixel_pos = quad.rect.xy + corner * quad.rect.zw;

    // Apply virtual resolution scaling if virtual_width and virtual_height are provided
    if (virtual_width > 0 && virtual_height > 0)
    {
        pixel_pos.x = pixel_pos.x * ((float)screen_width / (float)virtual_width);
        pixel_pos.y = pixel_pos.y * ((float)screen_height / (float)virtual_height);
    }

    // Direct3D NDC: X in [-1, 1], Y in [1, -1] (top-left is (0,0) in UI space)
    float ndc_x = (pixel_pos.x / (float)screen_width) * 2.0 - 1.0;
    float ndc_y = 1.0 - (pixel_pos.y / (float)screen_height) * 2.0;

    output.position = float4(ndc_x, ndc_y, 0.0, 1.0);
    output.uv = lerp(quad.uv.xy, quad.uv.zw, corner);
    output.color = quad.color;
    output.texture_index = quad.texture_index;
    return output;
}

float4 ps_main(PSInput input) : SV_Target0
{
    float4 tex_color = float4(1.0, 1.0, 1.0, 1.0);
    if (input.texture_index != 0xFFFFFFFF)
    {
        tex_color = g_textures[NonUniformResourceIndex(input.texture_index)].Sample(g_sampler, input.uv);
        // Handle A8 font / alpha mask textures: when RGB is 0.0 and alpha > 0.0, expand RGB to 1.0
        if (tex_color.r == 0.0 && tex_color.g == 0.0 && tex_color.b == 0.0)
        {
            tex_color.rgb = float3(1.0, 1.0, 1.0);
        }
    }

    float4 final_color = tex_color * input.color;
    return final_color;
}
