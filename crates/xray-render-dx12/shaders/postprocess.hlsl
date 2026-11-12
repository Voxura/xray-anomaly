// Phase 13: Post-Processing, HDR10 Tonemapping, Color Grading & Vignette Shader
// Renders via procedural full-screen triangle from SV_VertexID (0..2) without any vertex buffer upload.

cbuffer PostprocessConstants : register(b0)
{
    float2 screen_res_inv; // (1.0 / width, 1.0 / height)
    float time_seconds;
    float bloom_factor;
    uint scene_texture_index;
    uint bloom_texture_index;
    uint tonemap_mode; // 0 = Uncharted 2 / Filmic, 1 = ACES, 2 = Reinhard, 3 = Linear / Pass-through
    uint color_grading_enabled;
};

cbuffer TonemapConstants : register(b1)
{
    float4 tnmp_abcd;        // (a, b, c, d)
    float4 tnmp_efw_exposure;// (e, f, w, exposure)
    float4 tnmp_gamma_onoff; // (gamma, enabled, 0, 0)
};

cbuffer ImageCorrectionConstants : register(b2)
{
    float4 img_corrections;  // (exposure, gamma, saturation, 1.0)
    float4 img_color_grading;// (r, g, b, 0.0)
    float4 vignette_control; // (center_x, center_y, radius, intensity)
};

Texture2D g_textures[] : register(t0);
SamplerState g_sampler : register(s0);

struct PostprocessVSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Procedural fullscreen triangle (0..2) covering the entire NDC quad [-1, 1]
PostprocessVSOutput vs_postprocess(uint vertex_id : SV_VertexID)
{
    PostprocessVSOutput output;
    // Vertex 0: (-1, -1), UV: (0, 1)
    // Vertex 1: (-1,  3), UV: (0, -1)
    // Vertex 2: ( 3, -1), UV: (2, 1)
    float2 pos = float2((vertex_id == 2) ? 3.0 : -1.0, (vertex_id == 1) ? 3.0 : -1.0);
    output.position = float4(pos, 0.0, 1.0);
    output.uv = float2(pos.x * 0.5 + 0.5, 1.0 - (pos.y * 0.5 + 0.5));
    return output;
}

// Uncharted 2 rational shoulder-toe curve: F(x) = (x(Ax + CB) + DE) / (x(Ax + B) + DF) - E/F
float3 uncharted2_tonemap_partial(float3 x, float4 abcd, float4 efw)
{
    float a = abcd.x;
    float b = abcd.y;
    float c = abcd.z;
    float d = abcd.w;
    float e = efw.x;
    float f = max(efw.y, 1e-6);

    float3 num = x * (a * x + c * b) + (d * e);
    float3 den = x * (a * x + b) + (d * f);
    return (num / max(den, 1e-6)) - (e / f);
}

float3 uncharted2_tonemap(float3 color, float4 abcd, float4 efw)
{
    float exposure = efw.w;
    float w = efw.z;
    float3 curr = uncharted2_tonemap_partial(color * exposure, abcd, efw);
    float3 white_scale = 1.0 / max(uncharted2_tonemap_partial(float3(w, w, w), abcd, efw), 1e-6);
    return saturate(curr * white_scale);
}

float3 aces_tonemap(float3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float3 reinhard_tonemap(float3 x)
{
    return saturate(x / (1.0 + x));
}

float4 ps_postprocess(PostprocessVSOutput input) : SV_Target
{
    float4 scene_color = float4(0.0, 0.0, 0.0, 1.0);
    if (scene_texture_index != 0xFFFFFFFF)
    {
        scene_color = g_textures[NonUniformResourceIndex(scene_texture_index)].Sample(g_sampler, input.uv);
    }

    // Add bloom contribution
    float3 hdr_color = scene_color.rgb;
    if (bloom_texture_index != 0xFFFFFFFF && bloom_factor > 0.001)
    {
        float3 bloom = g_textures[NonUniformResourceIndex(bloom_texture_index)].Sample(g_sampler, input.uv).rgb;
        hdr_color += bloom * bloom_factor;
    }

    // Apply pre-tonemap exposure
    float exposure = max(img_corrections.x, 0.001);
    hdr_color *= exposure;

    // Tonemap HDR -> LDR [0.0..1.0]
    float3 ldr_color;
    if (tonemap_mode == 0)
    {
        // Uncharted 2 / Filmic
        ldr_color = uncharted2_tonemap(hdr_color, tnmp_abcd, tnmp_efw_exposure);
    }
    else if (tonemap_mode == 1)
    {
        // ACES Filmic
        ldr_color = aces_tonemap(hdr_color);
    }
    else if (tonemap_mode == 2)
    {
        // Reinhard
        ldr_color = reinhard_tonemap(hdr_color);
    }
    else
    {
        // Pass-through / clamped
        ldr_color = saturate(hdr_color);
    }

    // Color grading tint & saturation
    float luma = dot(ldr_color, float3(0.299, 0.587, 0.114));
    float saturation = max(img_corrections.z, 0.0);
    ldr_color = lerp(float3(luma, luma, luma), ldr_color, saturation);

    if (color_grading_enabled != 0)
    {
        ldr_color += img_color_grading.rgb;
    }

    // Gamma correction
    float gamma = max(img_corrections.y, 0.001);
    if (abs(gamma - 1.0) > 0.001)
    {
        ldr_color = pow(saturate(ldr_color), 1.0 / gamma);
    }

    // Radial screen vignette matching X-Ray combine_2_naa.ps
    if (vignette_control.x > 0.001 || vignette_control.y > 0.001)
    {
        float2 mid = input.uv - 0.5;
        float dist_norm = length(mid) * 2.0;
        float edge_radius = 1.41421356 * (1.0 - saturate(vignette_control.z));
        float t = (edge_radius > 1e-6) ? saturate(dist_norm / edge_radius) : 0.0;
        float center_mul = 1.0 - vignette_control.y;
        float corner_mul = 1.0 - vignette_control.x;
        float factor = lerp(center_mul, corner_mul, t);
        ldr_color *= factor;
    }

    return float4(saturate(ldr_color), 1.0);
}
