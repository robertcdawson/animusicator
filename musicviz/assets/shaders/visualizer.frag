#version 330 core

in vec2 TexCoord;

out vec4 FragColor;

// Audio feature uniforms
uniform float uOnset;     // Onset detection (0.0-1.0)
uniform float uEnergy;    // Audio energy/loudness (0.0-1.0)
uniform float uTime;      // Time in seconds

// Audio spectrum and data
uniform sampler2D uSpectrumTexture;  // Audio spectrum texture

void main() {
    // Calculate position relative to center
    vec2 center = vec2(0.5, 0.5);
    vec2 pos = TexCoord - center;
    float dist = length(pos);
    
    // Base color - sample spectrum texture for color variety
    vec3 baseColor = texture(uSpectrumTexture, vec2(dist * 2.0, 0.5)).rgb;
    
    // Create flowing rings based on time
    float rings = sin(dist * 30.0 - uTime * 0.5) * 0.5 + 0.5;
    
    // Pulse based on onset detection
    float pulse = 1.0 + uOnset * 2.0 * (1.0 - dist);
    
    // Final color with energy modulation for brightness
    vec3 color = baseColor * rings * pulse;
    color *= 0.6 + uEnergy * 0.8; // Energy affects brightness
    
    // Add glow effect on strong onsets
    float glow = uOnset * max(0.0, 1.0 - dist * 3.0);
    color += vec3(0.8, 0.7, 1.0) * glow;
    
    // Output final color with distance-based alpha for soft edges
    float alpha = smoothstep(1.0, 0.8, dist);
    FragColor = vec4(color, alpha);
} 