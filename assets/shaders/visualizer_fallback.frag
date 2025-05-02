#version 120

// Inputs from vertex shader
varying vec2 fragTexCoord;
varying vec2 fragPosition;

// Uniforms
uniform float time;       // Current time in seconds
uniform float uOnset;     // Audio onset detection (0.0 - 1.0)
uniform float uEnergy;    // Audio energy level (0.0 - 1.0)
uniform float uBeatPhase; // Phase of the beat (0.0 - 1.0)

// Simplified version of the glow function for older GLSL versions
float glow(vec2 position, vec2 center, float radius, float intensity) {
    float dist = length(position - center);
    return intensity * exp(-dist * dist / radius);
}

void main() {
    // Base coordinates (centered, -1 to 1 range)
    vec2 position = fragPosition;
    
    // Calculate distance from center
    float dist = length(position);
    
    // Simplified effects for compatibility
    float circle = smoothstep(0.5, 0.0, dist) * uBeatPhase;
    float burst = glow(position, vec2(0.0, 0.0), 2.0, uOnset);
    
    // Base color - simple dark blue
    vec3 color = vec3(0.1, 0.1, 0.2);
    
    // Add beat-reactive circle (cyan)
    color = mix(color, vec3(0.0, 0.7, 0.9), circle);
    
    // Add onset-reactive burst (white)
    color = mix(color, vec3(1.0), burst);
    
    // Apply energy to overall brightness
    color *= (0.5 + uEnergy);
    
    // Final color
    gl_FragColor = vec4(color, 1.0);
} 