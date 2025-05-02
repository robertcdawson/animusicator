#version 330 core

// Inputs from vertex shader
in vec2 fragTexCoord;
in vec2 fragPosition;

// Output color
out vec4 fragColor;

// Uniforms
uniform float time;       // Current time in seconds
uniform float uOnset;     // Audio onset detection (0.0 - 1.0)
uniform float uEnergy;    // Audio energy level (0.0 - 1.0)
uniform float uBeatPhase; // Phase of the beat (0.0 - 1.0)

// Optional audio feature uniforms (to be used in future versions)
uniform float uCentroid;  // Spectral centroid
uniform vec4 uContrast;   // Spectral contrast (simplified to 4 bands)
uniform vec4 uChroma;     // Chroma features (simplified to 4 values)
uniform vec4 uMFCC;       // MFCC features (simplified to 4 coefficients)

// Color palette
vec3 baseColor = vec3(0.1, 0.1, 0.2);
vec3 accentColor = vec3(0.0, 0.7, 0.9);
vec3 highlightColor = vec3(1.0, 0.3, 0.7);

// Function to create a glow effect
float glow(vec2 position, vec2 center, float radius, float intensity) {
    float dist = length(position - center);
    return intensity * exp(-dist * dist / radius);
}

void main() {
    // Base coordinates (centered, -1 to 1 range)
    vec2 position = fragPosition;
    
    // Calculate distance from center
    float dist = length(position);
    
    // Create a pulsing circle based on beat phase
    float circle = smoothstep(0.5 + 0.2 * sin(uBeatPhase * 6.28), 0.0, dist);
    
    // Create radial rays
    float angle = atan(position.y, position.x);
    float rays = abs(sin(angle * 8.0 + time * 2.0));
    rays = smoothstep(0.4, 0.6, rays) * 0.5;
    
    // Combine effects with audio reactivity
    float energyEffect = uEnergy * 1.5;  // Amplify energy for more impact
    
    // Base color
    vec3 color = baseColor;
    
    // Add beat-reactive circle
    color = mix(color, accentColor, circle * uBeatPhase);
    
    // Add energy-reactive rays
    color = mix(color, highlightColor, rays * energyEffect);
    
    // Add onset-reactive burst
    float burst = glow(position, vec2(0.0), 2.0, uOnset);
    color = mix(color, vec3(1.0), burst);
    
    // Apply energy to overall brightness
    color *= (0.5 + energyEffect);
    
    // Final color
    fragColor = vec4(color, 1.0);
} 