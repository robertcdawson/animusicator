#version 120

// Input vertex data (attributes)
attribute vec3 position;
attribute vec2 texCoord;

// Output data to fragment shader
varying vec2 fragTexCoord;
varying vec2 fragPosition;

// Uniforms
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform float time;

void main() {
    // Calculate final position in clip space
    gl_Position = projection * view * model * vec4(position, 1.0);
    
    // Pass texture coordinates to fragment shader
    fragTexCoord = texCoord;
    
    // Pass normalized position to fragment shader (useful for effects)
    fragPosition = position.xy;
} 