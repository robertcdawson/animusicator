#version 330 core

// Input vertex data (from VAO)
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

// Output data to fragment shader
out vec2 fragTexCoord;
out vec2 fragPosition;

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