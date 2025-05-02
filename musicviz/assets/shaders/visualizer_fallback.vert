#version 120

// Input vertex data (attributes)
attribute vec3 position;   // layout-equivalent to location 0
attribute vec2 texCoord;   // layout-equivalent to location 1

// Output data to fragment shader
varying vec2 fragTexCoord;

// Uniforms
uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

// Legacy names kept for compatibility but not used
// uniform mat4 projection;
// uniform mat4 view;
// uniform mat4 model;
// uniform float time;

void main()
{
    gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(position, 1.0);
    fragTexCoord = texCoord;
}
