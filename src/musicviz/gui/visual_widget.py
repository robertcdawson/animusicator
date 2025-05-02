#!/usr/bin/env python3
"""
OpenGL visualization widget for the Animusicator application.
"""

import os
import logging
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import (
    QSurfaceFormat,
    QPainter,
    QColor,
    QFont,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLBuffer,
    QOpenGLVertexArrayObject
)

# Import PyOpenGL functions directly
from OpenGL.GL import *

logger = logging.getLogger(__name__)

# Define basic GL constants since we're not using PyOpenGL for constants
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_TRIANGLES = 0x0004
GL_FLOAT = 0x1406

class VisualWidget(QOpenGLWidget):
    """
    QOpenGLWidget subclass for audio visualization.
    """
    
    # Add signal for status updates if needed elsewhere
    status_updated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set OpenGL format
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setVersion(3, 3)  # OpenGL 3.3
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        self.setFormat(fmt)
        
        # Initialize properties
        self.features = None
        self.vao = None
        self.vbo = None
        self.program = None
        self.shaders_loaded = False
        self.debug_overlay = False
        self.status_message = "Ready"
        self.show_status = True
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = 0
        
        # Set up FPS timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS
    
    def initializeGL(self):
        """Initialize OpenGL resources using PyOpenGL directly."""
        logger.info("Initializing OpenGL using PyOpenGL...")
        
        # Optionally, log context format info
        fmt = self.context().format()
        logger.info(f"OpenGL context version: {fmt.majorVersion()}.{fmt.minorVersion()}")
        
        try:
            # No FBO needed when using default framebuffer via Qt
            self._compile_shaders()
            self._setup_geometry()
            self.shaders_loaded = True
            logger.info("OpenGL initialization successful")
        except Exception as e:
            logger.error(f"Failed to initialize OpenGL: {e}")
            self.shaders_loaded = False
    
    def _compile_shaders(self):
        """Compile and link GLSL shaders using Qt's shader program."""
        vertex_shader = """
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec2 texCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(position, 1.0);
            TexCoord = texCoord;
        }
        """
        fragment_shader = """
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform float uTime;
        uniform float uOnset;
        uniform float uEnergy;
        void main() {
            vec2 center = vec2(0.5);
            float dist = distance(TexCoord, center);
            
            // Enhanced pulse effect - more sensitive for microphone input
            float pulse = sin(uTime * 3.0 + dist * 15.0) * 0.5 + 0.5;
            pulse *= pow(1.0 - dist * 1.8, 2.0);
            
            // Microphone-optimized onset response - more sensitive to small changes
            float onsetGlow = uOnset * 8.0 * (1.0 - dist * 1.5);
            pulse += onsetGlow;
            
            // Brighter base colors - better visibility for subtle audio
            vec3 color = vec3(
                0.7 + 0.5 * sin(uTime * 1.2 + TexCoord.x * 8.0),
                0.7 + 0.5 * cos(uTime * 0.8 + TexCoord.y * 8.0),
                0.8 + 0.4 * sin(uTime * 0.6 + dist * 10.0)
            );
            
            // Increase base energy level for microphone input
            float energyBoost = 0.6 + uEnergy * 2.0;  // Higher base value, more amplification
            color *= energyBoost;
            
            // Apply pulse with higher intensity
            color = mix(color * 0.4, color * 1.8, pulse);
            
            // Enhanced edge glow - wider and brighter
            float glow = smoothstep(1.0, 0.0, dist * 1.0);  // Wider glow
            color = mix(vec3(0.1, 0.1, 0.2), color, glow * 2.5);  // Add subtle blue background
            
            // Multiple reactive rings for more visual interest with quiet audio
            if (dist > 0.38 && dist < 0.42) {
                color += vec3(1.0, 0.8, 0.4) * (uOnset * 3.0 + 0.2);  // Inner ring - always slightly visible
            }
            if (dist > 0.48 && dist < 0.51) {
                color += vec3(0.4, 0.8, 1.0) * (uEnergy * 2.0 + 0.1);  // Outer ring - always slightly visible
            }
            
            // Add subtle animation even without audio input
            float idleAnimation = 0.15 * sin(uTime * 0.5 + TexCoord.x * 6.0 + TexCoord.y * 6.0);
            color += vec3(idleAnimation, idleAnimation * 0.8, idleAnimation * 1.2);
            
            // Final color adjustment - brighter overall
            color = min(color * 1.4, vec3(1.0));  // Higher boost with clamping
            
            FragColor = vec4(color, 1.0);
        }
        """
        # Use Qt's shader program
        self.program = QOpenGLShaderProgram(self)
        if not self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_shader):
            raise RuntimeError("Vertex shader compilation failed:\n" + self.program.log())
        if not self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_shader):
            raise RuntimeError("Fragment shader compilation failed:\n" + self.program.log())
        if not self.program.link():
            raise RuntimeError("Shader linking failed:\n" + self.program.log())
        logger.info("Shaders compiled successfully")
    
    def _setup_geometry(self):
        """Set up vertex array object and buffer using Qt classes."""
        # Define a full-screen quad (2 triangles)
        vertices = np.array([
            -1.0, -1.0, 0.0,   0.0, 0.0,
             1.0, -1.0, 0.0,   1.0, 0.0,
             1.0,  1.0, 0.0,   1.0, 1.0,
            -1.0, -1.0, 0.0,   0.0, 0.0,
             1.0,  1.0, 0.0,   1.0, 1.0,
            -1.0,  1.0, 0.0,   0.0, 1.0
        ], dtype=np.float32)
        self.vao = QOpenGLVertexArrayObject(self)
        self.vao.create()
        self.vao.bind()

        # Create VBO
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo.create()
        self.vbo.bind()
        self.vbo.allocate(vertices.tobytes(), vertices.nbytes)
        
        # Set up attribute arrays
        if not self.program.bind():
             logger.error("Failed to bind shader program in geometry setup")
             self.vao.release() # Release VAO before returning
             return

        self.program.enableAttributeArray(0)
        self.program.setAttributeBuffer(0, GL_FLOAT, 0, 3, 5 * np.dtype(np.float32).itemsize)
        self.program.enableAttributeArray(1)
        self.program.setAttributeBuffer(1, GL_FLOAT, 3 * np.dtype(np.float32).itemsize, 2, 5 * np.dtype(np.float32).itemsize)
        
        self.program.release()
        self.vbo.release() # Release VBO after VAO is configured
        
        self.vao.release() # Manually release VAO
        logger.info("Geometry setup completed")
    
    def paintGL(self):
        """Render the scene using PyOpenGL functions."""
        self.frame_count += 1
        import time
        current_time = time.time() # Use system time
        
        # Use OpenGL.GL functions directly
        glClearColor(0.07, 0.07, 0.07, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if not self.shaders_loaded or not self.program or not self.vao:
            # logger.warning("Shaders or geometry not ready in paintGL.") # Reduce log spam
            self._paint_fallback() 
            return
        
        # Bind Qt shader program and set uniforms
        if not self.program.bind():
             logger.error("Failed to bind shader program in paintGL")
             self._paint_fallback()
             return

        self.program.setUniformValue("uTime", current_time)
        
        # Set audio feature uniforms if available
        if self.features is not None:
            # Access attributes directly, provide defaults if None, and cast to float
            onset_val = float(getattr(self.features, 'onset', 0.0) or 0.0)
            energy_val = float(getattr(self.features, 'energy', 0.5) or 0.5)
            self.program.setUniformValue("uOnset", onset_val)
            self.program.setUniformValue("uEnergy", energy_val)
        else:
            # Ensure these are standard floats too
            self.program.setUniformValue("uOnset", 0.0)
            self.program.setUniformValue("uEnergy", 0.5)
        
        # Bind VAO (using Qt helper) and draw (using PyOpenGL)
        self.vao.bind()
        glDrawArrays(GL_TRIANGLES, 0, 6)
        self.vao.release() # Manually release VAO
        
        self.program.release()
        
        # Draw overlays after releasing the shader program
        if self.show_status:
             self._paint_status_overlay()
        if self.debug_overlay:
            self._paint_debug_overlay()
    
    def _paint_fallback(self):
        """Paint a fallback visualization when shaders are not available."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(0, 0, self.width(), self.height(), QColor(18, 18, 18))
        
        # Draw message
        painter.setPen(QColor(200, 200, 200))
        font = QFont("SF Pro Display", 14)
        font.setStyleHint(QFont.SansSerif)  # Fall back to system font if not available
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, "Shader initialization failed.\nFallback visualization active.")
        
        # Draw a simple visualization
        if self.features is not None:
            # Handle both dict and dataclass inputs gracefully
            if isinstance(self.features, dict):
                energy = self.features.get('energy', 0.5)
                onset  = self.features.get('onset', 0.0)
            else:
                energy = getattr(self.features, 'energy', 0.5)
                onset  = getattr(self.features, 'onset', 0.0)
            
            # Draw a circle whose size is affected by energy
            center_x = self.width() / 2
            center_y = self.height() / 2
            radius = min(self.width(), self.height()) * 0.2 * (0.5 + energy * 0.5)
            
            # Color affected by onset
            color = QColor(
                int(120 + onset * 135),
                int(120 + (1.0 - onset) * 135),
                int(180),
                200
            )
            
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        painter.end()
    
    def _paint_status_overlay(self):
        """Paint the status message overlay."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(220, 220, 220, 200))
        font = QFont("SF Pro Display", 12) # Use standard UI font
        font.setStyleHint(QFont.SansSerif)
        painter.setFont(font)
        
        # Draw status message bottom-left
        text_rect = self.rect().adjusted(10, 0, -10, -10) # Padding
        painter.drawText(text_rect, Qt.AlignBottom | Qt.AlignLeft, self.status_message)
        painter.end()
    
    def _paint_debug_overlay(self):
        """Paint debug information using QPainter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(220, 220, 220, 200))
        font = QFont("SF Mono", 10) # Use monospace font
        font.setStyleHint(QFont.Monospace)
        painter.setFont(font)

        # Calculate FPS (simple moving average might be better)
        current_timestamp = time.time()
        if current_timestamp - self.last_fps_update >= 1.0:
             self.fps = self.frame_count / (current_timestamp - self.last_fps_update)
             self.frame_count = 0
             self.last_fps_update = current_timestamp
        
        lines = [f"FPS: {self.fps:.1f}"]
        if self.features:
             # Access attributes directly
             lines.append(f"Onset: {getattr(self.features, 'onset', 0.0):.2f}")
             lines.append(f"Energy: {getattr(self.features, 'energy', 0.0):.2f}")
             # Add other features if they exist on the object
             # Example: lines.append(f"Centroid: {getattr(self.features, 'centroid', 0.0):.2f}")
        else:
             lines.append("No features yet")

        text_y = 20
        for line in lines:
             painter.drawText(10, text_y, line)
             text_y += 15
             
        painter.end() # Explicitly end painting
    
    def resizeGL(self, width, height):
        """Handle window resizing."""
        # Use PyOpenGL directly for viewport
        glViewport(0, 0, width, height)
        # You might need other resize logic here depending on your projection
    
    def update_features(self, features):
        """Receive new audio features and trigger an update."""
        self.features = features
        self.update()
    
    # New method to accept device name
    def set_device_name(self, name: str):
        """Update the status message with the current device name."""
        if name:
            self.status_message = f"Listening on: {name}"
        else:
            self.status_message = "Ready (Stopped)"
        self.status_updated.emit(self.status_message) # Emit signal if needed
        # No need to call self.update() directly, timer handles it
    
    def toggle_debug_overlay(self):
        """Toggle the debug overlay visibility."""
        self.debug_overlay = not self.debug_overlay
        return self.debug_overlay 