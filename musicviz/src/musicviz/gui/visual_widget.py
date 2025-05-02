#!/usr/bin/env python3
"""
OpenGL visualization widget for audio feature visualization.
"""

import os
import time
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtWidgets import QOpenGLWidget, QMessageBox
from PyQt5.QtGui import QColor, QPainter, QFont
import OpenGL.GL as gl
import logging
import math
import psutil

from ..visual.shaders import ShaderManager, ShaderCompilationError, ShaderLinkingError
from ..audio.feature_extractor import AudioFeatures

# Create logger
logger = logging.getLogger(__name__)


class VisualWidget(QOpenGLWidget):
    """
    OpenGL widget for audio visualization.
    
    This widget renders real-time visual effects based on audio features
    using OpenGL and GLSL shaders.
    """
    
    def __init__(self, parent=None, shader_dir=None):
        """
        Initialize the visualization widget.
        
        Args:
            parent: Parent widget
            shader_dir: Directory containing shader files (default: auto-detect)
        """
        super().__init__(parent)
        
        # Determine shader directory
        if shader_dir is None:
            # Try to locate assets/shaders relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            shader_dir = os.path.join(base_dir, 'assets', 'shaders')
            if not os.path.exists(shader_dir):
                logger.warning(f"Shader directory not found at {shader_dir}")
        
        self.shader_dir = shader_dir
        
        # Shader manager
        self.shader_manager = None
        self.shader_program = None
        
        # GL resources
        self.vao = None
        self.vbo = None
        self.ebo = None
        
        # Audio features
        self.features = None
        self.silence_threshold = 0.01  # Energy threshold for silence detection
        self.silence_counter = 0
        self.is_silent = False
        self.silence_start_time = 0.0
        
        # Animation timing
        self.start_time = time.time()
        self.time_offset = 0
        
        # Status info
        self.device_name = "No device"
        self.show_status = True
        self.show_debug_overlay = False
        self.fps = 0.0
        self.frame_times = []
        self.last_frame_time = time.time()
        
        # Animation timer for continuous updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS
        
        # Shader error handling
        self.shader_error_message = None
        
        # Default attribute locations if no shader yet
        self.pos_loc = 0
        self.tex_loc = 1
    
    def initializeGL(self):
        """Initialize OpenGL resources."""
        try:
            # Set up OpenGL state for basic rendering regardless of shaders
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            
            # Create shader manager
            self.shader_manager = ShaderManager(self.shader_dir)
            
            # Try to load shader program (may fail on macOS Metal)
            self.shader_program = None
            self.shader_error_message = None
            
            try:
                # Load shader program
                vertex_shader = 'visualizer.vert'
                fragment_shader = 'visualizer.frag'
                fallback_vertex = 'visualizer_fallback.vert'
                fallback_fragment = 'visualizer_fallback.frag'
                
                # Attempt to load shader.  The manager will register it under
                # either "main" (for modern contexts) or "main_fallback" (for
                # legacy contexts).  We keep track of which one succeeded so we
                # can refer to it consistently later.

                self.shader_program = self.shader_manager.load_with_fallback(
                    "main", 
                    vertex_shader, 
                    fragment_shader,
                    fallback_vertex,
                    fallback_fragment
                )

                # Determine which program name was actually stored
                if "main" in self.shader_manager.shader_programs:
                    self.active_program_name = "main"
                elif "main_fallback" in self.shader_manager.shader_programs:
                    self.active_program_name = "main_fallback"
                else:
                    self.active_program_name = None
            except (ShaderCompilationError, ShaderLinkingError) as e:
                logger.error(f"Shader compilation/linking error: {e}")
                self.shader_error_message = str(e)
                self.shader_program = None
                # We'll continue with fallback rendering
            except ValueError as e:
                logger.error(f"No compatible shaders available: {e}")
                self.shader_error_message = str(e)
                self.shader_program = None
            except Exception as e:
                logger.error(f"Failed to load shaders: {e}")
                self.shader_error_message = f"Shader initialization error: {e}"
                self.shader_program = None
            
            if self.shader_program is None:
                logger.error("Failed to load shader program, using fallback rendering")
                if self.shader_manager.has_compilation_errors():
                    errors = self.shader_manager.get_last_compilation_errors()
                    error_details = "\n".join(errors[:3])  # Show first 3 errors
                    logger.error(f"Shader compilation errors: {error_details}")
                    
                    # Schedule an error notification to be shown after widget is fully initialized
                    QTimer.singleShot(1000, lambda: self._show_shader_error(errors))
            
            # Setup quad geometry for advanced rendering
            try:
                # Set up geometry for a quad that fills the viewport
                vertices = np.array([
                    # positions        # texture coords
                    -1.0, -1.0, 0.0,   0.0, 0.0,  # bottom left
                     1.0, -1.0, 0.0,   1.0, 0.0,  # bottom right
                     1.0,  1.0, 0.0,   1.0, 1.0,  # top right
                    -1.0,  1.0, 0.0,   0.0, 1.0   # top left
                ], dtype=np.float32)
                
                indices = np.array([
                    0, 1, 2,  # first triangle
                    2, 3, 0   # second triangle
                ], dtype=np.uint32)
                
                # Check if the current context supports Vertex Array Objects (requires ≥GL 3.0)
                self.gl_major, self.gl_minor = self.shader_manager.get_gl_version()
                self.has_vao_support = (
                    self.gl_major >= 3 and hasattr(gl, "glGenVertexArrays")
                )

                if self.has_vao_support:
                    # Create and bind VAO
                    self.vao = gl.glGenVertexArrays(1)
                    gl.glBindVertexArray(self.vao)
                else:
                    self.vao = None  # We will use legacy state‐based attribute setup
                
                # Create and bind VBO
                self.vbo = gl.glGenBuffers(1)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
                
                # Create and bind EBO
                self.ebo = gl.glGenBuffers(1)
                gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
                gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
                
                # Define vertex attributes (query locations from shader if available)
                stride = 5 * 4  # bytes per vertex (5 floats)
                self.stride = stride

                # Try to query attribute locations from compiled shader (if available)
                if self.shader_program:
                    pos_loc = gl.glGetAttribLocation(self.shader_program, "position")
                    if pos_loc == -1:
                        pos_loc = 0  # Fallback
                    tex_loc = gl.glGetAttribLocation(self.shader_program, "texCoord")
                    if tex_loc == -1:
                        tex_loc = 1  # Fallback
                    self.pos_loc = pos_loc
                    self.tex_loc = tex_loc
                else:
                    pos_loc = 0
                    tex_loc = 1
                    self.pos_loc = pos_loc
                    self.tex_loc = tex_loc

                gl.glEnableVertexAttribArray(pos_loc)
                gl.glVertexAttribPointer(pos_loc, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)

                gl.glEnableVertexAttribArray(tex_loc)
                gl.glVertexAttribPointer(tex_loc, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(3 * 4))
                
                # Unbind VAO (if used)
                if self.has_vao_support and self.vao is not None:
                    gl.glBindVertexArray(0)
                
                # Set up OpenGL state
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                
                # Create a simple texture for spectrum data (will be updated each frame)
                self.spectrum_texture = gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.spectrum_texture)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
                
                # Initial texture data (gradient) – store as 8-bit to stay within
                # OpenGL 2.1 limits (float textures are not guaranteed).
                width, height = 64, 1
                gradient = np.zeros((height, width, 3), dtype=np.uint8)
                for i in range(width):
                    t = i / (width - 1)
                    r = int(t * 255)
                    g = int((1.0 - t) * 255)
                    b = 128  # constant 0.5 * 255
                    gradient[0, i] = (r, g, b)

                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D,
                    0,
                    gl.GL_RGB,
                    width,
                    height,
                    0,
                    gl.GL_RGB,
                    gl.GL_UNSIGNED_BYTE,
                    gradient
                )
                
                # If we have a shader program, try setting up the identity matrix uniforms
                if self.shader_program:
                    try:
                        # Create identity matrices for transform
                        identity = np.identity(4, dtype=np.float32)
                        self.shader_manager.use_program(self.active_program_name)
                        
                        # Set matrix uniforms (using identity matrices)
                        self.shader_manager.set_uniform_matrix4fv('uModelMatrix', identity)
                        self.shader_manager.set_uniform_matrix4fv('uViewMatrix', identity)
                        self.shader_manager.set_uniform_matrix4fv('uProjectionMatrix', identity)
                        
                        # Unbind shader
                        gl.glUseProgram(0)
                    except Exception as e:
                        logger.warning(f"Could not set matrix uniforms: {e}")
                
                self.has_advanced_rendering = True
                logger.info("Advanced rendering initialized successfully")
            except Exception as e:
                logger.error(f"Error setting up advanced rendering: {e}")
                self.has_advanced_rendering = False
            
        except Exception as e:
            logger.error(f"Error initializing OpenGL: {e}")
            self.has_advanced_rendering = False
            self.shader_error_message = f"OpenGL initialization error: {e}"
            
        # Always initialize some basic rendering properties
        self.use_fallback = not (self.shader_program is not None and self.has_advanced_rendering)
        if self.use_fallback:
            logger.info("Using fallback rendering mode")
    
    def paintGL(self):
        """Render the scene."""
        # Update FPS counter
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Keep a sliding window of the last 60 frame times
        self.frame_times.append(dt)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        
        # Calculate average FPS
        if self.frame_times:
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
        
        try:
            # Clear the screen
            gl.glClearColor(0.1, 0.1, 0.2, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            
            # Check if we should use advanced or fallback rendering
            if not self.use_fallback and self.shader_manager and self.shader_program:
                try:
                    # Use shader program for advanced rendering
                    self.shader_manager.use_program(self.active_program_name)
                    
                    # Set uniforms
                    current_time = time.time() - self.start_time + self.time_offset
                    # Provide both camel and plain 'time' uniforms for compatibility
                    self.shader_manager.set_uniform_1f('uTime', current_time)
                    self.shader_manager.set_uniform_1f('time', current_time)
                    
                    # Set matrix uniforms (using identity matrices)
                    try:
                        identity = np.identity(4, dtype=np.float32)
                        self.shader_manager.set_uniform_matrix4fv('uModelMatrix', identity)
                        self.shader_manager.set_uniform_matrix4fv('uViewMatrix', identity)
                        self.shader_manager.set_uniform_matrix4fv('uProjectionMatrix', identity)
                    except Exception:
                        # If setting matrix uniforms fails, don't worry - they might not be used
                        pass
                    
                    # Set audio feature uniforms if available
                    if self.features:
                        self.shader_manager.set_uniform_1f('uOnset', self.features.onset)
                        self.shader_manager.set_uniform_1f('uEnergy', self.features.energy)
                        self.shader_manager.set_uniform_1f('uBeatPhase', self.features.beat_phase)
                        
                        # Check for silence
                        self._check_silence(self.features.energy)
                        
                        # Pass silence status to shader
                        self.shader_manager.set_uniform_1f('uIsSilent', 1.0 if self.is_silent else 0.0)
                        if self.is_silent:
                            # Add pulsing silence indicator
                            silence_duration = time.time() - self.silence_start_time
                            self.shader_manager.set_uniform_1f('uSilencePulse', 
                                                              0.5 + 0.5 * math.sin(silence_duration * 1.5))
                        
                        # Optional spectral features (if available)
                        if self.features.centroid is not None:
                            self.shader_manager.set_uniform_1f('uCentroid', self.features.centroid)
                        
                        # Bind spectrum texture
                        gl.glActiveTexture(gl.GL_TEXTURE0)
                        gl.glBindTexture(gl.GL_TEXTURE_2D, self.spectrum_texture)
                        location = gl.glGetUniformLocation(self.shader_program, "uSpectrumTexture")
                        gl.glUniform1i(location, 0)
                    
                    # Draw quad (VAO if available, otherwise bind buffers manually)
                    if self.has_vao_support and self.vao is not None:
                        gl.glBindVertexArray(self.vao)
                        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
                        gl.glBindVertexArray(0)
                    else:
                        # Bind buffers manually for legacy path
                        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
                        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
                        gl.glEnableVertexAttribArray(self.pos_loc)
                        gl.glVertexAttribPointer(self.pos_loc, 3, gl.GL_FLOAT, gl.GL_FALSE, self.stride, None)
                        gl.glEnableVertexAttribArray(self.tex_loc)
                        gl.glVertexAttribPointer(self.tex_loc, 2, gl.GL_FLOAT, gl.GL_FALSE, self.stride, gl.ctypes.c_void_p(3 * 4))
                        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
                        gl.glDisableVertexAttribArray(self.pos_loc)
                        gl.glDisableVertexAttribArray(self.tex_loc)
                    
                    # Unbind shader program
                    gl.glUseProgram(0)
                except Exception as e:
                    logger.error(f"Error rendering with shader: {e}")
                    self._draw_fallback()
            else:
                # Use fallback rendering
                self._draw_fallback()
                
            # Draw status overlay
            self.drawStatusOverlay()
            
            # Draw silence indicator if needed
            if self.is_silent:
                self._draw_silence_indicator()
            
        except Exception as e:
            logger.error(f"Error rendering scene: {e}")
    
    def _draw_fallback(self):
        """Draw a simple fallback visualization when shaders fail."""
        try:
            # Use fixed-function pipeline
            gl.glUseProgram(0)
            
            # Calculate a time-based color
            current_time = time.time() - self.start_time + self.time_offset
            
            # Base color
            r, g, b = 0.3, 0.4, 0.6
            
            # Add time-based pulsing
            pulse = 0.2 * (math.sin(current_time * 2.0) * 0.5 + 0.5)
            
            # Add audio reactivity if available
            if self.features:
                # Use audio energy to modulate color
                energy = min(1.0, self.features.energy * 2.0)
                onset = min(1.0, self.features.onset * 2.0)
                
                r = 0.3 + energy * 0.5 + onset * 0.2
                g = 0.4 + pulse * 0.3
                b = 0.6 + energy * 0.4
            else:
                # Time-based color if no audio
                r = 0.3 + pulse * 0.2
                g = 0.4 + pulse * 0.3
                b = 0.6 + pulse * 0.2
                
            # Draw a rectangle that pulses with the beat
            try:
                # Use legacy OpenGL immediate mode
                gl.glColor4f(r, g, b, 1.0)
                
                # Draw a quad
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex2f(-0.8, -0.8)
                gl.glVertex2f(0.8, -0.8)
                gl.glVertex2f(0.8, 0.8)
                gl.glVertex2f(-0.8, 0.8)
                gl.glEnd()
                
                # Draw a pulsing inner quad if we have features
                if self.features and self.features.energy > 0.01:
                    # Highlight color
                    gl.glColor4f(1.0, 0.8, 0.3, 0.7)
                    
                    # Size based on energy
                    size = 0.3 + self.features.energy * 0.3
                    
                    # Draw inner quad
                    gl.glBegin(gl.GL_QUADS)
                    gl.glVertex2f(-size, -size)
                    gl.glVertex2f(size, -size)
                    gl.glVertex2f(size, size)
                    gl.glVertex2f(-size, size)
                    gl.glEnd()
            except Exception as e:
                logger.error(f"Error in legacy OpenGL rendering: {e}")
                
                # Even more basic fallback - just clear with a color
                gl.glClearColor(0.2, 0.3, 0.4, 1.0)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                
        except Exception as e:
            logger.error(f"Error in fallback rendering: {e}")
    
    def drawStatusOverlay(self):
        """Draw status information overlay."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        # Set font
        font = QFont("Helvetica", 10)
        painter.setFont(font)
        
        # Get widget size
        width = self.width()
        height = self.height()
        
        # Draw device name
        if self.show_status:
            painter.setPen(QColor(255, 255, 255, 180))
            device_text = f"Listening on: {self.device_name}"
            if self.use_fallback:
                if self.shader_error_message:
                    # Show a warning about shader compilation failure
                    device_text += " (Shader error - using fallback rendering)"
                else:
                    device_text += " (Basic rendering mode)"
            painter.drawText(10, height - 10, device_text)
        
        # Draw silence indicator if needed
        if self.is_silent and self.silence_counter > 20:
            self._draw_silence_indicator()
        
        # Draw debug overlay if enabled
        if self.show_debug_overlay:
            self._draw_debug_overlay(painter, width, height)
        
        painter.end()
    
    def resizeGL(self, width, height):
        """Handle widget resize events."""
        gl.glViewport(0, 0, width, height)
    
    def update_features(self, features):
        """
        Update audio features for visualization.
        
        Args:
            features: AudioFeatures object with current audio features
        """
        self.features = features
        self.update()  # Request a redraw
    
    def set_device_name(self, name):
        """Set the current audio device name for display."""
        self.device_name = name
    
    def toggle_status_overlay(self):
        """Toggle the status overlay visibility."""
        self.show_status = not self.show_status
        self.update()
    
    def reset_time(self):
        """Reset the animation time."""
        self.time_offset = 0
        self.start_time = time.time()
    
    def cleanup(self):
        """Clean up OpenGL resources."""
        if self.shader_manager:
            self.shader_manager.cleanup()
        
        # Delete buffers
        if self.vbo:
            gl.glDeleteBuffers(1, [self.vbo])
        if self.ebo:
            gl.glDeleteBuffers(1, [self.ebo])
        if self.vao:
            gl.glDeleteVertexArrays(1, [self.vao])
        if hasattr(self, 'spectrum_texture'):
            gl.glDeleteTextures(1, [self.spectrum_texture])
    
    def toggle_debug_overlay(self):
        """Toggle the performance metrics debug overlay."""
        self.show_debug_overlay = not self.show_debug_overlay
        self.update()
    
    def _check_silence(self, energy):
        """Check if audio is silent."""
        if energy < self.silence_threshold:
            self.silence_counter += 1
            if self.silence_counter > 60 and not self.is_silent:  # 1 second at 60 FPS
                self.is_silent = True
                self.silence_start_time = time.time()
        else:
            self.silence_counter = 0
            self.is_silent = False
    
    def _draw_silence_indicator(self):
        """Draw an indicator when no audio is detected."""
        try:
            # Use QPainter for text and indicator
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Calculate pulse effect
            silence_duration = time.time() - self.silence_start_time
            pulse = 0.5 + 0.5 * math.sin(silence_duration * 1.5)
            alpha = int(120 + 135 * pulse)
            
            # Create semi-transparent red color
            color = QColor(255, 60, 60, alpha)
            
            # Draw text in center
            font = QFont("Helvetica", 12, QFont.Bold)
            painter.setFont(font)
            painter.setPen(color)
            
            # Draw centered text
            rect = self.rect()
            painter.drawText(rect, Qt.AlignCenter, "No audio detected")
            
            # Small microphone icon or symbol below text
            icon_rect = QRect(rect.center().x() - 15, rect.center().y() + 30, 30, 30)
            painter.drawText(icon_rect, Qt.AlignCenter, "🔇")
            
            painter.end()
        except Exception as e:
            logger.error(f"Error drawing silence indicator: {e}")
    
    def _show_shader_error(self, errors):
        """
        Display shader compilation errors to the user.
        
        Args:
            errors: List of error messages
        """
        if not errors:
            return
        
        # Create a concise but informative error message
        if len(errors) == 1:
            message = errors[0]
        else:
            message = f"Found {len(errors)} shader compilation errors:\n\n" + "\n".join(errors[:3])
            if len(errors) > 3:
                message += f"\n\n...and {len(errors) - 3} more errors. See log for details."
        
        # Add information about fallback rendering
        message += "\n\nThe application will continue with simplified rendering."
        
        # Show error message dialog
        try:
            QMessageBox.warning(
                self, 
                "Shader Compilation Error",
                message,
                QMessageBox.Ok
            )
        except Exception as e:
            logger.error(f"Failed to show shader error dialog: {e}")
    
    def _draw_debug_overlay(self, painter, width, height):
        """
        Draw performance metrics and debug information.
        
        Args:
            painter: QPainter instance
            width: Widget width
            height: Widget height
        """
        # Set up debug info rectangle
        rect = QRect(10, 10, 300, 200)
        painter.fillRect(rect, QColor(0, 0, 0, 150))
        painter.setPen(QColor(255, 255, 255, 220))
        
        # Show FPS
        painter.drawText(rect.adjusted(10, 10, 0, 0), Qt.AlignLeft, f"FPS: {self.fps:.1f}")
        
        # Show rendering mode
        mode_text = "Rendering: Advanced (Shader)" if not self.use_fallback else "Rendering: Basic (Fallback)"
        painter.drawText(rect.adjusted(10, 30, 0, 0), Qt.AlignLeft, mode_text)
        
        # Show OpenGL info
        try:
            vendor = gl.glGetString(gl.GL_VENDOR).decode('utf-8')
            renderer = gl.glGetString(gl.GL_RENDERER).decode('utf-8')
            version = gl.glGetString(gl.GL_VERSION).decode('utf-8')
            painter.drawText(rect.adjusted(10, 50, 0, 0), Qt.AlignLeft, f"OpenGL: {version}")
            painter.drawText(rect.adjusted(10, 70, 0, 0), Qt.AlignLeft, f"GPU: {vendor} {renderer}")
        except Exception as e:
            painter.drawText(rect.adjusted(10, 50, 0, 0), Qt.AlignLeft, f"OpenGL info error: {e}")
        
        # Show CPU/RAM usage
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            painter.drawText(rect.adjusted(10, 90, 0, 0), Qt.AlignLeft, f"CPU: {cpu_percent:.1f}%")
            painter.drawText(rect.adjusted(10, 110, 0, 0), Qt.AlignLeft, f"RAM: {memory_percent:.1f}%")
        except Exception as e:
            painter.drawText(rect.adjusted(10, 90, 0, 0), Qt.AlignLeft, f"System stats error: {e}")
        
        # Show feature extraction info if available
        if self.features:
            # Show energy
            painter.drawText(rect.adjusted(10, 130, 0, 0), Qt.AlignLeft, f"Energy: {self.features.energy:.3f}")
            
            # Show onset
            painter.drawText(rect.adjusted(10, 150, 0, 0), Qt.AlignLeft, f"Onset: {self.features.onset:.3f}")
            
            # Show beat phase
            painter.drawText(rect.adjusted(10, 170, 0, 0), Qt.AlignLeft, f"Beat: {self.features.beat_phase:.3f}")
        
        # Show shader error if any
        if self.shader_error_message:
            # Truncate message to fit
            short_msg = self.shader_error_message[:50] + "..." if len(self.shader_error_message) > 50 else self.shader_error_message
            painter.setPen(QColor(255, 100, 100, 220))
            painter.drawText(rect.adjusted(10, 190, 0, 0), Qt.AlignLeft, f"Shader Error: {short_msg}") 