#!/usr/bin/env python3
"""
Simple test script to verify PyQt5 and OpenGL are working properly.
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import QOpenGLWidget
import OpenGL.GL as gl

class OpenGLTestWidget(QOpenGLWidget):
    """Basic OpenGL Widget for testing."""
    
    def initializeGL(self):
        """Initialize OpenGL."""
        print("OpenGL Initialization:")
        print(f"  Vendor: {gl.glGetString(gl.GL_VENDOR).decode()}")
        print(f"  Renderer: {gl.glGetString(gl.GL_RENDERER).decode()}")
        print(f"  Version: {gl.glGetString(gl.GL_VERSION).decode()}")
        print(f"  Shading Language: {gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION).decode()}")
        
        # Set clear color to blue
        gl.glClearColor(0.0, 0.0, 0.8, 1.0)
    
    def paintGL(self):
        """Render the scene."""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        # Draw a simple triangle
        gl.glBegin(gl.GL_TRIANGLES)
        gl.glColor3f(1.0, 0.0, 0.0)  # Red
        gl.glVertex2f(0.0, 0.5)
        gl.glColor3f(0.0, 1.0, 0.0)  # Green
        gl.glVertex2f(-0.5, -0.5)
        gl.glColor3f(0.0, 0.0, 1.0)  # Blue
        gl.glVertex2f(0.5, -0.5)
        gl.glEnd()
    
    def resizeGL(self, width, height):
        """Handle window resize events."""
        gl.glViewport(0, 0, width, height)

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("OpenGL Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Create OpenGL widget
        self.gl_widget = OpenGLTestWidget(self)
        self.setCentralWidget(self.gl_widget)

if __name__ == "__main__":
    # Set OpenGL format (optional)
    gl_format = QSurfaceFormat()
    gl_format.setVersion(2, 1)  # Use OpenGL 2.1
    gl_format.setProfile(QSurfaceFormat.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(gl_format)
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    print("PyQt5 and OpenGL test application started.")
    print("You should see a color gradient triangle on a blue background.")
    print("Close the window to exit.")
    
    # Start event loop
    sys.exit(app.exec_()) 