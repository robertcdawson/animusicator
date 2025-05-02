#!/usr/bin/env python
# shaders.py: Module for loading and compiling GLSL shaders

import os
import logging
from typing import Dict, Optional, Tuple, List

import OpenGL.GL as gl
from OpenGL.GL import shaders

# Set up logging
logger = logging.getLogger(__name__)

class ShaderCompilationError(Exception):
    """Exception raised when shader compilation fails."""
    
    def __init__(self, shader_type: str, filename: str, log: str):
        self.shader_type = shader_type
        self.filename = filename
        self.log = log
        message = f"Failed to compile {shader_type} shader '{filename}': {log}"
        super().__init__(message)


class ShaderLinkingError(Exception):
    """Exception raised when shader program linking fails."""
    
    def __init__(self, program_name: str, log: str):
        self.program_name = program_name
        self.log = log
        message = f"Failed to link shader program '{program_name}': {log}"
        super().__init__(message)


class ShaderManager:
    """
    Manages shader programs, including loading, compiling, and binding.
    Handles fallbacks for different OpenGL versions.
    """
    
    def __init__(self, shader_dir: str = "assets/shaders"):
        """
        Initialize shader manager.
        
        Args:
            shader_dir: Directory containing shader files
        """
        self.shader_dir = shader_dir
        self.current_program = None
        self.shader_programs: Dict[str, int] = {}
        self.gl_version = None
        self.compilation_errors: List[str] = []
        
    def _detect_gl_version(self) -> Tuple[int, int]:
        """
        Detect OpenGL version.
        
        Returns:
            tuple: (major_version, minor_version)
        """
        version_string = gl.glGetString(gl.GL_VERSION).decode('utf-8')
        logger.info(f"OpenGL Version: {version_string}")
        
        # Parse major and minor version
        try:
            version_parts = version_string.split(' ')[0].split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1])
            return (major, minor)
        except (IndexError, ValueError):
            logger.warning("Could not parse OpenGL version, assuming 2.1")
            return (2, 1)
    
    def get_gl_version(self) -> Tuple[int, int]:
        """
        Get the OpenGL version.
        
        Returns:
            tuple: (major_version, minor_version)
        """
        if self.gl_version is None:
            self.gl_version = self._detect_gl_version()
        return self.gl_version
    
    def _read_shader_source(self, shader_file: str) -> str:
        """
        Read shader source from file.
        
        Args:
            shader_file: Path to shader file
            
        Returns:
            str: Shader source code
            
        Raises:
            FileNotFoundError: If shader file is not found
        """
        full_path = os.path.join(self.shader_dir, shader_file)
        try:
            with open(full_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            error_msg = f"Shader file not found: {full_path}"
            logger.error(error_msg)
            self.compilation_errors.append(error_msg)
            raise
    
    def _compile_shader(self, source: str, shader_type: int, filename: str) -> int:
        """
        Compile a shader and handle errors.
        
        Args:
            source: Shader source code
            shader_type: GL_VERTEX_SHADER or GL_FRAGMENT_SHADER
            filename: Shader filename (for error reporting)
            
        Returns:
            int: Compiled shader object
            
        Raises:
            ShaderCompilationError: If compilation fails
        """
        shader_type_str = "vertex" if shader_type == gl.GL_VERTEX_SHADER else "fragment"
        
        try:
            shader = shaders.compileShader(source, shader_type)
            
            # Even if compilation succeeds, check the info log for warnings
            log = gl.glGetShaderInfoLog(shader)
            if log and log.strip():
                log_str = log.decode('utf-8') if isinstance(log, bytes) else str(log)
                logger.warning(f"Shader '{filename}' compiled with warnings: {log_str}")
            
            return shader
            
        except gl.GLError as e:
            # Get compilation error log
            log = gl.glGetShaderInfoLog(e.failed_object)
            log_str = log.decode('utf-8') if isinstance(log, bytes) else str(log)
            
            error_msg = f"Failed to compile {shader_type_str} shader '{filename}': {log_str}"
            logger.error(error_msg)
            self.compilation_errors.append(error_msg)
            
            raise ShaderCompilationError(shader_type_str, filename, log_str)
    
    def _link_program(self, program: int, name: str) -> None:
        """
        Link shader program and handle errors.
        
        Args:
            program: Shader program object
            name: Program name (for error reporting)
            
        Raises:
            ShaderLinkingError: If linking fails
        """
        try:
            # Check link status
            link_status = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
            if link_status == gl.GL_FALSE:
                log = gl.glGetProgramInfoLog(program)
                log_str = log.decode('utf-8') if isinstance(log, bytes) else str(log)
                
                error_msg = f"Failed to link shader program '{name}': {log_str}"
                logger.error(error_msg)
                self.compilation_errors.append(error_msg)
                
                raise ShaderLinkingError(name, log_str)
            
            # Check for warnings even if linking succeeded
            log = gl.glGetProgramInfoLog(program)
            if log and log.strip():
                log_str = log.decode('utf-8') if isinstance(log, bytes) else str(log)
                logger.warning(f"Shader program '{name}' linked with warnings: {log_str}")
                
        except gl.GLError as e:
            error_msg = f"OpenGL error while linking shader program '{name}': {e}"
            logger.error(error_msg)
            self.compilation_errors.append(error_msg)
            raise ShaderLinkingError(name, str(e))
    
    def load_program(self, name: str, vert_file: str, frag_file: str) -> int:
        """
        Load and compile a shader program.
        
        Args:
            name: Name for the shader program
            vert_file: Vertex shader filename
            frag_file: Fragment shader filename
            
        Returns:
            int: Shader program ID
            
        Raises:
            ShaderCompilationError: If shader compilation fails
            ShaderLinkingError: If program linking fails
            FileNotFoundError: If shader file is not found
        """
        try:
            # Clear previous errors for this program
            self.compilation_errors = [err for err in self.compilation_errors 
                                      if name not in err]
            
            # Read shader sources
            vert_source = self._read_shader_source(vert_file)
            frag_source = self._read_shader_source(frag_file)
            
            # Compile shaders with error handling
            vert_shader = self._compile_shader(vert_source, gl.GL_VERTEX_SHADER, vert_file)
            frag_shader = self._compile_shader(frag_source, gl.GL_FRAGMENT_SHADER, frag_file)
            
            # Link program manually to avoid implicit glValidateProgram call inside
            # PyOpenGL's `compileProgram`, which can fail on some macOS drivers when
            # a default framebuffer is not yet available (e.g. during widget
            # initialization).  We replicate the essential steps: create program,
            # attach shaders, link, then detach.  Validation is performed only on
            # newer GL versions later on.

            program = gl.glCreateProgram()
            gl.glAttachShader(program, vert_shader)
            gl.glAttachShader(program, frag_shader)
            gl.glLinkProgram(program)

            # Detach shaders (optional but keeps things tidy)
            gl.glDetachShader(program, vert_shader)
            gl.glDetachShader(program, frag_shader)
            
            # Check linking status
            self._link_program(program, name)
            
            # Store program
            self.shader_programs[name] = program
            logger.info(f"Successfully loaded shader program: {name}")
            
            # Validate program – on some older drivers (e.g. macOS OpenGL 2.1) this call
            # can raise an INVALID_OPERATION error even when the program is otherwise
            # usable.  We therefore run validation only on newer GL versions and treat
            # errors as warnings instead of aborting the whole load.
            try:
                major, minor = self.get_gl_version()
                if major >= 3:
                    gl.glValidateProgram(program)
                    validate_status = gl.glGetProgramiv(program, gl.GL_VALIDATE_STATUS)
                    if validate_status == gl.GL_FALSE:
                        log = gl.glGetProgramInfoLog(program)
                        log_str = log.decode('utf-8') if isinstance(log, bytes) else str(log)
                        logger.warning(f"Shader program '{name}' validation failed: {log_str}")
            except gl.GLError as e:
                # Validation is not critical for running – just log and continue.
                logger.warning(f"Shader program '{name}' validation raised GL error but will be ignored: {e}")
            except Exception as e:
                # Any other unexpected issue – log and keep going.
                logger.warning(f"Shader program '{name}' validation unexpected issue: {e}")
            
            return program
            
        except (ShaderCompilationError, ShaderLinkingError) as e:
            logger.error(f"Failed to load shader program '{name}': {e}")
            raise
        except gl.GLError as e:
            error_msg = f"OpenGL error when loading shader '{name}': {e}"
            logger.error(error_msg)
            self.compilation_errors.append(error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error loading shader '{name}': {e}"
            logger.error(error_msg)
            self.compilation_errors.append(error_msg)
            raise
    
    def load_with_fallback(self, name: str, 
                          vert_file: str, frag_file: str,
                          fallback_vert: Optional[str] = None, 
                          fallback_frag: Optional[str] = None) -> int:
        """
        Load shader with fallback for older OpenGL versions.
        
        Args:
            name: Name for the shader program
            vert_file: Main vertex shader filename
            frag_file: Main fragment shader filename
            fallback_vert: Fallback vertex shader filename
            fallback_frag: Fallback fragment shader filename
            
        Returns:
            int: Shader program ID
            
        Raises:
            ValueError: If no compatible shaders are available
        """
        major, minor = self.get_gl_version()
        
        # Try to load main shaders first if OpenGL version is sufficient
        if major >= 3 and minor >= 3:
            try:
                return self.load_program(name, vert_file, frag_file)
            except Exception as e:
                logger.warning(f"Failed to load main shader, trying fallback: {e}")
        else:
            logger.info(f"OpenGL version {major}.{minor} is below 3.3, using fallback shaders")
        
        # Fall back to compatibility shaders
        if fallback_vert and fallback_frag:
            try:
                return self.load_program(f"{name}_fallback", fallback_vert, fallback_frag)
            except Exception as e:
                error_msg = f"Failed to load both main and fallback shaders: {e}"
                logger.error(error_msg)
                self.compilation_errors.append(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = "No fallback shaders provided and main shaders failed to load"
            logger.error(error_msg)
            self.compilation_errors.append(error_msg)
            raise ValueError(error_msg)
    
    def get_last_compilation_errors(self) -> List[str]:
        """
        Get the list of compilation errors.
        
        Returns:
            List[str]: List of compilation error messages
        """
        return self.compilation_errors
    
    def has_compilation_errors(self) -> bool:
        """
        Check if there are any compilation errors.
        
        Returns:
            bool: True if there are compilation errors
        """
        return len(self.compilation_errors) > 0
            
    def use_program(self, name: str) -> None:
        """
        Activate a shader program.
        
        Args:
            name: Name of shader program to use
        """
        if name is None:
            gl.glUseProgram(0)  # Unbind any shader
            self.current_program = None
            return
            
        if name not in self.shader_programs:
            logger.warning(f"Shader program not found: {name}")
            gl.glUseProgram(0)  # Unbind any shader
            self.current_program = None
            return
            
        try:
            program = self.shader_programs[name]
            gl.glUseProgram(program)
            self.current_program = program
        except gl.GLError as e:
            logger.error(f"Error using shader program {name}: {e}")
            gl.glUseProgram(0)  # Unbind any shader
            self.current_program = None
    
    def set_uniform_1f(self, name: str, value: float) -> None:
        """
        Set a float uniform variable.
        
        Args:
            name: Uniform variable name
            value: Float value
        """
        if self.current_program is None:
            logger.warning("No active shader program when setting uniform")
            return
            
        try:
            location = gl.glGetUniformLocation(self.current_program, name)
            if location == -1:
                # Only log at debug level since this might be expected if the uniform isn't used
                logger.debug(f"Uniform '{name}' not found in shader program")
                return
            gl.glUniform1f(location, value)
        except gl.GLError as e:
            logger.error(f"Error setting uniform {name}: {e}")
    
    def set_uniform_3f(self, name: str, x: float, y: float, z: float) -> None:
        """
        Set a vec3 uniform variable.
        
        Args:
            name: Uniform variable name
            x, y, z: Vector components
        """
        if self.current_program is None:
            logger.warning("No active shader program when setting uniform")
            return
            
        try:
            location = gl.glGetUniformLocation(self.current_program, name)
            if location == -1:
                logger.debug(f"Uniform '{name}' not found in shader program")
                return
            gl.glUniform3f(location, x, y, z)
        except gl.GLError as e:
            logger.error(f"Error setting uniform {name}: {e}")
    
    def set_uniform_matrix4fv(self, name: str, matrix) -> None:
        """
        Set a mat4 uniform variable.
        
        Args:
            name: Uniform variable name
            matrix: 4x4 matrix (numpy array or compatible)
        """
        if self.current_program is None:
            logger.warning("No active shader program when setting uniform")
            return
            
        try:
            location = gl.glGetUniformLocation(self.current_program, name)
            if location == -1:
                logger.debug(f"Uniform '{name}' not found in shader program")
                return
            gl.glUniformMatrix4fv(location, 1, gl.GL_FALSE, matrix)
        except gl.GLError as e:
            logger.error(f"Error setting uniform {name}: {e}")
            
    def cleanup(self) -> None:
        """
        Clean up shader resources.
        """
        for program in self.shader_programs.values():
            gl.glDeleteProgram(program)
        self.shader_programs.clear()
        self.current_program = None 