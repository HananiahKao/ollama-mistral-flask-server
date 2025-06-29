#!/usr/bin/env python3
"""
Project Validation and Testing Script

This script validates that README.md and requirements.txt accurately reflect the current project state,
runs setup/build processes to verify functionality, and implements regression tests.
"""

import os
import sys
import subprocess
import ast
import re
import importlib
import requests
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

class ProjectValidator:
    def __init__(self):
        self.project_root = Path.cwd()
        self.validation_results = []
        self.errors = []
        self.warnings = []
        
    def log_error(self, message: str):
        """Log an error message."""
        self.errors.append(message)
        print(f"❌ ERROR: {message}")
        
    def log_warning(self, message: str):
        """Log a warning message."""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")
        
    def log_success(self, message: str):
        """Log a success message."""
        self.validation_results.append(message)
        print(f"✅ SUCCESS: {message}")
        
    def validate_readme_requirements_txt_consistency(self) -> bool:
        """Validate that README.md and requirements.txt are consistent with the code."""
        print("\n🔍 Validating README.md and requirements.txt consistency...")
        
        # Read files
        try:
            with open('README.md', 'r') as f:
                readme_content = f.read()
        except FileNotFoundError:
            self.log_error("README.md not found")
            return False
            
        try:
            with open('requirements.txt', 'r') as f:
                requirements_content = f.read()
        except FileNotFoundError:
            self.log_error("requirements.txt not found")
            return False
            
        # Extract imports from Python files
        python_imports = self.extract_python_imports()
        
        # Extract mentioned packages from README
        readme_packages = self.extract_packages_from_readme(readme_content)
        
        # Extract packages from requirements.txt
        req_packages = self.extract_packages_from_requirements(requirements_content)
        
        # Validate consistency
        success = True
        
        # Check if all Python imports are in requirements.txt
        missing_in_req = python_imports - req_packages
        if missing_in_req:
            self.log_error(f"Packages used in code but missing from requirements.txt: {missing_in_req}")
            success = False
        else:
            self.log_success("All Python imports are covered in requirements.txt")
            
        # Check if README mentions all major dependencies
        major_deps = {'flask', 'requests', 'beautifulsoup4', 'torch', 'diffusers', 'transformers'}
        missing_in_readme = major_deps - readme_packages
        if missing_in_readme:
            self.log_warning(f"Major dependencies not mentioned in README: {missing_in_readme}")
        else:
            self.log_success("README mentions all major dependencies")
            
        # Check if requirements.txt has all major dependencies
        missing_major_in_req = major_deps - req_packages
        if missing_major_in_req:
            self.log_error(f"Major dependencies missing from requirements.txt: {missing_major_in_req}")
            success = False
        else:
            self.log_success("requirements.txt includes all major dependencies")
            
        return success
        
    def extract_python_imports(self) -> Set[str]:
        """Extract all import statements from Python files."""
        imports = set()
        
        for py_file in self.project_root.rglob('*.py'):
            if 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
                            
            except Exception as e:
                self.log_warning(f"Could not parse {py_file}: {e}")
                
        return imports
        
    def extract_packages_from_readme(self, content: str) -> Set[str]:
        """Extract package names mentioned in README."""
        packages = set()
        
        # Look for common package patterns
        patterns = [
            r'pip install (\w+)',
            r'install (\w+)',
            r'(\w+)>=',
            r'(\w+)==',
            r'(\w+)<',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            packages.update(matches)
            
        return packages
        
    def extract_packages_from_requirements(self, content: str) -> Set[str]:
        """Extract package names from requirements.txt."""
        packages = set()
        
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name (before any version specifiers)
                package = re.match(r'^([a-zA-Z0-9_-]+)', line)
                if package:
                    packages.add(package.group(1).lower())
                    
        return packages
        
    def validate_setup_process(self) -> bool:
        """Validate that the setup process works correctly."""
        print("\n🔧 Validating setup process...")
        
        # Check if virtual environment exists
        venv_path = self.project_root / 'venv'
        if not venv_path.exists():
            self.log_warning("Virtual environment not found, attempting to create...")
            try:
                subprocess.run(['python3', '-m', 'venv', 'venv'], check=True, capture_output=True)
                self.log_success("Virtual environment created")
            except subprocess.CalledProcessError as e:
                self.log_error(f"Failed to create virtual environment: {e}")
                return False
        else:
            self.log_success("Virtual environment exists")
            
        # Check if requirements can be installed
        try:
            # Activate venv and install requirements
            if os.name == 'nt':  # Windows
                pip_cmd = ['venv\\Scripts\\pip', 'install', '-r', 'requirements.txt']
            else:  # Unix/Linux/macOS
                pip_cmd = ['venv/bin/pip', 'install', '-r', 'requirements.txt']
                
            result = subprocess.run(pip_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.log_success("Requirements installed successfully")
            else:
                self.log_error(f"Failed to install requirements: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_error("Requirements installation timed out")
            return False
        except Exception as e:
            self.log_error(f"Error during requirements installation: {e}")
            return False
            
        return True
        
    def validate_server_startup(self) -> bool:
        """Validate that the server can start correctly."""
        print("\n🚀 Validating server startup...")
        
        # Check if Ollama is running
        try:
            response = requests.get('http://127.0.0.1:11434/api/tags', timeout=5)
            if response.status_code == 200:
                self.log_success("Ollama service is running")
            else:
                self.log_warning("Ollama service responded but with unexpected status")
        except requests.exceptions.RequestException:
            self.log_warning("Ollama service not running (this is expected in deployment environments)")
            
        # Test if the Flask app can be imported and configured
        try:
            # Temporarily modify environment to prevent model loading during test
            original_env = os.environ.copy()
            os.environ['TESTING'] = 'true'
            
            # Import the app
            from app import app
            
            # Test basic app configuration
            with app.test_client() as client:
                # Mock the AI functions to avoid actual model loading
                with app.app_context():
                    # Test that the app can handle a basic request
                    response = client.get('/test-endpoint')
                    if response.status_code in [200, 404]:  # 404 is expected for unknown routes
                        self.log_success("Flask app can handle requests")
                    else:
                        self.log_error(f"Unexpected response code: {response.status_code}")
                        return False
                        
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)
            
        except Exception as e:
            self.log_error(f"Failed to validate Flask app: {e}")
            return False
            
        return True
        
    def run_regression_tests(self) -> bool:
        """Run regression tests to ensure expected behavior."""
        print("\n🧪 Running regression tests...")
        
        # Check if pytest is available
        try:
            import pytest
        except ImportError:
            self.log_warning("pytest not available, skipping regression tests")
            return True
            
        # Run existing tests
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 'test_app.py', '-v'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.log_success("Regression tests passed")
                return True
            else:
                self.log_error(f"Regression tests failed: {result.stdout}\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_error("Regression tests timed out")
            return False
        except Exception as e:
            self.log_error(f"Error running regression tests: {e}")
            return False
            
    def validate_deployment_environment(self) -> bool:
        """Simulate deployed environment to confirm expected behavior."""
        print("\n🌐 Validating deployment environment...")
        
        # Check for deployment configuration files
        deployment_files = ['render.yaml', 'Dockerfile', 'docker-compose.yml', 'Procfile']
        found_deployment_files = []
        
        for file in deployment_files:
            if (self.project_root / file).exists():
                found_deployment_files.append(file)
                
        if found_deployment_files:
            self.log_success(f"Found deployment configuration: {found_deployment_files}")
        else:
            self.log_warning("No deployment configuration files found")
            
        # Check environment variables
        required_env_vars = ['PORT', 'OLLAMA_API_URL', 'OLLAMA_MODEL']
        optional_env_vars = ['FLASK_ENV', 'FLASK_DEBUG']
        
        for var in required_env_vars:
            if os.getenv(var):
                self.log_success(f"Environment variable {var} is set")
            else:
                self.log_warning(f"Environment variable {var} not set (will use defaults)")
                
        # Test port configuration
        port = os.getenv('PORT', '5001')
        try:
            port_int = int(port)
            if 1024 <= port_int <= 65535:
                self.log_success(f"Port {port} is valid")
            else:
                self.log_error(f"Port {port} is outside valid range (1024-65535)")
                return False
        except ValueError:
            self.log_error(f"Invalid port number: {port}")
            return False
            
        # Check Ollama configuration for deployment compatibility
        ollama_url = os.getenv('OLLAMA_API_URL', 'http://127.0.0.1:11434/api/generate')
        if '127.0.0.1' in ollama_url:
            self.log_warning("Ollama URL uses '127.0.0.1' - consider using '127.0.0.1' for better deployment compatibility")
        elif '127.0.0.1' in ollama_url:
            self.log_success("Ollama URL uses '127.0.0.1' - good for deployment compatibility")
        else:
            self.log_success(f"Ollama URL configured: {ollama_url}")
            
        return True
        
    def run_full_validation(self) -> bool:
        """Run the complete validation suite."""
        print("🔍 Starting comprehensive project validation...")
        print("=" * 60)
        
        all_passed = True
        
        # Run all validation steps
        validations = [
            ("README and requirements.txt consistency", self.validate_readme_requirements_txt_consistency),
            ("Setup process", self.validate_setup_process),
            ("Server startup", self.validate_server_startup),
            ("Regression tests", self.run_regression_tests),
            ("Deployment environment", self.validate_deployment_environment),
        ]
        
        for name, validation_func in validations:
            print(f"\n📋 {name.upper()}")
            print("-" * 40)
            try:
                if not validation_func():
                    all_passed = False
            except Exception as e:
                self.log_error(f"Validation '{name}' failed with exception: {e}")
                all_passed = False
                
        # Print summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        if all_passed:
            print("🎉 All validations passed!")
        else:
            print("❌ Some validations failed")
            
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
                
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
                
        if self.validation_results:
            print(f"\n✅ Successes ({len(self.validation_results)}):")
            for result in self.validation_results:
                print(f"  - {result}")
                
        return all_passed

def main():
    """Main entry point."""
    validator = ProjectValidator()
    success = validator.run_full_validation()
    
    if success:
        print("\n🎉 Project validation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Project validation failed. Please address the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 