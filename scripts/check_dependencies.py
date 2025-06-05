#!/usr/bin/env python3
"""
Script untuk check dependencies dan environment setup
"""

import sys
import os
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(
            f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+"
        )
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✅ {package_name} - installed")
        return True
    except ImportError:
        print(f"❌ {package_name} - not installed")
        return False


def check_basic_dependencies():
    """Check basic dependencies"""
    print("\n📦 Checking basic dependencies...")

    basic_packages = ["fastapi", "uvicorn", "pydantic"]

    all_good = True
    for package in basic_packages:
        if not check_package(package):
            all_good = False

    return all_good


def check_advanced_dependencies():
    """Check advanced dependencies for full RAG system"""
    print("\n🔬 Checking advanced dependencies...")

    advanced_packages = [
        ("langchain", "langchain"),
        ("langchain-openai", "langchain_openai"),
        ("langchain-community", "langchain_community"),
        ("supabase", "supabase"),
        ("openai", "openai"),
        ("python-dotenv", "dotenv"),
    ]

    all_good = True
    for package_name, import_name in advanced_packages:
        if not check_package(package_name, import_name):
            all_good = False

    return all_good


def check_environment_file():
    """Check if environment file exists"""
    print("\n🔧 Checking environment configuration...")

    env_file = Path(".env")
    env_example = Path("env.example")

    if env_file.exists():
        print("✅ .env file found")
        return True
    elif env_example.exists():
        print("⚠️ .env file not found, but env.example exists")
        print("💡 Copy env.example to .env and update with your API keys")
        return False
    else:
        print("❌ No environment configuration found")
        return False


def install_basic_dependencies():
    """Install basic dependencies"""
    print("\n📥 Installing basic dependencies...")

    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "fastapi",
                "uvicorn[standard]",
                "pydantic",
            ]
        )
        print("✅ Basic dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install basic dependencies")
        return False


def main():
    """Main function"""
    print("🔍 Checking Question Answering RAG System Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        print("\n❌ Python version check failed")
        sys.exit(1)

    # Check basic dependencies
    basic_ok = check_basic_dependencies()

    # Check advanced dependencies
    advanced_ok = check_advanced_dependencies()

    # Check environment
    env_ok = check_environment_file()

    print("\n" + "=" * 50)
    print("📋 Summary:")

    if basic_ok:
        print("✅ Basic API functionality available")
        print("   You can run: make run-health")
    else:
        print("❌ Basic dependencies missing")
        response = input("\n🔧 Install basic dependencies? (y/n): ")
        if response.lower() == "y":
            if install_basic_dependencies():
                print("✅ Now you can run: make run-health")

    if advanced_ok and env_ok:
        print("✅ Full RAG system available")
        print("   You can run: make run-api")
    elif advanced_ok and not env_ok:
        print("⚠️ Advanced dependencies OK, but environment setup needed")
        print("   1. Copy env.example to .env")
        print("   2. Update .env with your API keys")
        print("   3. Then run: make run-api")
    else:
        print("❌ Full RAG system not available")
        print("   Install requirements: pip install -r requirements.txt")

    print("\n🚀 Available commands:")
    print("   make run-health     - Simple health check API")
    print("   make run-api        - Full RAG API (needs env setup)")
    print("   make run-multi-api  - Multi-model RAG API (needs env setup)")
    print("   make help           - Show all available commands")


if __name__ == "__main__":
    main()
