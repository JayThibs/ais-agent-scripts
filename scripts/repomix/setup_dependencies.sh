#!/bin/bash

# Setup script for planwithgemini.py dependencies
# This script installs repomix and provides instructions for gemini CLI

set -e

echo "Setting up dependencies for planwithgemini.py..."
echo

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install Node.js and npm first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Install repomix globally
echo "📦 Installing repomix..."
npm install -g repomix

if command -v repomix &> /dev/null; then
    echo "✅ repomix installed successfully!"
    repomix --version
else
    echo "❌ repomix installation failed. Try running with sudo:"
    echo "   sudo npm install -g repomix"
    exit 1
fi

echo
echo "📦 Checking for Gemini CLI..."

# Check if gemini is installed
if command -v gemini &> /dev/null; then
    echo "✅ Gemini CLI is already installed!"
else
    echo "⚠️  Gemini CLI is not installed."
    echo
    echo "To install Gemini CLI, you need to:"
    echo "1. Visit the Gemini CLI repository"
    echo "2. Follow the installation instructions for your platform"
    echo
    echo "Alternative: You can modify the script to use other LLMs like:"
    echo "- ollama (for local models)"
    echo "- anthropic CLI"
    echo "- openai CLI"
fi

echo
echo "✅ Setup complete!"
echo
echo "You can now use planwithgemini.py:"
echo "  python scripts/repomix/planwithgemini.py --help"
echo
echo "Example usage:"
echo "  python scripts/repomix/planwithgemini.py auto \"implement caching layer\""