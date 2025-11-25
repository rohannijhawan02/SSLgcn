"""Quick check of installed packages"""

packages = {
    'pandas': 'Data processing',
    'numpy': 'Numerical operations',
    'rdkit': 'Chemistry library (REQUIRED)',
    'torch': 'PyTorch (REQUIRED for graphs)',
    'dgl': 'Deep Graph Library (REQUIRED for graphs)',
    'matplotlib': 'Visualization',
    'seaborn': 'Visualization'
}

print("="*60)
print("Checking Required Packages...")
print("="*60)

missing = []
installed = []

for package, description in packages.items():
    try:
        __import__(package)
        print(f"✓ {package:<15} - {description}")
        installed.append(package)
    except ImportError:
        print(f"✗ {package:<15} - {description} (NOT INSTALLED)")
        missing.append(package)

print("\n" + "="*60)
print(f"Summary: {len(installed)}/{len(packages)} packages installed")
print("="*60)

if missing:
    print("\n⚠️  Missing packages:", ', '.join(missing))
    print("\nTo install missing packages, run:")
    print("  pip install -r requirements.txt")
    print("\nOr install individually:")
    for pkg in missing:
        print(f"  pip install {pkg}")
else:
    print("\n🎉 All packages installed! You're ready to go!")
    print("\nNext step: Run the test suite")
    print("  python test_graph_conversion.py")
