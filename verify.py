import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CPU Threads: {torch.get_num_threads()}")
    
    # Your Intel Core Ultra 7 155H has 22 threads (16 cores)
    # Let's optimize for it
    torch.set_num_threads(16)  # Use performance cores
    print(f"   Optimized for Intel Core Ultra 7 155H")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

packages = ['numpy', 'pandas', 'sklearn', 'matplotlib']
for pkg in packages:
    try:
        mod = __import__(pkg)
        print(f"✅ {pkg}: installed")
    except:
        print(f"❌ {pkg}: not installed")