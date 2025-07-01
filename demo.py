import importlib

# List of libraries to check and their common import names
# Note: 'imblearn' imports as 'imblearn', 'dvc-gs' is a DVC plugin and doesn't
# have a direct __version__ attribute accessible this way, but DVC itself does.
# Flask is 'flask'.
libraries = {
    "scikit-learn": "sklearn",
    "imbalanced-learn": "imblearn", # Actual package name is imbalanced-learn, import name is imblearn
    "pandas": "pandas",
    "numpy": "numpy",
    "seaborn": "seaborn",
    "matplotlib": "matplotlib",
    "mlflow": "mlflow",
    "dvc": "dvc",
    "Flask": "flask"
}

print("Checking installed versions of specified libraries:\n")

for package_name, import_name in libraries.items():
    try:
        # Attempt to import the module
        module = importlib.import_module(import_name)

        # Check for __version__ attribute
        if hasattr(module, '__version__'):
            print(f"{package_name} ({import_name}): {module.__version__}")
        else:
            print(f"{package_name} ({import_name}): Version information not found (no __version__ attribute).")
    except ImportError:
        print(f"{package_name} ({import_name}): Not installed or cannot be imported.")
    except Exception as e:
        print(f"{package_name} ({import_name}): An error occurred while checking version: {e}")

print("\nNote: For 'dvc-gs', it's a DVC plugin. Its version is tied to the DVC version,")
print("and it doesn't typically expose a separate __version__ attribute via direct import.")
print("The DVC version displayed above should indicate its compatibility.")
