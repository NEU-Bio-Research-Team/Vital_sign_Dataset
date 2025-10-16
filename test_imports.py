#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.

Run this script from the project root directory:
    python test_imports.py
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all module imports."""
    print("🧪 Testing imports...")
    
    try:
        from utils import setup_plotting, load_vitaldb_data, preprocess_data
        print("✅ utils module imported successfully")
    except Exception as e:
        print(f"❌ utils module import failed: {e}")
        return False
    
    try:
        from train import get_default_model_configs, hyperparameter_tuning
        print("✅ train module imported successfully")
    except Exception as e:
        print(f"❌ train module import failed: {e}")
        return False
    
    try:
        from evaluate import evaluate_models, calculate_comprehensive_metrics
        print("✅ evaluate module imported successfully")
    except Exception as e:
        print(f"❌ evaluate module import failed: {e}")
        return False
    
    try:
        from visualization import plot_roc_curves, plot_pr_curves
        print("✅ visualization module imported successfully")
    except Exception as e:
        print(f"❌ visualization module import failed: {e}")
        return False
    
    try:
        from shap_explainer import explain_model_with_shap
        print("✅ shap_explainer module imported successfully")
    except ImportError as e:
        if "shap" in str(e):
            print("⚠️  shap_explainer module requires SHAP library")
            print("   Install with: pip install shap")
        else:
            print(f"❌ shap_explainer module import failed: {e}")
            return False
    except Exception as e:
        print(f"❌ shap_explainer module import failed: {e}")
        return False
    
    print("\n🎉 All imports working correctly!")
    print("\n📝 Next steps:")
    print("   1. Install SHAP: pip install shap")
    print("   2. Run example notebooks: jupyter notebook notebooks/example_train.ipynb")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
