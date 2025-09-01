#!/usr/bin/env python3
"""
Test Script for Voice Recognition System

This script tests the basic functionality of the system.
"""

import os
from pathlib import Path
from service import ServiceState, PipelineService


def test_system():
    """Test the voice recognition system."""
    print("🧪 Testing Voice Recognition System...")
    
    try:
        # Test 1: Service State
        print("\n1. Testing Service State...")
        state = ServiceState()
        print(f"   ✅ GPU Available: {state.is_gpu_available()}")
        print(f"   ✅ Device: {state.DEVICE}")
        print(f"   ✅ Sample Rate: {state.TARGET_SAMPLE_RATE}Hz")
        print(f"   ✅ ASR Backend: {state.ASR_BACKEND}")
        print(f"   ✅ Model Size: {state.DEFAULT_MODEL_SIZE}")
        
        # Test 2: Pipeline Service
        print("\n2. Testing Pipeline Service...")
        pipeline = PipelineService(state)
        print("   ✅ Pipeline initialized successfully")
        
        # Test 3: Service Info
        print("\n3. Getting Service Information...")
        info = pipeline.get_pipeline_info()
        print(f"   ✅ ASR Service: {info['services']['asr']['model_loaded']}")
        print(f"   ✅ Feature Service: {info['services']['features']['resemblyzer_available']}")
        print(f"   ✅ Enrollment Service: {info['services']['enrollment']['profiles_loaded']} profiles")
        
        # Test 4: Check Directories
        print("\n4. Checking Directory Structure...")
        paths = state.get_paths()
        for name, path in paths.items():
            exists = path.exists()
            print(f"   ✅ {name.capitalize()}: {path} ({'✓' if exists else '✗'})")
        
        print("\n🎉 All tests passed! System is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base64_service():
    """Test the Base64 service specifically."""
    print("\n🔍 Testing Base64 Service...")
    
    try:
        from service import Base64Service, ServiceState
        
        state = ServiceState()
        base64_service = Base64Service(state)
        
        # Test Base64 validation
        test_string = "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT"
        
        validation = base64_service.validate_base64_string(test_string)
        print(f"   ✅ Base64 Validation: {validation['is_valid']}")
        print(f"   ✅ Estimated Size: {validation['estimated_size_mb']:.2f}MB")
        
        info = base64_service.get_base64_info(test_string)
        print(f"   ✅ Detected Format: {info.get('detected_format', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Base64 test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Voice Recognition System Test")
    print("=" * 50)
    
    # Run main tests
    main_success = test_system()
    
    # Run Base64 tests
    base64_success = test_base64_service()
    
    # Summary
    print("\n" + "=" * 50)
    if main_success and base64_success:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Test with audio: python service_cli.py -f audio.mp3")
        print("3. Test with Base64: python service_cli.py --base64-file audio_base64.txt")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Verify all service files are present")
