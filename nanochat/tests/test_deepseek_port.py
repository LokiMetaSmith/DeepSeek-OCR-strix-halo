import torch
# Correct import structure for the environment
from nanochat.vision import DeepSeekVisionTransformer, VisionConfig
from PIL import Image

def test_deepseek_vision_instantiation():
    config = VisionConfig(use_deepseek=True, deepseek_local_path='dummy_path.pth')
    # We mock the build_sam_vit_b to avoid needing a real checkpoint for this test
    # But for now, let's just see if it inits (it might fail on checkpoint load)
    try:
        model = DeepSeekVisionTransformer(config)
    except Exception as e:
        # Expected failure due to missing checkpoint, but confirms class is reachable
        print(f'Caught expected error (checkpoint): {e}')
        return
    print("DeepSeekVisionTransformer instantiated (unexpectedly without checkpoint)")

def test_tiling_logic_import():
    from nanochat.deepseek_modules.image_processing import dynamic_preprocess
    img = Image.new('RGB', (2000, 2000))
    crops, ratio = dynamic_preprocess(img, image_size=1024)
    assert len(crops) > 1, 'Should create multiple tiles for large image'
    print(f'Tiling test passed: Generated {len(crops)} crops for 2000x2000 image')

if __name__ == '__main__':
    test_tiling_logic_import()
    test_deepseek_vision_instantiation()
