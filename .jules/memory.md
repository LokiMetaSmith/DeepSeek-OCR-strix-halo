## Archived: 2026-10-31

* The current task involves porting DeepSeek-OCR features into the `nanochat` repository (LokiMetaSmith fork).
* The `GPT` class in `nanochat/gpt.py` has been updated to support `vision_backbone='deepseek'`, which instantiates `DeepSeekVisionTransformer` and sets the default `vision_projector` to `nn.Identity()`.
* The ported DeepSeek-OCR modules in `nanochat` require the `easydict` Python package.
* The DeepSeek-OCR source code contains hardcoded CUDA calls (e.g., `.cuda()`), which will need to be patched with device-agnostic code to run on the AMD ROCm platform.
* The DeepSeek-OCR model produces different output formats depending on the prompt. For clean text, use "\nFree OCR.". For structured markdown, use "\n<|grounding|>Convert the document to markdown.". For bounding box coordinates, use "\n<|grounding|>OCR this image.".
* I must use the `request_user_input` and `message_user` tools to ask the user clarifying questions during the planning phase.
* Before starting a task, I must enter a 'deep planning mode' to fully understand the user's requirements by asking clarifying questions. I should only create a plan after I have absolute certainty about the user's goals.
* The project's goal is to get the DeepSeek-OCR model working on the user's specialized local hardware, which is an AMD Strix Halo (gfx1151) GPU, using ROCm 7.1.0. I do not have direct access to this hardware.
* A `DeepSeekVisionTransformer` class has been added to `nanochat/vision.py`, implementing the Hybrid Encoder (SAM + CLIP) and high-resolution tiling logic.
* My role is to act as a remote specialist: I will create diagnostic and installation scripts for the user to run on their local hardware. The user will provide the output back to me for analysis and to inform the next steps, such as creating code patches.
* Ported DeepSeek modules in `nanochat` have `flash_attn` dependencies disabled (commented out) to ensure compatibility with standard PyTorch on the user's AMD ROCm hardware.
* The `nanochat` source code resides in `nanochat/nanochat/`, and the new DeepSeek modules are located in `nanochat/nanochat/deepseek_modules/`.
* If a hard blocker is encountered, I should report it, research the problem, and also attempt to implement a (potentially flawed) solution to help inform the user's own research.
* The project requires creating extensive notes in a `notes.md` file, which should only ever be appended to. Final deliverables should include setup and run scripts, and a detailed `README.md`.
