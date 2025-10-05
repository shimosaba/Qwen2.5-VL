# !pip list | grep lmdeploy
# !pip list | grep transformers
# !pip list | grep torch
# !pip list | grep flash_attn

# lmdeploy                  0.10.1
# transformers              4.57.0
# torch                     2.8.0
# torchvision               0.23.0
# flash_attn                2.8.3

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
import time

SYSTEM_PROMPT = """You are an assistant specialized in OCR.
Please output OCR results only.
"""

PROMPT = """Please OCR and categorize image.
Categories are as follows:
- Company Name
- Department
- Position
- Qualification
- First Name
- Last Name
- First Name kana
- Last Name kana
- Email
- URL
- Phone Number
- Fax Number
- Mobile Number
- Post Code
- Prefecture
- Municipalities
- Street address
- Building Name
- Logo
- Other

Format Example 1: 
Company Name: XXX株式会社
Department: XXX部

Format Example 2, For a single line, use the following:
Prefecture: XX県 Municipalities: XXX市
"""

def main():

    MODEL = 'OpenGVLab/InternVL3-38B'

    backend = TurbomindEngineConfig(session_len=8196,
                                cache_max_entry_count=0.2
                                )

    pipe = pipeline(MODEL, 
                    backend_config=backend, 
                    log_level="ERROR")



    start = time.perf_counter()
    for i in range(5):
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"/gdrive/VLM/{i}.png"}},
            ]}
        ]
        
        gen_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=1024,
        )
        
        # モデルの実行
        output = pipe(messages, gen_config=gen_config)
        print("="*50)
        print(output.text)
        print("="*50)

    end = time.perf_counter()
    print(f"処理時間: {(end - start) / 10:.2f}秒")


if __name__ == "__main__":
    main()