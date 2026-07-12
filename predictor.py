import os
from google import genai
from google.genai import types

# The robot gets its key from Render
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def get_smart_predictions(asset_list):
    """
    Asks Gemini for a smart score for each asset.
    """
    # 1. We ask the smart Flash model
    prompt = f"""
    Analyze the current market sentiment for these assets: {asset_list}.
    Give a score from -1.0 (bad) to +1.0 (good) for each.
    Return ONLY a dictionary like this: {{'ASSET': score}}
    """
    
    try:
        # 2. Ask the model
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
        )
        
        # 3. Clean up the answer
        text = response.text.replace("```python", "").replace("```", "").strip()
        predictions = eval(text)
        return predictions
    except Exception as e:
        print(f"Oops! The Brain is resting: {e}")
        # Give a safe neutral score if Gemini is busy
        return {asset: 0.05 for asset in asset_list}
