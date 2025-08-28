import os
import google.generativeai as genai
genai.configure(api_key="AIzaSyAn26uG2YjfKgD7b9B0td39KxcmdSQZZ48")

# Initialize the model
model = genai.GenerativeModel("gemini-2.5-flash")


def analyze_hair_results(analysis_results):
    """
    This function takes the extracted hair analysis results (dictionary)
    and generates dynamic AI-based suggestions using Gemini.
    
    Parameters:
        analysis_results (dict): Example:
            {
                "hair_type": "Dry",
                "scalp_condition": "Oily",
                "issues": ["Hair Fall", "Dandruff"]
            }
    
    Returns:
        dict: {
            "analysis_summary": "...",
            "ai_suggestions": "..."
        }
    """

    # Prompt for Gemini
    prompt = f"""
    You are a professional trichologist AI.
    Based on the following hair analysis results, provide a clear summary 
    and actionable personalized suggestions:

    Hair Type: {analysis_results.get("hair_type", "Not specified")}
    Scalp Condition: {analysis_results.get("scalp_condition", "Not specified")}
    Issues: {", ".join(analysis_results.get("issues", [])) if analysis_results.get("issues") else "None"}

    Format your response as:
    1. **Summary of Condition**
    2. **AI Recommendations** (with lifestyle, diet, and product tips)
    """

    try:
        response = model.generate_content(prompt)

        return {
            "analysis_summary": "AI-generated personalized hair report:",
            "ai_suggestions": response.text.strip() if response.text else "No suggestions generated."
        }

    except Exception as e:
        return {
            "analysis_summary": "Error while generating report",
            "ai_suggestions": str(e)
        }


if __name__ == "__main__":
    sample_results = {
        "hair_type": "Dry",
        "scalp_condition": "Oily",
        "issues": ["Hair Fall", "Dandruff"]
    }

    report = analyze_hair_results(sample_results)
    print("=== Hair Analysis Report ===")
    print(report["analysis_summary"])
    print(report["ai_suggestions"])
