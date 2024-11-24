from typing import Dict, Union

class Prompts:
    """Manages system prompts for RSS feed analysis."""
    
    ANALYSIS: Dict[str, Union[Dict[str, str], str]] = {
        "default": {
            "system": """Analyze the RSS content and create exactly 15 news summaries with categories. 
Format MUST be exactly as follows (including numbering, markdown, dates, and categories):

1. **[Exact Article Title](article_url)** (YYYY-MM-DD)
   Category: <category>
   Brief summary of the article.

2. **[Exact Article Title](article_url)** (YYYY-MM-DD)
   Category: <category>
   Brief summary of the article.

(Continue exactly like this for all 15 items)

Categories should be one of: Technology, Business, Politics, Science, Health, Entertainment, Sports, World News

Important:
- Must provide exactly 15 items
- Must use numbers followed by period (1., 2., etc.)
- Must wrap titles in bold (**) and markdown links []()
- Must include the article date in parentheses (YYYY-MM-DD) format
- Must include category on a new line after the title
- Must include one summary paragraph after the category
- If article date is not available, omit the parentheses entirely
""",
            "user": "Please analyze and summarize the following RSS feed content in 15 points:"
        }
    }

    @classmethod
    def get_prompt(cls, prompt_type: str = "default", prompt_key: str = "system") -> str:
        """Get a specific prompt by type and key."""
        if prompt_type not in cls.ANALYSIS:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        prompt = cls.ANALYSIS[prompt_type]
        return prompt[prompt_key] if isinstance(prompt, dict) else prompt