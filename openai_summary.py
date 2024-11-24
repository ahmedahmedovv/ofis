from datetime import datetime
import logging
from openai import OpenAI
from typing import Dict, Any, Optional
import re
import yaml
import email.utils
from dateutil import parser
from dotenv import load_dotenv
import os

class OpenAISummaryHandler:
    def __init__(self, config: Dict[str, Any], scoring_config: Dict[str, Any]):
        """
        Initialize OpenAI summary handler with configurations
        
        Args:
            config (dict): Main configuration dictionary containing OpenAI settings
            scoring_config (dict): Scoring configuration dictionary
        """
        load_dotenv()  # Load environment variables
        self.config = config
        if not self._validate_scoring_config(scoring_config):
            raise ValueError("Invalid scoring configuration")
        self.scoring_config = scoring_config
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # Use environment variable
        
    def _validate_scoring_config(self, scoring_config: Dict[str, Any]) -> bool:
        """Validate scoring configuration"""
        required_sections = ['weights', 'recency', 'content_length', 'keywords', 
                            'reliable_sources', 'thresholds']
        
        # Check all required sections exist
        if not all(section in scoring_config for section in required_sections):
            missing = [s for s in required_sections if s not in scoring_config]
            logging.error(f"Missing sections in scoring config: {missing}")
            return False
        
        # Validate weights sum to 1.0
        weights = scoring_config['weights']
        weight_sum = sum(weights.values())
        if not 0.99 <= weight_sum <= 1.01:  # Allow for small floating point differences
            logging.error(f"Weights must sum to 1.0, got {weight_sum}")
            return False
        
        return True

    def prepare_feed_content(self, feed_data: Dict[str, Any]) -> str:
        """Prepare and score feed content for analysis using scoring config"""
        logging.info(f"Starting feed content preparation with {len(feed_data.get('feeds', []))} feeds")
        
        if not feed_data.get('feeds'):
            logging.error("No feeds found in feed_data")
            return "No valid feed data available for analysis"
        
        scored_entries = []
        self.links = []
        
        for feed in feed_data['feeds']:
            logging.debug(f"Processing feed: {feed.get('feed_title', 'Unknown feed')}")
            for entry in feed['entries']:
                score = self._calculate_entry_score(entry)
                logging.debug(f"Calculated score {score} for article: {entry.get('title', 'No title')}")
                
                if score > 0:  # Only include articles that meet minimum threshold
                    scored_entries.append({
                        'title': entry.get('title', 'No title').strip(),
                        'link': entry.get('link', '').strip(),
                        'description': entry.get('description', '').strip(),
                        'published': entry.get('published', ''),
                        'score': score
                    })
        
        logging.info(f"Found {len(scored_entries)} entries that meet the minimum score threshold")
        
        # Sort by score and take top candidates
        scored_entries.sort(key=lambda x: x['score'], reverse=True)
        top_entries = scored_entries[:self.scoring_config['thresholds']['top_candidates']]
        logging.info(f"Selected top {len(top_entries)} entries for analysis")
        
        # Format content for OpenAI
        content = []
        for entry in top_entries:
            if entry['title'] and entry['link']:
                self.links.append({
                    'title': entry['title'],
                    'link': entry['link']
                })
            content.append(f"[{entry['title']}]({entry['link']})")
            content.append(f"Score: {entry['score']:.2f}")
            content.append(f"Published: {entry['published']}")
            if entry['description']:
                content.append(f"Summary: {entry['description'][:300]}...")
            content.append("---")
        
        final_content = "\n".join(content)
        logging.debug(f"Prepared content length: {len(final_content)} characters")
        return final_content

    def _calculate_entry_score(self, entry: Dict[str, Any]) -> float:
        """Calculate a news article's importance score using scoring config"""
        base_score = 0.0
        weights = self.scoring_config['weights']
        
        logging.debug(f"Scoring article: {entry.get('title', 'No title')}")
        
        # 1. RECENCY SCORE
        if 'published' in entry:
            try:
                # Try multiple date parsing methods
                date_str = entry['published']
                try:
                    # Try RFC 2822 first (common in RSS)
                    parsed_date = email.utils.parsedate_to_datetime(date_str)
                except (TypeError, ValueError):
                    try:
                        # Try ISO format
                        parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except ValueError:
                        # Fall back to dateutil parser
                        parsed_date = parser.parse(date_str)
                
                # Ensure timezone awareness
                if parsed_date.tzinfo is None:
                    parsed_date = parsed_date.replace(tzinfo=datetime.now().astimezone().tzinfo)
                
                hours_since_published = (datetime.now(parsed_date.tzinfo) - parsed_date).total_seconds() / 3600
                
                max_age = self.scoring_config['recency']['max_age_hours']
                min_score = self.scoring_config['recency']['min_score']
                
                recency_score = max(min_score, 1 - (hours_since_published / max_age))
                base_score += recency_score * weights['recency']
                
                logging.debug(f"Recency score: {recency_score:.2f}")
            except Exception as e:
                logging.warning(f"Couldn't calculate recency score for date '{entry['published']}': {str(e)}")
                base_score += self.scoring_config['recency']['min_score'] * weights['recency']

        # 2. CONTENT LENGTH SCORE
        description = entry.get('description', '')
        if description:
            ideal_length = self.scoring_config['content_length']['ideal_length']
            min_score = self.scoring_config['content_length']['min_score']
            
            length_score = max(min_score, min(len(description) / ideal_length, 1))
            base_score += length_score * weights['content_length']
            logging.debug(f"Length score: {length_score:.2f}")

        # 3. KEYWORDS SCORE
        title = entry.get('title', '').lower()
        important_keywords = self.scoring_config['keywords']['important']
        matching_keywords = [keyword for keyword in important_keywords if keyword in title]
        keyword_score = len(matching_keywords) / len(important_keywords)
        base_score += keyword_score * weights['keywords']
        logging.debug(f"Keyword score: {keyword_score:.2f}")

        # 4. SOURCE RELIABILITY SCORE
        source_url = entry.get('link', '')
        reliable_sources = self.scoring_config['reliable_sources']
        is_reliable = any(source in source_url for source in reliable_sources)
        source_score = 1.0 if is_reliable else 0.0
        base_score += source_score * weights['source_reliability']
        logging.debug(f"Source reliability score: {source_score:.2f}")

        # 5. CATEGORY SCORING
        title_and_description = f"{entry.get('title', '').lower()} {entry.get('description', '').lower()}"
        category_boost = self._calculate_category_boost(title_and_description)
        
        # Apply category boost to base score
        final_score = base_score * category_boost
        logging.debug(f"Category boost: {category_boost:.2f}")
        logging.debug(f"Final score after category boost: {final_score:.2f}")
        
        return final_score

    def _calculate_category_boost(self, text: str) -> float:
        """Calculate category-based boost for article"""
        max_boost = 1.0
        
        for category, config in self.scoring_config['categories'].items():
            category_keywords = config['keywords']
            matching_keywords = sum(1 for keyword in category_keywords if keyword in text)
            
            if matching_keywords > 0:
                category_weight = config.get('weight', 1.0)
                boost = min(category_weight, 1.0 + (matching_keywords / len(category_keywords)))
                max_boost = max(max_boost, boost)
                logging.debug(f"Category {category} boost: {boost:.2f}")
        
        return max_boost

    def generate_analysis(self, feed_data: Dict[str, Any], prompt_type: str = "default") -> str:
        """Generate analysis using OpenAI API"""
        try:
            from prompts import Prompts
            
            # Prepare the content
            content = self.prepare_feed_content(feed_data)
            
            # Get the prompts from Prompts class
            system_prompt = Prompts.get_prompt(prompt_type, "system")
            user_prompt = Prompts.get_prompt(prompt_type, "user")
            
            # Get analysis from OpenAI
            response = self.client.chat.completions.create(
                model=self.config['openai']['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + "\n" + content}
                ],
                temperature=0.7,
                max_tokens=self.config['openai']['max_tokens']
            )
            
            # Extract the analysis text
            analysis = response.choices[0].message.content
            return analysis
            
        except Exception as e:
            logging.error(f"Error in generate_analysis: {str(e)}")
            return f"Error generating analysis: {str(e)}"

    def format_analysis(self, analysis: str, feed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format the analysis output with metadata and links"""
        # Replace missing or incorrect links with proper ones
        for link_info in self.links:
            title = re.escape(link_info['title'])
            url = link_info['link']
            
            # Pattern to find titles without proper links
            patterns = [
                f'\\*\\*([^*]*{title}[^*]*)\\*\\*(?!\\]\\()',  # Bold title without link
                f'\\[([^]]*{title}[^]]*)\\]\\([^)]+\\)',  # Incorrect link
            ]
            
            for pattern in patterns:
                analysis = re.sub(pattern, f'**[\\1]({url})**', analysis)

        # Extract summaries and categories
        summaries = []
        pattern = r'\d+\.\s+\*\*\[([^\]]+)\]\(([^)]+)\)\*\*\s*(?:\(([^)]+)\))?\s*Category:\s*(\w+(?:\s+\w+)*)\s*([^\n]+)'
        
        matches = re.finditer(pattern, analysis)
        for match in matches:
            title, url, date, category, summary = match.groups()
            summaries.append({
                "title": title,
                "url": url,
                "date": date or "",
                "category": category.strip(),
                "summary": summary.strip()
            })

        # Save to JSON
        import json
        with open('news_summaries.json', 'w', encoding='utf-8') as f:
            json.dump({"summaries": summaries}, f, indent=2, ensure_ascii=False)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_text = f"""RSS Feed Analysis
Generated: {timestamp}
==================================================

{analysis}

==================================================
Sources analyzed: {len(feed_data['feeds'])} feeds
Total articles: {sum(len(feed['entries']) for feed in feed_data['feeds'])}
JSON output saved to: news_summaries.json
"""
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "sources_analyzed": len(feed_data['feeds']),
                "feed_sources": {
                    "successful": len(feed_data['feeds']),
                    "total_attempted": total_feeds_attempted,
                    "success_rate": f"{success_rate:.1f}%"
                }
            },
            "summary": [
                {
                    "number": i + 1,
                    "title": item["title"],
                    "link": item["url"],
                    "published": item["date"],
                    "category": self._determine_category(item["title"], item["summary"]),  # Add category
                    "summary": item["summary"]
                }
                for i, item in enumerate(summaries)
            ]
        }

    def _determine_category(self, title: str, summary: str) -> str:
        """Determine the category based on content analysis"""
        categories = ["Technology", "Business", "Politics", "Science", 
                     "Health", "Entertainment", "Sports", "World News"]
        
        # Add logic to determine category based on content
        # For now, let's use a simple keyword-based approach
        keywords = {
            "Technology": ["tech", "software", "cyber", "digital", "AI"],
            "Business": ["business", "economy", "market", "finance"],
            "Politics": ["government", "election", "political", "minister"],
            "Science": ["research", "study", "scientific", "discovery"],
            "Health": ["health", "medical", "disease", "treatment"],
            "Entertainment": ["movie", "film", "music", "celebrity"],
            "Sports": ["sport", "game", "tournament", "championship"],
            "World News": ["international", "global", "world"]
        }
        
        content = (title + " " + summary).lower()
        for category, words in keywords.items():
            if any(word.lower() in content for word in words):
                return category
        
        return "World News"  # Default category

    def analyze_by_category(self, feed_data: Dict[str, Any], category: str) -> Optional[str]:
        """
        Generate specialized analysis based on category
        
        Args:
            feed_data (dict): Feed data to analyze
            category (str): Analysis category (technical, news, trends, etc.)
            
        Returns:
            Optional[str]: Generated analysis or None if category not found
        """
        try:
            return self.generate_analysis(feed_data, category)
        except ValueError:
            logging.error(f"Invalid analysis category: {category}")
            return None

    def batch_analyze(self, feed_data: Dict[str, Any], categories: list) -> Dict[str, str]:
        """
        Generate multiple analyses for different categories
        
        Args:
            feed_data (dict): Feed data to analyze
            categories (list): List of analysis categories
            
        Returns:
            dict: Dictionary of category-analysis pairs
        """
        results = {}
        for category in categories:
            analysis = self.analyze_by_category(feed_data, category)
            if analysis:
                results[category] = analysis
        return results 