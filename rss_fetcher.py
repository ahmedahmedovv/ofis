import feedparser
from datetime import datetime, timedelta
import os
import json
import yaml
import logging
from time import mktime
import urllib.robotparser
import urllib.request
from html import unescape
import time
from openai import OpenAI
import glob
from prompts import Prompts
from openai_summary import OpenAISummaryHandler
from typing import Dict, Any, Optional
from pathlib import Path
import re
from deep_translator import GoogleTranslator
import langdetect

class FeedCache:
    def __init__(self, expiry_time: int = 3600):
        self.cache = {}
        self.expiry_time = expiry_time

    def get(self, url: str) -> Optional[Dict]:
        if url in self.cache:
            data, timestamp = self.cache[url]
            if time.time() - timestamp < self.expiry_time:
                return data
        return None

    def set(self, url: str, data: Dict):
        self.cache[url] = (data, time.time())

def setup_logging(config):
    """Setup logging configuration with directory creation"""
    try:
        # Create logs directory if it doesn't exist
        log_file = config['logging']['file']
        log_dir = os.path.dirname(log_file)
        
        # Create directory if it doesn't exist
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config['logging']['level']),
            format=config['logging']['format'],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler() if config['logging']['console_output'] else None
            ]
        )
        
        logging.info("Logging setup completed")
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        # Fallback to basic console logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

def can_fetch_url(url, config):
    """Check robots.txt for URL fetchability"""
    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{urllib.parse.urlparse(url).scheme}://{urllib.parse.urlparse(url).netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(config['feed_options']['user_agent'], url)
    except Exception as e:
        logging.warning(f"Could not check robots.txt for {url}: {str(e)}")
        return True

def process_entry_value(value, field_config):
    """Process entry value according to field configuration"""
    if not value:
        return field_config['default']
    
    if field_config.get('strip_html', False):
        from html.parser import HTMLParser
        h = HTMLParser()
        value = h.unescape(value)
        # Basic HTML tag stripping - could be more sophisticated
        import re
        value = re.sub('<[^<]+?>', '', value)
    
    if field_config.get('max_length'):
        value = value[:field_config['max_length']]
        
    if field_config.get('type') == 'list' and isinstance(value, str):
        value = [v.strip() for v in value.split(',')]
        
    return value

def fetch_feeds(urls, config):
    """Fetch RSS feeds and return formatted results as a dictionary"""
    results = []
    errors = {}  # Track errors for each URL
    
    for url in urls:
        try:
            # Add the actual feed fetching implementation here
            if not can_fetch_url(url, config):
                errors[url] = "URL blocked by robots.txt"
                continue
                
            feed = feedparser.parse(url)
            if feed.entries:
                results.append({
                    'feed_url': url,
                    'feed_title': feed.feed.get('title', 'Untitled Feed'),
                    'entries': feed.entries
                })
            
        except urllib.error.HTTPError as e:
            error_msg = f"HTTP Error {e.code}: {e.reason}"
            errors[url] = error_msg
            logging.error(f"Error fetching {url}: {error_msg}")
            
        except urllib.error.URLError as e:
            error_msg = f"URL Error: {str(e.reason)}"
            errors[url] = error_msg
            logging.error(f"Error fetching {url}: {error_msg}")
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            errors[url] = error_msg
            logging.error(f"Error fetching {url}: {error_msg}")
    
    return {
        'timestamp': datetime.now().isoformat(),
        'feeds': results,
        'errors': errors,  # Include errors in the return data
        'stats': {
            'total_attempted': len(urls),
            'successful': len(results),
            'failed': len(errors),
            'success_rate': f"{(len(results) / len(urls) * 100):.1f}%"
        }
    }

def save_results(content, config):
    """Save results to a JSON file with timestamp"""
    output_dir = config['output_options']['directory']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime(config['output_options']['timestamp_format'])
    filename = os.path.join(
        output_dir,
        config['output_options']['filename_format'].format(timestamp=timestamp)
    )
    
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(
            content,
            file,
            indent=config['output_options']['indent'],
            ensure_ascii=config['output_options']['ensure_ascii']
        )
    
    return filename

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def get_latest_feed_file(directory):
    """Get the most recent feed JSON file"""
    files = [f for f in os.listdir(directory) if f.startswith('rss_feeds_') and f.endswith('.json')]
    if not files:
        raise FileNotFoundError("No feed files found")
    
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return os.path.join(directory, latest_file)

def prepare_feed_content(feed_data):
    """Prepare feed content for analysis in a structured format"""
    if not feed_data.get('feeds'):
        return "No valid feed data available for analysis"
    
    content = []
    
    # Add timestamp using feed_data timestamp if available, otherwise current time
    timestamp = feed_data.get('timestamp', datetime.now().isoformat())
    content.append(f"Analysis timestamp: {timestamp}")
    content.append(f"Number of feeds: {len(feed_data['feeds'])}\n")
    
    for feed in feed_data['feeds']:
        content.append(f"## Feed: {feed['feed_title']}")
        content.append(f"Source: {feed['feed_url']}")
        
        content.append("\n### Recent Entries:")
        for entry in feed['entries']:
            content.append(f"- **{entry.get('title', 'No title')}**")
            # Fix date handling
            published_date = entry.get('published', entry.get('date', 'No date available'))
            content.append(f"  Date: {published_date}")
            if 'description' in entry:
                # Clean and truncate description
                desc = entry.get('description', '').replace('\n', ' ').strip()
                content.append(f"  Summary: {desc[:300]}...")
            content.append("---")
        content.append("\n")
    
    return "\n".join(content)

def load_prompts():
    """Load prompts from prompts.yaml"""
    with open('prompts.yaml', 'r') as file:
        return yaml.safe_load(file)

def ensure_directories_exist(config):
    """Create all necessary directories"""
    directories = [
        'logs',  # Base logs directory
        'results',  # For feed results
        'debug/raw_feeds',  # For debug data
        os.path.dirname(config['logging']['file'])  # Full path for log file
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")
            except Exception as e:
                logging.error(f"Error creating directory {directory}: {str(e)}")

def save_analysis(analysis: str, config: Dict[str, Any]) -> str:
    """Save analysis to JSON file"""
    # Create output filename with timestamp
    timestamp = datetime.now().strftime(config['output_options']['timestamp_format'])
    output_file = os.path.join(
        config['output_options']['directory'],
        f"analysis_{timestamp}.json"
    )

    # Get the actual number of feeds attempted from rss_sources.txt
    try:
        with open('rss_sources.txt', 'r', encoding='utf-8') as f:
            total_feeds_attempted = sum(1 for line in f if line.strip() and not line.startswith('#'))
    except FileNotFoundError:
        total_feeds_attempted = 0

    # Get successful feeds count from the latest feed file
    try:
        latest_feed_file = max(
            glob.glob(os.path.join(config['output_options']['directory'], 'rss_feeds_*.json')),
            key=os.path.getctime
        )
        with open(latest_feed_file, 'r', encoding='utf-8') as f:
            feed_data = json.load(f)
            successful_feeds = len(feed_data.get('feeds', []))
    except (FileNotFoundError, ValueError):
        successful_feeds = 0

    # Calculate success rate
    success_rate = (successful_feeds / total_feeds_attempted * 100) if total_feeds_attempted > 0 else 0

    analysis_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "sources_analyzed": successful_feeds,
            "feed_sources": {
                "successful": successful_feeds,
                "total_attempted": total_feeds_attempted,
                "success_rate": f"{success_rate:.1f}%"
            }
        },
        "summary": []
    }

    try:
        # Parse the analysis content
        pattern = r'(\d+)\.\s+\*\*\[(.*?)\]\((.*?)\)\*\*\s*\((\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|)\)\s*(.*?)(?=(?:\d+\.|$))'
        matches = re.finditer(pattern, analysis, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            number, title, link, date_str, summary = match.groups()
            
            # Convert date format if needed
            published = None
            if date_str:
                try:
                    if '/' in date_str:
                        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                    else:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    published = date_obj.isoformat()
                except ValueError:
                    logging.warning(f"Could not parse date: {date_str}")
            
            article_data = {
                "number": int(number),
                "title": title.strip(),
                "link": link.strip(),
                "published": published,
                "summary": summary.strip()
            }
            analysis_data["summary"].append(article_data)
            
        logging.info(f"Found {len(analysis_data['summary'])} articles from {successful_feeds} sources")
        
    except Exception as e:
        logging.error(f"Error parsing analysis: {str(e)}")
        logging.debug(f"Analysis content that caused error: {analysis}")
        
    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    return output_file

def translate_text(text: str, target_lang: str = 'en') -> str:
    """Translate text to target language if it's not already in English"""
    try:
        # Skip translation if text is already in English
        if not text or langdetect.detect(text) == 'en':
            return text
            
        translator = GoogleTranslator(source='auto', target=target_lang)
        # Split long text into chunks if needed (Google Translator has character limits)
        max_chars = 5000
        if len(text) > max_chars:
            chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
            translated_chunks = [translator.translate(chunk) for chunk in chunks]
            return ' '.join(translated_chunks)
        else:
            return translator.translate(text)
    except Exception as e:
        logging.warning(f"Translation failed: {str(e)}")
        return text

def fetch_and_save_feeds(config):
    """Fetch feeds and save them to a file"""
    try:
        # Read RSS sources
        with open('rss_sources.txt', 'r', encoding='utf-8') as file:
            urls = [line.strip() for line in file if line.strip() and not line.startswith('#')]
        
        if not urls:
            logging.error("No URLs found in rss_sources.txt")
            return None
            
        # Create results directory if it doesn't exist
        os.makedirs(config['output_options']['directory'], exist_ok=True)
        
        # Prepare feed data structure with timestamp
        feed_data = {
            'timestamp': datetime.now().isoformat(),
            'feeds': []
        }
        
        # Fetch feeds
        for url in urls:
            try:
                logging.info(f"Fetching feed: {url}")
                
                # Setup request with headers
                headers = {'User-Agent': config['feed_options']['user_agent']}
                req = urllib.request.Request(url, headers=headers)
                
                # Fetch feed with timeout
                with urllib.request.urlopen(req, timeout=config['feed_options']['timeout']) as response:
                    feed_content = response.read()
                    feed = feedparser.parse(feed_content)
                    
                    if feed.entries:
                        # Translate feed entries
                        translated_entries = []
                        for entry in feed.entries[:config['feed_options']['max_entries_per_feed']]:
                            translated_entry = {
                                'title': translate_text(entry.get('title', 'No title')),
                                'link': entry.get('link', ''),
                                'description': translate_text(entry.get('description', 'No description')),
                                'published': entry.get('published', ''),
                                'original_title': entry.get('title', 'No title'),  # Keep original title
                                'original_description': entry.get('description', 'No description')  # Keep original description
                            }
                            translated_entries.append(translated_entry)
                        
                        feed_data['feeds'].append({
                            'feed_url': url,
                            'feed_title': translate_text(feed.feed.get('title', 'Untitled Feed')),
                            'original_feed_title': feed.feed.get('title', 'Untitled Feed'),
                            'fetch_time': datetime.now().isoformat(),
                            'entries': translated_entries
                        })
                        logging.info(f"Successfully fetched and translated {len(feed.entries)} entries from {url}")
                    
            except Exception as e:
                logging.error(f"Error fetching {url}: {str(e)}")
                continue
        
        if not feed_data['feeds']:
            logging.error("No feeds were successfully fetched")
            return None
            
        # Save feeds to file
        timestamp = datetime.now().strftime(config['output_options']['timestamp_format'])
        output_file = os.path.join(
            config['output_options']['directory'],
            f"rss_feeds_{timestamp}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(feed_data, f, indent=config['output_options']['indent'])
            
        logging.info(f"Feeds saved to: {output_file}")
        return output_file
        
    except Exception as e:
        logging.error(f"Error in fetch_and_save_feeds: {str(e)}")
        return None

def validate_feed(feed):
    """Validate feed structure and content"""
    required_fields = ['title', 'entries']
    return all(field in feed.feed for field in required_fields)

def validate_entry(entry):
    """Validate and clean entry data"""
    required_fields = {
        'title': 'No title',
        'description': 'No description',
        'published': None
    }
    
    cleaned_entry = {}
    for field, default in required_fields.items():
        cleaned_entry[field] = entry.get(field, default)
        
    # Handle date specifically
    if not cleaned_entry['published']:
        # Try alternative date fields
        for date_field in ['date', 'updated', 'created']:
            if date_field in entry:
                cleaned_entry['published'] = entry[date_field]
                break
        if not cleaned_entry['published']:
            cleaned_entry['published'] = 'No date available'
    
    return cleaned_entry

def load_configs() -> Dict[str, Any]:
    """Load all configuration files"""
    config_files = {
        'main': 'config.yaml',
        'scoring': 'scoring_config.yaml'
    }
    
    configs = {}
    for key, filename in config_files.items():
        try:
            with open(filename, 'r') as file:
                configs[key] = yaml.safe_load(file)
        except FileNotFoundError:
            logging.error(f"Configuration file {filename} not found")
            raise
            
    return configs

def main():
    start_time = time.time()  # Start timing
    try:
        # Load all configurations
        configs = load_configs()
        config = configs['main']
        
        # Setup logging
        setup_logging(config)
        
        # Ensure directories exist
        ensure_directories_exist(config)
        
        # Fetch and save feeds
        feed_file = fetch_and_save_feeds(config)
        if not feed_file:
            logging.error("Failed to fetch feeds")
            return
            
        logging.info(f"Feeds saved to: {feed_file}")
        
        # Load the newly fetched feed data
        with open(feed_file, 'r', encoding='utf-8') as f:
            feed_data = json.load(f)
        
        # Initialize OpenAI handler with both configs
        openai_handler = OpenAISummaryHandler(config, configs['scoring'])
        
        # Generate analysis
        analysis = openai_handler.generate_analysis(feed_data)
        
        # Save analysis
        output_file = save_analysis(analysis, config)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print results
        print(f"\nAnalysis saved to: {output_file}")
        print("\nAnalysis Summary:")
        print("=" * 50)
        print(analysis)
        print("\nExecution Statistics:")
        print("=" * 50)
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Processed {len(feed_data['feeds'])} feeds")
        if 'feeds' in feed_data and feed_data['feeds']:
            total_entries = sum(len(feed['entries']) for feed in feed_data['feeds'])
            print(f"Total entries processed: {total_entries}")
            print(f"Average time per feed: {execution_time/len(feed_data['feeds']):.2f} seconds")
        
    except Exception as e:
        end_time = time.time()
        print(f"An error occurred after {end_time - start_time:.2f} seconds: {str(e)}")
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 