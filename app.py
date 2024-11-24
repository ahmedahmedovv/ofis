from flask import Flask, render_template, jsonify
from rss_fetcher import load_config, fetch_and_save_feeds, save_analysis, load_configs
from openai_summary import OpenAISummaryHandler
import json
import os
from datetime import datetime
import logging
import humanize
from pathlib import Path

app = Flask(__name__)

def get_latest_analysis():
    """Get the most recent analysis file from results directory"""
    results_dir = 'results'
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "sources_analyzed": 0,
            },
            "summary": [],
            "message": "✨ No analysis available yet. Click refresh to fetch news! ✨"
        }
    
    analysis_files = [f for f in os.listdir(results_dir) if f.startswith('analysis_') and f.endswith('.json')]
    if not analysis_files:
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "sources_analyzed": 0,
            },
            "summary": [],
            "message": "✨ No analysis available yet. Click refresh to fetch news! ✨"
        }
    
    latest_file = max(analysis_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    with open(os.path.join(results_dir, latest_file), 'r') as f:
        return json.load(f)

def generate_report(feed_data, analysis_data, config):
    """Generate detailed report about the feed fetching and analysis process"""
    
    # Get the actual number of feeds attempted from rss_sources.txt
    try:
        with open('rss_sources.txt', 'r', encoding='utf-8') as f:
            total_feeds_attempted = sum(1 for line in f if line.strip() and not line.startswith('#'))
    except FileNotFoundError:
        total_feeds_attempted = len(feed_data['feeds'])  # Fallback to successful feeds count
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "feed_statistics": {
            "total_feeds_attempted": total_feeds_attempted,  # Use the actual count
            "successful_feeds": len(feed_data['feeds']),
            "total_news_items": sum(len(feed['entries']) for feed in feed_data['feeds']),
            "success_rate": f"{(len(feed_data['feeds']) / total_feeds_attempted * 100):.1f}%" if total_feeds_attempted > 0 else "0%",
            "feeds_detail": []
        },
        "analysis_statistics": {
            "articles_analyzed": len(analysis_data['summary']),
            "data_transfer": {
                "feed_data_size": 0,
                "analysis_data_size": 0
            }
        },
        "timing": {
            "fetch_started": feed_data.get('timestamp'),
            "analysis_completed": analysis_data['metadata']['generated_at']
        }
    }

    # Calculate data sizes
    feed_data_size = len(json.dumps(feed_data).encode('utf-8'))
    analysis_data_size = len(json.dumps(analysis_data).encode('utf-8'))
    
    report["analysis_statistics"]["data_transfer"] = {
        "feed_data_size": humanize.naturalsize(feed_data_size),
        "analysis_data_size": humanize.naturalsize(analysis_data_size),
        "total_size": humanize.naturalsize(feed_data_size + analysis_data_size)
    }

    # Detailed feed information
    for feed in feed_data['feeds']:
        feed_detail = {
            "feed_url": feed['feed_url'],
            "feed_title": feed['feed_title'],
            "articles_count": len(feed['entries']),
            "fetch_time": feed.get('fetch_time'),
            "oldest_article": None,
            "newest_article": None
        }

        # Get article date range if available
        articles_with_dates = [
            entry for entry in feed['entries'] 
            if entry.get('published')
        ]
        
        if articles_with_dates:
            feed_detail["oldest_article"] = min(
                articles_with_dates, 
                key=lambda x: x['published']
            )['published']
            feed_detail["newest_article"] = max(
                articles_with_dates, 
                key=lambda x: x['published']
            )['published']

        report["feed_statistics"]["feeds_detail"].append(feed_detail)

    # Create report directory if it doesn't exist
    report_dir = Path('reports')
    report_dir.mkdir(exist_ok=True)

    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = report_dir / f'fetch_report_{timestamp}.json'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Generate human-readable summary
    summary_file = report_dir / f'fetch_summary_{timestamp}.txt'
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("RSS Feed Fetch and Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Feed Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total feeds attempted: {report['feed_statistics']['total_feeds_attempted']}\n")
        f.write(f"Successful feeds: {report['feed_statistics']['successful_feeds']}\n")
        f.write(f"Success rate: {report['feed_statistics']['success_rate']}\n")
        f.write(f"Total news items: {report['feed_statistics']['total_news_items']}\n\n")
        
        f.write("Analysis Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Articles analyzed: {report['analysis_statistics']['articles_analyzed']}\n")
        f.write(f"Feed data size: {report['analysis_statistics']['data_transfer']['feed_data_size']}\n")
        f.write(f"Analysis data size: {report['analysis_statistics']['data_transfer']['analysis_data_size']}\n")
        f.write(f"Total data transferred: {report['analysis_statistics']['data_transfer']['total_size']}\n\n")
        
        f.write("Individual Feed Details:\n")
        f.write("-" * 20 + "\n")
        for feed in report["feed_statistics"]["feeds_detail"]:
            f.write(f"\nFeed: {feed['feed_title']}\n")
            f.write(f"URL: {feed['feed_url']}\n")
            f.write(f"Articles: {feed['articles_count']}\n")
            if feed['oldest_article'] and feed['newest_article']:
                f.write(f"Date range: {feed['oldest_article']} to {feed['newest_article']}\n")
            f.write("\n")

    return report_file, summary_file

@app.route('/')
def index():
    """Display the latest analysis, grouped by category"""
    data = get_latest_analysis()
    
    # Group news by category
    if data.get('summary'):
        categorized_news = {}
        for item in data['summary']:
            # Extract category from the item's summary text
            category_line = [line for line in item['summary'].split('\n') 
                           if line.startswith('Category:')]
            category = 'Uncategorized'
            if category_line:
                category = category_line[0].replace('Category:', '').strip()
            
            if category not in categorized_news:
                categorized_news[category] = []
            categorized_news[category].append(item)
        
        # Sort categories alphabetically, but put "Uncategorized" at the end
        categories = list(categorized_news.keys())
        categories.sort(key=lambda x: ('Z' + x) if x == 'Uncategorized' else x)
        
        # Create sorted dictionary
        sorted_categories = {cat: categorized_news[cat] for cat in categories}
        
        # Update data structure
        data['categorized_summary'] = sorted_categories
    
    return render_template('index.html', data=data)

@app.route('/refresh', methods=['POST'])
def refresh_feeds():
    """Endpoint to trigger feed refresh"""
    try:
        # Load configurations
        configs = load_configs()
        
        # Fetch and save feeds
        feed_file = fetch_and_save_feeds(configs['main'])
        if not feed_file:
            return jsonify({'error': 'Failed to fetch feeds'}), 500
            
        # Load the feed data
        with open(feed_file, 'r', encoding='utf-8') as f:
            feed_data = json.load(f)
        
        # Initialize OpenAI handler and generate analysis
        openai_handler = OpenAISummaryHandler(configs['main'], configs['scoring'])
        analysis = openai_handler.generate_analysis(feed_data)
        
        # Save analysis
        output_file = save_analysis(analysis, configs['main'])
        
        # Load the saved analysis data
        with open(output_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # Generate report
        report_file, summary_file = generate_report(feed_data, analysis_data, configs['main'])
        
        logging.info(f"Report generated: {report_file}")
        logging.info(f"Summary generated: {summary_file}")
        
        return jsonify({'success': True, 'message': 'Feeds refreshed successfully'})
        
    except Exception as e:
        logging.error(f"Error in refresh_feeds: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.template_filter('datetime')
def format_datetime(value):
    """Format datetime string to a more readable format"""
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime('%B %d, %Y at %I:%M %p')
    except:
        return value

if __name__ == '__main__':
    app.run(debug=True) 