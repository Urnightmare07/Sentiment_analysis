# Sentiment_analysis

Overview
This project uses the YouTube API to scrape comments from a YouTube video and then applies sentiment analysis techniques to classify those comments as positive, negative, or neutral. The purpose of the project is to help content creators or marketers gauge the public opinion on videos and videos' content.

Features
Scrape comments from YouTube videos using the YouTube API

Perform sentiment analysis on the scraped comments

Classify comments as positive, negative, or neutral

Display sentiment distribution (e.g., pie charts, graphs)

Optionally, save results to a file for further analysis

Tech Stack
Python - Programming language

TensorFlow / PyTorch - Deep learning models for sentiment analysis

YouTube Data API - API for fetching video comments

pandas - For data manipulation and analysis

Matplotlib / Seaborn - For data visualization

TextBlob / NLTK / Huggingface Transformers - For sentiment analysis

Installation
Step 1: Set up a virtual environment
bash
python -m venv venv
Step 3: Install dependencies
pip install -r requirements.txt
Step 4: Get YouTube API credentials
Go to the Google Developer Console.

Create a new project and enable the YouTube Data API v3.

Generate your API key and replace it in config.py or as an environment variable.

Step 5: Configure the settings
Update config.py with the video ID you want to analyze.

Usage
1. Scrape YouTube Comments
You can use the following command to start scraping comments for sentiment analysis:

bash
Copy
Edit
python scrape_comments.py --video_id <YOUR_VIDEO_ID>
This command will fetch comments from the video and save them in a CSV file.

2. Analyze Sentiment
After scraping comments, you can analyze the sentiment by running:

bash
Copy
Edit
python sentiment_analysis.py --input_file comments.csv --output_file sentiment_results.csv
This will output the sentiment classification (positive, negative, neutral) for each comment.

3. Visualize Results
To generate a visualization of sentiment distribution, run:

bash
Copy
Edit
python visualize_results.py --input_file sentiment_results.csv
This will generate a pie chart or bar graph showing the distribution of positive, negative, and neutral sentiments.

Dependencies
requests - For making HTTP requests to the YouTube API

pandas - For data manipulation

numpy - For numerical operations

TextBlob - For basic sentiment analysis

transformers - For advanced transformer-based models (optional)

matplotlib / seaborn - For visualizations

Use the following command to install the required
