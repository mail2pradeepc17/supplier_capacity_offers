import os
import logging
import re
import pandas as pd
import concurrent.futures
from config import settings
import google.generativeai as genai
from flask import Flask, request, render_template
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Configure Gemini API
def get_gemini_api_key():
    return settings.GEMINI_API_KEY

GEMINI_API_KEY = get_gemini_api_key()
print("--->" + str(len(GEMINI_API_KEY)))  # check for token length

# Configure GenAI Model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")  # Using available model

# Load CSV data
csv_path = os.path.join(os.path.dirname(__file__), 'offers.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"offers.csv not found at {csv_path}")

df_offers = pd.read_csv(csv_path)
logging.info(f"Loaded {len(df_offers)} offers from CSV")

# Use ThreadPoolExecutor to avoid blocking Flask main thread
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def extract_locations(text):
    known_locations = {
        'pune', 'mumbai', 'nagpur', 'nashik', 'thane', 'aurangabad',
        'delhi', 'chennai', 'bangalore', 'hyderabad', 'kolkata'
    }
    
    # First check for explicit "from X to Y" patterns
    from_to_match = re.search(r'from\s+([a-z]+)\s+to\s+([a-z]+)', text.lower())
    if from_to_match:
        return [from_to_match.group(1), from_to_match.group(2)]
    
    # Then check for standalone locations
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word in known_locations]

def extract_tonnage(text):
    match = re.search(r'(\d+)\s*(?:tons?|tonnage|quintals?)', text.lower())
    if match:
        return int(match.group(1))
    return None

def extract_date(text):
    # Look for dates in various formats
    date_patterns = [
        r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})',  # DD/MM/YYYY or similar
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})',
        r'(\d{1,2})\s+(?:of\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?,?\s*(\d{4})?'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text.lower(), re.IGNORECASE)
        if match:
            try:
                if pattern == date_patterns[0]:  # Date in numeric format
                    day, month, year = match.groups()
                    day = int(day)
                    month = int(month)
                    year = int(year)
                    if year < 100:  # Handle 2-digit years
                        year += 2000
                    return datetime(year, month, day)
                else:  # Date with month name
                    day = int(match.group(2)) if pattern == date_patterns[1] else int(match.group(1))
                    month_str = match.group(1) if pattern == date_patterns[1] else match.group(2)
                    month = datetime.strptime(month_str[:3], '%b').month
                    year_str = match.group(3) if pattern == date_patterns[1] else match.group(3)
                    year = int(year_str) if year_str else None
                    if year and year < 100:
                        year += 2000
                    elif not year:
                        year = datetime.now().year  # Default to current year if not specified
                    return datetime(year, month, day)
            except ValueError:
                continue
    
    return None

# Match logic
def get_gemini_match_score(query_desc, offer_desc, offer_from=None, offer_to=None):
    logging.info(f"Scoring match:\nQuery: {query_desc}\nOffer: {offer_desc}")
    
    base_score = 0
    q = query_desc.lower()
    o = offer_desc.lower()

    # --- LOCATION MATCHING ---
    q_locs = extract_locations(q)
    o_locs = extract_locations(o)
    
    matched_locs = set(q_locs) & set(o_locs)
    base_score += len(matched_locs) * 2  # Base location match
    
    # --- DIRECTION MATCHING ---
    expected_from = None
    expected_to = None
    
    from_to_match = re.search(r'from\s+(\w+)\s+to\s+(\w+)', q)
    if from_to_match:
        expected_from = from_to_match.group(1).lower()
        expected_to = from_to_match.group(2).lower()
    
    # Check if direction matches
    direction_match = False
    reverse_direction = False
    
    if expected_from and expected_to and offer_from and offer_to:
        expected_from = expected_from.title()
        expected_to = expected_to.title()
        offer_from = offer_from.title()
        offer_to = offer_to.title()
        
        if expected_from == offer_from and expected_to == offer_to:
            direction_match = True
        elif expected_from == offer_to and expected_to == offer_from:
            reverse_direction = True
            
    if direction_match:
        base_score += 4  # Full bonus for matching direction
    elif reverse_direction:
        base_score -= 2  # Penalize reverse direction less severely
    
    # --- CAPACITY TYPE MATCHING ---
    if any(kw in q for kw in ['truck', 'transport', 'shipping']) and \
       any(kw in o for kw in ['truck', 'container', 'haul']):
        base_score += 2
        
    if any(kw in q for kw in ['storage', 'warehouse']) and \
       any(kw in o for kw in ['storage', 'warehouse']):
        base_score += 2
        
    if any(kw in q for kw in ['packaging', 'pack']) and \
       any(kw in o for kw in ['packaging', 'box', 'wrap']):
        base_score += 2
    
    # --- TONNAGE MATCHING ---
    q_ton = extract_tonnage(q)
    o_ton = extract_tonnage(o)
    
    if q_ton and o_ton:
        diff = abs(q_ton - o_ton)
        if diff == 0:
            base_score += 4
        elif diff <= 2:  # Smaller difference gets almost full points
            base_score += 3
        elif diff <= 5:  # Moderate difference gets some points
            base_score += 2
        elif diff <= 10:  # Larger difference still gets minor points
            base_score += 1
    
    # --- DATE MATCHING ---
    q_date = extract_date(q)
    o_date = extract_date(o)
    
    if q_date and o_date:
        days_diff = abs((q_date - o_date).days)
        if days_diff == 0:
            base_score += 3  # Exact date match
        elif days_diff <= 7:  # Within a week
            base_score += 2
        elif days_diff <= 15:  # Within two weeks
            base_score += 1
    
    # --- COMMON WORD BONUS ---
    q_words = set(re.findall(r'\b\w+\b', q))
    o_words = set(re.findall(r'\b\w+\b', o))
    common = q_words & o_words
    base_score += min(len(common), 5)  # Increased weight for common words
    
    final_score = max(0, min(10, base_score))
    
    logging.debug(f"Final score: {final_score} | Direction match: {direction_match}, Reverse: {reverse_direction}")
    return final_score

# Results display logic

def run_search(query):
    logging.info(f"Running search for query: '{query}'")
    
    # Normalize query
    query_lower = query.lower()
    
    # Step 1: Extract key information from query
    from_loc = None
    to_loc = None
    
    # Try to extract locations from query using improved method
    locations = extract_locations(query_lower)
    if len(locations) >= 2:
        from_loc = locations[0].title()
        to_loc = locations[1].title()
    
    # Extract tonnage requirement
    q_ton = extract_tonnage(query_lower)
    
    # Extract date requirement
    q_date = extract_date(query_lower)
    
    # Step 2: Filter by location if we have both locations
    if from_loc and to_loc:
        filtered = df_offers[
            (df_offers['location_from'].str.contains(from_loc, case=False)) &
            (df_offers['location_to'].str.contains(to_loc, case=False))
        ]
    else:
        # Fallback: try basic substring match
        filtered = df_offers[df_offers['description'].str.contains(query, case=False, na=False)]
    
    if len(filtered) == 0:
        logging.warning("No pre-filtered matches found. Using fallback: location-based match only.")
        
        # Fallback: use all offers from known locations
        possible_locations = [loc for loc in ['pune', 'mumbai', 'nagpur', 'nashik', 'aurangabad', 'thane'] 
                             if loc in query_lower]
        if possible_locations:
            location_filter = df_offers['location_from'].apply(lambda x: any(loc in str(x).lower() for loc in possible_locations)) | \
                             df_offers['location_to'].apply(lambda x: any(loc in str(x).lower() for loc in possible_locations))
            filtered = df_offers[location_filter]
    
    if len(filtered) == 0:
        logging.warning("Still no matches found after fallback.")
        return []
    
    # Step 3: Gemini scoring only on filtered set
    filtered = filtered.head(50)  # Limit to 50 for performance
    
    # Add additional context to the description based on query parameters
    def score_with_context(desc):
        additional_context = ""
        if q_ton:
            additional_context += f"Target tonnage: {q_ton} tons. "
        if q_date:
            additional_context += f"Target date: {q_date.strftime('%Y-%m-%d')}. "
        
        return get_gemini_match_score(f"{query} {additional_context}", desc)
    
    filtered['relevance'] = filtered['description'].apply(score_with_context)
    
    matches = filtered.sort_values(by='relevance', ascending=False).head(10)
    logging.info(f"Found top {len(matches)} matches.")
    return matches.to_dict(orient='records')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_query = request.form['query']
        return search_matches(user_query)
    return render_template('index.html')


@app.route('/search')
def search_matches():
    query = request.args.get('q')
    
    if not query or len(query.strip()) < 5:
        logging.warning("Invalid query input")
        return "Please enter a valid search query.", 400
    
    try:
        future = executor.submit(run_search, query)
        matches = future.result(timeout=10)  # Reduced timeout to 10 seconds
        return render_template('results.html', query=query, matches=matches)
    except concurrent.futures.TimeoutError:
        logging.error("Gemini match timed out")
        return "Server timeout while processing request. Try refining your query.", 504
    except Exception as e:
        logging.exception("Unexpected error during search")
        return f"Internal server error: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True, threaded=True)