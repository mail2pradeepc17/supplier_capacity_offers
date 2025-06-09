import os
import logging
import re
import pandas as pd
import concurrent.futures
from config import settings
import google.generativeai as genai
from flask import Flask, request, render_template

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

#  Configure Gemini API
def get_gemini_api_key():
    return settings.GEMINI_API_KEY

GEMINI_API_KEY = get_gemini_api_key()
print("--->" + str(len(GEMINI_API_KEY)))  # check for token length

# Configure GenAI Model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")     # Gemini 2.0 Flash

# Load CSV data
csv_path = os.path.join(os.path.dirname(__file__), 'offers.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"offers.csv not found at {csv_path}")

df_offers = pd.read_csv(csv_path)
logging.info(f"Loaded {len(df_offers)} offers from CSV")

# Use ThreadPoolExecutor to avoid blocking Flask main thread
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def extract_locations(text):
    """Extract location names from text (simple word matching)"""
    # You can expand this list as needed
    known_locations = {
        'pune', 'mumbai', 'nagpur', 'nashik', 'thane', 'aurangabad',
        'delhi', 'chennai', 'bangalore', 'hyderabad', 'kolkata'
    }
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word in known_locations]

def extract_tonnage(text):
    """Extract numeric tonnage from text"""
    match = re.search(r'(\d+)\s*(?:tons?|tonnage|quintals?)', text.lower())
    if match:
        return int(match.group(1))
    return None

def get_gemini_match_score(query_desc, offer_desc):
    """
    Simulate Gemini match score using smart keyword + numeric matching
    """
    logging.info(f"Scoring match:\nQuery: {query_desc}\nOffer: {offer_desc}")

    score = 0

    q = query_desc.lower()
    o = offer_desc.lower()

    # --- LOCATION MATCHING (Dynamic!) ---
    q_locs = extract_locations(q)
    o_locs = extract_locations(o)

    matched_locs = set(q_locs) & set(o_locs)
    score += len(matched_locs) * 3  # +3 per matched location

    # --- CAPACITY TYPE MATCHING ---
    if any(kw in q for kw in ['truck', 'transport', 'shipping']) and \
       any(kw in o for kw in ['truck', 'container', 'haul']):
        score += 2

    if any(kw in q for kw in ['storage', 'warehouse']) and \
       any(kw in o for kw in ['storage', 'warehouse']):
        score += 2

    if any(kw in q for kw in ['packaging', 'pack']) and \
       any(kw in o for kw in ['packaging', 'box', 'wrap']):
        score += 2

    # --- TONNAGE MATCHING ---
    q_ton = extract_tonnage(q)
    o_ton = extract_tonnage(o)

    if q_ton and o_ton:
        diff = abs(q_ton - o_ton)
        if diff == 0:
            score += 4  # Perfect match
        elif diff <= 5:
            score += 3  # Close match
        else:
            score += 1  # At least in ballpark

    # --- COMMON WORD BONUS ---
    q_words = set(re.findall(r'\b\w+\b', q))
    o_words = set(re.findall(r'\b\w+\b', o))
    common = q_words & o_words
    score += min(len(common), 3)  # Up to +3 for keyword overlap

    # Clamp score between 0 and 10
    return max(0, min(10, score))

def run_search(query):
    logging.info(f"Running search for query: '{query}'")

    # Normalize query
    query_lower = query.lower()

    # Step 1: Location-based filter first (from/to)
    from_loc = None
    to_loc = None

    # Try to extract locations from query
    for loc in ['pune', 'mumbai', 'nagpur', 'nashik', 'aurangabad', 'thane']:
        if loc in query_lower.split():
            if from_loc is None:
                from_loc = loc.title()
            elif to_loc is None:
                to_loc = loc.title()

    # Filter by location if we can detect both
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
        if from_loc:
            filtered = df_offers[df_offers['location_from'].str.contains(from_loc, case=False)]

    if len(filtered) == 0:
        logging.warning("Still no matches found after fallback.")
        return []

    # Step 2: Gemini scoring only on filtered set
    filtered = filtered.head(50)  # Limit to 50 for performance
    filtered['relevance'] = filtered['description'].apply(
        lambda desc: get_gemini_match_score(query, desc)
    )

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