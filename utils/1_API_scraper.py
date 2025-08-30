import os
import requests
import brotli
import json
import re
import pandas as pd
import time

#################################################################
############################ HEADERS ############################

HEADERS = {
    "accept": "application/json",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "en-US,en;q=0.9",
    "dnt": "1",
    # The 'if-none-match' header has been removed to ensure the server
    # always sends the full data with a 200 OK status.
    "priority": "u=1, i",
    "referer": "https://ifsc.results.info/event/1442/",
    "sec-ch-ua": '"Chromium";v="139", "Not;A=Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0.0.0 Safari/537.36"
    ),
    "x-csrf-token": "-NI_zRsP_QmfGvAFg20awKnhXuNKbnxWDXrqFjUlQdzZAur00dAAZ29hz9nBbdnhpUAL8FIcTeDRaJKfidTOCg",
}

# ---------- COOKIES ----------
COOKIES_STRING = (
    "_ga=GA1.2.730144941.1748434369; _gid=GA1.2.482734398.1756487631; _gat_UA-157153964-4=1; _verticallife_resultservice_session=nrbwkpfOjl8W5IiYs1hFWC8H3GaUi0IppQvXrWcKA9pUfQwxmK9SJBntlSdsP9g7cBJHBYjgXk0E%2F1%2BtDTnABdUE1G526mTNvLQqrgt%2F3xpSHRrjdBakAGPirBYR5hFnvUBJnmNXCIkxR6uIIRnpAIblBd3s3zYQOBZoB0OqByWaYW166oJsEDIdoCvi2mX98eAHAPfisQQqg%2BOGNrVnbDoPmATeVQTP0msEQdbJl2HfNiB54HSytLjD%2FRz7iRsbAWG15uFsvj71dPDw3NVv2w6F%2BKMncAIEOq%2F90ynCaaJGpst8deR831AzrA%3D%3D--GFtsaOQbx42hE1Qu--k6c7JRlfmPETwfxB19MVCA%3D%3D; _ga_3CW645GZWB=GS2.2.s1756487631$o68$g1$t1756489467$j60$l0$h0"
)

COOKIES = {key: value for part in COOKIES_STRING.split("; ") if "=" in part for key, value in [part.split("=", 1)]}

# ---------- API HELPER with Brotli Decompression ----------
BASE_API = "https://ifsc.results.info/"



#################################################################
######################### GET_API_DATA ##########################


def get_api_data(endpoint: str = "") -> dict:
    """
    Fetch and decode JSON data from an IFSC API endpoint, handling Brotli compression.
    If endpoint is relative (e.g., '/seasons'), it is joined with BASE_API.
    """
    url = endpoint if endpoint.startswith("http") else BASE_API + endpoint.lstrip("/")
    response = requests.get(url, headers=HEADERS, cookies=COOKIES)

    response.raise_for_status()

    json_text = ""

    # Check for Brotli compression and attempt to decompress
    if response.headers.get('Content-Encoding') == 'br':
        try:
            decompressed_content = brotli.decompress(response.content)
            json_text = decompressed_content.decode('utf-8')
        except brotli.error:
            json_text = response.text
        except UnicodeDecodeError as e:
            # Handle decoding errors
            print(f"Unicode decode error: {e}")
            return {}
    else:
        # If not Brotli, use the standard response text (requests handles gzip/deflate)
        json_text = response.text

    # Check for empty response body after decompression and before JSON parsing
    if not json_text:
        print(f"Warning: Received 200 OK but the response body for '{url}' is empty or could not be decompressed.")
        return {}

    try:
        # Attempt to parse the JSON text
        return json.loads(json_text)
    except json.JSONDecodeError as err:
        print(f"JSON Decode Error: {err} from URL: {url}")
        print("Response text was:", json_text)
        return {}


#################################################################
######################### GET_ALL_YEARS #########################

def get_worldcup_leagues() -> pd.DataFrame:
    """
    Fetches all seasons and filters for "World Cups" leagues that are not youth events.
    """
    info_data = get_api_data("api/v1/")
    seasons = info_data['seasons']
    results = []

    for s in seasons:
        year = s['name']
        
        for league in s['leagues']:
            
            # 4. Filter for only World Cups
            if 'World Cups' in league['name'] and 'Youth' not in league['name']:
                results.append({
                    "year": int(year),
                    "league_name": league['name'],
                    "url": league['url']
                    })
    df = pd.DataFrame(results)
    df.to_csv('leagues.csv', index=False)
    return df


#################################################################
################### GET_ALL_EVENTS_IN_A_YEAR ####################

def _clean_location(event_name: str, year: int, event_url: str) -> str:
    """
    Extracts and cleans the location name from an event name based on the year.
    This handles historical data inconsistencies in the API.
    # Could be faster, but idk where tf is 'location' in data
    # HARD-FUCKING CODE
    """
    
    if 1990 < year <= 1997:
        # Extract text between '-' and a number.
        match = re.search(r'-(.*?)\d', event_name)
        return match.group(1).strip() if match else event_name
    elif 1998 <= year <= 2008:
        # Extract text between '-' and '('.
        return event_name.split('-')[1].split('(')[0].strip()
    
    else: # Default behavior for recent years (>= 2007)
        
        # fetch from the event's API .
        location = get_api_data(event_url).get('location', '')

        cleaned = re.sub(r'(WCH|WC|Wc)', '', location).strip()
        location = re.sub(r'\d+', '', cleaned)
        
        # Edge cases
        if 'combined' in event_name.lower():
            return event_name.split('-')[1].split('(')[0].strip()
        if event_name == 'IFSC Climbing World Championships  - Qinghai (CHN) 2009 - 15m Speed':
            return 'Qinghai'
            
        return location
    
def _process_disciplines(event_data: dict, base_event: dict) -> list:
    """
    Processes the disciplines(bouldering, lead, and speed) within a single event and returns a list of dictionaries
    for each discipline/round.
    """
    events_list = []
    
    # Loop through each discipline within the event
    for discipline in event_data['d_cats']:
        discipline_name, gender = discipline['name'].split()
        
        # For UIAA-era events (1990-2006)
        if 1990 < base_event['year'] < 2007:
            events_list.append({
                **base_event,
                "discipline": discipline_name.capitalize(),
                "gender": gender,
                "event_results": discipline['result_url']
                # no round_results and scores
            })
        
        # IFSC-era events
        elif base_event['year'] >= 2007:
            # Loop through each round (qualis, semis, finals)
            for round in discipline['category_rounds']:
                events_list.append({
                    **base_event,
                    "discipline": discipline_name.capitalize(),
                    "gender": gender,
                    "round": round['name'],
                    "event_results": discipline['result_url'],
                    "category_round_results": round['result_url'],
                })
    return events_list
def get_events_from_league(league_url: str) -> pd.DataFrame:
    """
    Retrieves all events from a given league URL, extracts key information,
    and returns a DataFrame.
    
    Args:
        league_url: The API endpoint for a specific league (e.g., 'leagues/228').
    
    Returns:
        A pandas DataFrame of processed events.
    """
    data = get_api_data(league_url)
    year = int(data['season'])
    
    all_events = []
    # Loop through each event(salt-lake, chamonix, seoul, ...) in the league
    for event in data['events']:
        base_event_data = {
            "event_name": event['event'],
            "event_id": event['event_id'],
            "year": year,
            "start_date": event['local_start_date'],
            "event_results": event.get('result_url', None),
        }
        
        base_event_data['location'] = _clean_location(base_event_data['event_name'], year, event['url'])

        # Process the disciplines and rounds 
        all_events.extend(_process_disciplines(event, base_event_data))

    # Save and return the DataFrame
    os.makedirs('API_Event_metadata', exist_ok=True)
    filename = f"{year}_event_meta"
    df = pd.DataFrame(all_events)
    
    column_order = ['event_name', 'event_id', 'year', 'location', 'discipline', 'gender', 'round', 'start_date', 'category_round_results', 'event_results']
    # Reorder columns and handle missing ones (e.g., 'round' for older data)
    df = df[[c for c in column_order if c in df.columns]]
    
    df.to_csv(f"./API_Event_metadata/{filename}.csv")
    print(f"✅ Saved {filename}")
    
    return df



#################################################################
################## GET_RESULTS_FROM_EACH_EVENT ##################

def _create_base_row(athlete: dict, event_meta: dict) -> dict:
    """Creates a base row dictionary for an athlete's result."""
    base_row = {
        "name": athlete.get('name', ''),
        "country": athlete.get('country', ''),
        "round_rank": int(athlete['rank']) if athlete.get('rank') is not None else None,
        "round_score": " ".join(athlete.get('score', '').split()),
    }
    base_row.update(event_meta)
    return base_row

def _process_speed_final(athlete: dict, row: dict) -> dict:
    """Parses speed final results from elimination stages."""
    for stage in athlete.get('speed_elimination_stages', []):
        row[f"{stage['name']}_winner"] = stage.get('winner') == 1
        try:
            row[f"{stage['name']}_time"] = stage['time'] / 1000 if stage.get('time') else stage.get('score')
        except Exception as e:
            print(f"⚠️ Error parsing speed stage for athlete {athlete.get('name', '')}: {e}")
    return row

def _process_combined_stages(athlete: dict, row: dict) -> dict:
    """Parses combined (Boulder&Lead) stage results."""
    for stage in athlete.get('combined_stages', []):
        if 'ascents' not in stage:
            row[f"{stage['stage_name']}_score"] = stage['stage_score']
            row[f"{stage['stage_name']}_rank"] = stage['stage_rank']
        elif stage['stage_name'] == 'Boulder':
            for ascent in stage.get('ascents', []):
                route_name = ascent.get("route_name", "")
                digits = ''.join(filter(str.isdigit, route_name))
                if not digits:
                    continue
                p = int(digits)
                row[f"P{p}_Top"] = ascent["top_tries"] if ascent.get("top") else "X"
                row[f"P{p}_Zone"] = ascent["zone_tries"] if ascent.get("zone") else "X"
                if ascent.get("low_zone"):
                    row[f"P{p}_LowZone"] = ascent.get("low_zone_tries", "X")
        elif stage['stage_name'] == 'Lead':
            for ascent in stage.get('ascents', []):
                route_name = ascent.get("route_name", "")
                digits = ''.join(filter(str.isdigit, route_name))
                if not digits:
                    continue
                p = int(digits)
                row[f"Route_{p}"] = ascent["score"]
    return row

def _process_lead_pre_2020(athlete: dict, row: dict) -> dict:
    """Parses Lead results for events up to 2019."""
    scores = athlete.get('score', '')
    if "|" in scores:
        route_A, route_B = [i.split()[0] for i in scores.split("|")]
        row[f"Route_1"] = route_A
        row[f"Route_2"] = route_B
    else:
        row[f"Route_1"] = scores.split()[0]
    return row

def _process_ascents(athlete: dict, event: dict, row: dict) -> dict:
    """Parses ascent-based results for Boulder, Lead, and Speed (non-final)."""
    for ascent in athlete.get('ascents', []):
        route_name = ascent.get("route_name", "")
        if event['discipline'] == 'Boulder':
            digits = ''.join(filter(str.isdigit, route_name))
            if not digits:
                continue
            p = int(digits)
            row[f"P{p}_Top"] = ascent["top_tries"] if ascent.get("top") else "X"
            row[f"P{p}_Zone"] = ascent["zone_tries"] if ascent.get("zone") else "X"
        elif event['discipline'] == 'Lead':
            digits = ''.join(filter(str.isdigit, route_name))
            if not digits:
                continue
            p = int(digits)
            row[f"Route_{p}"] = ascent["score"]
        elif event['discipline'] == 'Speed':
            row[f"Quali_time_{ascent['route_name']}"] = (
                ascent['time_ms'] / 1000 if ascent.get('time_ms')
                else 'DNS' if ascent.get('dns')
                else 'DNF' if ascent.get('dnf')
                else None
            )
    return row

def parse_round_result(event: dict) -> pd.DataFrame:
    """
    Parses a single event's round results and saves them to a CSV file.
    The function handles different data structures for UIAA and IFSC eras.
    
    Args:
        event: A dictionary containing event metadata and URLs.
        
    Returns:
        A pandas DataFrame with the parsed results.
    """
    if event['year'] <= 2006:
        print(f"🔍 Parsing UIAA: {event['year']} | {event.get('location', '')} | {event['discipline']} | {event['gender']}")
        result_url = event['event_results']
    else:
        print(f"🔍 Parsing IFSC: {event['year']} | {event.get('location', '')} | {event['discipline']} | {event['gender']} | {event.get('round', 'N/A')}")
        result_url = event['category_round_results']

    data = get_api_data(result_url)
    if 'cancel' in data.get('event', '').lower():
        print(f'📘Empty: {event["start_date"]}_{event.get("location", "")}_{event["discipline"]}_{event["gender"]}')
        return pd.DataFrame()

    results = []
    # Loop through each athlete's result
    for athlete in data.get('ranking', []):
        # Create a base row with name, country, and rank/score
        row = _create_base_row(athlete, event)
        
        # Apply discipline-specific parsing logic
        if event['year'] <= 2006:
            # For UIAA events, only the base row is needed
            results.append(row)
        elif event['discipline'] == 'Speed' and event.get("round") == 'Final':
            row = _process_speed_final(athlete, row)
            results.append(row)
        elif event['discipline'] == 'Boulder&lead':
            row = _process_combined_stages(athlete, row)
            results.append(row)
        elif event['discipline'] == 'Lead' and event['year'] <= 2019:
            row = _process_lead_pre_2020(athlete, row)
            results.append(row)
        else:
            row = _process_ascents(athlete, event, row)
            results.append(row)
    
    # Create the DataFrame and save it to a CSV
    df = pd.DataFrame(results)
    
    filename = f"{event['start_date']}_{event.get('location', 'no-loc')}_{event['discipline']}_{event['gender']}_{event.get('round', 'no-round')}"
    os.makedirs(f"API_Results_Expanded/{event['year']}/", exist_ok=True)
    df.to_csv(f"./API_Results_Expanded/{event['year']}/{filename}.csv", index=False)
    
    print(f"✅ Saved: {filename}")
    return df




# ---------- USAGE ----------
if __name__ == "__main__":
    # # verify that api is working
    # data = get_api_data("api/v1/events/1361")
    # print(data.keys())
    
    leagues_df = pd.read_csv('leagues.csv')
    start_time = time.time()
    for index, row in leagues_df.iterrows():
        year = row['year']
        league_url = row['url']
        
        print(f"\n--- Processing all events for {year} World Cup leagues ---")
        try:
            events_df = get_events_from_league(league_url)
            print(f"Successfully fetched {len(events_df)} events for {year}.")

            if not events_df.empty:
                print(f"--- Parsing round results for all events in {year} ---")
                for event_index, event_row in events_df.iterrows():
                    print(f"  > Processing event {event_index + 1}/{len(events_df)} for {year}: {event_row['event_name']}")
                    parse_round_result(event_row.to_dict())
            else:
                print(f"No events found for {year}.")

        except requests.exceptions.RequestException as e:
            print(f"⚠️ An error occurred while fetching data for {year}: {e}")
        except Exception as e:
            print(f"⚠️ An unexpected error occurred while processing data for {year}: {e}")

    
    print(f"Fetching took a while, it was: {int((time.time()-start_time)//60)} minutes")