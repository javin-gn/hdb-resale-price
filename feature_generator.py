import os
import json
import time
import io
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from datetime import datetime
import re
import pyproj
import math

def FeatureGenerator(address): #Code for dagster
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  #Code for dagster
    coords_cache_path = os.path.join(SCRIPT_DIR, "data/coords_cache.json") #Code for dagster
    mrt_cache_path = os.path.join(SCRIPT_DIR, "data/mrt_lrt_cache.json") #Code for dagster
    schools_cache_path = os.path.join(SCRIPT_DIR, "data/primary_schools_cache.json") #Code for dagster
    shopping_malls_cache_path = os.path.join(SCRIPT_DIR, "data/shopping_malls_cache.json") #Code for dagster
   
    
    print("🚀 Starting the HDB Resale Data Pipeline...") #Code for dagster

    load_dotenv()

    print("""
    # ===============================================================================================================================
    # STEP 1: Connect to OneMap API and Retrieve Geospatial Coordinates for HDB Address Dataset
    # ===============================================================================================================================
    """)

    df = pd.DataFrame()

    def get_current_onemap_token():
        """
        Automated credential authentication wrapper for the OneMap gateway engine.
        """
        print("🔑 Authenticating developer credentials with OneMap Security Gateway...")
        login_url = "https://www.onemap.gov.sg/api/auth/post/getToken"
       

        # Securely retrieve constants from your root `.env` system environment file
        email = os.getenv("ONEMAP_EMAIL")
        password = os.getenv("ONEMAP_PASSWORD")

        if not email or not password:
            raise ValueError(
                "❌ Missing Credentials! Ensure ONEMAP_EMAIL and ONEMAP_PASSWORD "
                "are defined inside your workspace root `.env` configuration file."
            )
        
        payload = {
            "email" : email,
            "password" : password
        }

        try:
            response_obj = requests.post(login_url, json=payload)
            if response_obj.status_code != 200:
                print(f"❌ Handshake Denied: Status {response_obj.status_code}")
                return None
            
            response = response_obj.json()
            token = response.get("access_token")
            print("✅ Handshake successful. Valid 72-hour API token generated.")
            return token
        
        except Exception as e:
            print(f"❌ Security gateway transmission failed: {e}")
            return None

    access_token = get_current_onemap_token()
    headers = {"Authorization": access_token} 
    

    if os.path.exists(coords_cache_path):
        with open(coords_cache_path, "r") as f:
            cache = json.load(f)
    else:
        cache = {}
    
    if access_token:

        row_data = {}


        search_url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=Y&pageNum=1"

        try:
            res = requests.get(search_url, headers=headers).json()
            
            if res.get("results") and len(res["results"]) > 0:
                result_node = res["results"][0]
                
                # 3. Update the cache dictionary
                cache[address] = {
                    "lat": float(result_node["LATITUDE"]),
                    "lon": float(result_node["LONGITUDE"]),
                    "x": float(result_node["X"]),
                    "y": float(result_node["Y"])
                }
                
                # 4. Persist the updated cache to the file
                with open(coords_cache_path, "w") as f:
                    json.dump(cache, f, indent=4)
                    
                print(f"Successfully geocoded '{address}' and updated cache.")
            else:
                print(f"No results found for: {address}")

        except Exception as e:
            print(f"An error occurred while geocoding: {e}")
                
    
 
    row_data = cache[address].copy()
    row_data["full_address"] = address
    
    # Create the DataFrame directly from the data
    df = pd.DataFrame([row_data])
    # Map coordinates from cache directly back into the dataframe rows
    df["lat"] = df["full_address"].map(lambda x: cache.get(x, {}).get("lat", np.nan))
    df["lon"] = df["full_address"].map(lambda x: cache.get(x, {}).get("lon", np.nan))
    df["x"] = df["full_address"].map(lambda x: cache.get(x, {}).get("x", np.nan)) # ◄ NEW
    df["y"] = df["full_address"].map(lambda x: cache.get(x, {}).get("y", np.nan)) # ◄ NEW

    # Drop rows that failed to geocode to prevent math errors
    df = df.dropna(subset=["lat", "lon"]).copy()
    df = df.dropna(subset=["x", "y"]).copy()
    
  

    print("""
    # ===============================================================================================================================
    # STEP 2: Polycentric Matrix Math Engine 
    # ===============================================================================================================================
    """)
    # UPDATED GOVT POLICY ECONOMIC NODES (URA Master Plan Hierarchy)

    # --- REVISED HUBS DICTIONARY (SVY21 METRES) ---
    hubs = {
        "Jurong_Lake_District": (18033.4, 35359.6),
        "Tampines_Regional_Centre": (41243.5, 36980.2),
        "Woodlands_Regional_Centre": (22802.1, 44710.8),
        "Paya_Lebar_Central": (34820.6, 34105.9),
        "One_North_Hub": (23348.9, 31711.2),
        "Punggol_Digital_District": (35720.4, 41920.5),
        "Bishan_Sub_Regional_Centre": (29718.3, 37280.4),
        "Seletar_Aerospace_Park": (31740.1, 42510.9),
        "Changi_Business_Park": (43720.8, 34810.1),
        "International_Business_Park": (18810.2, 35120.7)
    }

    # --- AUTHORITATIVE NON-"SHOPPING" RETAIL NODES (SVY21 METRES) ---
    # --- AUTHORITATIVE NON-"SHOPPING" RETAIL NODES (CORRECTED SVY21 METRES) ---
    FIXED_MALLS = [
        {"mall_name": "Amk Hub", "mall_x": 29184.22, "mall_y": 39105.82}, 
        {"mall_name": "Ion Orchard", "mall_x": 27807.51, "mall_y": 32395.71},
        {"mall_name": "Ngee Ann City", "mall_x": 27931.33, "mall_y": 32247.16},
        {"mall_name": "Vivocity", "mall_x": 26458.74, "mall_y": 29013.91},
        {"mall_name": "Plaza Singapura", "mall_x": 29088.22, "mall_y": 32881.54},
        {"mall_name": "Bugis Junction", "mall_x": 30200.70, "mall_y": 32857.73},
        {"mall_name": "Nex", "mall_x": 32306.21, "mall_y": 36743.79},
        {"mall_name": "Jem", "mall_x": 17972.14, "mall_y": 35431.11},
        {"mall_name": "Westgate", "mall_x": 17871.91, "mall_y": 35572.82},
        {"mall_name": "Tampines Mall", "mall_x": 41280.55, "mall_y": 36802.40}
    ]


    ###############################################
    # Fetch Primary School from OneMap API or cache
    ###############################################
    def fetch_primary_schools_svy21(token):
    # Define a tracking path inside your working repository
        # cache_file = "scripts/primary_schools_cache.json" # dagster
        cache_file = schools_cache_path
        
        # ─── STEP A: CHECK LOCAL STORAGE CACHE WITH 30-DAY TTL ───
        # 30 days in seconds = 30 * 24 * 60 * 60 = 2,592,000 seconds
        EXPIRATION_PERIOD_SECONDS = 2592000

        if os.path.exists(cache_file):
            # Calculate how many seconds ago the file was modified
            file_age_seconds = time.time() - os.path.getmtime(cache_file)

            if file_age_seconds < EXPIRATION_PERIOD_SECONDS:
                print("\n💾 Loading Primary School data from local JSON storage cache...")
                try:
                    with open(cache_file, "r") as f:
                        cached_data = json.load(f)
                    df_sch = pd.DataFrame(cached_data)
                    print(f"✔ Cache hit. Loaded {len(df_sch)} official primary schools instantly.")
                    return df_sch
                except Exception as e:
                    print(f"⚠ Cache corrupted, falling back to network request: {e}")
            
            else:
                days_old = round(file_age_seconds / (24 * 3600), 1)
                print(f"\n⏳ School cache file exists but is outdated ({days_old} days old). Triggering auto-refresh...")

        # ─── STEP B: FALLBACK TO LIVE NETWORK CALLS ───
        print("\n🌐 Cache miss! Downloading Primary School dataset from official OneMap API...")

        schools = []
        current_page = 1
        total_pages = 1
        headers = {"Authorization": token} 
        
        while current_page <= total_pages:
            # Construct the URL with the active page number variable    
            url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal=PRIMARY%20SCHOOL&returnGeom=Y&getAddrDetails=Y&pageNum={current_page}"   
        
            try:
                response = requests.get(url, headers=headers)
            
                # Guard check: Ensure the server returned a valid 200 OK code
                if response.status_code != 200:
                    print(f"❌ Server Error on Page {current_page}: Received status code {response.status_code}")
                    break
                
                res = response.json()
                # pprint(res)
        
                # Update total pages dynamically from the API's first metadata response
                if current_page == 1:
                    total_pages = int(res.get("totalNumPages", 1))
                    print(f"Total pages to retrieve: {total_pages}")

                # Extract items from results dictionary array
                items = res.get("results", [])
                for item in items:
                    name = item.get("SEARCHVAL", "").upper()

                    # 2. Establish strict string exclusions for student care, enrichment, and preschools
                    is_student_care = "STUDENT CARE" in name or "ENRICHMENT" in name or "PRESCHOOL" in name

                    # Verify that it is an official Primary School asset
                    if "PRIMARY SCHOOL" in name and not is_student_care:
                        schools.append({
                            "school_name": name.title(),
                            "sch_x": float(item["X"]),
                            "sch_y": float(item["Y"])
                        })

                print(f"Processed Page {current_page}/{total_pages}...")
                current_page += 1
                time.sleep(0.2)  # Short pause to satisfy OneMap rate-limiting rules

            except Exception as e:
                print(f"❌ Connection or JSON parsing failed on page {current_page}: {e}")
                break

        df_sch = pd.DataFrame(schools)
        print(f"✅ Filter complete. Isolated {len(df_sch)} official primary schools.")

        # ─── STEP C: SAVE DOWNLOADED OBJECT TO CACHE ───
        if not df_sch.empty:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                # Convert dataframe records to a clean dictionary list object for tracking
                with open(cache_file, "w") as f:
                    json.dump(df_sch.to_dict(orient="records"), f, indent=4)
                print(f"💾 Successfully saved schools collection cache to: {cache_file}")
            except Exception as e:
                print(f"⚠ Failed to save local backup file: {e}")
                
        return df_sch

    ###############################################
    # Fetch MRT/LRT from OneMap API or cache
    ###############################################
    def fetch_MRT_LRT_svy21(token):
    # Define a tracking path inside your working repository
        # cache_file = "scripts/mrt_lrt_cache.json" # dagster
        cache_file = mrt_cache_path
        
        # ─── STEP A: CHECK LOCAL STORAGE CACHE WITH 30-DAY TTL ───
        # 30 days in seconds = 30 * 24 * 60 * 60 = 2,592,000 seconds
        EXPIRATION_PERIOD_SECONDS = 2592000

        if os.path.exists(cache_file):
            # Calculate how many seconds ago the file was modifled
            file_age_seconds = time.time() - os.path.getmtime(cache_file)

            if file_age_seconds < EXPIRATION_PERIOD_SECONDS:            
                print("\n💾 Loading MRT/LRT data from local JSON storage cache...")
                try:
                    with open(cache_file, "r") as f:
                        cached_data = json.load(f)
                    df_stations = pd.DataFrame(cached_data)
                    print(f"✔ Cache hit. Loaded {len(df_stations)} MRT/LRT instantly.")
                    return df_stations
                except Exception as e:
                    print(f"⚠ Cache corrupted, falling back to network request: {e}")

            else:
                days_old = round(file_age_seconds / (24*3600),1)
                print(f"\n⏳ Cache file exists but is outdated ({days_old} days old). Triggering auto-refresh...")

        # ─── STEP B: FALLBACK TO LIVE NETWORK CALLS ───
        print("\n🌐 Cache miss! Downloading MRT/LRT dataset from official OneMap API...")

        search_queries = ["MRT STATION", "LRT STATION"]
        stations_dict = {}
        headers = {"Authorization": token} 

        for query in search_queries:
            encoded_query = query.replace(" ", "%20")
            current_page = 1
            total_pages = 1
        
            print(f"🛰 Scanning endpoint layer for: {query}")

            while current_page <= total_pages:
                # Construct the URL with the active page number variable    
                url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={encoded_query}&returnGeom=Y&getAddrDetails=Y&pageNum={current_page}"   
            
                try:
                    response = requests.get(url, headers=headers)
                
                    # Guard check: Ensure the server returned a valid 200 OK code
                    if response.status_code != 200:
                        print(f"❌ Server Error on Page {current_page}: Received status code {response.status_code}")
                        break
                    
                    res = response.json()
                    # pprint(res)
            
                    # Update total pages dynamically from the API's first metadata response
                    if current_page == 1:
                        total_pages = int(res.get("totalNumPages", 1))
                        print(f"Total pages to retrieve: {total_pages}")

                    # Extract items from results dictionary array
                    items = res.get("results", [])
                    for item in items:
                        name = item.get("SEARCHVAL", "").upper()

                        # 2. Establish strict string exclusions for student care, enrichment, and preschools
                        # is_student_care = "STUDENT CARE" in name or "ENRICHMENT" in name or "PRESCHOOL" in name

                        # 🌟 EXCLUSION FLAGS: Identify exits, bus terminals, and depots
                        is_station = "MRT STATION" in name or "LRT STATION" in name
                        is_noise = "EXIT" in name or "BUS INTERCHANGE" in name or "DEPOT" in name
                        # Verify that it is an official Primary School asset
                        if is_station and not is_noise:
                            # Title case the name cleanly (e.g., "Ang Mo Kio Mrt Station")
                            clean_name = name.strip().title()

                            stations_dict[clean_name] = {
                                "station_name": clean_name,
                                "station_x": float(item["X"]),
                                "station_y": float(item["Y"])
                            }

                    print(f"Processed Page {current_page}/{total_pages}...")
                    current_page += 1
                    time.sleep(0.2)  # Short pause to satisfy OneMap rate-limiting rules

                except Exception as e:
                    print(f"❌ Connection or JSON parsing failed on page {current_page}: {e}")
                    break

        df_stations = pd.DataFrame(list(stations_dict.values()))
        print(f"✅ Filter complete. Isolated {len(df_stations)} MRT/LRT stations.")

        # ─── STEP C: SAVE DOWNLOADED OBJECT TO CACHE ───
        if not df_stations.empty:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                # Convert dataframe records to a clean dictionary list object for tracking
                with open(cache_file, "w") as f:
                    json.dump(df_stations.to_dict(orient="records"), f, indent=4)
                print(f"💾 Successfully saved schools collection cache to: {cache_file}")
            except Exception as e:
                print(f"⚠ Failed to save local backup file: {e}")
                
        return df_stations


    def fetch_shopping_malls_svy21(token):
        """
        Automated Themes API ingestion engine with 30-Day TTL caching layer 
        built exclusively to map Singapore's shopping mall network infrastructure.
        """
        # ─── STEP A: CHECK LOCAL STORAGE CACHE WITH 30-DAY TTL ───
        # 30 days in seconds = 30 * 24 * 60 * 60 = 2,592,000 seconds
        # cache_file = "scripts/shopping_malls_cache.json" # dagster
        cache_file = shopping_malls_cache_path
        EXPIRATION_PERIOD_SECONDS = 2592000 # 30 Days

        if os.path.exists(cache_file):
            file_age_seconds = time.time() - os.path.getmtime(cache_file)
            
            if file_age_seconds < EXPIRATION_PERIOD_SECONDS:            
                print("\n💾 Loading Shopping mall data from local JSON storage cache...")
                try:
                    with open(cache_file, "r") as f:
                        cached_data = json.load(f)
                    df_mall = pd.DataFrame(cached_data)
                    print(f"✔ Cache hit. Loaded {len(df_mall)} shopping mall instantly.")
                    return df_mall
                except Exception as e:
                    print(f"⚠ Cache corrupted, falling back to network request: {e}")

            else:
                days_old = round(file_age_seconds / (24*3600),1)
                print(f"\n⏳ Cache file exists but is outdated ({days_old} days old). Triggering auto-refresh...")

        # ─── STEP B: FALLBACK TO LIVE NETWORK CALLS ───
        print("\n🌐 Cache miss! Requesting full retail layer from OpenStreetMap Overpass API...")
        
        # Initialize coordinate transformer (WGS84 Lat/Lng -> Singapore SVY21)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3414", always_xy=True)
        
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        # ─── FIXED: ADDING HEADERS TO PREVENT 406 BLOCKS ───
        headers = {
            # Overpass requires a descriptive custom User-Agent. Do not use stock python-requests.
            "User-Agent": "SingaporeMallDataFetcher/1.0 (contact: your_email@domain.com)",
            "Accept": "application/json"
        }
        
        overpass_query = """
        [out:json][timeout:30];
        area["ISO3166-1"="SG"]->.searchArea;
        (
        node["shop"="mall"](area.searchArea);
        way["shop"="mall"](area.searchArea);
        rel["shop"="mall"](area.searchArea);
        );
        out center;
        """
        
        shopping_malls = []
        
        try:
            # Pass the newly defined headers alongside your data payload
            response = requests.post(overpass_url, data={'data': overpass_query}, headers=headers, timeout=35)
            
            if response.status_code != 200:
                print(f"❌ Overpass API Error: Received status code {response.status_code}")
                if response.status_code == 406:
                    print("💡 Tip: The server rejected the client identity. Ensure the custom User-Agent header is set correctly.")
                return []
                
            res = response.json()
            elements = res.get('elements', [])
            
            print(f"Total entries fetched from OpenStreetMap: {len(elements)}")
            
            for item in elements:
                tags = item.get('tags', {})
                name = tags.get('name') or tags.get('name:en')
                
                if not name:
                    continue
                    
                if 'center' in item:
                    lat = item['center']['lat']
                    lon = item['center']['lon']
                else:
                    lat = item.get('lat')
                    lon = item.get('lon')
                    
                if lat and lon:
                    try:
                        mall_x, mall_y = transformer.transform(lon, lat)
                        
                        shopping_malls.append({
                            "mall_name": name.title(),
                            "mall_x": round(float(mall_x), 3),
                            "mall_y": round(float(mall_y), 3)
                        })
                    except Exception as conversion_error:
                        print(f"⚠️ Coordinate projection failed for {name}: {conversion_error}")
                        
            # Deduplicate results
            unique_malls = {m['mall_name'].lower(): m for m in shopping_malls}.values()
            shopping_malls = list(unique_malls)
            
            print(f"✅ Successfully compiled and projected {len(shopping_malls)} unique malls to SVY21 format.")
            
        except Exception as e:
            print(f"❌ Connection or parsing failed during Overpass API execution: {e}")
        
        # [Insert this directly inside fetch_shopping_malls_svy21, after Step B's while loop completes]
        df_mall = pd.DataFrame(shopping_malls)
            
        # --- FIXED INJECTION METHOD ---
        print("\nInjecting fixed non-'shopping' destination nodes into baseline framework...")
        df_fixed = pd.DataFrame(FIXED_MALLS)
            
        # Deduplicate to prevent overlapping duplicate values if your keyword script also catches them
        if not df_mall.empty:
            df_mall = pd.concat([df_mall, df_fixed], ignore_index=True)
            df_mall = df_mall.drop_duplicates(subset=["mall_name"]).reset_index(drop=True)
        else:
            df_mall = df_fixed

        print(f"✅ Filter complete. Isolated {len(df_mall)} shopping mall.")

        # ─── STEP C: SAVE DOWNLOADED OBJECT TO CACHE ───
        if not df_mall.empty:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                # Convert dataframe records to a clean dictionary list object for tracking
                with open(cache_file, "w") as f:
                    json.dump(df_mall.to_dict(orient="records"), f, indent=4)
                print(f"💾 Successfully saved shopping mall collection cache to: {cache_file}")
            except Exception as e:
                print(f"⚠ Failed to save local backup file: {e}")
                
        return df_mall


    # =====================================================================
    # PHASE 1: HUB DISTANCE TRACKING MATRIX
    # =====================================================================

 

    def haversine_km(lat1, lon1, lat2, lon2):
        # Use np.radians instead of math.radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        # Use np.sin, np.cos, np.sqrt, np.arcsin
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371 * c


    print("\n📐 Computing distances to URA Master Plan Economic Hubs...")


    # Compute straight-line distances using the flat cartesian coorindates
    for hub_name, (hub_x, hub_y) in hubs.items():
        col_name = f"dist_to_{hub_name.lower()}"
        df[col_name] = haversine_km(df["x"], df["y"], hub_x, hub_y)

    # Calculate proximity to the absolute closest economic hub
    dist_cols = [f"dist_to_{hub_name.lower()}" for hub_name in hubs.keys()]
    df["min_distance_to_regional_hub_km"] = df[dist_cols].min(axis=1)
    df["closest_regional_hub_name"] = df[dist_cols].idxmin(axis=1).str.replace("dist_", "")

    print("   ✅ Commercial Hub straight-line matrix calculations complete!")

    # =====================================================================
    # PHASE 2: PRIMARY SCHOOL DISTANCE TRACKING MATRIX
    # =====================================================================
    print("\n📐 Executing straight-line matrix calculations for Primary Schools...")

    df_schools = fetch_primary_schools_svy21(access_token)

    if not df_schools.empty:
        # Convert your 231k HDB rows into arrays for vector math
        # 1. Extract inputs into clean, flat NumPy arrays
        hdb_x = df["x"].to_numpy() # Shape: (231770)
        hdb_y = df["y"].to_numpy() 

        # Convert your primary school coordinates into arrays
        sch_x = df_schools["sch_x"].to_numpy()     # Shape: (NumPrimarySchools,)
        sch_y = df_schools["sch_y"].to_numpy()     # Shape: (NumPrimarySchools,)
        sch_names = df_schools["school_name"].to_numpy()

        # 2. Instantiate blank pre-allocated storage structures for speed
        total_rows = len(df)
        schools_1km = np.zeros(total_rows, dtype=int)
        schools_2km = np.zeros(total_rows, dtype=int)
        closest_names = []
        closest_distances = np.zeros(total_rows, dtype=float)

        # 3. Process in chunks of 20,000 rows to completely eliminate OOM crashes
        chunk_size = 20000
        print(f"Processing calculations in blocks of {chunk_size} HDB rows...")

        for i in range(0, total_rows,chunk_size):
            end_idx = min(i + chunk_size, total_rows)

            # Reshape coordinates for broadcasting: Shape (Chunk_Rows, 1)
            chunk_x = hdb_x[i:end_idx][:, np.newaxis]
            chunk_y = hdb_y[i:end_idx][:, np.newaxis]

            # Broadcast subtraction: (Chunk_rows, 1) - (1, 197) -> Shape (Chunk_Rows, 197)
            dx_sch = chunk_x - sch_x
            dy_sch = chunk_y - sch_y
            dist_matrix_sch = np.sqrt(dx_sch**2 + dy_sch**2)

            schools_1km[i:end_idx] = np.sum(dist_matrix_sch <= 1000.0, axis = 1)
            schools_2km[i:end_idx] = np.sum(dist_matrix_sch <= 2000.0, axis = 1)

            # Pull min index map positions and exact distance values
            min_idx = np.argmin(dist_matrix_sch, axis=1)
            closest_names.extend(sch_names[min_idx])
            closest_distances[i:end_idx] = np.min(dist_matrix_sch, axis=1) / 1000.0

    # 4. Save results back into your primary dataframe
        df["primary_schools_within_1km"] = schools_1km
        df["primary_schools_within_2km"] = schools_2km
        df["closest_primary_school_name"] = closest_names
        df["dist_to_closest_primary_school_km"] = closest_distances

        print("Primary School proximity calculations successfully completed! ✅")
        
    else:
        print("⚠ Calculation skipped: Primary School DataFrame is empty.")

       

    # =====================================================================
    # PHASE 3: MAJOR TRANSPORT NODES DISTANCE TRACKING MATRIX (MRT,LRT)
    # =====================================================================
    print("\n📐 Executing straight-line matrix calculations for MRT/LRT...")

    df_train_stations = fetch_MRT_LRT_svy21(access_token)
    print(df_train_stations.head(5))

    if not df_train_stations.empty:
        # Separate between MRT and LRT
        is_LRT_mask = df_train_stations['station_name'].str.contains("Lrt|LRT",case=False, na=False)

        df_LRT = df_train_stations[is_LRT_mask].copy()
        df_MRT = df_train_stations[~is_LRT_mask].copy()

        print(f"Isolated: {len(df_MRT)} MRT Stations vs. {len(df_LRT)} LRT Stations.")

        # Convert your 231k HDB rows into arrays for vector math
        # 1. Extract inputs into clean, flat NumPy arrays
        hdb_x = df["x"].to_numpy() # Shape: (231770)
        hdb_y = df["y"].to_numpy() 

        # Extract mrt arrays
        mrt_x = df_MRT["station_x"].to_numpy()    
        mrt_y = df_MRT["station_y"].to_numpy()     
        mrt_names = df_MRT["station_name"].to_numpy()

        # Extract lrt arrays
        lrt_x = df_LRT["station_x"].to_numpy()    
        lrt_y = df_LRT["station_y"].to_numpy()     
        lrt_names = df_LRT["station_name"].to_numpy()

        # 2. Instantiate blank pre-allocated storage structures for speed
        total_rows = len(df)
        mrt_within_500m = np.zeros(total_rows, dtype=int)
        mrt_within_1km = np.zeros(total_rows, dtype=int)
        closest_mrt_names = []
        closest_mrt_distances = np.zeros(total_rows, dtype=float)

        lrt_within_500m = np.zeros(total_rows, dtype=int)
        closest_lrt_names = []
        closest_lrt_distances = np.zeros(total_rows, dtype=float)

        # 3. Process in chunks of 20,000 rows to completely eliminate OOM crashes
        chunk_size = 20000
        print(f"Processing calculations in blocks of {chunk_size} HDB rows...")

        for i in range(0, total_rows,chunk_size):
            end_idx = min(i + chunk_size, total_rows)

            # Reshape coordinates for broadcasting: Shape (Chunk_Rows, 1)
            chunk_x = hdb_x[i:end_idx][:, np.newaxis]
            chunk_y = hdb_y[i:end_idx][:, np.newaxis]

            # ---- COMPUTE MRT METRICS ONLY ----
            
            # Broadcast subtraction: (Chunk_rows, 1) - (1, 197) -> Shape (Chunk_Rows, 197)
            if len(mrt_names) > 0: 
                dx_mrt = chunk_x - mrt_x
                dy_mrt = chunk_y - mrt_y
                dist_mrt = np.sqrt(dx_mrt**2 + dy_mrt**2)

                mrt_within_500m[i:end_idx] = np.sum(dist_mrt <= 500.0, axis = 1)
                mrt_within_1km[i:end_idx] = np.sum(dist_mrt <= 1000.0, axis = 1)

                # Pull min index map positions and exact distance values
                min_mrt_idx = np.argmin(dist_mrt, axis=1)
                closest_mrt_names.extend(mrt_names[min_mrt_idx])
                closest_mrt_distances[i:end_idx] = np.min(dist_mrt, axis=1) / 1000.0

            # ---- COMPUTE LRT METRICS ONLY ----
            if len(lrt_names) > 0: 
                dx_lrt = chunk_x - lrt_x
                dy_lrt = chunk_y - lrt_y
                dist_lrt = np.sqrt(dx_lrt**2 + dy_lrt**2)

                lrt_within_500m[i:end_idx] = np.sum(dist_lrt <= 500.0, axis = 1)
            
                # Pull min index map positions and exact distance values
                min_lrt_idx = np.argmin(dist_lrt, axis=1)
                closest_lrt_names.extend(lrt_names[min_lrt_idx])
                closest_lrt_distances[i:end_idx] = np.min(dist_lrt, axis=1) / 1000.0 

    # 4. Save results back into your primary dataframe
        df["mrt_within_500m"] = mrt_within_500m
        df["mrt_within_1km"] = mrt_within_1km
        # Check if we actually captured any MRT names before binding
        if len(closest_mrt_names) == total_rows:
            df["closest_mrt_name"] = closest_mrt_names
            df["dist_to_closest_mrt_km"] = closest_mrt_distances
        else:
            df["closest_mrt_name"] = "None"
            df["dist_to_closest_mrt_km"] = np.nan
            
        df["lrt_within_500m"] = lrt_within_500m
        
        # Check if we actually captured any LRT names before binding
        if len(closest_lrt_names) == total_rows:
            df["closest_lrt_name"] = closest_lrt_names
            df["dist_to_closest_lrt_km"] = closest_lrt_distances
        else:
            df["closest_lrt_name"] = "None"
            df["dist_to_closest_lrt_km"] = np.nan

        print("MRT and LRT separated proximity calculations successfully completed! 🚇✅")
   
    else:
        print("⚠ Calculation skipped: Could not load the transit station elements.")


    # =====================================================================
    # PHASE 4: COMMERCIAL LIFESTYLE PROXIMITY MATRIX (SHOPPING MALLS ONLY)
    # =====================================================================
    print("\n📐 Executing straight-line matrix calculations for Shopping Malls...")

    df_malls = fetch_shopping_malls_svy21(access_token)

    if not df_malls.empty:
        hdb_x = df["x"].to_numpy()
        hdb_y = df["y"].to_numpy()
        total_rows = len(df)
        chunk_size = 20000

        # Extract target array points
        mall_x = df_malls["mall_x"].to_numpy()
        mall_y = df_malls["mall_y"].to_numpy()
        mall_names = df_malls["mall_name"].to_numpy()

        # Pre-allocate feature columns for operational execution speed
        malls_within_500m = np.zeros(total_rows, dtype=int)
        malls_within_1km = np.zeros(total_rows, dtype=int)
        # closest_mall_names = []
        closest_mall_names = np.empty(total_rows, dtype=object) 
        closest_mall_distances = np.zeros(total_rows, dtype=float)

        print(f"Processing shopping mall allocations in blocks of {chunk_size} HDB rows...")
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk_x = hdb_x[i:end_idx][:, np.newaxis]
            chunk_y = hdb_y[i:end_idx][:, np.newaxis]
            # chunk_x = hdb_x[i:end_idx].reshape(-1, 1)
            # chunk_y = hdb_y[i:end_idx].reshape(-1, 1)

            # Cartesian broadcasting math (planar metric system calculations)
            # dx_m = chunk_x - mall_x.flatten()
            # dy_m = chunk_y - mall_y.flatten()
            dx_m = chunk_x - mall_x
            dy_m = chunk_y - mall_y
            dist_m = np.sqrt(dx_m**2 + dy_m**2)
            
            # Accumulate density maps based on SVY21 metric bounds (metres)
            malls_within_500m[i:end_idx] = np.sum(dist_m <= 500.0, axis=1)
            malls_within_1km[i:end_idx] = np.sum(dist_m <= 1000.0, axis=1)
            
            # Map nearest asset fields across rows
            min_m_idx = np.argmin(dist_m, axis=1)
            # closest_mall_names.extend((mall_names[min_m_idx]))
            closest_mall_names[i:end_idx] = mall_names[min_m_idx]
            closest_mall_distances[i:end_idx] = np.min(dist_m, axis=1) / 1000.0

        # Bind fields seamlessly back to central tracking database structure
        df["malls_within_500m"] = malls_within_500m
        df["malls_within_1km"] = malls_within_1km
        df["closest_shopping_mall_name"] = closest_mall_names
        df["dist_to_closest_shopping_mall_km"] = closest_mall_distances
        
        print("Shopping Mall matrix calculations completed successfully! 🛍️✅")
        
    else:
        print("⚠ Calculation skipped: Shopping Mall DataFrame could not be compiled.")
    
    print("""
    # ===============================================================================================================================
    # STEP 3: Connect to Data.gov.sg Collection Endpoint & Extract Dataset IDs
    # ===============================================================================================================================
    """)

   
    

    # The corrected mapping dictionary for genDs.py
    hdb_towns = {
        "TAP": "Tampines",
        "WL" : "Woodlands",
        "JW" : "Jurong West",
        "SK" : "Sengkang",
        "YS" : "Yishun",
        "PG" : "Punggol",
        "HG" : "Hougang",
        "CCK": "Choa Chu Kang",
        "BD" : "Bedok",
        "BM" : "Bukit Merah",
        "PRC": "Pasir Ris",
        "BB" : "Bukit Batok",
        "TP" : "Toa Payoh",
        "AMK": "Ang Mo Kio",
        "BP" : "Bukit Panjang",
        "KWN": "Kallang/Whampoa",
        "GL" : "Geylang",
        "SB" : "Sembawang",
        "QT" : "Queenstown",
        "BH" : "Bishan",
        "JE" : "Jurong East",
        "CL" : "Clementi",
        "SGN": "Serangoon",
        "TG" : "Tengah",
        "CT" : "Central Area",
        "MP" : "Marine Parade",
        "BT" : "Bukit Timah"
    }



    def normalize_for_hdb_api(onemap_street_name):
        # 1. Force Uppercase to match data.gov.sg schema
        street = onemap_street_name.upper().strip()

        # 2. Dictionary mapping of OneMap values -> HDB Dataset abbreviations
        abbreviations = {
            " AVENUE": " AVE",
            " ROAD": " RD",
            " DRIVE": " DR",
            " STREET": " ST",
            " BOULEVARD": " BLVD",
            " CRESCENT": " CRES",
            " CLOSE": " CL",
            " CENTRAL": " CTRL",
            " BUKIT": " BT",
            " TANJONG": " TG",
            " KAMPONG": " KG",
            " NORTH": " NTH"
        }

        # Replace words sequentially
        for full_word, short_word in abbreviations.items():
            if full_word in street:
                street = street.replace(full_word, short_word)

        return street
    
    def parse_singapore_address(address_str):
        # Ensure it's a standard string and force uppercase
        address_str = str(address_str).upper().strip()
        
        # 1. Match digits followed by optional letters at the very start (e.g., "635" or "635B")
        # \d+ matches the numbers, [A-Z]* matches any trailing suffix letters
        block_match = re.match(r"^(\d+[A-Z]*)", address_str)
        block_no = block_match.group(1) if block_match else ""
        
        # 2. Clean the street name dynamically
        # Remove the starting block prefix completely to isolate the street part
        street_part = re.sub(r"^\d+[A-Z]*\s+", "", address_str)
        
        # Strip out country and 6-digit postal code markers
        street_part = re.sub(r"\bSINGAPORE\s+\d{6}\b|\b\d{6}\b", "", street_part).strip()
        
        # 3. Isolate the core street name by locating the last valid suffix indicator
        suffixes = r"\b(DRIVE|DR|AVENUE|AVE|STREET|ST|ROAD|RD|CRESCENT|CRES|CLOSE|CL|CENTRAL|CTRL|LINK|WAY|LOOP|PLACE|PL)\b"
        
        matches = list(re.finditer(suffixes, street_part))
        if matches:
            last_match = matches[-1]
            end_idx = last_match.end()
            
            # Capture trailing sector numbers if they exist (e.g., "DRIVE 2", "AVE 12")
            following_text = street_part[end_idx:].strip()
            digit_match = re.match(r"^(\d+)", following_text)
            if digit_match:
                end_idx += digit_match.end() + 1 
                
            street_name = street_part[:end_idx].strip()
        else:
            street_name = street_part

        return block_no, street_name.title()
    
    def calculate_remaining_lease(lease_commencement_year):
        # 1. Get the current calendar year dynamically (e.g., 2026)
        current_year = datetime.now().year
        
        # 2. Calculate elapsed years since the lease started
        elapsed_years = current_year - int(lease_commencement_year)
        
        # 3. Subtract from the initial 99-year HDB master lease leasehold
        remaining_lease_years = 99 - elapsed_years
        
        # Prevent negative values just in case of data anomalies
        return max(0, remaining_lease_years)
    
    blk, street = parse_singapore_address(address)

    cleaned_street = normalize_for_hdb_api(street)
    print("Connecting to Data.gov.sg V2 Collection API...")
    #print(blk)
    #print(cleaned_street)
    url = "https://data.gov.sg/api/action/datastore_search"

    filter_conditions = {
        "blk_no":  blk, 
        "street": cleaned_street 
    }
    

    params = {
        "resource_id": "d_17f5382f26140b1fdae0ba2ef6239d2f",
        "filters": json.dumps(filter_conditions),
        "fields": "year_completed,bldg_contract_town"
    }

    MATURE_ESTATES = {
    "ANG MO KIO","BEDOK","BISHAN","BUKIT MERAH","BUKIT TIMAH","CENTRAL AREA",
    "CLEMENTI","GEYLANG","KALLANG/WHAMPOA","MARINE PARADE","PASIR RIS",
    "QUEENSTOWN","SERANGOON","TAMPINES","TOA PAYOH",
    }

    

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract the records list
        records = data.get('result', {}).get('records', [])
        #print(json.dumps(records))
        # Check if the records list is not empty
        if records:
            df["town"] = (hdb_towns.get(records[0].get("bldg_contract_town"), records[0].get("bldg_contract_town"))).upper()
            df["lease_commence_date"] = records[0].get("year_completed"),
            df["remaining_lease"] = calculate_remaining_lease(records[0].get("year_completed"))
            df["is_mature"]= df["town"].str.upper().isin(MATURE_ESTATES).astype(int)
            CBD_LAT, CBD_LNG = 1.2841, 103.8516  # Raffles Place MRT
            df["dist_cbd_km"] = haversine_km(df["lat"], df["lon"], CBD_LAT, CBD_LNG)
            
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        
    
    
    try:
        #json_string = df.to_json(orient='records')
        #return json_string
        result_dict = df.to_dict(orient='records')[0] 
        #print(result_dict)
        return result_dict
    except NameError:
        print("✅ Script complete.")
        return 0

# if __name__ == "__main__":
#     genDS()
