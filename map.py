from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
import numpy as np
import requests
from json import JSONEncoder

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder

# File to store pincode data
PINCODE_DATA_FILE = 'india_pincodes.json'

def download_pincode_data():
    """Download Indian pincode data with latitude and longitude"""
    # Check if file already exists
    if os.path.exists(PINCODE_DATA_FILE):
        print("Pincode data already downloaded.")
        return
        
    # URL to download from (GitHub repository with pincode data)
    url = "https://raw.githubusercontent.com/mithunsasidharan/India-Pincode-Latitude-Longitude/master/pincodes.json"
    
    try:
        print("Downloading pincode data...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the JSON data and reformat it for easier lookup
        try:
            data = response.json()
            # The data from GitHub might have different structure, so let's normalize it
            formatted_data = {}
            for item in data:
                pincode = str(item.get('pincode', '')).strip()
                if pincode and 'latitude' in item and 'longitude' in item:
                    lat = float(item['latitude'])
                    lng = float(item['longitude'])
                    if lat != 0 and lng != 0:  # Skip entries with 0,0 coordinates
                        formatted_data[pincode] = {"lat": lat, "lng": lng}
            
            # Save the formatted data
            with open(PINCODE_DATA_FILE, 'w') as f:
                json.dump(formatted_data, f)
            
            print(f"Pincode data downloaded successfully! ({len(formatted_data)} pincodes)")
        except Exception as e:
            print(f"Error parsing pincode data: {e}")
            create_sample_pincode_data()
    except Exception as e:
        print(f"Error downloading pincode data: {e}")
        create_sample_pincode_data()

def create_sample_pincode_data():
    """Create a sample pincode data file with major cities"""
    print("Creating a smaller sample file instead.")
    
    sample_data = {
        # Format: "pincode": {"lat": latitude, "lng": longitude}
        "110001": {"lat": 28.6269, "lng": 77.2101},  # Delhi
        "400001": {"lat": 18.9322, "lng": 72.8264},  # Mumbai
        "560001": {"lat": 12.9734, "lng": 77.5922},  # Bangalore
        "560008": {"lat": 12.9901, "lng": 77.5876},  # High Grounds, Bangalore
        "600001": {"lat": 13.0836, "lng": 80.2825},  # Chennai
        "700001": {"lat": 22.5726, "lng": 88.3639},  # Kolkata
        "500001": {"lat": 17.3616, "lng": 78.4747},  # Hyderabad
        "380001": {"lat": 23.0225, "lng": 72.5714},  # Ahmedabad
        "226001": {"lat": 26.8467, "lng": 80.9462},  # Lucknow
        "800001": {"lat": 25.5941, "lng": 85.1376},  # Patna
        "302001": {"lat": 26.9124, "lng": 75.7873},  # Jaipur
        "641001": {"lat": 11.0168, "lng": 76.9558},  # Coimbatore
        "695001": {"lat": 8.5241, "lng": 76.9366},   # Thiruvananthapuram
        "144001": {"lat": 31.3260, "lng": 75.5762}   # Jalandhar
    }
    
    with open(PINCODE_DATA_FILE, 'w') as f:
        json.dump(sample_data, f)
    
    print("Created sample pincode data file.")

def get_pincode_coordinates(pincode):
    """Get lat/lng for a pincode from our data file"""
    pincode_str = str(pincode)
    
    # Load pincode data from file
    try:
        with open(PINCODE_DATA_FILE, 'r') as f:
            pincode_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is invalid, download/create it
        download_pincode_data()
        with open(PINCODE_DATA_FILE, 'r') as f:
            pincode_data = json.load(f)
    
    # Return coordinates if pincode exists in our data
    if pincode_str in pincode_data:
        return pincode_data[pincode_str]
    
    # If pincode not found, use approximation based on first digit
    first_digit = int(pincode_str[0]) if pincode_str else 0
    first_two = int(pincode_str[:2]) if len(pincode_str) >= 2 else first_digit
    
    # Indian pincode zones (approximation if not found in database)
    if first_digit == 1:  # Delhi, Haryana, Punjab region
        lat = np.random.uniform(28.0, 32.0)
        lng = np.random.uniform(74.0, 78.0)
    elif first_digit == 2:  # Uttar Pradesh, Uttarakhand
        lat = np.random.uniform(25.0, 31.0)
        lng = np.random.uniform(77.0, 84.0)
    elif first_digit == 3:  # Rajasthan, Gujarat
        lat = np.random.uniform(22.0, 28.0)
        lng = np.random.uniform(69.0, 77.0)
    elif first_digit == 4:  # Maharashtra, Goa
        lat = np.random.uniform(15.0, 21.0)
        lng = np.random.uniform(72.0, 80.0)
    elif first_digit == 5:  # Andhra Pradesh, Telangana, Karnataka
        if first_two >= 56 and first_two <= 59:  # Karnataka
            lat = np.random.uniform(12.0, 18.0)
            lng = np.random.uniform(74.0, 78.0)
        else:  # Andhra Pradesh, Telangana
            lat = np.random.uniform(14.0, 20.0)
            lng = np.random.uniform(77.0, 84.0)
    elif first_digit == 6:  # Tamil Nadu, Kerala
        lat = np.random.uniform(8.0, 14.0)
        lng = np.random.uniform(76.0, 80.0)
    elif first_digit == 7:  # West Bengal, Orissa, North East
        lat = np.random.uniform(17.0, 28.0)
        lng = np.random.uniform(84.0, 96.0)
    elif first_digit == 8:  # Central/Eastern states
        lat = np.random.uniform(17.0, 27.0)
        lng = np.random.uniform(75.0, 88.0)
    else:  # Default to central India
        lat = np.random.uniform(20.0, 25.0)
        lng = np.random.uniform(75.0, 82.0)
    
    # Add small random variation to avoid exact overlaps
    lat += np.random.uniform(-0.5, 0.5)
    lng += np.random.uniform(-0.5, 0.5)
    
    print(f"Warning: Pincode {pincode_str} not found in database, using approximation")
    return {"lat": lat, "lng": lng}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        try:
            # Read the Excel file
            df = pd.read_excel(file)
            
            # Check if 'pincode' column exists
            if 'pincode' not in df.columns and 'Pincode' not in df.columns:
                return jsonify({'error': 'File does not contain a pincode column'})
            
            # Standardize column name
            pincode_col = 'pincode' if 'pincode' in df.columns else 'Pincode'
            
            # Count occurrences of each pincode
            pincode_counts = df[pincode_col].value_counts().reset_index()
            pincode_counts.columns = ['pincode', 'count']
            
            # Add lat/lng to each pincode
            geo_data = []
            for _, row in pincode_counts.iterrows():
                coords = get_pincode_coordinates(row['pincode'])
                # Convert NumPy/pandas types to standard Python types
                geo_data.append({
                    'pincode': int(row['pincode']) if isinstance(row['pincode'], (np.int64, np.int32)) else str(row['pincode']),
                    'count': int(row['count']),  # Convert np.int64 to standard int
                    'lat': float(coords['lat']),  # Ensure it's a standard float
                    'lng': float(coords['lng'])   # Ensure it's a standard float
                })
            
            return jsonify({'success': True, 'data': geo_data})
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'})
    
    return jsonify({'error': 'File must be an Excel file (.xlsx or .xls)'})

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Download pincode data at startup
    download_pincode_data()
    
    # Create index.html template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Pincode Map Bubble Chart</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-form {
            margin: 20px 0;
            text-align: center;
        }
        .upload-form input[type="file"] {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .upload-form button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .upload-form button:hover {
            background-color: #45a049;
        }
        #map-container {
            width: 100%;
            height: 600px;
            position: relative;
            margin-top: 20px;
        }
        .error-message {
            color: red;
            margin: 10px 0;
            text-align: center;
        }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border-left-color: #09f;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .info-box {
            padding: 15px;
            margin: 20px 0;
            background-color: #f8f9fa;
            border-left: 4px solid #17a2b8;
            border-radius: 4px;
        }
        .legend {
            line-height: 18px;
            color: #555;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        }
        .legend i {
            width: 18px;
            height: 18px;
            float: left;
            margin-right: 8px;
            opacity: 0.7;
        }
        .not-found-msg {
            color: #ff8c00;
            font-style: italic;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .search-box {
            margin: 10px 0;
            text-align: center;
        }
        .search-box input {
            padding: 8px;
            width: 200px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .search-box button {
            padding: 8px 15px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .search-box button:hover {
            background-color: #0069d9;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pincode Map Bubble Chart Generator</h1>
        
        <div class="info-box">
            <p><strong>How it works:</strong> Upload an Excel file with a column named "pincode" or "Pincode". The app will display the frequency of each pincode as bubbles on the map of India.</p>
            <p>The app uses real geographical coordinates for Indian pincodes from open source data. If a pincode is not found in the database, an approximate location is used based on the pincode's regional prefix.</p>
        </div>
        
        <div class="upload-form">
            <input type="file" id="excel-file" accept=".xlsx, .xls">
            <button onclick="uploadFile()">Generate Map</button>
        </div>
        
        <div class="search-box">
            <input type="text" id="search-pincode" placeholder="Search pincode...">
            <button onclick="searchPincode()">Search</button>
        </div>
        
        <div id="error-message" class="error-message"></div>
        <div id="spinner" class="spinner"></div>
        <div id="not-found-message" class="not-found-msg"></div>
        
        <div id="map-container"></div>
    </div>

    <script>
        // Initialize the map centered on India
        const map = L.map('map-container').setView([23.0, 80.0], 5);
        
        // Add the OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);
        
        // Store circle markers and data
        let bubbles = [];
        let mapData = [];
        
        function uploadFile() {
            const fileInput = document.getElementById('excel-file');
            const errorMessage = document.getElementById('error-message');
            const spinner = document.getElementById('spinner');
            const notFoundMsg = document.getElementById('not-found-message');
            
            // Clear previous messages
            errorMessage.textContent = '';
            notFoundMsg.textContent = '';
            
            // Remove existing bubbles from map
            bubbles.forEach(bubble => map.removeLayer(bubble));
            bubbles = [];
            mapData = [];
            
            if (!fileInput.files.length) {
                errorMessage.textContent = 'Please select an Excel file';
                return;
            }
            
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);
            
            // Show loading spinner
            spinner.style.display = 'block';
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide spinner
                spinner.style.display = 'none';
                
                if (data.error) {
                    errorMessage.textContent = data.error;
                    return;
                }
                
                // Save data for search functionality
                mapData = data.data;
                
                createBubbleMap(data.data);
            })
            .catch(error => {
                spinner.style.display = 'none';
                errorMessage.textContent = 'Error uploading file: ' + error.message;
            });
        }
        
        function searchPincode() {
            const searchInput = document.getElementById('search-pincode');
            const notFoundMsg = document.getElementById('not-found-message');
            const pincode = searchInput.value.trim();
            
            if (!pincode) {
                notFoundMsg.textContent = 'Please enter a pincode to search';
                return;
            }
            
            if (!mapData || mapData.length === 0) {
                notFoundMsg.textContent = 'Please upload data first';
                return;
            }
            
            notFoundMsg.textContent = '';
            
            // Find the pincode in our data
            const found = mapData.find(item => String(item.pincode) === pincode);
            
            if (found) {
                // Center map on the found pincode
                map.setView([found.lat, found.lng], 12);
                
                // Highlight the bubble by creating a temporary marker
                const highlightMarker = L.circleMarker([found.lat, found.lng], {
                    radius: 15,
                    color: '#ff0000',
                    weight: 3,
                    opacity: 1,
                    fillOpacity: 0
                }).addTo(map);
                
                // Flash animation
                let opacity = 1;
                const flashInterval = setInterval(() => {
                    opacity = opacity === 1 ? 0.2 : 1;
                    highlightMarker.setStyle({ opacity: opacity });
                }, 500);
                
                // Remove after 5 seconds
                setTimeout(() => {
                    clearInterval(flashInterval);
                    map.removeLayer(highlightMarker);
                }, 5000);
            } else {
                notFoundMsg.textContent = `Pincode ${pincode} not found in your uploaded data`;
            }
        }
        
        function createBubbleMap(data) {
            if (!data || data.length === 0) {
                document.getElementById('error-message').textContent = 'No data to display';
                return;
            }
            
            // Get min and max counts for scaling bubbles
            const counts = data.map(d => d.count);
            const minCount = Math.min(...counts);
            const maxCount = Math.max(...counts);
            
            // Scale function for bubble size
            function getRadius(count) {
                // Calculate a size between 5 and 25 based on count
                return 5 + ((count - minCount) / (maxCount - minCount)) * 20;
            }
            
            // Color scale
            function getColor(count) {
                const ratio = (count - minCount) / (maxCount - minCount);
                if (ratio < 0.2) return "#feebe2";
                if (ratio < 0.4) return "#fbb4b9";
                if (ratio < 0.6) return "#f768a1";
                if (ratio < 0.8) return "#c51b8a";
                return "#7a0177";
            }
            
            // Add bubbles to map
            data.forEach(item => {
                const circle = L.circleMarker([item.lat, item.lng], {
                    radius: getRadius(item.count),
                    fillColor: getColor(item.count),
                    color: "#000",
                    weight: 1,
                    opacity: 0.8,
                    fillOpacity: 0.6
                }).addTo(map);
                
                // Add popup with information
                circle.bindPopup(`<strong>Pincode:</strong> ${item.pincode}<br><strong>Count:</strong> ${item.count}`);
                
                // Store reference for later removal
                bubbles.push(circle);
            });
            
            // Add a legend to the map
            const legend = L.control({position: 'bottomright'});
            
            legend.onAdd = function (map) {
                const div = L.DomUtil.create('div', 'info legend');
                const grades = [
                    minCount,
                    minCount + (maxCount - minCount) * 0.2,
                    minCount + (maxCount - minCount) * 0.4,
                    minCount + (maxCount - minCount) * 0.6,
                    minCount + (maxCount - minCount) * 0.8
                ];
                
                div.innerHTML = '<h4>Frequency</h4>';
                
                // Loop through our density intervals and generate a label with a colored square for each interval
                for (let i = 0; i < grades.length; i++) {
                    const nextGrade = i < grades.length - 1 ? grades[i + 1] : maxCount;
                    div.innerHTML +=
                        '<i style="background:' + getColor(grades[i] + 1) + '"></i> ' +
                        Math.round(grades[i]) + (i < grades.length - 1 ? '&ndash;' + Math.round(nextGrade) : '+') + '<br>';
                }
                
                return div;
            };
            
            // Remove existing legend if present
            const legends = document.querySelectorAll('.legend');
            legends.forEach(el => el.remove());
            
            legend.addTo(map);
            
            // Zoom to fit all bubbles
            if (bubbles.length > 0) {
                const group = L.featureGroup(bubbles);
                map.fitBounds(group.getBounds().pad(0.1));
            }
        }
    </script>
</body>
</html>
        ''')
    
    app.run(debug=True)