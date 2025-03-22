import os
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

def get_model_name_from_summary(model_dir):
    """
    Get the model name from the summary.json file if it exists.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Model name string or None if not found
    """
    summary_path = os.path.join(model_dir, "results", "summary.json")
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            if "model" in summary_data:
                return summary_data["model"]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None

def extract_model_name(model_dir):
    """
    Extract model name from the directory name (first part before underscore).
    
    Args:
        model_dir: Model directory name
        
    Returns:
        Model name string
    """
    return model_dir.split('_')[0] if '_' in model_dir else model_dir

def create_geoguessr_visualization(responses_dir="../responses"):
    """
    Create an HTML visualization of GeoGuessr benchmark results,
    showing all models' guesses for each location with company-specific colors.
    
    Args:
        responses_dir: Directory containing model response folders
    """
    # Dictionary to store all data by location
    all_data = {}
    model_names = []
    test_datasets = set()
    model_metadata = {}
    
    # Define company colors
    company_colors = {
        "Anthropic": "#FF7F50",  # Orange
        "Google": "#4285F4",     # Google Blue
        "OpenAI": "#858585",     # White 
        "Meta": "#add4ed",       # Facebook Blue
        "Mistral": "#FFDE59",    # Yellow
        "Alibaba": "#ffd796",    # Alibaba Orange
        "Other": "#A020F0"       # Purple for others
    }
    
    # Model name patterns to identify companies
    company_patterns = {
        "Anthropic": ["Claude"],
        "Google": ["Gemini"],
        "OpenAI": ["GPT", "o1"],
        "Meta": ["Llama"],
        "Mistral": ["Mistral", "Pixtral"],
        "Alibaba": ["Qwen"]
    }
    
    # Get all model directories
    try:
        model_dirs = [d for d in os.listdir(responses_dir) 
                     if os.path.isdir(os.path.join(responses_dir, d))]
    except FileNotFoundError:
        print(f"Directory not found: {responses_dir}")
        model_dirs = []
    
    # If no models found, use sample data for demonstration
    if not model_dirs:
        print("No model directories found. Using sample data from paste.txt.")
        # Create a sample model with metadata
        model_name = "Sample Model"
        model_names.append(model_name)
        model_metadata[model_name] = {
            "company": "Sample",
            "color": "#A020F0",
            "test_dataset": "sample"
        }
        test_datasets.add("sample")
        
        sample_data = parse_sample_data("paste.txt")
        location_data = organize_by_location(sample_data, model_name, "sample")
        all_data = location_data
    else:
        # Process each model directory to extract metadata and data
        location_data = defaultdict(dict)
        
        for model_dir in model_dirs:
            # Extract test dataset from directory name
            test_dataset = extract_test_dataset(model_dir)
            test_datasets.add(test_dataset)
            
            # Try to get model name from summary.json for DISPLAY ONLY
            display_name = get_model_name_from_summary(os.path.join(responses_dir, model_dir))
            
            # If summary.json doesn't exist or doesn't contain model name, 
            # extract it from directory name
            if not display_name:
                display_name = extract_model_name(model_dir)
                
            # Determine company based on display name
            company = determine_company(display_name, company_patterns)
            
            # Get color for the company
            color = company_colors.get(company, company_colors["Other"])
            
            # Store model metadata - IMPORTANT: use model_dir as the key, not the display name
            model_names.append(model_dir)  # Keep directory name as the identifier
            model_metadata[model_dir] = {
                "company": company,
                "color": color,
                "test_dataset": test_dataset,
                "display_name": display_name  # Store display name separately
            }
            
            # Read model data
            csv_path = os.path.join(responses_dir, model_dir, "results", "detailed.csv")
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    model_data = list(reader)
                    
                # IMPORTANT: Use model_dir as the key here, not display_name
                for entry in model_data:
                    # Round coordinates for the key
                    rounded_lat = round(float(entry['lat_true']), 5)
                    rounded_lng = round(float(entry['lng_true']), 5)
                    loc_key = f"{rounded_lat},{rounded_lng}"
                    entry["test_dataset"] = test_dataset
                    
                    # Use model_dir as the key for storing data
                    location_data[loc_key][model_dir] = entry
            else:
                print(f"CSV file not found for model {model_dir}: {csv_path}")
        
        all_data = dict(location_data)
    
    # Generate HTML content
    html_content = generate_html(all_data, model_names, model_metadata, list(test_datasets))
    
    # Write HTML to file
    with open('visualization.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Visualization created: visualization.html")

def determine_company(model_name, company_patterns):
    """
    Determine the company based on model name patterns.
    
    Args:
        model_name: Name of the model
        company_patterns: Dictionary of company name patterns
        
    Returns:
        Company name string
    """
    for company, patterns in company_patterns.items():
        for pattern in patterns:
            if pattern in model_name:
                return company
    return "Other"

def extract_test_dataset(model_dir):
    """
    Extract test dataset name from model directory name.
    Dataset is whatever is within the second underscore.
    
    Args:
        model_dir: Model directory name
        
    Returns:
        Test dataset name
    """
    # Split by underscore and get the second part if it exists
    parts = model_dir.split('_')
    if len(parts) >= 2:
        return parts[1]  # Return the second part (index 1)
    
    # Fallback to the old pattern matching for backward compatibility
    match = re.search(r'(acw(?:-\d{2}-\d{2}-\d{2})?)', model_dir)
    if match:
        return match.group(1)
    
    return "unknown"

def organize_by_location(data, model_name, test_dataset):
    """
    Organize data by location for a single model with coordinate rounding.
    """
    location_data = defaultdict(dict)
    
    for entry in data:
        entry["test_dataset"] = test_dataset  # Add test dataset to entry
        # Round coordinates to 5 decimal places for the key
        rounded_lat = round(float(entry['lat_true']), 8)
        rounded_lng = round(float(entry['lng_true']), 8)
        loc_key = f"{rounded_lat},{rounded_lng}"
        location_data[loc_key][model_name] = entry
    
    return dict(location_data)

def parse_sample_data(file_path):
    """
    Parse the sample data provided in the paste.txt file.
    
    Args:
        file_path: Path to the sample data file
    
    Returns:
        List of dictionaries containing the parsed data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        print(f"Sample file not found: {file_path}")
        return []

def generate_html(location_data, model_names, model_metadata, test_datasets):
    """Generate the HTML content for the visualization."""
    # Convert data to JSON for use in JavaScript
    json_data = json.dumps(location_data)
    json_models = json.dumps(model_names)
    json_metadata = json.dumps(model_metadata)
    json_datasets = json.dumps(test_datasets)
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GeoBench Visualization</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <link rel="stylesheet" href="style.css" id="theme-css">
    </head>
    <body>
        <div id="container">
            <div id="header">
                <div id="filters">
                    <div class="filter-group">
                        <label for="dataset-filter">Test Dataset: </label>
                        <select id="dataset-filter"></select>
                    </div>
                    <div class="filter-group">
                        <label for="location-selector">Location: </label>
                        <select id="location-selector"></select>
                    </div>
                </div>
                <div id="location-info"></div>
            </div>
            <div id="map">
                <div id="legend"></div>
            </div>
            <div id="no-data-message">
                No data available for the selected dataset. Please choose a different dataset.
            </div>
            <div id="navigation">
                <button class="nav-button" id="prev-button">← Previous</button>
                <span id="location-counter"></span>
                <button class="nav-button" id="next-button">Next →</button>
            </div>
            <div id="info-panel">
                <div id="model-comparisons"></div>
            </div>
        </div>
        
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            // Data from Python
            const locationData = {json_data};
            const modelNames = {json_models};
            const modelMetadata = {json_metadata};
            const testDatasets = {json_datasets};
            
            // State variables
            let currentLocationIndex = 0;
            let filteredLocationKeys = [];
            let map = null;
            let trueMarker = null;
            let guessMarkers = [];
            let connectionLines = [];
            let currentDataset = 'all'; 
            
            // Initialize the visualization
            function init() {{
                // Setup map
                map = L.map('map', {{attributionControl: false}}).setView([0, 0], 2);
                L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                    maxZoom: 19
                }}).addTo(map);
                
                // Create legend
                createLegend();
                
                // Populate dataset filter
                populateDatasetFilter();
                
                // Apply initial filtering
                applyDatasetFilter();
                
                // Event listeners
                document.getElementById('dataset-filter').addEventListener('change', (e) => {{
                    currentDataset = e.target.value;
                    currentLocationIndex = 0;
                    applyDatasetFilter();
                }});
                
                document.getElementById('location-selector').addEventListener('change', (e) => {{
                    const selectedIndex = filteredLocationKeys.indexOf(e.target.value);
                    if (selectedIndex !== -1) {{
                        currentLocationIndex = selectedIndex;
                        showLocation(e.target.value);
                    }}
                }});
                
                document.getElementById('prev-button').addEventListener('click', () => {{
                    if (currentLocationIndex > 0) {{
                        currentLocationIndex--;
                        showLocation(filteredLocationKeys[currentLocationIndex]);
                    }}
                }});
                
                document.getElementById('next-button').addEventListener('click', () => {{
                    if (currentLocationIndex < filteredLocationKeys.length - 1) {{
                        currentLocationIndex++;
                        showLocation(filteredLocationKeys[currentLocationIndex]);
                    }}
                }});
                
                // Keyboard navigation with arrow keys
                document.addEventListener('keydown', (e) => {{
                    if (e.key === 'ArrowLeft') {{
                        if (currentLocationIndex > 0) {{
                            currentLocationIndex--;
                            showLocation(filteredLocationKeys[currentLocationIndex]);
                        }}
                    }} else if (e.key === 'ArrowRight') {{
                        if (currentLocationIndex < filteredLocationKeys.length - 1) {{
                            currentLocationIndex++;
                            showLocation(filteredLocationKeys[currentLocationIndex]);
                        }}
                    }}
                }});
            }}
            
            function createLegend() {{
                const legend = document.getElementById('legend');
                legend.innerHTML = '<h4 style="margin-top: 0; margin-bottom: 10px;">Legend</h4>';
                
                // Add true location to legend
                const trueLegendItem = document.createElement('div');
                trueLegendItem.className = 'legend-item';
                trueLegendItem.innerHTML = `
                    <span class="color-box" style="background-color: green;"></span>
                    <span>Location</span>
                `;
                legend.appendChild(trueLegendItem);
                
                // Group models by company and display name to avoid duplicates
                const companiesModels = {{}};
                const displayedModels = new Set(); // Track which display names we've already shown
                
                // First pass - group by company
                modelNames.forEach(model => {{
                    const company = modelMetadata[model].company;
                    if (!companiesModels[company]) {{
                        companiesModels[company] = [];
                    }}
                    
                    // Get display name
                    const displayName = modelMetadata[model].display_name || model;
                    
                    // Create a unique key for this display name to avoid duplicates
                    const displayKey = `${{company}}_${{displayName}}`;
                    
                    // Only add if we haven't already added this display name
                    if (!displayedModels.has(displayKey)) {{
                        displayedModels.add(displayKey);
                        companiesModels[company].push({{
                            dirName: model,
                            displayName: displayName
                        }});
                    }}
                }});
                
                // Add companies and their models to legend
                Object.keys(companiesModels).sort().forEach(company => {{
                    // Skip if the company has no models after deduplication
                    if (companiesModels[company].length === 0) return;
                    
                    // Company header
                    const companySection = document.createElement('div');
                    companySection.className = 'company-section';
                    
                    const companyHeader = document.createElement('div');
                    companyHeader.className = 'company-header';
                    companyHeader.textContent = company;
                    companySection.appendChild(companyHeader);
                    
                    // Add models for this company
                    companiesModels[company].forEach(modelInfo => {{
                        const modelItem = document.createElement('div');
                        modelItem.className = 'legend-item';
                        
                        modelItem.innerHTML = `
                            <span class="color-box" style="background-color: ${{modelMetadata[modelInfo.dirName].color}};"></span>
                            <span>${{modelInfo.displayName}}</span>
                        `;
                        companySection.appendChild(modelItem);
                    }});
                    
                    legend.appendChild(companySection);
                }});
            }}
            
            function populateDatasetFilter() {{
                const datasetFilter = document.getElementById('dataset-filter');
                
                // Add 'All' option
                const allOption = document.createElement('option');
                allOption.value = 'all';
                allOption.textContent = 'All Datasets';
                datasetFilter.appendChild(allOption);
                
                // Add dataset options
                testDatasets.sort().forEach(dataset => {{
                    const option = document.createElement('option');
                    option.value = dataset;
                    option.textContent = dataset;
                    datasetFilter.appendChild(option);
                }});
            }}
            
            function applyDatasetFilter() {{
                // Filter locations based on dataset
                filteredLocationKeys = Object.keys(locationData).filter(locationKey => {{
                    if (currentDataset === 'all') return true;
                    
                    // Check if this location has data from the selected dataset
                    const modelsData = locationData[locationKey];
                    return Object.values(modelsData).some(modelData => 
                        modelData.test_dataset === currentDataset
                    );
                }});
                
                // Re-populate location selector
                populateLocationSelector();
                
                // Show message if no data available
                const noDataMessage = document.getElementById('no-data-message');
                const navigationDiv = document.getElementById('navigation');
                const infoPanel = document.getElementById('info-panel');
                const mapDiv = document.getElementById('map');
                
                if (filteredLocationKeys.length === 0) {{
                    noDataMessage.style.display = 'block';
                    navigationDiv.style.display = 'none';
                    infoPanel.style.display = 'none';
                    mapDiv.style.opacity = '0.3';
                }} else {{
                    noDataMessage.style.display = 'none';
                    navigationDiv.style.display = 'flex';
                    infoPanel.style.display = 'block';
                    mapDiv.style.opacity = '1';
                    
                    // Show first location
                    currentLocationIndex = 0;
                    showLocation(filteredLocationKeys[0]);
                }}
            }}
            
            function populateLocationSelector() {{
                const locationSelector = document.getElementById('location-selector');
                locationSelector.innerHTML = ''; // Clear existing options
                
                filteredLocationKeys.forEach((locationKey, index) => {{
                    const option = document.createElement('option');
                    option.value = locationKey;
                    
                    // Get country from any model's data
                    const modelsData = locationData[locationKey];
                    const firstModelData = modelsData[Object.keys(modelsData)[0]];
                    const country = firstModelData.country_true;
                    
                    option.textContent = `${{index + 1}}: ${{country}} (${{locationKey}})`;
                    locationSelector.appendChild(option);
                }});
            }}
            
            function showLocation(locationKey) {{
                // Update selector
                document.getElementById('location-selector').value = locationKey;
                
                // Update location counter
                document.getElementById('location-counter').textContent = 
                    `Location ${{currentLocationIndex + 1}} of ${{filteredLocationKeys.length}}`;
                
                // Clear previous markers and lines
                if (trueMarker) map.removeLayer(trueMarker);
                guessMarkers.forEach(marker => map.removeLayer(marker));
                connectionLines.forEach(line => map.removeLayer(line));
                guessMarkers = [];
                connectionLines = [];
                
                // Get location coordinates
                const [trueLat, trueLng] = locationKey.split(',').map(parseFloat);
                
                // Get all model data for this location
                const modelsData = locationData[locationKey];
                const firstModelData = modelsData[Object.keys(modelsData)[0]];
                
                // Update location info
                document.getElementById('location-info').textContent = 
                    `${{firstModelData.country_true}} (${{trueLat.toFixed(4)}}, ${{trueLng.toFixed(4)}})`;
                
                // Add true location marker
                trueMarker = L.marker([trueLat, trueLng], {{
                    icon: L.icon({{
                        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
                        iconSize: [25, 41],
                        iconAnchor: [12, 41],
                        popupAnchor: [1, -34],
                        shadowSize: [41, 41]
                    }})
                }}).addTo(map).bindPopup(firstModelData.country_true);
                
                // Clear model comparisons panel
                const modelComparisons = document.getElementById('model-comparisons');
                modelComparisons.innerHTML = '';
                
                // Bounds for map view
                const points = [[trueLat, trueLng]];
                
                // Filter models by dataset if needed
                const filteredModels = modelNames.filter(model => {{
                    if (currentDataset === 'all') return true;
                    if (!modelsData[model]) return false;
                    return modelsData[model].test_dataset === currentDataset;
                }});
                
                // Process each model's guess
                filteredModels.forEach(model => {{
                    if (modelsData[model]) {{
                        const modelData = modelsData[model];
                        
                        // Skip models from different datasets if filtering
                        if (currentDataset !== 'all' && modelData.test_dataset !== currentDataset) {{
                            return;
                        }}
                        
                        // Add model comparison details
                        addModelComparisonDetails(model, modelData);
                        
                        // Only add guess marker and line if there was a valid guess
                        if (modelData.lat_guess && modelData.lng_guess && modelData.refused !== 'True') {{
                            const guessLat = parseFloat(modelData.lat_guess);
                            const guessLng = parseFloat(modelData.lng_guess);
                            
                            // Add point to bounds
                            points.push([guessLat, guessLng]);
                            
                            // Get color from metadata
                            const modelColor = modelMetadata[model].color;
                            
                            // Create custom icon with model color
                            const customIcon = L.divIcon({{
                                html: `<div style="background-color: ${{modelColor}}; width: 15px; height: 15px; border-radius: 50%; border: 2px solid white;"></div>`,
                                className: 'custom-marker',
                                iconSize: [25, 25],
                                iconAnchor: [8, 8]
                            }});
                            
                            // Add marker
                            const marker = L.marker([guessLat, guessLng], {{ icon: customIcon }})
                                .addTo(map)
                                .bindPopup(`${{modelMetadata[model].display_name}}`);
                            guessMarkers.push(marker);
                            
                            // Add line connecting true location and guess
                            const line = L.polyline([[trueLat, trueLng], [guessLat, guessLng]], {{
                                color: modelColor,
                                dashArray: '5, 10',
                                weight: 2,
                                opacity: 0.7
                            }}).addTo(map);
                            connectionLines.push(line);
                        }}
                    }}
                }});
                
                // Fit map to show all points
                if (points.length > 1) {{
                    const bounds = L.latLngBounds(points);
                    map.fitBounds(bounds, {{ padding: [50, 50] }});
                }} else {{
                    // If only true location, zoom to it
                    map.setView([trueLat, trueLng], 5);
                }}
            }}
            
            function addModelComparisonDetails(model, modelData) {{
                const modelComparisons = document.getElementById('model-comparisons');
                
                const modelDiv = document.createElement('div');
                modelDiv.className = 'model-details';
                
                const isCorrectCountry = modelData.country_correct === 'True';
                const distance = parseFloat(modelData.distance_km);
                const score = parseInt(modelData.score);
                
                // Use the display name from metadata if available
                const displayName = modelMetadata[model].display_name || model;
                
                modelDiv.innerHTML = `
                    <div class="model-header">
                        <span class="model-color" style="background-color: ${{modelMetadata[model].color}};"></span>
                        <span>${{displayName}}</span>
                        <span class="stats-summary">
                            Dataset: ${{modelData.test_dataset}} | 
                            Company: ${{modelMetadata[model].company}}
                        </span>
                    </div>
                    <div>
                        <p>Guessed: ${{modelData.country_guess || 'No guess'}} 
                        (${{modelData.lat_guess || 'N/A'}}, ${{modelData.lng_guess || 'N/A'}})</p>
                        <p>Distance: ${{distance.toFixed(2)}} km | Score: ${{score.toLocaleString()}}</p>
                        <p class="${{isCorrectCountry ? 'correct' : 'incorrect'}}">
                            Correct country: ${{isCorrectCountry ? 'Yes ✓' : 'No ✗'}}
                        </p>
                        ${{modelData.refused === 'True' ? '<p><strong>Note:</strong> Model refused to guess</p>' : ''}}
                        ${{modelData.error_message ? `<p><strong>Error:</strong> ${{modelData.error_message}}</p>` : ''}}
                    </div>
                `;
                
                modelComparisons.appendChild(modelDiv);
            }}
            
            // Initialize on page load
            window.onload = init;
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    create_geoguessr_visualization()