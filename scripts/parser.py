import re
import os
import argparse
from dataclasses import dataclass
from typing import Tuple

from geo2p.canon import COUNTRY_ALIASES

@dataclass
class Guess:
    country: str
    lat: float
    lng: float
    
    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.lat, self.lng)

def parse_response(response: str) -> Guess:
    # Look for the final answer section that matches the required format
    final_answer_match = re.search(
        # Country part:
        r"(?:^|\n)(?:\*\*)?(?:country|Country)(?:\*\*)?:\s*"  # Keyword (optionally **keyword**), colon, initial space
        r"(\*+)?\s*([^,\r\n]+?)\s*(\*+)?"                  # Optional value wrapper (* or ** etc.), internal spaces, country name, internal spaces, optional value wrapper
        r"\s*(?:\n|$)"                                      # Trailing space and newline/end
        # Separator:
        r".*?"
        # Latitude part:
        r"(?:^|\n)(?:\*\*)?(?:lat|Lat)(?:\*\*)?:\s*"          # Keyword (optionally **keyword**), colon, initial space
        r"(\*+)?\s*([-+]?\d+\.?\d*?)\s*(\*+)?"              # Optional value wrapper (* or ** etc.), internal spaces, lat number, internal spaces, optional value wrapper
        r"\s*(?:\n|$)"                                      # Trailing space and newline/end
        # Separator:
        r".*?"
        # Longitude part:
        r"(?:^|\n)(?:\*\*)?(?:lng|Lng)(?:\*\*)?:\s*"          # Keyword (optionally **keyword**), colon, initial space
        r"(\*+)?\s*([-+]?\d+\.?\d*?)\s*(\*+)?"              # Optional value wrapper (* or ** etc.), internal spaces, lng number, internal spaces, optional value wrapper
        r"\s*(?:\n|$)",                                     # Trailing space and newline/end
        response,
        re.MULTILINE | re.DOTALL
    )
    if not final_answer_match:
        raise ValueError("Response missing required fields in final answer format")

    try:
        country = final_answer_match.group(2).strip() # Country name is now group 2
        lat_str = final_answer_match.group(5).strip() # Lat number is now group 5
        lng_str = final_answer_match.group(8).strip() # Lng number is now group 8
        
        lat = float(lat_str)
        lng = float(lng_str)
    except (AttributeError, IndexError, ValueError) as e:
        raise ValueError(f"Failed to parse final answer: {e}")
    
    if not -90 <= lat <= 90:
        raise ValueError(f"Invalid latitude value: {lat} (must be between -90 and 90)")
    if not -180 <= lng <= 180:
        raise ValueError(f"Invalid longitude value: {lng} (must be between -180 and 180)")
        
    return Guess(country=country, lat=lat, lng=lng)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse model responses and count valid ones.")
    parser.add_argument("response_folder", type=str, help="Path to the response folder containing an 'output' subdirectory with .txt files.")
    args = parser.parse_args()

    output_dir = os.path.join(args.response_folder, "output")
    if not os.path.isdir(output_dir):
        print(f"Error: Output directory '{output_dir}' not found.")
        exit(1)

    valid_responses = 0
    invalid_countries = 0
    total_files = 0

    for filename in os.listdir(output_dir):
        if filename.endswith(".txt"):
            total_files += 1
            file_path = os.path.join(output_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                g = parse_response(content)
                print(os.path.splitext(os.path.basename(filename))[0], g.country, g.lat, g.lng)
                valid_responses += 1
                if COUNTRY_ALIASES.get(g.country.lower()) is None:
                    invalid_countries += 1
            except ValueError as e:
                print(f"Invalid response in {filename}: {e}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Found {valid_responses} valid responses out of {total_files} files in {output_dir}") 
    print(f"Invalid countries: {invalid_countries}")