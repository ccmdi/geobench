import os
import json
import asyncio
import logging
import argparse
from PIL import Image
import aiohttp
from typing import List, Dict, Optional

from geoguessr import GeoGuessr
from pano import Pano

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetGenerator:    
    def __init__(self, output_dir: str = "dataset"):
        """Initialize the dataset generator with output directory."""
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.metadata_file = os.path.join(output_dir, "metadata.json")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.metadata = self._load_metadata()

        self.geoguessr = GeoGuessr()
    
    def _load_metadata(self) -> List[Dict]:
        """Load existing metadata if available."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"bounds": None, "images": []}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    async def generate_dataset(self, num_locations: int = 10, map_id: str = None):
        """Generate a dataset with the specified number of locations."""
        logging.info(f"Generating dataset with {num_locations} locations")
        
        bound = None
        locations_added = 0
        
        while locations_added < num_locations:
            game = await self.geoguessr.create_geoguessr_game(map_id=map_id)
            
            if not game:
                logging.error("Failed to create GeoGuessr game. Retrying...")
                await asyncio.sleep(2)
                continue
            
            if not bound:
                bound = game['bounds']
                self.metadata['bounds'] = bound

            while game and game['round'] <= 5 and locations_added < num_locations:
                logging.info(f"Processing round {game['round']}")
                
                try:
                    round_data = game['rounds'][game['round'] - 1]
                    lat = round_data['lat']
                    lng = round_data['lng']
                    pano_id = round_data.get('panoId')
                    heading = round_data.get('heading', 0)
                    pitch = round_data.get('pitch', 0)
                    
                    country = await self._get_country(lat, lng)
                    if not country:
                        logging.warning(f"Couldn't determine country for location: {lat}, {lng}. Skipping.")
                        game = await self.geoguessr.guess_and_advance()
                        continue
                    
                    # Process and save panorama
                    image_filename = f"{len(self.metadata['images']) + 1}.jpg"
                    image_path = os.path.join(self.images_dir, image_filename)
                    
                    await self._save_pano_view(pano_id, lat, lng, heading, pitch, image_path)
                    
                    self.metadata['images'].append({
                        "image_path": f"images/{image_filename}",
                        "country": country,
                        "lat": lat,
                        "lng": lng
                    })
                    
                    self._save_metadata()
                    
                    locations_added += 1
                    logging.info(f"Added location {locations_added}/{num_locations}: {country} ({lat}, {lng})")
                    
                    game = await self.geoguessr.guess_and_advance()
                
                except Exception as e:
                    logging.error(f"Error processing round: {e}")
                    try:
                        game = await self.geoguessr.guess_and_advance()
                    except:
                        break
        
        logging.info(f"Dataset generation complete. Generated {locations_added} locations.")
    
    async def _save_pano_view(self, pano_id, lat, lng, heading, pitch, output_path):
        """Get panorama view and save it to a file."""
        try:
            pano = Pano(pano_id=pano_id) if pano_id else Pano(lat=lat, lng=lng)
            
            image_data = await pano.get_panorama(heading, pitch, FOV=110)
            
            if image_data is None:
                logging.error("Failed to get panorama image")
                return False
            
            image = Image.fromarray(image_data)
            image.save(output_path, quality=95)
            return True
            
        except Exception as e:
            logging.error(f"Error processing panorama: {e}")
            return False
    
    async def _get_country(self, lat: float, lng: float) -> Optional[str]:
        """Get country name for a location using reverse geocoding."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.bigdatacloud.net/data/reverse-geocode-client"
                params = {
                    "latitude": lat,
                    "longitude": lng,
                    "localityLanguage": "en"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('countryName')
            return None
        except Exception as e:
            logging.error(f"Error getting country: {e}")
            return None

async def main():
    parser = argparse.ArgumentParser(description="Generate GeoGuessr dataset for LLM benchmarking")
    parser.add_argument("--num", type=int, default=50, help="Number of locations to generate")
    parser.add_argument("--output", type=str, default="default", help="Output directory")
    parser.add_argument("--map", type=str, default=None, help="GeoGuessr map ID")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(output_dir="dataset/"+str(args.output))
    await generator.generate_dataset(num_locations=args.num, map_id=args.map)

if __name__ == "__main__":
    asyncio.run(main())