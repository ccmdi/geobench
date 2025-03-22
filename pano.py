from PIL import Image
from e2p import Equirectangular
import asyncio, aiohttp
import io
import numpy as np
import logging

GSV_PANO_URL = "https://geo0.ggpht.com/cbk"

class Pano:
    """
    A GSV panorama, with a unique ID and image file.
    """
    def __init__(self, pano_id=None, lat=None, lng=None):
        self.zoom = 4
        self.dimensions = None
        self.driving_direction = None

        if lat is not None and lng is not None:
            self.pano_id = None
            self.lat = lat
            self.lng = lng
        else:
            self.pano_id = self.convert_pano_id(pano_id)
            self.lat = None
            self.lng = None
            
        self.panorama = None
        self.img = None
    
    
    async def get_panorama(self, heading, pitch, FOV=110):
        if self.pano_id is None:
            self.pano_id = await self.get_panoid()

        if self.panorama is None:
            await self.get_pano_metadata()
            self.panorama = await self._fetch_and_build_panorama()

        equ = Equirectangular(self.panorama)
        result = equ.GetPerspective(FOV, (heading - self.driving_direction) % 360, pitch, 1080, 1920)

        return result

    async def fetch_single_tile(self, session, x, y, retries=3):
        params = {
            "cb_client": "apiv3",
            "panoid": self.pano_id,
            "output": "tile",
            "zoom": self.zoom,
            "x": x,
            "y": y
        }
        
        for attempt in range(retries):
            try:
                async with session.get(GSV_PANO_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        logging.error(f"Error fetching tile {x},{y}: Status {response.status}")
                        if attempt < retries - 1:
                            await asyncio.sleep(1)
                            continue
                        return None
                    data = await response.read()
                    tile = Image.open(io.BytesIO(data))
                    return tile
            except Exception as e:
                logging.error(f"Exception fetching tile {x},{y}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)
                    continue
                return None

    async def _fetch_and_build_panorama(self):        
        if self.dimensions[1] == 8192:  # Gen 4
            max_x, max_y = 16, 8
        elif self.dimensions[1] == 6656:  # Gen 3
            max_x, max_y = 13, 6.5
        else:  # Fallback
            max_x, max_y = 7, 4
        
        async with aiohttp.ClientSession() as session:
            # Get tiles based on determined dimensions
            raw_tiles = await asyncio.gather(
                *[self.fetch_single_tile(session, x, y) 
                for y in range(int(max_y) if max_y == 8 else 7)  # Handle Gen 3's 6.5
                for x in range(max_x)]
            )
            
            if max_y == 6.5:  # Handle Gen 3's half row
                tiles = []
                # First 6 rows
                for y in range(6):
                    for x in range(13):
                        idx = y * max_x + x
                        tiles.append(raw_tiles[idx])
                        
                # Half of 7th row
                for x in range(13):
                    idx = 6 * max_x + x
                    tile = raw_tiles[idx]
                    if tile is None:
                        continue
                    tile_array = np.array(tile)
                    half_height = tile_array.shape[0] // 2
                    half_tile = Image.fromarray(tile_array[:half_height])
                    tiles.append(half_tile)
                
                return self._stitch_panorama(tiles, max_x, max_y)
            else:
                return self._stitch_panorama(raw_tiles, max_x, max_y)
    
    def _stitch_panorama(self, tiles, max_x, max_y):
        is_half_height = max_y % 1 != 0
        full_height = int(max_y)
        
        tile_width, tile_height = tiles[0].size
        if is_half_height:
            last_row_height = tile_height // 2
            total_height = (full_height * tile_height) + last_row_height
        else:
            total_height = int(max_y * tile_height)
            
        total_width = int(max_x * tile_width)
        
        full_panorama = Image.new('RGB', (total_width, total_height))
        
        for idx, img in enumerate(tiles):
            x = (idx % int(max_x)) * tile_width
            y = (idx // int(max_x)) * tile_height
            full_panorama.paste(img, (x, y))
            
        return np.array(full_panorama)

    async def get_panoid(self):
        url = "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/SingleImageSearch"

        headers = {
            "Content-Type": "application/json+protobuf"
        }
        radius = 50
        payload = f'[["apiv3"],[[null,null,{self.lat},{self.lng}],{radius}],[[null,null,null,null,null,null,null,null,null,null,[null,null]],null,null,null,null,null,null,null,[1],null,[[[2,true,2]]]],[[2,6]]]'

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload, headers=headers) as response:
                try:
                    data = await response.json()
                    return data[1][1][1]
                except Exception as e:
                    logging.error(f"Error getting panoid: {e}")
    
    async def get_pano_metadata(self):
        if self.dimensions and self.driving_direction:
            return self.dimensions, self.driving_direction
        
        url = "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/GetMetadata"
    
        headers = {
            "Content-Type": "application/json+protobuf"
        }
        
        request_data = [
            ["apiv3",None,None,None,"US",None,None,None,None,None,[[0]]],
            ["en","US"],
            [[[2,self.pano_id]]],
            [[1,2,3,4,8,6]]
        ]

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=request_data, headers=headers) as response:
                try:
                    data = await response.json()

                    self.dimensions = data[1][0][2][3][0][4][0]  # [height, width]
                    self.driving_direction = data[1][0][5][0][1][2][0]  # Driving direction
                    logging.info(f"Metadata: {self.dimensions}, {self.driving_direction}")
                    
                    return self.dimensions, self.driving_direction
                except Exception as e:
                    logging.error(f"Error getting metadata: {e}")
                    return None

    @staticmethod
    def convert_pano_id(pano_id):
        """
        Convert GeoGuessr hex-encoded pano ID to base64-encoded pano ID.
        """
        try:
            decoded = bytes.fromhex(pano_id).decode()
            return decoded
        except ValueError:
            return pano_id
        except Exception as e:
            logging.error(f"Error: {e}")
            return pano_id

    @staticmethod
    def add_compass(image: np.ndarray, heading: float, output_path: str = 'image.jpg'):
        """
        Add a compass overlay to an image.
        
        Args:
            image: numpy array of the image
            heading: Heading angle in degrees (0-360)
            output_path: Path to save the resulting image
        """
        try:
            # Convert numpy array to PIL Image
            main_image = Image.fromarray(image)
            compass = Image.open('compass.png')
            
            compass_size = int(min(main_image.size) * 0.15)  # 15% of smaller dimension
            compass = compass.resize((compass_size, compass_size), Image.Resampling.LANCZOS)
            
            compass = compass.convert('RGBA')
            rotated_compass = compass.rotate(heading, expand=False, resample=Image.Resampling.BICUBIC)
            
            # Calculate position (bottom left with padding)
            padding = int(compass_size * 0.2)  # 20% of compass size as padding
            position = (padding, main_image.size[1] - compass_size - padding)
            
            result = main_image.convert('RGBA')
            result.paste(rotated_compass, position, rotated_compass)
            
            result = result.convert('RGB')
            return result
            
        except Exception as e:
            logging.error(f"Error adding compass overlay: {e}")
    
    def to_dict(self):
        """Return a JSON-serializable representation of the Pano"""
        return {
            'pano_id': self.pano_id,
            'zoom': self.zoom,
            'dimensions': self.dimensions,
            'driving_direction': self.driving_direction
        }
