import os
import aiohttp
import logging

from dotenv import load_dotenv
load_dotenv()

NCFA = os.environ.get("NCFA")

class GeoGuessr:
    def __init__(self):
        self.headers = {
            "Content-Type": "application/json",
            "Cookie": f"_ncfa={NCFA}"
        }
        self.game = None
        
    async def create_geoguessr_game(self, map_id = None):
        """
        Creates a new GeoGuessr game.

        Returns:
            str: Game token
        """
        self.map_id = map_id

        async with aiohttp.ClientSession() as session:
            game_response = await session.post(
                "https://www.geoguessr.com/api/v3/games",
                headers=self.headers,
                json={
                    "map": self.map_id,
                    "type": "standard",
                    "timeLimit": 0,
                    "forbidMoving": True,
                    "forbidZooming": True,
                    "forbidRotating": True
                }
            )
            if game_response.status != 200:
                print(game_response)
                logging.error(f"Error creating game: {game_response.status}")
                self.game = None
                return None
            
            self.game = await game_response.json()
            return self.game
    
    async def guess_and_advance(self):
        """
        Guesses at (0, 0) and advances to the next round.
        """
        logging.info("guess_and_advance")
        if self.game['round'] >= 5:
            await self.create_geoguessr_game(map_id=self.map_id)
        if not self.game:
            return None
        
        game_id = self.game['token']
        

        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"https://www.geoguessr.com/api/v3/games/{game_id}",
                headers=self.headers,
                json={
                    "token": game_id,
                    "lat": 0,
                    "lng": 0,
                    "timedOut": False,
                    "stepsCount": 0
                }
            )
            if response.status != 200:
                logging.error(f"Error getting game data: {response.status}")
                return None
            
            self.game = await response.json()
            
            # Advance to next round
            self.game = await session.get(
                f"https://www.geoguessr.com/api/v3/games/{game_id}",
                headers=self.headers
            )

            if response.status != 200:
                logging.error(f"Error advancing: {response.status}")
                return None

            self.game = await self.game.json()
            logging.info(str(self.game['round']) +  " - round begin")
            return self.game
