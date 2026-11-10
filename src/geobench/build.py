import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import httpx
from charoite import StreetViewError

from geobench.geoguessr import GeoGuessr, Location

REVERSE_GEOCODE_URL = "https://api.bigdatacloud.net/data/reverse-geocode-client"
SOURCE_WIDTH = 6656
VIEW_FOV = 110.0
VIEW_SIZE = (1920, 1080)


async def country_at(http: httpx.AsyncClient, location: Location) -> str | None:
    params = {"latitude": location.lat, "longitude": location.lng, "localityLanguage": "en"}
    response = await http.get(REVERSE_GEOCODE_URL, params=params)
    response.raise_for_status()
    return response.json().get("countryName") or None


async def build(out: Path, count: int, map_id: str) -> None:
    metadata_path = out / "metadata.json"
    (out / "images").mkdir(parents=True, exist_ok=True)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"bounds": None, "images": []}
    target = len(metadata["images"]) + count
    transport = httpx.AsyncHTTPTransport(retries=3)
    async with httpx.AsyncClient(transport=transport, timeout=30, follow_redirects=True) as http:
        locations = GeoGuessr(http, os.environ["NCFA"]).locations(map_id)
        async for bounds, location in locations:
            if len(metadata["images"]) >= target:
                break
            try:
                country = await country_at(http, location)
                if not country:
                    logging.warning("No country for %s, %s; skipping", location.lat, location.lng)
                    continue
                pano = await location.panorama(http.request)
                sphere = await pano.equirectangular(http.request, SOURCE_WIDTH)
            except (StreetViewError, httpx.HTTPError) as e:
                logging.warning("Skipping %s, %s: %s", location.lat, location.lng, e)
                continue
            view = await asyncio.to_thread(sphere.view, location.heading, location.pitch, VIEW_FOV, VIEW_SIZE)
            name = f"{len(metadata['images']) + 1}.jpg"
            view.save(out / "images" / name, quality=95)
            metadata["bounds"] = metadata["bounds"] or bounds
            metadata["images"].append({"image_path": f"images/{name}", "country": country, "lat": location.lat, "lng": location.lng})
            metadata_path.write_text(json.dumps(metadata, indent=2))
            logging.info("Added %d/%d: %s (%s, %s)", len(metadata["images"]), target, country, location.lat, location.lng)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GeoBench map from GeoGuessr rounds")
    parser.add_argument("--num", type=int, default=50, help="Locations to add")
    parser.add_argument("--output", required=True, help="Map folder name under dataset/")
    parser.add_argument("--map", required=True, help="GeoGuessr map id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(build(Path("dataset") / args.output, args.num, args.map))


if __name__ == "__main__":
    main()
