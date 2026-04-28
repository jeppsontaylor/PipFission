import os
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("OANDA_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
URL = "https://api-fxpractice.oanda.com/v3/instruments/USD_JPY/positionBook"

async def main():
    async with aiohttp.ClientSession() as session:
        print(f"Testing {URL}...")
        async with session.get(URL, headers=HEADERS) as resp:
            print(f"Status: {resp.status}")
            text = await resp.text()
            if resp.status == 200:
                print("Success! Data preview:")
                print(text[:500])
            else:
                print("Failed:")
                print(text)

if __name__ == "__main__":
    asyncio.run(main())
