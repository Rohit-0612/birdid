"""
Bird Call Audio Downloader — uses xeno-canto API (free, no key needed)
Run: python3 download_bird_audio.py
Downloads sample .mp3 files for testing the audio feature
"""
import requests
import os

AUDIO_DIR = "/Users/swayam/Desktop/birdsproject/bird_audio_samples"
os.makedirs(AUDIO_DIR, exist_ok=True)

TEST_SPECIES = [
    "Barn Swallow",
    "American Robin",
    "Blue Jay",
    "Northern Cardinal",
    "Mallard"
]

def download_xenocanto(species_name, save_dir, max_files=3):
    query = species_name.replace(' ', '+')
    url = f"https://xeno-canto.org/api/2/recordings?query={query}"
    try:
        resp = requests.get(url, timeout=10).json()
        recordings = resp.get('recordings', [])[:max_files]
        for rec in recordings:
            audio_url = 'https:' + rec['file']
            safe_name = species_name.replace(' ', '_')
            filename = os.path.join(save_dir, f"{safe_name}_{rec['id']}.mp3")
            r = requests.get(audio_url, timeout=30)
            with open(filename, 'wb') as f:
                f.write(r.content)
            print(f"  Downloaded: {filename}")
    except Exception as e:
        print(f"  Error for {species_name}: {e}")

for species in TEST_SPECIES:
    print(f"Downloading: {species}")
    download_xenocanto(species, AUDIO_DIR)

print(f"\nDone! Audio files saved to: {AUDIO_DIR}")
