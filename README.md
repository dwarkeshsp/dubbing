# Podcast Dubbing Tool

Dubs podcast episodes into Spanish, Hindi, and Chinese using ElevenLabs.

## Setup (one time)

1. **Install Python 3.10+** if not already installed

2. **Create virtual environment:**
   ```bash
   cd /Users/dwarkesh/Documents/dubbing
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set API key** (add to your shell profile, e.g. `~/.zshrc`):
   ```bash
   export ELEVEN_API_KEY="your-api-key-here"
   ```

## Usage

Always activate the virtual environment first:
```bash
cd /Users/dwarkesh/Documents/dubbing
source venv/bin/activate
```

### Dub a new episode (before upload)

For a local video or audio file:
```bash
python dub_podcast.py -s /path/to/episode.mp4
python dub_podcast.py -s /path/to/episode.mp3
```

### Dub multiple sources

Create a `sources.txt` file with one source per line:
```
# YouTube URLs
https://www.youtube.com/watch?v=ABC123
https://www.youtube.com/watch?v=DEF456

# Local files
/path/to/new_episode.mp4
~/Desktop/interview.mp3
```

Then run:
```bash
python dub_podcast.py -f sources.txt
```

### Dub existing YouTube videos

Specific URLs:
```bash
python dub_podcast.py -s https://www.youtube.com/watch?v=VIDEO_ID
```

Last N episodes from channel:
```bash
python dub_podcast.py -n 5
```

### Options

| Flag | Description |
|------|-------------|
| `-s` | Sources (URLs or file paths) |
| `-f` | File containing sources |
| `-n` | Number of recent channel episodes |
| `-l` | Languages (default: es hi zh) |
| `-o` | Output directory (default: dubbed_episodes) |

## Output

Each episode creates a folder with:
```
dubbed_episodes/
└── Episode Name/
    ├── Spanish.mp3    # Dubbed audio
    ├── Spanish.srt    # Subtitles for YouTube
    ├── Spanish.json   # Full transcript
    ├── Hindi.mp3
    ├── Hindi.srt
    ├── Hindi.json
    ├── Chinese.mp3
    ├── Chinese.srt
    └── Chinese.json
```

## Notes

- **Skips existing:** Re-running won't re-dub episodes that already have output files
- **Resumes:** If interrupted, run again to resume pending jobs
- **Quota:** Stops gracefully if ElevenLabs quota is exceeded
- **Supported formats:** .mp4, .mov, .mkv, .avi, .webm, .mp3, .wav, .m4a, .aac, .flac, .ogg
