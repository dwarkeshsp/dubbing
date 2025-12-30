#!/usr/bin/env python3
"""
Dub YouTube videos or local audio/video files into multiple languages using ElevenLabs API.

Usage:
    python dub_podcast.py -f sources.txt           # From sources file (URLs or local files)
    python dub_podcast.py -s URL1 /path/to/video   # Specific sources
    python dub_podcast.py -n 5                     # Last 5 from channel
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import yt_dlp

logging.getLogger("yt_dlp").setLevel(logging.ERROR)

# Configuration
ELEVENLABS_API = "https://api.elevenlabs.io/v1"
LANGUAGES = {"es": "Spanish", "hi": "Hindi", "zh": "Chinese"}
DEFAULT_CHANNEL = "https://www.youtube.com/@DwarkeshPatel/videos"
DEFAULT_EPISODES = 3
POLL_INTERVAL = 30
MAX_CONCURRENT_JOBS = 6
MAX_RETRIES = 3
JOBS_FILE = "pending_jobs.json"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


@dataclass
class Source:
    id: str
    title: str
    url: str | None = None  # YouTube URL
    local_path: Path | None = None  # Local file path

    @property
    def safe_title(self) -> str:
        return "".join(c if c.isalnum() or c in " -_" else "_" for c in self.title)[:80]

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "url": self.url, "local_path": str(self.local_path) if self.local_path else None}

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        return cls(id=d["id"], title=d["title"], url=d.get("url"), local_path=Path(d["local_path"]) if d.get("local_path") else None)


# Alias for backward compatibility in job persistence
Video = Source


@dataclass
class DubbingJob:
    id: str
    source: Source
    lang: str
    retries: int = 0

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source.to_dict(), "lang": self.lang, "retries": self.retries}

    @classmethod
    def from_dict(cls, d: dict) -> "DubbingJob":
        source_data = d.get("source") or d.get("video")  # backward compat
        return cls(id=d["id"], source=Source.from_dict(source_data), lang=d["lang"], retries=d.get("retries", 0))

    @property
    def video(self) -> Source:
        return self.source  # backward compat


@dataclass
class SourceResult:
    source: Source
    audio_path: Path | None = None
    success: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def video(self) -> Source:
        return self.source  # backward compat


class QuotaExceededError(Exception):
    pass


# Job persistence for resume
def save_jobs(jobs: list[DubbingJob], output_dir: Path):
    (output_dir / JOBS_FILE).write_text(json.dumps([j.to_dict() for j in jobs], indent=2))


def load_jobs(output_dir: Path) -> list[DubbingJob]:
    path = output_dir / JOBS_FILE
    if not path.exists():
        return []
    try:
        return [DubbingJob.from_dict(d) for d in json.loads(path.read_text())]
    except Exception:
        return []


def clear_jobs(output_dir: Path):
    path = output_dir / JOBS_FILE
    if path.exists():
        path.unlink()


# Source handling functions
def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def convert_video_to_audio(video_path: Path, output_path: Path) -> bool:
    """Convert video file to MP3 using ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(output_path), "-y"],
            capture_output=True,
            check=True,
        )
        return output_path.exists()
    except Exception:
        return False


def fetch_channel_videos(channel_url: str, count: int) -> list[Source]:
    """Fetch latest videos from a YouTube channel."""
    print(f"\nFetching {count} videos from channel...")
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "playlistend": count}) as ydl:
        result = ydl.extract_info(channel_url, download=False)
        if not result:
            return []
        sources = []
        for entry in result.get("entries", []):
            if entry and entry.get("id"):
                source = Source(id=entry["id"], title=entry.get("title", "Unknown"), url=f"https://www.youtube.com/watch?v={entry['id']}")
                sources.append(source)
                print(f"  {source.title[:60]}")
        return sources


def fetch_youtube_info(url: str) -> Source | None:
    """Fetch info for a YouTube URL."""
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "ignoreerrors": True}) as ydl:
        try:
            result = ydl.extract_info(url, download=False)
            if result:
                return Source(id=result["id"], title=result.get("title", "Unknown"), url=url)
        except Exception:
            pass
    return None


def parse_local_file(path_str: str) -> Source | None:
    """Create a Source from a local file path."""
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        return None
    if not (is_video_file(path) or is_audio_file(path)):
        return None
    title = path.stem  # filename without extension
    return Source(id=f"local_{hash(str(path)) & 0xFFFFFFFF:08x}", title=title, local_path=path)


def fetch_sources(inputs: list[str]) -> list[Source]:
    """Fetch sources from a mix of URLs and local file paths."""
    print(f"\nProcessing {len(inputs)} sources...")
    sources = []
    for inp in inputs:
        inp = inp.strip()
        if not inp:
            continue

        if is_url(inp):
            source = fetch_youtube_info(inp)
            if source:
                sources.append(source)
                print(f"  [URL] {source.title[:55]}")
            else:
                print(f"  [FAIL] Could not fetch: {inp[:50]}")
        else:
            source = parse_local_file(inp)
            if source:
                sources.append(source)
                print(f"  [FILE] {source.title[:55]}")
            else:
                print(f"  [FAIL] Not found or unsupported: {inp[:50]}")
    return sources


def prepare_audio(source: Source, output_dir: Path) -> Path | None:
    """Get audio for a source - download from YouTube or convert/copy local file."""
    output_path = output_dir / f"{source.safe_title}.mp3"
    if output_path.exists():
        return output_path

    # Local file
    if source.local_path:
        if is_audio_file(source.local_path):
            # Copy or just use directly if already mp3
            if source.local_path.suffix.lower() == ".mp3":
                import shutil
                shutil.copy(source.local_path, output_path)
            else:
                # Convert to mp3
                subprocess.run(
                    ["ffmpeg", "-i", str(source.local_path), "-acodec", "libmp3lame", "-q:a", "2", str(output_path), "-y"],
                    capture_output=True,
                )
            if output_path.exists():
                print(f"  [AUDIO] {source.safe_title[:50]} ({output_path.stat().st_size / 1024 / 1024:.1f}MB)")
                return output_path
        elif is_video_file(source.local_path):
            # Extract audio from video
            if convert_video_to_audio(source.local_path, output_path):
                print(f"  [CONVERT] {source.safe_title[:50]} ({output_path.stat().st_size / 1024 / 1024:.1f}MB)")
                return output_path
        print(f"  [FAIL] {source.safe_title[:50]}")
        return None

    # YouTube URL
    if source.url:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(output_dir / f"{source.safe_title}.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([source.url])
            if output_path.exists():
                print(f"  [DL] {source.safe_title[:50]} ({output_path.stat().st_size / 1024 / 1024:.1f}MB)")
                return output_path
        except Exception:
            pass
        print(f"  [DL FAIL] {source.safe_title[:50]}")

    return None


# ElevenLabs API functions
async def submit_job(session: aiohttp.ClientSession, api_key: str, source: Source, audio_path: Path, lang: str) -> DubbingJob | None:
    data = aiohttp.FormData()
    data.add_field("file", open(audio_path, "rb"), filename=audio_path.name, content_type="audio/mpeg")
    data.add_field("target_lang", lang)
    data.add_field("source_lang", "en")
    data.add_field("name", f"{source.title[:50]} - {LANGUAGES[lang]}")
    data.add_field("watermark", "true")

    try:
        async with session.post(f"{ELEVENLABS_API}/dubbing", headers={"xi-api-key": api_key}, data=data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"  [SUBMIT] {source.safe_title[:40]} -> {LANGUAGES[lang]}")
                return DubbingJob(id=result["dubbing_id"], source=source, lang=lang)
            error = await resp.text()
            if "quota_exceeded" in error.lower():
                raise QuotaExceededError(error)
            print(f"  [FAIL] {source.safe_title[:40]} -> {LANGUAGES[lang]}: {error[:80]}")
    except QuotaExceededError:
        raise
    except Exception as e:
        print(f"  [ERROR] {source.safe_title[:40]} -> {LANGUAGES[lang]}: {e}")
    return None


async def check_status(session: aiohttp.ClientSession, api_key: str, job: DubbingJob) -> tuple[str, str | None]:
    try:
        async with session.get(f"{ELEVENLABS_API}/dubbing/{job.id}", headers={"xi-api-key": api_key}) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result.get("status", "unknown"), result.get("error")
    except Exception:
        pass
    return "error", None


async def download_result(session: aiohttp.ClientSession, api_key: str, job: DubbingJob, output_dir: Path) -> bool:
    episode_dir = output_dir / job.video.safe_title
    episode_dir.mkdir(parents=True, exist_ok=True)
    lang_name = LANGUAGES[job.lang]

    try:
        # Download audio
        async with session.get(f"{ELEVENLABS_API}/dubbing/{job.id}/audio/{job.lang}", headers={"xi-api-key": api_key}) as resp:
            if resp.status != 200:
                return False
            content = await resp.read()
            (episode_dir / f"{lang_name}.mp3").write_bytes(content)
            print(f"  [DONE] {job.video.safe_title[:40]} -> {lang_name} ({len(content) / 1024 / 1024:.1f}MB)")

        # Download transcript JSON
        async with session.get(f"{ELEVENLABS_API}/dubbing/{job.id}/transcript/{job.lang}", headers={"xi-api-key": api_key}, params={"format_type": "json"}) as resp:
            if resp.status == 200:
                (episode_dir / f"{lang_name}.json").write_text(json.dumps(await resp.json(), indent=2, ensure_ascii=False))

        # Download SRT for YouTube
        async with session.get(f"{ELEVENLABS_API}/dubbing/{job.id}/transcript/{job.lang}", headers={"xi-api-key": api_key}, params={"format_type": "srt"}) as resp:
            if resp.status == 200:
                (episode_dir / f"{lang_name}.srt").write_text(await resp.text(), encoding="utf-8")

        return True
    except Exception:
        return False


# Main processing
async def process_all(channel_url: str | None, num_episodes: int, languages: list[str], output_dir: Path, sources: list[str] | None = None):
    api_key = os.environ.get("ELEVEN_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: Set ELEVEN_API_KEY or ELEVENLABS_API_KEY")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_temp"
    temp_dir.mkdir(exist_ok=True)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        # Resume pending jobs
        pending = load_jobs(output_dir)
        if pending:
            print(f"\n{'='*60}\nResuming {len(pending)} pending jobs\n{'='*60}")
            await poll_and_download(session, api_key, pending, output_dir, {}, [])
            if load_jobs(output_dir):
                print("Jobs still pending. Run again to continue.")
                return

        # Get sources (YouTube URLs or local files)
        source_list = fetch_sources(sources) if sources else fetch_channel_videos(channel_url, num_episodes)
        if not source_list:
            print("No sources found.")
            return

        print(f"\nDubbing {len(source_list)} sources into {', '.join(LANGUAGES[l] for l in languages)}")

        # Track results
        results: dict[str, SourceResult] = {s.id: SourceResult(source=s) for s in source_list}

        # Check existing and build work queue
        work_queue: list[tuple[Source, str, int]] = []  # (source, lang, retries)
        for source in source_list:
            result = results[source.id]
            episode_dir = output_dir / source.safe_title
            for lang in languages:
                if (episode_dir / f"{LANGUAGES[lang]}.mp3").exists():
                    result.skipped.append(lang)
                else:
                    work_queue.append((source, lang, 0))
            if result.skipped:
                print(f"  [SKIP] {source.safe_title[:50]} ({', '.join(LANGUAGES[l] for l in result.skipped)})")

        if not work_queue:
            print("\nAll sources already dubbed!")
            return

        # Prepare audio for sources that need it
        sources_needing_audio = list({s.id: s for s, _, _ in work_queue}.values())
        print(f"\n{'='*60}\nPreparing audio ({len(sources_needing_audio)} sources)\n{'='*60}")

        with ThreadPoolExecutor(max_workers=3) as pool:
            for source, path in pool.map(lambda s: (s, prepare_audio(s, temp_dir)), sources_needing_audio):
                results[source.id].audio_path = path

        # Build final queue with audio paths
        final_queue: list[tuple[Source, str, Path, int]] = []
        for source, lang, retries in work_queue:
            if results[source.id].audio_path:
                final_queue.append((source, lang, results[source.id].audio_path, retries))
            else:
                results[source.id].failed.append(lang)

        # Submit and process jobs
        print(f"\n{'='*60}\nSubmitting jobs (max {MAX_CONCURRENT_JOBS} concurrent)\n{'='*60}")
        await submit_and_poll(session, api_key, final_queue, output_dir, results)

        # Cleanup temp files
        for result in results.values():
            if result.audio_path and result.audio_path.exists():
                result.audio_path.unlink()
        try:
            temp_dir.rmdir()
        except OSError:
            pass

        # Summary
        print(f"\n{'='*60}\nSummary\n{'='*60}")
        total_success = total_skipped = total_failed = 0
        for result in results.values():
            total_success += len(result.success)
            total_skipped += len(result.skipped)
            total_failed += len(result.failed)
            print(f"\n{result.video.title[:55]}")
            if result.success:
                print(f"  Success: {', '.join(LANGUAGES[l] for l in result.success)}")
            if result.skipped:
                print(f"  Skipped: {', '.join(LANGUAGES[l] for l in result.skipped)}")
            if result.failed:
                print(f"  Failed: {', '.join(LANGUAGES[l] for l in result.failed)}")

        print(f"\n{'='*60}")
        print(f"Total: {total_success} success, {total_skipped} skipped, {total_failed} failed")
        print(f"Output: {output_dir.absolute()}")


async def submit_and_poll(session, api_key, work_queue, output_dir, results):
    jobs: list[DubbingJob] = []
    job_results: dict[str, VideoResult] = {}
    quota_exceeded = False
    total_done = 0

    # Submit initial batch
    while work_queue and len(jobs) < MAX_CONCURRENT_JOBS and not quota_exceeded:
        video, lang, audio_path, retries = work_queue.pop(0)
        try:
            job = await submit_job(session, api_key, video, audio_path, lang)
            if job:
                job.retries = retries
                jobs.append(job)
                job_results[job.id] = results[video.id]
                save_jobs(jobs, output_dir)
            else:
                results[video.id].failed.append(lang)
        except QuotaExceededError:
            quota_exceeded = True
            results[video.id].failed.append(lang)
            for v, l, _, _ in work_queue:
                results[v.id].failed.append(l)
            work_queue.clear()
        await asyncio.sleep(0.3)

    if not jobs:
        return

    print(f"\n{len(jobs)} active, {len(work_queue)} queued")
    if quota_exceeded:
        print("(Quota exceeded - stopped submissions)")

    # Poll loop
    await poll_and_download(session, api_key, jobs, output_dir, job_results, work_queue, results, quota_exceeded)


async def poll_and_download(session, api_key, jobs, output_dir, job_results, work_queue=None, results=None, quota_exceeded=False):
    work_queue = work_queue or []
    results = results or {}
    pending = jobs.copy()
    total_done = 0
    elapsed = 0

    while pending or work_queue:
        save_jobs(pending, output_dir)
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        still_pending = []
        for job in pending:
            status, error = await check_status(session, api_key, job)
            result = job_results.get(job.id) or results.get(job.video.id)

            if status == "dubbed":
                if await download_result(session, api_key, job, output_dir):
                    if result:
                        result.success.append(job.lang)
                else:
                    if result:
                        result.failed.append(job.lang)
                total_done += 1
            elif status == "failed":
                is_quota = error and "quota" in error.lower()
                if not is_quota and job.retries < MAX_RETRIES and results:
                    audio_path = results[job.video.id].audio_path
                    if audio_path:
                        print(f"  [RETRY {job.retries + 1}/{MAX_RETRIES}] {job.video.safe_title[:30]} -> {LANGUAGES[job.lang]}")
                        work_queue.append((job.video, job.lang, audio_path, job.retries + 1))
                else:
                    print(f"  [FAIL] {job.video.safe_title[:40]} -> {LANGUAGES[job.lang]}: {error or 'unknown'}")
                    if result:
                        result.failed.append(job.lang)
                total_done += 1
            else:
                still_pending.append(job)

        pending = still_pending

        # Submit more if slots available
        while work_queue and len(pending) < MAX_CONCURRENT_JOBS and not quota_exceeded:
            video, lang, audio_path, retries = work_queue.pop(0)
            try:
                job = await submit_job(session, api_key, video, audio_path, lang)
                if job:
                    job.retries = retries
                    pending.append(job)
                    job_results[job.id] = results[video.id]
                else:
                    results[video.id].failed.append(lang)
            except QuotaExceededError:
                quota_exceeded = True
                results[video.id].failed.append(lang)
                for v, l, _, _ in work_queue:
                    results[v.id].failed.append(l)
                work_queue.clear()
            await asyncio.sleep(0.3)

        print(f"  [{elapsed}s] {len(pending)} active, {total_done} done" + (f", {len(work_queue)} queued" if work_queue else ""))

    clear_jobs(output_dir)


def load_sources_from_file(filepath: Path) -> list[str]:
    """Load sources from file - one per line (URLs or local file paths)."""
    if not filepath.exists():
        return []
    return [line.strip() for line in filepath.read_text().strip().split("\n") if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="Dub YouTube videos or local files using ElevenLabs")
    parser.add_argument("-c", "--channel", default=DEFAULT_CHANNEL, help="YouTube channel URL")
    parser.add_argument("-n", "--episodes", type=int, default=DEFAULT_EPISODES, help="Number of episodes from channel")
    parser.add_argument("-l", "--languages", nargs="+", default=list(LANGUAGES.keys()), help="Target languages (es, hi, zh)")
    parser.add_argument("-o", "--output", type=Path, default=Path("dubbed_episodes"), help="Output directory")
    parser.add_argument("-s", "--sources", nargs="+", help="Sources: YouTube URLs or local video/audio files")
    parser.add_argument("-f", "--file", type=Path, help="File with sources (one per line)")
    args = parser.parse_args()

    sources = (args.sources or []) + (load_sources_from_file(args.file) if args.file else [])

    print("=" * 60)
    print("Podcast Dubbing Tool")
    print("=" * 60)
    if sources:
        print(f"Sources:   {len(sources)}")
    else:
        print(f"Channel:   {args.channel}")
        print(f"Episodes:  {args.episodes}")
    print(f"Languages: {', '.join(args.languages)}")
    print(f"Output:    {args.output}")

    asyncio.run(process_all(
        channel_url=args.channel if not sources else None,
        num_episodes=args.episodes,
        languages=args.languages,
        output_dir=args.output,
        sources=sources or None,
    ))


if __name__ == "__main__":
    main()
