#!/usr/bin/env python3
"""
Dub YouTube videos into multiple languages using ElevenLabs API.

Usage:
    python dub_podcast.py -f urls.txt              # Dub videos from file
    python dub_podcast.py -u URL1 URL2             # Dub specific URLs
    python dub_podcast.py -n 5                     # Dub last 5 from channel
"""

import argparse
import asyncio
import json
import logging
import os
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


@dataclass
class Video:
    id: str
    title: str
    url: str

    @property
    def safe_title(self) -> str:
        return "".join(c if c.isalnum() or c in " -_" else "_" for c in self.title)[:80]

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "url": self.url}

    @classmethod
    def from_dict(cls, d: dict) -> "Video":
        return cls(id=d["id"], title=d["title"], url=d["url"])


@dataclass
class DubbingJob:
    id: str
    video: Video
    lang: str
    retries: int = 0

    def to_dict(self) -> dict:
        return {"id": self.id, "video": self.video.to_dict(), "lang": self.lang, "retries": self.retries}

    @classmethod
    def from_dict(cls, d: dict) -> "DubbingJob":
        return cls(id=d["id"], video=Video.from_dict(d["video"]), lang=d["lang"], retries=d.get("retries", 0))


@dataclass
class VideoResult:
    video: Video
    audio_path: Path | None = None
    success: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


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


# YouTube functions
def fetch_videos(channel_url: str, count: int) -> list[Video]:
    print(f"\nFetching {count} videos from channel...")
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "playlistend": count}) as ydl:
        result = ydl.extract_info(channel_url, download=False)
        if not result:
            return []
        videos = []
        for entry in result.get("entries", []):
            if entry and entry.get("id"):
                video = Video(id=entry["id"], title=entry.get("title", "Unknown"), url=f"https://www.youtube.com/watch?v={entry['id']}")
                videos.append(video)
                print(f"  {video.title[:60]}")
        return videos


def fetch_video_info(url: str) -> Video | None:
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "ignoreerrors": True}) as ydl:
        try:
            result = ydl.extract_info(url, download=False)
            if result:
                return Video(id=result["id"], title=result.get("title", "Unknown"), url=url)
        except Exception:
            pass
    return None


def fetch_videos_from_urls(urls: list[str]) -> list[Video]:
    print(f"\nFetching info for {len(urls)} videos...")
    videos = []
    for url in urls:
        video = fetch_video_info(url)
        if video:
            videos.append(video)
            print(f"  {video.title[:60]}")
        else:
            print(f"  [FAIL] Could not fetch: {url[:50]}")
    return videos


def download_audio(video: Video, output_dir: Path) -> Path | None:
    output_path = output_dir / f"{video.safe_title}.mp3"
    if output_path.exists():
        return output_path

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / f"{video.safe_title}.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video.url])
        if output_path.exists():
            print(f"  [DL] {video.safe_title[:50]} ({output_path.stat().st_size / 1024 / 1024:.1f}MB)")
            return output_path
    except Exception:
        pass
    print(f"  [DL FAIL] {video.safe_title[:50]}")
    return None


# ElevenLabs API functions
async def submit_job(session: aiohttp.ClientSession, api_key: str, video: Video, audio_path: Path, lang: str) -> DubbingJob | None:
    data = aiohttp.FormData()
    data.add_field("file", open(audio_path, "rb"), filename=audio_path.name, content_type="audio/mpeg")
    data.add_field("target_lang", lang)
    data.add_field("source_lang", "en")
    data.add_field("name", f"{video.title[:50]} - {LANGUAGES[lang]}")
    data.add_field("watermark", "true")

    try:
        async with session.post(f"{ELEVENLABS_API}/dubbing", headers={"xi-api-key": api_key}, data=data) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"  [SUBMIT] {video.safe_title[:40]} -> {LANGUAGES[lang]}")
                return DubbingJob(id=result["dubbing_id"], video=video, lang=lang)
            error = await resp.text()
            if "quota_exceeded" in error.lower():
                raise QuotaExceededError(error)
            print(f"  [FAIL] {video.safe_title[:40]} -> {LANGUAGES[lang]}: {error[:80]}")
    except QuotaExceededError:
        raise
    except Exception as e:
        print(f"  [ERROR] {video.safe_title[:40]} -> {LANGUAGES[lang]}: {e}")
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
async def process_all(channel_url: str | None, num_episodes: int, languages: list[str], output_dir: Path, urls: list[str] | None = None):
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

        # Get videos
        videos = fetch_videos_from_urls(urls) if urls else fetch_videos(channel_url, num_episodes)
        if not videos:
            print("No videos found.")
            return

        print(f"\nDubbing {len(videos)} videos into {', '.join(LANGUAGES[l] for l in languages)}")

        # Track results
        results: dict[str, VideoResult] = {v.id: VideoResult(video=v) for v in videos}

        # Check existing and build work queue
        work_queue: list[tuple[Video, str, int]] = []  # (video, lang, retries)
        for video in videos:
            result = results[video.id]
            episode_dir = output_dir / video.safe_title
            for lang in languages:
                if (episode_dir / f"{LANGUAGES[lang]}.mp3").exists():
                    result.skipped.append(lang)
                else:
                    work_queue.append((video, lang, 0))
            if result.skipped:
                print(f"  [SKIP] {video.safe_title[:50]} ({', '.join(LANGUAGES[l] for l in result.skipped)})")

        if not work_queue:
            print("\nAll videos already dubbed!")
            return

        # Download audio for videos that need it
        videos_needing_audio = list({v.id: v for v, _, _ in work_queue}.values())
        print(f"\n{'='*60}\nDownloading audio ({len(videos_needing_audio)} videos)\n{'='*60}")

        with ThreadPoolExecutor(max_workers=3) as pool:
            for video, path in pool.map(lambda v: (v, download_audio(v, temp_dir)), videos_needing_audio):
                results[video.id].audio_path = path

        # Build final queue with audio paths
        final_queue: list[tuple[Video, str, Path, int]] = []
        for video, lang, retries in work_queue:
            if results[video.id].audio_path:
                final_queue.append((video, lang, results[video.id].audio_path, retries))
            else:
                results[video.id].failed.append(lang)

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


def load_urls_from_file(filepath: Path) -> list[str]:
    if not filepath.exists():
        return []
    return [line.strip() for line in filepath.read_text().strip().split("\n") if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="Dub YouTube videos using ElevenLabs")
    parser.add_argument("-c", "--channel", default=DEFAULT_CHANNEL, help="YouTube channel URL")
    parser.add_argument("-n", "--episodes", type=int, default=DEFAULT_EPISODES, help="Number of episodes")
    parser.add_argument("-l", "--languages", nargs="+", default=list(LANGUAGES.keys()), help="Target languages")
    parser.add_argument("-o", "--output", type=Path, default=Path("dubbed_episodes"), help="Output directory")
    parser.add_argument("-u", "--urls", nargs="+", help="Specific video URLs")
    parser.add_argument("-f", "--file", type=Path, help="File with video URLs")
    args = parser.parse_args()

    urls = (args.urls or []) + (load_urls_from_file(args.file) if args.file else [])

    print("=" * 60)
    print("YouTube Dubbing Tool")
    print("=" * 60)
    if urls:
        print(f"Videos:    {len(urls)} URLs")
    else:
        print(f"Channel:   {args.channel}")
        print(f"Episodes:  {args.episodes}")
    print(f"Languages: {', '.join(args.languages)}")
    print(f"Output:    {args.output}")

    asyncio.run(process_all(
        channel_url=args.channel if not urls else None,
        num_episodes=args.episodes,
        languages=args.languages,
        output_dir=args.output,
        urls=urls or None,
    ))


if __name__ == "__main__":
    main()
