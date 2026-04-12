"""
YouTube Data API v3 scraper for the ClickLens project.

Collects video metadata and thumbnails across gaming, travel, and fitness niches
for training a YouTube thumbnail click-through predictor.
"""

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("YOUTUBE_API_KEY")

NICHE_CONFIG = {
    "gaming": {
        "search_queries": [
            "gaming youtube",
            "let's play",
            "game review",
            "esports highlights",
        ],
        "seed_channel_ids": [
            "UCOpNcN46UbXVtpKMrmU4Abg",   # PewDiePie
            "UCX6OQ3DkcsbYNE6H8uQQuVA",   # MrBeast Gaming
            "UCIPPMRA040LQr5QPyJEbmXA",   # Mark Rober
            "UCYzPXprvl5Y-Sf0g4vX-m6g",   # jacksepticeye
            "UC-lHJZR3Gqxm24_Vd_AJ5Yw",   # PewDiePie (old)
        ],
    },
    "travel": {
        "search_queries": [
            "travel vlog",
            "travel guide",
            "backpacking",
            "travel tips",
        ],
        "seed_channel_ids": [
            "UCnTsUMBOA8E-OHJE-UrFOnA",   # Kara and Nate
            "UCyEd6QBSgat5kkC6svyjudA",   # Lost LeBlanc
            "UC0Ize0RLIbGdP5Iy5ox10VQ",   # Drew Binsky
            "UCFr3sz2t3bDp6Cux08B93KQ",   # Vagabrothers
            "UCt_NLJ4McJlCyYM-dSPRo7Q",   # Flying The Nest
        ],
    },
    "fitness": {
        "search_queries": [
            "fitness workout",
            "gym routine",
            "home workout",
            "bodybuilding",
        ],
        "seed_channel_ids": [
            "UCe0TLA0EsQbE-MjuHXevj2A",   # ATHLEAN-X
            "UCERm5yFZ1SptUEU4wZ2vJvw",   # Jeff Nippard
            "UCZIIRX8rkNjVpP-oLMHpeDw",   # Blogilates
            "UCqjwF8rxRsotnojGl4gM0Zw",   # Natacha Oceane
            "UCaBqRxHEMPKC4VlPCKgijbQ",   # Jeremy Ethier
        ],
    },
}

DEFAULT_MAX_CHANNELS = 50
DEFAULT_MAX_VIDEOS = 50

RATE_LIMIT_SLEEP = 0.25  # seconds between API calls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_duration_seconds(duration_str: str) -> int:
    """Convert ISO 8601 duration (e.g. PT1H2M3S) to total seconds."""
    match = re.match(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str or ""
    )
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def is_short_or_livestream(video: dict) -> bool:
    """Return True if the video should be filtered out as a Short or livestream."""
    title = video.get("snippet", {}).get("title", "")
    duration_iso = video.get("contentDetails", {}).get("duration", "")
    duration_sec = parse_duration_seconds(duration_iso)

    # Shorts: < 60 seconds or #Shorts in title
    if duration_sec < 60 or "#shorts" in title.lower():
        return True

    # Livestreams: has liveStreamingDetails or duration > 4 hours
    if video.get("liveStreamingDetails") is not None:
        return True
    if duration_sec > 4 * 3600:
        return True

    return False


def passes_view_and_age_filters(video: dict) -> bool:
    """Return True if the video passes the minimum view-count and age filters."""
    stats = video.get("statistics", {})
    view_count = int(stats.get("viewCount", 0))
    if view_count < 1000:
        return False

    published_str = video.get("snippet", {}).get("publishedAt", "")
    if not published_str:
        return False

    try:
        published_dt = datetime.fromisoformat(
            published_str.replace("Z", "+00:00")
        )
    except ValueError:
        return False

    age = datetime.now(timezone.utc) - published_dt
    if age.total_seconds() < 24 * 3600:
        return False

    return True


def download_thumbnail(video_id: str, thumbnails: dict, dest_dir: Path) -> tuple:
    """Download the best available thumbnail; return (local_path, url)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{video_id}.jpg"

    # Prefer maxresdefault, fall back to hqdefault
    for key in ("maxres", "high", "medium", "default"):
        thumb = thumbnails.get(key)
        if thumb:
            url = thumb["url"]
            break
    else:
        return None, None

    # Also try the direct maxresdefault URL which isn't always in the API response
    maxres_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    hqdefault_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

    for try_url in (maxres_url, hqdefault_url, url):
        try:
            resp = requests.get(try_url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 1000:
                dest_path.write_bytes(resp.content)
                return str(dest_path), try_url
        except requests.RequestException:
            continue

    return None, None


# ---------------------------------------------------------------------------
# YouTube API wrappers
# ---------------------------------------------------------------------------


def build_youtube_client():
    """Build and return a YouTube Data API v3 client."""
    if not API_KEY:
        logger.error(
            "YOUTUBE_API_KEY not set. Please set it in your .env file or "
            "as an environment variable."
        )
        sys.exit(1)
    return build("youtube", "v3", developerKey=API_KEY)


def search_channels(youtube, query: str, max_results: int = 25) -> list[str]:
    """Search for channel IDs matching a query."""
    channel_ids = []
    page_token = None

    while len(channel_ids) < max_results:
        try:
            request = youtube.search().list(
                q=query,
                type="channel",
                part="snippet",
                maxResults=min(50, max_results - len(channel_ids)),
                pageToken=page_token,
            )
            response = request.execute()
            time.sleep(RATE_LIMIT_SLEEP)
        except HttpError as e:
            logger.warning("API error searching channels for '%s': %s", query, e)
            break
        except Exception as e:
            logger.warning("Unexpected error searching channels for '%s': %s", query, e)
            break

        for item in response.get("items", []):
            cid = item.get("snippet", {}).get("channelId") or item.get("id", {}).get("channelId")
            if cid and cid not in channel_ids:
                channel_ids.append(cid)

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return channel_ids


def get_channel_details(youtube, channel_ids: list[str]) -> dict:
    """Return a mapping of channel_id -> {title, subscriber_count, uploads_playlist_id}."""
    details = {}
    # API allows up to 50 IDs per call
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i : i + 50]
        try:
            request = youtube.channels().list(
                id=",".join(batch),
                part="snippet,statistics,contentDetails",
            )
            response = request.execute()
            time.sleep(RATE_LIMIT_SLEEP)
        except HttpError as e:
            logger.warning("API error fetching channel details: %s", e)
            continue
        except Exception as e:
            logger.warning("Unexpected error fetching channel details: %s", e)
            continue

        for item in response.get("items", []):
            cid = item["id"]
            uploads_playlist = (
                item.get("contentDetails", {})
                .get("relatedPlaylists", {})
                .get("uploads")
            )
            details[cid] = {
                "title": item["snippet"]["title"],
                "subscriber_count": int(
                    item.get("statistics", {}).get("subscriberCount", 0)
                ),
                "uploads_playlist_id": uploads_playlist,
            }

    return details


def get_video_ids_from_playlist(
    youtube, playlist_id: str, max_results: int = 50
) -> list[str]:
    """Retrieve recent video IDs from a channel's uploads playlist."""
    video_ids = []
    page_token = None

    while len(video_ids) < max_results:
        try:
            request = youtube.playlistItems().list(
                playlistId=playlist_id,
                part="contentDetails",
                maxResults=min(50, max_results - len(video_ids)),
                pageToken=page_token,
            )
            response = request.execute()
            time.sleep(RATE_LIMIT_SLEEP)
        except HttpError as e:
            logger.warning("API error fetching playlist %s: %s", playlist_id, e)
            break
        except Exception as e:
            logger.warning("Unexpected error fetching playlist %s: %s", playlist_id, e)
            break

        for item in response.get("items", []):
            vid = item["contentDetails"]["videoId"]
            video_ids.append(vid)

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return video_ids


def get_video_details(youtube, video_ids: list[str]) -> list[dict]:
    """Fetch full details for a list of video IDs (batched 50 at a time)."""
    all_videos = []

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        try:
            request = youtube.videos().list(
                id=",".join(batch),
                part="snippet,contentDetails,statistics,liveStreamingDetails",
            )
            response = request.execute()
            time.sleep(RATE_LIMIT_SLEEP)
        except HttpError as e:
            logger.warning("API error fetching video details: %s", e)
            continue
        except Exception as e:
            logger.warning("Unexpected error fetching video details: %s", e)
            continue

        all_videos.extend(response.get("items", []))

    return all_videos


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def discover_channels(youtube, niche: str, max_channels: int) -> list[str]:
    """Discover channel IDs for a niche via search queries + seed channels."""
    config = NICHE_CONFIG[niche]
    channel_ids = list(config["seed_channel_ids"])

    queries = config["search_queries"]
    per_query = max(1, (max_channels - len(channel_ids)) // len(queries))

    logger.info(
        "Discovering channels for '%s' (seed=%d, per_query=%d)",
        niche,
        len(channel_ids),
        per_query,
    )

    for query in tqdm(queries, desc=f"Searching channels [{niche}]"):
        found = search_channels(youtube, query, max_results=per_query)
        for cid in found:
            if cid not in channel_ids:
                channel_ids.append(cid)
            if len(channel_ids) >= max_channels:
                break
        if len(channel_ids) >= max_channels:
            break

    channel_ids = channel_ids[:max_channels]
    logger.info("Discovered %d channels for '%s'", len(channel_ids), niche)
    return channel_ids


def scrape_niche(
    youtube,
    niche: str,
    max_channels: int = DEFAULT_MAX_CHANNELS,
    max_videos: int = DEFAULT_MAX_VIDEOS,
) -> list[dict]:
    """Full scraping pipeline for one niche. Returns list of metadata dicts."""
    logger.info("=== Starting scrape for niche: %s ===", niche)

    # 1. Discover channels
    channel_ids = discover_channels(youtube, niche, max_channels)

    # 2. Get channel details (subscriber count, uploads playlist)
    logger.info("Fetching channel details for %d channels ...", len(channel_ids))
    channel_info = get_channel_details(youtube, channel_ids)
    logger.info("Got details for %d channels", len(channel_info))

    # 3. For each channel, fetch recent videos
    raw_dir = PROJECT_ROOT / "data" / "raw" / niche
    raw_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for cid in tqdm(channel_info, desc=f"Channels [{niche}]"):
        info = channel_info[cid]
        uploads_pid = info.get("uploads_playlist_id")
        if not uploads_pid:
            logger.debug("No uploads playlist for channel %s, skipping", cid)
            continue

        # 3a. Get video IDs from uploads playlist
        video_ids = get_video_ids_from_playlist(
            youtube, uploads_pid, max_results=max_videos
        )
        if not video_ids:
            continue

        # 3b. Get full video details
        videos = get_video_details(youtube, video_ids)

        for video in videos:
            # Apply filters
            if is_short_or_livestream(video):
                continue
            if not passes_view_and_age_filters(video):
                continue

            video_id = video["id"]
            snippet = video["snippet"]
            stats = video.get("statistics", {})

            # Compute video age
            published_str = snippet.get("publishedAt", "")
            try:
                published_dt = datetime.fromisoformat(
                    published_str.replace("Z", "+00:00")
                )
                video_age_days = (
                    datetime.now(timezone.utc) - published_dt
                ).days
            except ValueError:
                video_age_days = None

            # Download thumbnail
            thumbnails = snippet.get("thumbnails", {})
            thumb_path, thumb_url = download_thumbnail(
                video_id, thumbnails, raw_dir
            )

            if thumb_path is None:
                logger.debug("Could not download thumbnail for %s", video_id)
                continue

            record = {
                "video_id": video_id,
                "channel_id": cid,
                "channel_title": info["title"],
                "title": snippet.get("title", ""),
                "niche": niche,
                "view_count": int(stats.get("viewCount", 0)),
                "subscriber_count": info["subscriber_count"],
                "published_at": published_str,
                "video_age_days": video_age_days,
                "thumbnail_path": thumb_path,
                "thumbnail_url": thumb_url,
            }
            records.append(record)

    logger.info(
        "Collected %d valid videos for niche '%s'", len(records), niche
    )
    return records


def run_pipeline(niches: list[str], max_channels: int, max_videos: int):
    """Execute the full scraping pipeline across all requested niches."""
    youtube = build_youtube_client()

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    for niche in niches:
        if niche not in NICHE_CONFIG:
            logger.warning("Unknown niche '%s', skipping.", niche)
            continue
        records = scrape_niche(youtube, niche, max_channels, max_videos)
        all_records.extend(records)

    if not all_records:
        logger.warning("No videos collected. Check your API key and quota.")
        return

    df = pd.DataFrame(all_records)
    csv_path = processed_dir / "video_metadata.csv"
    df.to_csv(csv_path, index=False)
    logger.info(
        "Saved %d video records to %s", len(df), csv_path
    )

    # Print summary
    logger.info("--- Summary ---")
    for niche in niches:
        subset = df[df["niche"] == niche] if niche in df["niche"].values else pd.DataFrame()
        logger.info(
            "  %s: %d videos, %d unique channels",
            niche,
            len(subset),
            subset["channel_id"].nunique() if len(subset) > 0 else 0,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape YouTube video metadata and thumbnails for ClickLens."
    )
    parser.add_argument(
        "--niches",
        nargs="+",
        default=list(NICHE_CONFIG.keys()),
        choices=list(NICHE_CONFIG.keys()),
        help="Niches to scrape (default: all).",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=DEFAULT_MAX_CHANNELS,
        help=f"Max channels per niche (default: {DEFAULT_MAX_CHANNELS}).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=DEFAULT_MAX_VIDEOS,
        help=f"Max videos per channel (default: {DEFAULT_MAX_VIDEOS}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(
        "ClickLens YouTube Scraper starting: niches=%s, max_channels=%d, max_videos=%d",
        args.niches,
        args.max_channels,
        args.max_videos,
    )
    run_pipeline(args.niches, args.max_channels, args.max_videos)
