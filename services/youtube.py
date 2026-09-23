"""
YouTube tutorial search service without requiring an API key.
Scrapes YouTube search results for educational lecture videos.
"""

import json
import re
import urllib.parse
import urllib.request


def search_youtube(query: str, max_results: int = 3) -> list[dict]:
    """Search YouTube for educational tutorials matching the query without API key."""
    try:
        clean_query = query.strip()
        search_terms = clean_query + " tutorial lecture"
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(search_terms)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=3.5) as response:
            html = response.read().decode("utf-8")
            match = re.search(r"var ytInitialData = ({.*?});</script>", html)
            if not match:
                return []
            data = json.loads(match.group(1))
            contents = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
            videos = []
            for section in contents:
                items = section.get("itemSectionRenderer", {}).get("contents", [])
                for item in items:
                    v = item.get("videoRenderer")
                    if v and "videoId" in v:
                        vid_id = v.get("videoId")
                        title = v.get("title", {}).get("runs", [{}])[0].get("text", "")
                        channel = v.get("ownerText", {}).get("runs", [{}])[0].get("text", "")
                        duration = v.get("lengthText", {}).get("simpleText", "")
                        if vid_id and title and "#shorts" not in title.lower() and "#short" not in title.lower():
                            videos.append({
                                "id": vid_id,
                                "title": title,
                                "channel": channel,
                                "duration": duration,
                                "url": f"https://www.youtube.com/watch?v={vid_id}",
                                "thumbnail": f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg",
                            })
                            if len(videos) >= max_results:
                                break
                if len(videos) >= max_results:
                    break
            return videos
    except Exception as e:
        print(f"YouTube search notice: {e}")
        return []
