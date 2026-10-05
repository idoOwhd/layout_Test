#!/usr/bin/env python3
"""Fetch every referenced issue/PR and retain independent, auditable records.

Source verification does not count as a GPU experiment or a native PR replay.
No downloaded source is executed by this program.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import urllib.error
import urllib.request


def fetch(url: str, token: str = "") -> tuple[str, dict]:
    headers = {"User-Agent": "layout-Test-RQ1-source-audit",
               "Accept": "application/vnd.github+json" if "api.github.com/" in url else "text/html,*/*"}
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = "Bearer " + token
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=40) as r:
        data = r.read().decode("utf-8", errors="replace")
        return data, dict(r.headers)


def audit(url: str, destination: Path, token: str) -> dict:
    match = re.fullmatch(r"https://github.com/([^/]+/[^/]+)/(pull|issues)/(\d+)", url)
    if not match:
        return {"url": url, "source_status": "not_issue_or_pr"}
    repo, kind, number = match.groups()
    name = repo.replace("/", "__") + "__" + kind + "__" + number
    row = {"url": url, "repo": repo, "kind": kind, "number": int(number),
           "source_status": "unverified", "native_reproduction_status": "not_run"}
    errors = []
    for origin in ("api", "html"):
        try:
            endpoint = f"https://api.github.com/repos/{repo}/issues/{number}" if origin == "api" else url
            text, headers = fetch(endpoint, token)
            path = destination / (name + (".json" if origin == "api" else ".html"))
            path.write_text(text, encoding="utf-8")
            row.update(source_status="verified", fetch_origin=origin,
                       artifact=str(path), sha256=hashlib.sha256(text.encode()).hexdigest())
            if origin == "api":
                issue = json.loads(text)
                row.update(title=issue.get("title"), body=issue.get("body") or "",
                           created_at=issue.get("created_at"), updated_at=issue.get("updated_at"),
                           state=issue.get("state"), canonical_url=issue.get("html_url"),
                           actual_kind="pull" if "pull_request" in issue else "issues")
                if row["actual_kind"] != kind:
                    row["source_status"] = "link_kind_mismatch"
            else:
                title = re.search(r"<title>(.*?)</title>", text, re.S)
                row["title"] = html.unescape(title.group(1)).strip() if title else ""
                row["body"] = ""
                row["source_status"] = "html_verified_body_pending"
            break
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            errors.append(f"{origin}: {type(error).__name__}: {error}")
    if kind == "pull" and row["source_status"] in {"verified", "html_verified_body_pending"}:
        try:
            diff, _ = fetch(url + ".diff")
            path = destination / (name + ".diff")
            path.write_text(diff, encoding="utf-8")
            row["diff_artifact"] = str(path)
            row["changed_files"] = re.findall(r"^diff --git a/(.*?) b/", diff, re.M)
            row["test_files"] = [p for p in row["changed_files"]
                                 if "test" in p.lower() or "bench" in p.lower()]
        except (urllib.error.URLError, TimeoutError) as error:
            errors.append(f"diff: {type(error).__name__}: {error}")
    row["fetch_errors"] = errors
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path,
                        default=Path(__file__).with_name("rq1_shared_github_urls.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-from", type=Path,
                        help="Reuse verified records; retry failures into a NEW directory")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    urls = sorted(set(catalog["urls"]))
    rows = []
    if args.retry_from:
        previous=json.loads((args.retry_from/"source_audit.json").read_text())
        for row in previous["records"]:
            if row["url"] not in urls or row.get("source_status")!="verified":continue
            row=dict(row)
            for key in ("artifact","diff_artifact"):
                if row.get(key):
                    source=Path(row[key]);target=args.output_dir/source.name
                    shutil.copyfile(source,target);row[key]=str(target)
            rows.append(row)
    done={r["url"] for r in rows}
    token = os.environ.get("GITHUB_TOKEN", "")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(audit, url, args.output_dir, token): url for url in urls if url not in done}
        for job in as_completed(jobs):
            try:
                row = job.result()
            except Exception as error:
                row = {"url": jobs[job], "source_status": "audit_error",
                       "error": f"{type(error).__name__}: {error}"}
            rows.append(row)
            print(f"[{len(rows)}/{len(urls)}] {row['source_status']} {row['url']}", flush=True)
    rows.sort(key=lambda r: r["url"])
    (args.output_dir / "source_audit.json").write_text(
        json.dumps({"source": catalog["source"], "expected": len(urls), "records": rows},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = ["# 全链接源码审计", "", "源码可访问不是性能验证成功。每个链接独立记录。", "",
              "| URL | status | title | native replay |", "|---|---|---|---|"]
    for row in rows:
        title = str(row.get("title", "")).replace("|", "\\|").replace("\n", " ")
        report.append(f"| {row['url']} | {row['source_status']} | {title} | not_run |")
    (args.output_dir / "SOURCE_AUDIT_CN.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    return 0 if len(rows) == len(urls) else 2


if __name__ == "__main__":
    raise SystemExit(main())
