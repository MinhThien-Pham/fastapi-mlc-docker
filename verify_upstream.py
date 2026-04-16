#!/usr/bin/env python3
"""Manual upstream verification for fastapi-mlc-docker.

Verifies the mlc-cli SHA running in Docker by executing smoke and full
integration tests, then records the result.

Usage:
    python verify_upstream.py          # verify + commit, no push
    python verify_upstream.py --push   # verify + commit + push
"""

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

METADATA = Path(".upstream-sha.json")
REPO_URL = "https://github.com/ballinyouup/mlc-cli.git"
API_URL = "http://localhost:8000"
SMOKE = "tests/integration/test_smoke.py"
FULL = "tests/integration/test_full_pipeline.py"
LABEL = "upstream-contract-fail"
MARKER = "<!-- verify-upstream-status -->"


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def die(msg):
    print(f"\n[FAIL] {msg}"); sys.exit(1)


# ── Preflight ────────────────────────────────────────────────────────────────

def commits_ahead_of_remote():
    """Return the number of local commits ahead of origin/main."""
    r = sh(["git", "rev-list", "--count", "origin/main..HEAD"])
    if r.returncode != 0:
        return -1  # unknown
    return int(r.stdout.strip())


def preflight(want_push=False):
    print("=== Preflight Checks ===\n")

    if not METADATA.is_file():
        die(f"{METADATA} not found")
    print("  ✓ Metadata file exists")

    r = sh(["docker", "compose", "ps", "--status", "running", "--format", "{{.Name}}"])
    if r.returncode != 0 or "web" not in r.stdout:
        die("Docker 'web' service not running. Run: docker compose up -d")
    print("  ✓ Docker 'web' service running")

    try:
        import httpx
        httpx.get(f"{API_URL}/health", timeout=5).raise_for_status()
    except Exception as e:
        die(f"API unreachable at {API_URL}: {e}")
    print("  ✓ API reachable")

    r = sh(["git", "status", "--porcelain", str(METADATA)])
    if r.stdout.strip():
        die(f"{METADATA} has uncommitted changes")
    print("  ✓ Metadata file clean in git")

    r = sh(["git", "branch", "--show-current"])
    if r.stdout.strip() != "main":
        die(f"Not on 'main' (on '{r.stdout.strip()}')")
    print(f"  ✓ On branch: main")

    if want_push:
        ahead = commits_ahead_of_remote()
        if ahead > 0:
            die(f"Local main is {ahead} commit(s) ahead of origin/main.\n"
                f"       Push or rebase those first, then re-run with --push.")
        print("  ✓ No unrelated local commits ahead of remote")

    print()


# ── SHA helpers ──────────────────────────────────────────────────────────────

def read_verified_sha():
    return json.loads(METADATA.read_text()).get("verified_sha", "")

def read_container_sha():
    r = sh(["docker", "compose", "exec", "-T", "web",
            "bash", "-c", "cd /workspace/mlc-cli && git rev-parse HEAD"])
    if r.returncode != 0:
        die(f"Cannot read SHA from container: {r.stderr.strip()}")
    return r.stdout.strip()


# ── Test execution ───────────────────────────────────────────────────────────

def run_test(label, script):
    print(f"\n{'='*60}\n  {label}\n{'='*60}\n")
    ok = subprocess.run([sys.executable, script]).returncode == 0
    print(f"\n--- {label}: {'PASSED' if ok else 'FAILED'} ---")
    return ok


# ── Metadata + Git ───────────────────────────────────────────────────────────

def commit_metadata(sha):
    data = {"repo": REPO_URL, "verified_sha": sha,
            "verified_date": datetime.now(timezone.utc).astimezone().isoformat()}
    METADATA.write_text(json.dumps(data, indent=2) + "\n")
    subprocess.run(["git", "add", str(METADATA)], check=True)
    subprocess.run(["git", "commit", "-m",
                    f"chore: verify upstream mlc-cli {sha[:12]}"], check=True)
    print(f"\n[OK] Committed verification for {sha[:12]}")

def git_push():
    r = subprocess.run(["git", "push"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[WARN] Push failed: {r.stderr.strip()}")
        return False
    print("[OK] Pushed to remote")
    return True


# ── Issue handling ───────────────────────────────────────────────────────────

def gh_ok():
    return sh(["gh", "--version"]).returncode == 0

def find_open_issue(sha):
    """Find an open issue for a specific SHA.

    Prefers an issue whose title or body mentions this SHA.
    Falls back to a single open labeled issue (safe, unambiguous).
    Returns None if multiple non-matching issues exist (ambiguous).
    """
    short = sha[:12]
    r = sh(["gh", "issue", "list", "--label", LABEL,
            "--state", "open", "--json", "number,title,body", "--limit", "10"])
    if r.returncode != 0:
        return None
    issues = json.loads(r.stdout)
    # Prefer an issue that mentions this specific SHA
    for issue in issues:
        text = issue.get("title", "") + issue.get("body", "")
        if short in text or sha in text:
            return issue["number"]
    # Fallback: only safe if exactly one open issue exists
    if len(issues) == 1:
        return issues[0]["number"]
    return None

def find_status_comment_id(issue_num):
    r = sh(["gh", "api",
            f"repos/{{owner}}/{{repo}}/issues/{issue_num}/comments",
            "--jq", f'[.[] | select(.body | contains("{MARKER}"))][0].id // empty'])
    return r.stdout.strip() or None if r.returncode == 0 else None

def status_body(sha, smoke, full, verified, pushed):
    ts = datetime.now(timezone.utc).astimezone().isoformat()
    def fmt(v): return "⏭️ not run" if v is None else ("✅ passed" if v else "❌ failed")
    yn = lambda v: "✅ yes" if v else "❌ no"
    return (f"{MARKER}\n### Verification Status\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| **SHA tested** | `{sha}` |\n"
            f"| **Smoke test** | {fmt(smoke)} |\n"
            f"| **Full test** | {fmt(full)} |\n"
            f"| **Verified locally** | {yn(verified)} |\n"
            f"| **Pushed** | {yn(pushed)} |\n"
            f"| **Last updated** | {ts} |\n")

def upsert_status(issue_num, body):
    cid = find_status_comment_id(issue_num)
    if cid:
        sh(["gh", "api", "-X", "PATCH",
            f"repos/{{owner}}/{{repo}}/issues/comments/{cid}",
            "-f", f"body={body}"])
    else:
        sh(["gh", "issue", "comment", str(issue_num), "--body", body])

def handle_issues(sha, smoke, full, verified, pushed):
    if not gh_ok():
        print("[WARN] gh CLI not available — skipping issue handling")
        return
    body = status_body(sha, smoke, full, verified, pushed)
    issue_num = find_open_issue(sha)

    if verified and pushed:
        if issue_num:
            upsert_status(issue_num, body)
            sh(["gh", "issue", "close", str(issue_num),
                "--comment", "Upstream verified and pushed. Closing."])
            print(f"[OK] Closed issue #{issue_num}")
    elif verified:
        if issue_num:
            upsert_status(issue_num, body)
            print(f"[OK] Updated status on issue #{issue_num}")
    else:
        if issue_num:
            upsert_status(issue_num, body)
            print(f"[OK] Updated status on issue #{issue_num}")
        else:
            sh(["gh", "label", "create", LABEL, "--color", "E11D48",
                "--description", "Upstream mlc-cli contract check failed"])
            r = sh(["gh", "issue", "create",
                    "--title", f"⚠️ Upstream mlc-cli verification failed ({sha[:12]})",
                    "--label", LABEL,
                    "--body", f"Manual verification failed for `{sha}`.\n\n{body}"])
            if r.returncode == 0:
                try:
                    new_num = int(r.stdout.strip().rstrip("/").split("/")[-1])
                    print(f"[OK] Created issue #{new_num}")
                except (ValueError, IndexError):
                    pass


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Manual upstream verification")
    ap.add_argument("--push", action="store_true", help="Push after successful verification")
    args = ap.parse_args()

    print("=== Manual Upstream Verification ===\n")
    preflight(want_push=args.push)

    verified_sha = read_verified_sha()
    container_sha = read_container_sha()
    print(f"Verified SHA:  {verified_sha}")
    print(f"Container SHA: {container_sha}")

    if container_sha == verified_sha:
        print("\n✅ Already verified. Nothing to do.")
        return

    print(f"\n⚠️  Container has {container_sha[:12]}, verified is {verified_sha[:12]}")
    print("Running verification tests...\n")

    smoke_ok = run_test("Smoke Integration Test", SMOKE)
    if not smoke_ok:
        print("\n❌ Smoke failed — skipping full test.")
        handle_issues(container_sha, False, None, False, False)
        sys.exit(1)

    full_ok = run_test("Full Integration Test", FULL)
    if not full_ok:
        print("\n❌ Full test failed.")
        handle_issues(container_sha, True, False, False, False)
        sys.exit(1)

    print("\n✅ All tests passed.")
    commit_metadata(container_sha)

    pushed = args.push and git_push()
    handle_issues(container_sha, True, True, True, pushed)

    if pushed:
        print("\n=== Verification Complete (committed + pushed) ===")
    else:
        print("\n=== Verification Complete (committed locally) ===")
        if not args.push:
            print("Run with --push to also push, or: git push")


if __name__ == "__main__":
    main()
