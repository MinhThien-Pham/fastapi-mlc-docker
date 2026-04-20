#!/usr/bin/env python3
"""Manual upstream verification for fastapi-mlc-docker.

Tests the latest upstream mlc-cli HEAD as a candidate and promotes it
to the pinned (approved) SHA if smoke + full integration tests pass.

State model:
  pinned_sha  — the approved upstream version this project uses
  candidate   — upstream HEAD, fetched at verify time, tested intentionally

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
    r = sh(["git", "rev-list", "--count", "origin/main..HEAD"])
    return int(r.stdout.strip()) if r.returncode == 0 else -1

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

def read_pinned_sha():
    return json.loads(METADATA.read_text()).get("pinned_sha", "")

def fetch_upstream_head():
    """Get the latest SHA from Bryan's repo (the candidate)."""
    r = sh(["git", "ls-remote", REPO_URL, "HEAD"])
    if r.returncode != 0:
        die(f"Cannot fetch upstream HEAD: {r.stderr.strip()}")
    return r.stdout.split()[0]

def read_container_sha():
    r = sh(["docker", "compose", "exec", "-T", "web",
            "bash", "-c", "cd /workspace/mlc-cli && git rev-parse HEAD"])
    if r.returncode != 0:
        die(f"Cannot read SHA from container: {r.stderr.strip()}")
    return r.stdout.strip()

def checkout_in_container(sha):
    """Update the container's mlc-cli to a specific SHA."""
    print(f"\n[INFO] Updating container mlc-cli to {sha[:12]}...")
    r = sh(["docker", "compose", "exec", "-T", "web",
            "bash", "-c",
            f"cd /workspace/mlc-cli && git fetch origin && git checkout {sha}"])
    if r.returncode != 0:
        die(f"Failed to checkout {sha[:12]} in container: {r.stderr.strip()}")
    print(f"  ✓ Container now at {sha[:12]}")


# ── Test execution ───────────────────────────────────────────────────────────

def run_test(label, script):
    print(f"\n{'='*60}\n  {label}\n{'='*60}\n")
    ok = subprocess.run([sys.executable, script]).returncode == 0
    print(f"\n--- {label}: {'PASSED' if ok else 'FAILED'} ---")
    return ok


# ── Metadata + Git ───────────────────────────────────────────────────────────

def commit_metadata(sha):
    data = {"repo": REPO_URL, "pinned_sha": sha,
            "pinned_date": datetime.now(timezone.utc).astimezone().isoformat()}
    METADATA.write_text(json.dumps(data, indent=2) + "\n")
    subprocess.run(["git", "add", str(METADATA)], check=True)
    subprocess.run(["git", "commit", "-m",
                    f"chore: pin upstream mlc-cli to {sha[:12]}"], check=True)
    print(f"\n[OK] Pinned upstream to {sha[:12]}")

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
    """Find an open issue for a specific SHA (or single open labeled issue)."""
    short = sha[:12]
    r = sh(["gh", "issue", "list", "--label", LABEL,
            "--state", "open", "--json", "number,title,body", "--limit", "10"])
    if r.returncode != 0:
        return None
    issues = json.loads(r.stdout)
    for issue in issues:
        text = issue.get("title", "") + issue.get("body", "")
        if short in text or sha in text:
            return issue["number"]
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
            f"| **Candidate SHA** | `{sha}` |\n"
            f"| **Smoke test** | {fmt(smoke)} |\n"
            f"| **Full test** | {fmt(full)} |\n"
            f"| **Promoted to pinned** | {yn(verified)} |\n"
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
                "--comment", "Candidate verified and promoted. Closing."])
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
                    "--body", f"Manual verification failed for candidate `{sha}`.\n\n{body}"])
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

    pinned_sha = read_pinned_sha()
    candidate_sha = fetch_upstream_head()
    print(f"Pinned SHA:    {pinned_sha}")
    print(f"Candidate SHA: {candidate_sha}  (upstream HEAD)")

    if candidate_sha == pinned_sha:
        print("\n✅ Upstream HEAD matches pinned SHA. Nothing to do.")
        return

    print(f"\n⚠️  Candidate {candidate_sha[:12]} differs from pinned {pinned_sha[:12]}")

    original_container_sha = read_container_sha()

    # Checkout candidate in container
    checkout_in_container(candidate_sha)

    # Verify the container is now at the candidate
    actual = read_container_sha()
    if actual != candidate_sha:
        die(f"Container SHA {actual[:12]} != candidate {candidate_sha[:12]} after checkout")

    print("\nRunning verification tests against candidate...\n")

    verification_success = False
    try:
        # Smoke test
        smoke_ok = run_test("Smoke Integration Test", SMOKE)
        if not smoke_ok:
            print("\n❌ Smoke failed — skipping full test.")
            handle_issues(candidate_sha, False, None, False, False)
            sys.exit(1)

        # Full test
        full_ok = run_test("Full Integration Test", FULL)
        if not full_ok:
            print("\n❌ Full test failed.")
            handle_issues(candidate_sha, True, False, False, False)
            sys.exit(1)
            
        verification_success = True

    except SystemExit:
        # Re-raise SystemExit so sys.exit(1) calls propagate normally after finally block
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error during verification: {e}")
        sys.exit(1)

    finally:
        if not verification_success:
            print(f"\n[INFO] Rolling back container to original SHA {original_container_sha[:12]}...")
            checkout_in_container(original_container_sha)

    # Both passed — promote candidate to pinned
    print("\n✅ All tests passed. Promoting candidate to pinned.")
    commit_metadata(candidate_sha)

    pushed = args.push and git_push()
    handle_issues(candidate_sha, True, True, True, pushed)

    if pushed:
        print("\n=== Verification Complete (pinned + pushed) ===")
    else:
        print("\n=== Verification Complete (pinned locally) ===")
        if not args.push:
            print("Run with --push to also push, or: git push")


if __name__ == "__main__":
    main()
