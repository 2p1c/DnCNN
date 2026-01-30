# scripts/git_auto.py
import subprocess
import sys
import datetime

def run_command(command):
    try:
        result = subprocess.run(
            command, check=True, text=True, capture_output=True, shell=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing '{command}': {e.stderr}")
        sys.exit(1)

def main():
    # 1. Check status
    status = run_command("git status --porcelain")
    if not status:
        print("[OK] No changes to commit.")
        return

    # 2. Get commit message from args or use default
    if len(sys.argv) > 1:
        commit_msg = sys.argv[1]
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"chore: auto update at {timestamp}"

    print(f"[Run] Staging and Committing with message: '{commit_msg}'...")

    # 3. Git Operations
    run_command("git add .")
    run_command(f'git commit -m "{commit_msg}"')
    
    # 4. Push (Safe check)
    try:
        print("[Run] Pushing to remote...")
        run_command("git push")
        print("[OK] Success! Code pushed to remote.")
    except Exception:
        print("[WARN] Commit success, but Push failed (maybe no upstream branch?).")

if __name__ == "__main__":
    main()