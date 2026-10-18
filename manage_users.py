"""
Manage who can sign in.

    python manage_users.py list
    python manage_users.py add alice            # prompts for a password (min 8 chars)
    python manage_users.py add bob --admin      # admins can see every user's documents
    python manage_users.py passwd alice
    python manage_users.py remove alice
"""
import argparse
import getpass
import sys

import auth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["list", "add", "passwd", "remove"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--admin", action="store_true")
    ap.add_argument("--password", help="avoid on shared machines: it ends up in shell history")
    args = ap.parse_args()

    if args.action == "list":
        for u in auth.list_users():
            print(f"{u['name']:<20} {'admin' if u['admin'] else ''}")
        return
    if not args.name:
        sys.exit("a username is required")
    if args.action == "remove":
        print("removed" if auth.remove_user(args.name) else "no such user")
        return
    password = args.password or getpass.getpass("Password: ")
    if not args.password and getpass.getpass("Repeat: ") != password:
        sys.exit("passwords differ")
    existing = {u["name"]: u["admin"] for u in auth.list_users()}
    if args.action == "add" and args.name in existing:
        sys.exit("user exists; use passwd")
    if args.action == "passwd" and args.name not in existing:
        sys.exit("no such user")
    auth.create_user(args.name, password, admin=args.admin or existing.get(args.name, False))
    print("saved")


if __name__ == "__main__":
    main()
