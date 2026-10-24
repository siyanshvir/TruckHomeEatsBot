"""Run the self-correcting LearningBrain (learner.py) instead of the regular Brain.

    python3 learn_bot.py                         # play the live game, learning every day
    python3 learn_bot.py --practice --days 48    # one 48-day practice market
    python3 learn_bot.py --train 5 --days 48     # 5 practice markets in a row, lessons carry over
    python3 learn_bot.py --lessons               # print what has been learned so far
    python3 learn_bot.py --forget                # start over with no lessons

Lessons are kept in lessons.json next to this file. Uses bot.py for all the server plumbing.
"""
import argparse
import os
import sys

import bot
import learner


def practice(args, seed=None):
    body = {"days": args.days}
    if seed:
        body["seed"] = seed
    if args.rivals is not None:
        body["rivals"] = args.rivals
    status, res = bot.call("POST", "/api/v1/practice", body)
    if status not in (200, 201):
        sys.exit("could not start practice (%d): %s" % (status, res.get("error")))
    print("practice market %s started (%d days)" % (res["id"], args.days))
    final = bot.play(res["base"])
    print("practice over: cash $%.2f" % final["you"]["cash"])
    for t in sorted(final["trucks"], key=lambda t: -t["cash"]):
        print("  %-24s $%10.2f" % (t["truck_name"], t["cash"]))
    return final["you"]["cash"]


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description="Night Shift bot that learns from the trucks that beat it")
    ap.add_argument("--practice", action="store_true", help="play one private practice market")
    ap.add_argument("--train", type=int, default=0, help="play this many practice markets in a row")
    ap.add_argument("--seed", help="practice seed (with --train, seeds seed, seed+1, ...)")
    ap.add_argument("--days", type=int, default=48, help="practice market length")
    ap.add_argument("--rivals", type=int, help="practice house rivals")
    ap.add_argument("--memory", default=learner.DEFAULT_MEMORY, help="lessons file")
    ap.add_argument("--quiet", action="store_true", help="don't print each day's lessons")
    ap.add_argument("--lessons", action="store_true", help="print the lessons learned so far and exit")
    ap.add_argument("--forget", action="store_true", help="delete the lessons file and exit")
    args = ap.parse_args()

    if args.forget:
        if os.path.exists(args.memory):
            os.remove(args.memory)
        print("lessons cleared")
        return
    brain = learner.LearningBrain(memory_path=args.memory, verbose=not args.quiet)
    if args.lessons:
        print(learner.summary(brain.lessons))
        for e in brain.lessons["journal"][-10:]:
            print("day %s rank %s profit $%.0f: %s" % (e["day"], e["rank"], e["our_profit"], " | ".join(e["lessons"]) or "-"))
        return
    bot.BRAIN = brain
    if not bot.KEY:
        sys.exit("Set NIGHTSHIFT_KEY first (you get it when you register).")
    if args.train:
        results = []
        for i in range(args.train):
            seed = str(int(args.seed) + i) if args.seed else None
            results.append(practice(args, seed))
            print(learner.summary(brain.lessons))
        print("training cash by market: " + ", ".join("$%.0f" % c for c in results))
    elif args.practice:
        practice(args, args.seed)
        print(learner.summary(brain.lessons))
    else:
        print("playing the live game at %s with the learning brain (Ctrl+C to stop)" % bot.URL)
        bot.play("/api/v1")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
