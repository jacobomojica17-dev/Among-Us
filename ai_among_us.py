"""
Mini Among Us for LLMs.
Each player is a different AI model. The script is the game engine:
it tracks rooms, tasks, kills, meetings and votes, and asks each model for its move.

Setup:
    pip install openai
    export API_KEY="your-openrouter-key"     # Windows PowerShell: $env:API_KEY="..."
    python ai_among_us.py

Local models (Ollama):
    export BASE_URL="http://localhost:11434/v1" API_KEY="ollama" MODELS="llama3.1"
"""
import json
import os
import random
import re
from collections import Counter, defaultdict

from openai import OpenAI

# ---------------- CONFIG ----------------
BASE_URL = os.getenv("BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("API_KEY", "")
# Comma-separated list in the MODELS env var, or edit this default.
# Check exact model IDs at https://openrouter.ai/models
MODELS = os.getenv(
    "MODELS",
    "openai/gpt-4o-mini,anthropic/claude-haiku-4.5,google/gemini-2.5-flash,deepseek/deepseek-chat",
).split(",")

NUM_PLAYERS = 8
NUM_IMPOSTORS = 2
TASKS_PER_PLAYER = 3
KILL_COOLDOWN = 2        # rounds between kills
MAX_ROUNDS = 30
DISCUSSION_ROUNDS = 2    # how many times each player speaks in a meeting
CONFIRM_EJECTS = True    # reveal the role of ejected players

COLORS = ["Red", "Blue", "Green", "Pink", "Orange", "Yellow", "Black", "White", "Purple", "Cyan"]

# Simplified Skeld map
EDGES = """Cafeteria-Weapons Cafeteria-Admin Cafeteria-MedBay Cafeteria-UpperEngine
Weapons-Navigation Navigation-Shields Shields-Communications Shields-Storage
Communications-Storage Storage-Admin Storage-Electrical Storage-LowerEngine
Electrical-LowerEngine LowerEngine-UpperEngine LowerEngine-Reactor LowerEngine-Security
UpperEngine-Reactor UpperEngine-Security UpperEngine-MedBay Reactor-Security""".split()
MAP = defaultdict(set)
for e in EDGES:
    a, b = e.split("-")
    MAP[a].add(b)
    MAP[b].add(a)
ROOMS = sorted(MAP)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# ---------------- HELPERS ----------------
def ask(player, prompt):
    """Send a prompt to the player's model and parse a JSON reply."""
    msgs = [{"role": "system", "content": player.system}, {"role": "user", "content": prompt}]
    for _ in range(2):  # one retry
        try:
            r = client.chat.completions.create(
                model=player.model, messages=msgs, temperature=1.0, max_tokens=500
            )
            m = re.search(r"\{.*\}", r.choices[0].message.content or "", re.S)
            if m:
                return json.loads(m.group())
        except Exception as ex:
            print(f"  (error from {player.name}/{player.model}: {ex})")
    return {}


def match(text, choices):
    """Case-insensitive match of text against a collection of strings."""
    t = str(text).strip().lower().replace(" ", "")
    for c in choices:
        if c.lower().replace(" ", "") == t:
            return c
    return None


class Player:
    def __init__(self, name, model, impostor):
        self.name, self.model, self.impostor = name, model, impostor
        self.room = "Cafeteria"
        self.alive = True
        self.cooldown = KILL_COOLDOWN
        self.tasks = {r: False for r in random.sample(ROOMS, TASKS_PER_PLAYER)}  # fake for impostors
        self.memory = []
        self.system = ""

    def remember(self, msg):
        self.memory.append(msg)

    def mem_text(self, n=25):
        return "\n".join(self.memory[-n:]) or "(nothing yet)"


# ---------------- GAME ----------------
class Game:
    def __init__(self):
        roles = [True] * NUM_IMPOSTORS + [False] * (NUM_PLAYERS - NUM_IMPOSTORS)
        random.shuffle(roles)
        names = random.sample(COLORS, NUM_PLAYERS)
        self.players = [Player(n, MODELS[i % len(MODELS)], r) for i, (n, r) in enumerate(zip(names, roles))]
        self.bodies = {}  # victim name -> room
        self.events = []
        imps = [p.name for p in self.players if p.impostor]
        for p in self.players:
            p.system = self.system_prompt(p, imps)

    def system_prompt(self, p, imps):
        base = (
            f"You are {p.name}, a player in a text-based game of Among Us on The Skeld. "
            f"There are {NUM_PLAYERS} players and {NUM_IMPOSTORS} impostors. "
            "Crewmates win by finishing all their tasks or voting out every impostor. "
            "Impostors win when they equal the number of crewmates. "
            "Always answer with a single JSON object and nothing else. "
        )
        if p.impostor:
            mates = [i for i in imps if i != p.name]
            base += (
                f"YOUR SECRET ROLE: IMPOSTOR. Your partner is {', '.join(mates) or 'nobody'}. "
                "Kill crewmates when nobody else can see you, fake tasks, and lie convincingly in meetings. "
                "Never admit you are an impostor."
            )
        else:
            base += (
                "YOUR ROLE: CREWMATE. Do your tasks, notice suspicious behavior, "
                "and share what you saw honestly in meetings."
            )
        return base

    def log(self, kind, text, **extra):
        self.events.append({"kind": kind, "text": text, **extra})
        print(text)

    def alive_players(self):
        return [p for p in self.players if p.alive]

    def winner(self):
        alive = self.alive_players()
        imps = [p for p in alive if p.impostor]
        crew = [p for p in alive if not p.impostor]
        if not imps:
            return "Crewmates (all impostors ejected)"
        if len(imps) >= len(crew):
            return "Impostors"
        if all(all(p.tasks.values()) for p in crew):
            return "Crewmates (all tasks done)"
        return None

    # ---- one player's turn ----
    def take_turn(self, p, rnd):
        others = [o.name for o in self.alive_players() if o is not p and o.room == p.room]
        here_bodies = [n for n, r in self.bodies.items() if r == p.room]
        p.remember(
            f"R{rnd}: In {p.room}. You see: {', '.join(others) or 'nobody'}."
            + (f" DEAD BODY: {', '.join(here_bodies)}!" if here_bodies else "")
        )
        todo = [r for r, d in p.tasks.items() if not d]
        prompt = (
            f"Round {rnd}. You are in {p.room}. Adjacent rooms: {', '.join(sorted(MAP[p.room]))}.\n"
            f"Players here: {', '.join(others) or 'nobody'}. Bodies here: {', '.join(here_bodies) or 'none'}.\n"
            f"Your remaining tasks (rooms): {', '.join(todo) or 'none'}"
            + (" (these are fake, use them as cover)" if p.impostor else "")
            + ".\n"
        )
        if p.impostor:
            prompt += f"Kill cooldown: {'READY' if p.cooldown <= 0 else str(p.cooldown) + ' rounds'}.\n"
        prompt += (
            f"\nYour memory:\n{p.mem_text()}\n\n"
            "Choose ONE action. Reply with JSON only:\n"
            '{"thought": "your private reasoning", "action": "move|task|kill|report|wait", "target": "room or player name, else empty"}\n'
            "- move: target = an adjacent room\n- task: do a task in the current room\n"
            "- kill: (impostor only) target = a crewmate in this room\n- report: report a body in this room"
        )
        out = ask(p, prompt)
        thought = out.get("thought", "")
        action = str(out.get("action", "wait")).lower().strip()
        target = out.get("target", "") or ""
        self.log("thought", f"  [{p.name} ({'IMP' if p.impostor else 'crew'}) thinks] {thought}")

        if action == "move":
            room = match(target, MAP[p.room])
            if room:
                p.room = room
                p.remember(f"R{rnd}: You moved to {room}.")
                self.log("action", f"{p.name} -> {room}")
            else:
                p.remember(f"R{rnd}: Invalid move, you stayed put.")
        elif action == "task":
            if p.tasks.get(p.room) is False:
                p.tasks[p.room] = True
                p.remember(f"R{rnd}: You {'faked' if p.impostor else 'completed'} a task in {p.room}.")
                self.log("action", f"{p.name} does a task in {p.room}")
            else:
                p.remember(f"R{rnd}: No task available here.")
        elif action == "kill" and p.impostor:
            victims = [o.name for o in self.alive_players() if o is not p and not o.impostor and o.room == p.room]
            v = match(target, victims)
            if v and p.cooldown <= 0:
                victim = next(o for o in self.players if o.name == v)
                victim.alive = False
                victim.remember("You were killed.")
                self.bodies[v] = p.room
                p.cooldown = KILL_COOLDOWN
                p.remember(f"R{rnd}: You killed {v} in {p.room}.")
                for w in self.alive_players():  # witnesses
                    if w is not p and w.room == p.room:
                        w.remember(f"R{rnd}: You SAW {p.name} kill {v} in {p.room}!")
                self.log("kill", f"*** {p.name} killed {v} in {p.room} ***")
            else:
                p.remember(f"R{rnd}: Kill failed (cooldown or no valid target).")
        elif action == "report" and here_bodies:
            return ("report", here_bodies)
        return None

    # ---- meeting ----
    def meeting(self, reporter, bodies_found):
        alive = self.alive_players()
        header = f"{reporter.name} reported the body of {', '.join(bodies_found)} in {reporter.room}."
        self.log("meeting", f"\n===== MEETING: {header} =====")
        for p in alive:
            p.remember(f"MEETING: {header}")
            p.room = "Cafeteria"
        self.bodies.clear()

        chat = []
        names = ", ".join(p.name for p in alive)
        for _ in range(DISCUSSION_ROUNDS):
            for p in random.sample(alive, len(alive)):
                prompt = (
                    f"MEETING. {header} Alive players: {names}.\n"
                    f"Discussion so far:\n{chr(10).join(chat) or '(nobody has spoken)'}\n\n"
                    f"Your memory:\n{p.mem_text()}\n\n"
                    'Reply with JSON only: {"thought": "private reasoning", "say": "1-3 sentences said out loud"}'
                )
                out = ask(p, prompt)
                self.log("thought", f"  [{p.name} thinks] {out.get('thought', '')}")
                line = f"{p.name}: {out.get('say', '...')}"
                chat.append(line)
                self.log("chat", line)

        votes = Counter()
        for p in alive:
            prompt = (
                f"VOTING. Alive players: {names}.\nDiscussion:\n{chr(10).join(chat)}\n\n"
                f"Your memory:\n{p.mem_text()}\n\n"
                'Reply with JSON only: {"thought": "private reasoning", "vote": "player name or skip"}'
            )
            out = ask(p, prompt)
            v = match(out.get("vote", "skip"), [a.name for a in alive if a is not p]) or "skip"
            votes[v] += 1
            self.log("vote", f"{p.name} votes {v}")

        top = votes.most_common(2)
        ejected = None
        if top and top[0][0] != "skip" and (len(top) == 1 or top[0][1] > top[1][1]):
            ejected = next(p for p in self.players if p.name == top[0][0])
        if ejected:
            ejected.alive = False
            role = ("an IMPOSTOR" if ejected.impostor else "NOT an impostor") if CONFIRM_EJECTS else "ejected"
            result = f"{ejected.name} was ejected ({role})."
        else:
            result = "Nobody was ejected."
        self.log("result", result)
        for p in self.alive_players():
            p.remember("MEETING CHAT:\n" + "\n".join(chat))
            p.remember(f"MEETING RESULT: {result}")

    # ---- main loop ----
    def run(self):
        print("Players:", ", ".join(f"{p.name}={p.model}" for p in self.players))
        print("Impostors (secret):", ", ".join(p.name for p in self.players if p.impostor), "\n")
        for rnd in range(1, MAX_ROUNDS + 1):
            self.log("round", f"\n--- Round {rnd} ---")
            for p in random.sample(self.players, len(self.players)):
                if not p.alive:
                    continue
                result = self.take_turn(p, rnd)
                if result:
                    self.meeting(p, result[1])
                    break
                if self.winner():
                    break
            for p in self.players:
                p.cooldown = max(0, p.cooldown - 1)
            w = self.winner()
            if w:
                self.log("end", f"\n### {w} win! ###")
                break
        else:
            self.log("end", "\n### Draw: round limit reached ###")
        with open("game_log.json", "w") as f:
            json.dump(self.events, f, indent=2)
        print("Saved game_log.json")


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("Set the API_KEY environment variable first (see the top of this file).")
    Game().run()
