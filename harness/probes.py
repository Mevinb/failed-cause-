import random
import re

FEWSHOT = (
    "Answer each question with a short answer.\n"
    "Q: What is the capital of Italy?\nA: Rome\n"
    "Q: What is 7 + 5?\nA: 12\n"
    "Q: {q}\nA:"
)

QA = {
    "fact": [
        ("What is the capital of Japan?", ["tokyo"]),
        ("What is the chemical symbol for gold?", ["au"]),
        ("Who wrote Romeo and Juliet?", ["shakespeare"]),
        ("What is the largest planet in the solar system?", ["jupiter"]),
        ("At what temperature in Celsius does water boil at sea level?", ["100"]),
        ("How many legs does a spider have?", ["8", "eight"]),
        ("Who wrote the novel 1984?", ["orwell"]),
        ("What is the currency of the United Kingdom?", ["pound"]),
        ("Which gas do plants absorb from the air for photosynthesis?", ["carbon dioxide", "co2"]),
        ("Who was the first person to walk on the Moon?", ["armstrong"]),
        ("What is the capital of Australia?", ["canberra"]),
        ("How many continents are there?", ["7", "seven"]),
    ],
    "arith": [
        ("What is 23 + 19?", ["42"]),
        ("What is 9 * 8?", ["72"]),
        ("What is 144 / 12?", ["12"]),
        ("What is 100 - 37?", ["63"]),
        ("What is 15 * 15?", ["225"]),
        ("What is 2 to the power of 10?", ["1024"]),
        ("What is the square root of 81?", ["9"]),
        ("What is 17 + 26?", ["43"]),
        ("What is 6 * 7 + 3?", ["45"]),
        ("What is 1000 - 1?", ["999"]),
    ],
    "seq": [
        ("What comes next: 2, 4, 8, 16, ?", ["32"]),
        ("What comes next: 1, 1, 2, 3, 5, 8, ?", ["13"]),
        ("What comes next: Monday, Tuesday, Wednesday, ?", ["thursday"]),
        ("What comes next: A, C, E, G, ?", ["i"]),
        ("What comes next: 10, 20, 30, ?", ["40"]),
        ("What comes next: January, March, May, ?", ["july"]),
    ],
    "lang": [
        ("What is the French word for 'cat'?", ["chat"]),
        ("What is the Spanish word for 'water'?", ["agua"]),
        ("What is the German word for 'dog'?", ["hund"]),
        ("What is the opposite of 'hot'?", ["cold"]),
        ("What is the plural of 'mouse'?", ["mice"]),
        ("What is the past tense of 'go'?", ["went"]),
        ("Repeat exactly: blue tiger seven lamp", ["blue tiger seven lamp"]),
        ("Repeat exactly: quartz meadow ninety violin", ["quartz meadow ninety violin"]),
    ],
}

FILLER = [
    "The grass is green.", "The sky is blue.", "The sun is yellow.", "Here we go.",
    "There and back again.", "The river flows slowly to the sea.", "Birds sing in the morning.",
    "A cat sleeps on the warm wall.", "Clouds drift over the quiet hills.",
]


def passkey_prompt(key, n_sent, depth, seed):
    rng = random.Random(seed)
    sents = [rng.choice(FILLER) for _ in range(n_sent)]
    sents.insert(int(n_sent * depth), f"The pass key is {key}. Remember it. {key} is the pass key.")
    return (
        "There is a pass key hidden inside a lot of irrelevant text. Find it and memorize it.\n"
        + " ".join(sents)
        + "\nQ: What is the pass key?\nA:"
    )


PASSKEY = [
    (48213, 120, 0.1), (77051, 120, 0.5), (30962, 120, 0.9),
    (61487, 260, 0.1), (25830, 260, 0.5), (93174, 260, 0.9),
]

FREEFORM = [
    "The ocean at night is",
    "Here is a short story about a robot who discovers music:\n",
]


def build():
    items = []
    for cat, qs in QA.items():
        for i, (q, ans) in enumerate(qs):
            items.append({"id": f"{cat}{i}", "cat": cat, "prompt": FEWSHOT.format(q=q), "answers": ans, "n": 12})
    for i, (key, n, d) in enumerate(PASSKEY):
        items.append({"id": f"pk{i}", "cat": "passkey", "prompt": passkey_prompt(key, n, d, i), "answers": [str(key)], "n": 10})
    return items


def score(output, answers):
    line = output.strip().split("\n")[0].lower() if output.strip() else ""
    return any(re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", line) for a in answers)
