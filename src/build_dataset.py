"""
STEP 2 - Create CSV dataset
Context-Aware Multi-Meaning Slang Classification Using Graph Theory

Builds a starter labeled dataset: each row = one sentence containing a slang
word, tagged with its intended meaning in that sentence, rough context words,
a topic/category, and a sentiment label.

This is a STARTER dataset (8 slang words x 2 meanings x ~10 sentences = ~160 rows).
Expand later by adding more slang words / meanings / sentences.
"""

import pandas as pd
import re

STOPWORDS = set("""
a an the is are was were be been being to of in on at for with and or but
this that these those i you he she it we they my your his her its our their
just so very really much more most as if then than there here up down out
about into over under again do does did have has had can could will would
should not no nor
""".split())

# ---------------------------------------------------------------------------
# slang -> meaning -> (category, sentiment, [sentences])
# ---------------------------------------------------------------------------
DATA = {
    "fire": {
        "amazing/excellent": ("Music/Social", "positive", [
            "This song is fire, I can't stop replaying it.",
            "Bro that new sneaker drop is absolute fire.",
            "Her verse on the last track was fire.",
            "The new season of the show is fire, you have to watch it.",
            "That comeback he just made was fire.",
            "This playlist is straight fire for a road trip.",
            "The chef's new dish is fire, best thing I've eaten all year.",
            "Their live performance last night was fire.",
            "The dunk he just landed was fire.",
            "This meme is fire, send it to the group chat.",
        ]),
        "literal combustion": ("Emergency/News", "negative", [
            "Fire broke out in the building late last night.",
            "Firefighters battled the fire for over six hours.",
            "The forest fire has spread across three counties.",
            "A fire alarm went off in the office this morning.",
            "The fire destroyed most of the warehouse before it was contained.",
            "Residents were evacuated after a fire started in the kitchen.",
            "The fire department responded to the fire within minutes.",
            "Smoke from the fire was visible for miles.",
            "The candle tipped over and started a small fire.",
            "Investigators are trying to determine what caused the fire.",
        ]),
    },
    "sick": {
        "amazing/impressive": ("Sports/Social", "positive", [
            "That match was sick, best game I've seen all season.",
            "His new trick on the skateboard was sick.",
            "The graphics in this game are sick.",
            "That was a sick goal in the last minute.",
            "Her new tattoo design is sick.",
            "The stunt in that movie trailer looked sick.",
            "This new phone camera is sick, look at the detail.",
            "That guitar solo was absolutely sick.",
            "The choreography in the music video is sick.",
            "He pulled off a sick combo in the fighting game.",
        ]),
        "ill/unwell": ("Health", "negative", [
            "He is feeling sick and stayed home from school.",
            "She's been sick with a fever since yesterday.",
            "I felt sick after eating the leftover food.",
            "The kids got sick from the flu going around school.",
            "He looked really sick during the meeting today.",
            "She called in sick to work this morning.",
            "My stomach feels sick after that long car ride.",
            "He's too sick to travel this weekend.",
            "The whole family got sick with a cold last week.",
            "She's been sick in bed for three days straight.",
        ]),
    },
    "cooked": {
        "in trouble/exhausted": ("Exams/Social", "negative", [
            "I'm cooked after that exam, I barely finished on time.",
            "We are so cooked if the boss finds out about this.",
            "He's cooked, he didn't study at all for the final.",
            "I'm cooked, I have three deadlines due tomorrow.",
            "She's cooked after pulling an all-nighter for the project.",
            "If we miss this flight we're absolutely cooked.",
            "I'm cooked, my legs are done after that marathon.",
            "He's cooked, the coach benched him after that mistake.",
            "We're cooked if this presentation doesn't go well.",
            "I'm cooked, I forgot to submit the assignment.",
        ]),
        "food prepared": ("Food", "neutral", [
            "The chicken is cooked and ready to serve.",
            "Make sure the rice is fully cooked before eating.",
            "The steak was cooked medium rare, just how I like it.",
            "She cooked dinner for the whole family tonight.",
            "The vegetables are cooked, you can turn off the stove.",
            "He cooked breakfast for everyone this morning.",
            "The pasta needs to be cooked for ten more minutes.",
            "The fish was cooked perfectly on the grill.",
            "The eggs are cooked, they're ready on the plate.",
            "The stew has been cooked slowly all afternoon.",
        ]),
    },
    "ghost": {
        "to stop responding/disappear": ("Relationships/Social", "negative", [
            "He totally ghosted me after our third date.",
            "She ghosted the group chat and never replied.",
            "My friend ghosted me and I haven't heard from him in weeks.",
            "The recruiter ghosted me after the final interview.",
            "He ghosted our project team right before the deadline.",
            "She got ghosted by the guy she matched with online.",
            "My roommate ghosted me on rent this month.",
            "The client ghosted us after we sent the invoice.",
            "He ghosted everyone at the party without saying bye.",
            "I got ghosted by the seller after paying the deposit.",
        ]),
        "literal spirit/apparition": ("Supernatural", "neutral", [
            "The old house is said to be haunted by a ghost.",
            "They claimed to see a ghost in the abandoned hallway.",
            "The legend says a ghost still wanders the castle at night.",
            "Visitors reported hearing a ghost in the attic.",
            "The movie is about a ghost trapped in an old mansion.",
            "She swears she saw a ghost near the cemetery gate.",
            "The tour guide told us a ghost story about the hotel.",
            "A ghost supposedly appears in the mirror at midnight.",
            "Local folklore describes a ghost that guards the bridge.",
            "The children were scared of the ghost story around the campfire.",
        ]),
    },
    "lit": {
        "exciting/fun": ("Party/Social", "positive", [
            "The party last night was absolutely lit.",
            "This concert is lit, the crowd is going wild.",
            "That festival was lit from start to finish.",
            "The energy in the stadium tonight is lit.",
            "Their new album drop is lit, everyone's talking about it.",
            "The DJ set at the club was lit.",
            "Game night at his place was so lit.",
            "This new bar has a really lit atmosphere.",
            "The graduation party was lit, best night of the summer.",
            "That halftime show was lit.",
        ]),
        "illuminated/on fire": ("Literal", "neutral", [
            "The candles were lit around the room for the dinner.",
            "She lit the fireplace to warm up the living room.",
            "The streetlights were lit as the sun went down.",
            "He lit a match to start the campfire.",
            "The lanterns were lit for the evening festival.",
            "The stove was lit and ready for cooking.",
            "The torches were lit along the pathway.",
            "She lit a candle for the birthday cake.",
            "The bonfire was lit as everyone gathered around.",
            "The fireworks were lit right at midnight.",
        ]),
    },
    "dead": {
        "extremely funny": ("Humor/Social", "positive", [
            "I'm dead, that joke was too funny.",
            "This meme has me dead, I can't stop laughing.",
            "She sent a video that left me dead laughing.",
            "I'm dead, he actually said that in the meeting.",
            "That comedian's set had the whole crowd dead.",
            "I'm dead, look at what the cat just did.",
            "This caption is sending me, I'm dead.",
            "He's dead after seeing that plot twist coming.",
            "I'm dead, that impression was too accurate.",
            "The group chat is dead from that joke.",
        ]),
        "no longer alive": ("Literal", "negative", [
            "The plant has been dead for weeks now.",
            "Scientists confirmed the ancient tree is dead.",
            "The battery in my phone is completely dead.",
            "The old oak tree in the yard finally died and is dead.",
            "The signal on my phone has been dead all day.",
            "The engine is dead, the car won't start.",
            "The garden looks dead after the frost last night.",
            "His grandfather passed away and is now dead.",
            "The flowers are dead from lack of water.",
            "The Wi-Fi connection has been dead since morning.",
        ]),
    },
    "salty": {
        "upset/bitter": ("Social", "negative", [
            "He's still salty about losing the game last night.",
            "She got salty after her comment was ignored.",
            "Don't be salty just because you lost the bet.",
            "He's salty that he wasn't picked for the team.",
            "They got salty after the referee's bad call.",
            "She's been salty ever since the breakup.",
            "He's salty about not getting the promotion.",
            "The fans got salty after the team's early exit.",
            "She's salty that her idea got rejected in the meeting.",
            "He's salty about missing out on the concert tickets.",
        ]),
        "literal taste": ("Food", "neutral", [
            "The soup tastes too salty for my liking.",
            "The chips were extremely salty straight out of the bag.",
            "This broth is salty, add a bit more water.",
            "The fries at that restaurant are always salty.",
            "The seawater tasted salty when I accidentally swallowed some.",
            "The bacon was crispy and salty this morning.",
            "The popcorn was buttery and salty at the theater.",
            "The pretzels are salty, they go great with beer.",
            "The pickles were more salty than sour.",
            "The ham was too salty for the sandwich.",
        ]),
    },
    "killer": {
        "excellent/impressive": ("Social", "positive", [
            "That was a killer performance on stage tonight.",
            "She has a killer smile that lights up the room.",
            "He gave a killer presentation at the conference.",
            "This app has a killer new feature everyone loves.",
            "That was a killer play in the final quarter.",
            "The band delivered a killer set at the festival.",
            "He has a killer resume for this job.",
            "That's a killer deal on the new laptop.",
            "She wrote a killer essay for her application.",
            "The team pulled off a killer comeback in the second half.",
        ]),
        "literal murderer": ("Crime/News", "negative", [
            "Police are still searching for the killer.",
            "The killer was arrested after a two-week manhunt.",
            "Investigators believe the killer knew the victim.",
            "The trial of the accused killer begins next month.",
            "Witnesses described the killer fleeing the scene.",
            "The serial killer was finally identified through DNA evidence.",
            "The killer left no fingerprints at the crime scene.",
            "Detectives are questioning a suspect linked to the killer.",
            "The documentary examines the case of the notorious killer.",
            "The killer was sentenced to life in prison.",
        ]),
    },
}


def extract_context_words(sentence: str, slang: str, window: int = 4) -> str:
    """Very rough context-word extractor: content words near the slang term.
    This is a placeholder — Step 5 (spaCy) will replace this with real
    dependency-based context extraction."""
    tokens = re.findall(r"[A-Za-z']+", sentence.lower())
    if slang not in tokens:
        return ""
    idx = tokens.index(slang)
    lo, hi = max(0, idx - window), min(len(tokens), idx + window + 1)
    nearby = [t for t in tokens[lo:hi] if t != slang and t not in STOPWORDS]
    # de-duplicate while preserving order
    seen = set()
    ordered = []
    for w in nearby:
        if w not in seen:
            seen.add(w)
            ordered.append(w)
    return ", ".join(ordered)


def main():
    rows = []
    row_id = 1
    for slang, meanings in DATA.items():
        for meaning, (category, sentiment, sentences) in meanings.items():
            for sentence in sentences:
                context_words = extract_context_words(sentence, slang)
                rows.append({
                    "ID": row_id,
                    "Sentence": sentence,
                    "Slang": slang,
                    "Meaning": meaning,
                    "Context_Words": context_words,
                    "Category": category,
                    "Sentiment": sentiment,
                })
                row_id += 1

    df = pd.DataFrame(rows)
    out_path = "../data/raw/slang_dataset.csv"
    df.to_csv(out_path, index=False)

    print(f"Rows: {len(df)}")
    print(f"Slang words: {df['Slang'].nunique()}")
    print(df.groupby(['Slang', 'Meaning']).size())
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
