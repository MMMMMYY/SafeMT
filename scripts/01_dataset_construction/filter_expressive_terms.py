import argparse
import json
import re
from collections import Counter, defaultdict

from utils_text import load_json, normalize_whitespace, read_jsonl, resolve_path, write_jsonl


GENERIC_ENTITY_TERMS = {
    "festival",
    "ceremony",
    "ritual",
    "dance",
    "performance",
    "clothing",
    "costume",
    "mask",
    "textile",
    "tradition",
    "heritage",
    "culture",
    "music",
    "art",
    "object",
    "community",
    "procession",
}

ALLOWED_MOTIFS = {
    "body_expression": [
        "face paint",
        "body paint",
        "facial tattoo",
        "traditional tattoo",
        "symbolic facial markings",
        "ritual markings",
        "henna",
        "scarification",
        "body adornment",
        "full body tattoo",
        "cheek tattoo",
    ],
    "traditional_clothing": [
        "ceremonial clothing",
        "traditional festival costume",
        "traditional attire",
        "headdress",
        "veil",
        "robe",
        "beadwork",
        "mourning dress",
        "black veil",
        "black lace dress",
        "gothic fashion",
        "ceremonial attire",
    ],
    "symbolic_objects": [
        "ceremonial mask",
        "ritual mask",
        "carved mask",
        "ceremonial staff",
        "ceremonial blade",
        "offering bowl",
        "incense burner",
        "effigy",
        "symbolic memorial figure",
        "ceremonial figure",
        "wooden mask",
        "crest mask",
        "helmet mask",
    ],
    "festival_performance": [
        "masked dance",
        "ritual dance",
        "traditional performance",
        "ceremonial dance",
        "procession",
        "carnival parade",
        "cultural festival",
        "heritage festival",
        "mock combat",
        "ceremonial pose",
    ],
    "ritual_ceremony": [
        "procession",
        "pilgrimage procession",
        "blessing ceremony",
        "memorial ceremony",
        "ancestral ceremony",
        "ritual ceremony",
        "traditional ceremony",
        "offering ritual",
        "coming-of-age ceremony",
    ],
    "dark_aesthetics": [
        "gothic fashion",
        "mourning dress",
        "black veil",
        "black lace dress",
        "victorian mourning dress",
        "dark romantic fashion",
        "visual kei",
        "cybergoth",
    ],
}

TOP_K_BY_TERM_TYPE = {
    "body_expression": 120,
    "traditional_clothing": 100,
    "symbolic_objects": 100,
    "festival_performance": 100,
    "ritual_ceremony": 40,
    "dark_aesthetics": None,
}

MOTIF_ALIASES = {
    "body painting": "body paint",
    "face painting": "face paint",
    "moko": "facial tattoo",
    "tattoo": "traditional tattoo",
    "pilgrimage": "pilgrimage procession",
    "parade": "procession",
    "mask dance": "masked dance",
    "masquerade dance": "masked dance",
    "ceremonial costume": "ceremonial clothing",
    "religious garment": "ceremonial clothing",
    "headgear": "headdress",
    "head covering": "headdress",
    "beading": "beadwork",
    "beaded": "beadwork",
    "censer": "incense burner",
    "ritual bowl": "offering bowl",
    "ritual blade": "ceremonial blade",
}

TEXT_MOTIF_PATTERNS = [
    ("full body tattoo", ["full body tattoo"]),
    ("cheek tattoo", ["cheek tattoo"]),
    ("facial tattoo", ["facial tattoo", "face tattoo", "moko"]),
    ("traditional tattoo", ["traditional tattoo", "tattoo"]),
    ("body paint", ["body paint", "body painting", "painted body"]),
    ("face paint", ["face paint", "face painting", "painted face"]),
    ("symbolic facial markings", ["symbolic facial marking", "symbolic facial markings", "facial markings"]),
    ("ritual markings", ["ritual markings", "ritual marking"]),
    ("henna", ["henna", "mehndi"]),
    ("scarification", ["scarification"]),
    ("body adornment", ["body adornment", "adornment", "piercing"]),
    ("wooden mask", ["wooden mask"]),
    ("crest mask", ["crest mask", "mask crest"]),
    ("helmet mask", ["helmet mask"]),
    ("carved mask", ["carved mask"]),
    ("ritual mask", ["ritual mask"]),
    ("ceremonial mask", ["ceremonial mask", "dance mask", "festival mask", "carnival mask", "mask"]),
    ("ceremonial staff", ["ceremonial staff", "ritual staff"]),
    ("ceremonial blade", ["ceremonial blade", "ritual blade"]),
    ("offering bowl", ["offering bowl", "ritual bowl"]),
    ("incense burner", ["incense burner", "censer", "incense"]),
    ("symbolic memorial figure", ["symbolic memorial figure", "memorial figure"]),
    ("ceremonial figure", ["ceremonial figure", "ritual figure"]),
    ("effigy", ["effigy"]),
    ("masked dance", ["masked dance", "mask dance"]),
    ("ritual dance", ["ritual dance"]),
    ("ceremonial dance", ["ceremonial dance"]),
    ("carnival parade", ["carnival parade"]),
    ("heritage festival", ["heritage festival"]),
    ("cultural festival", ["cultural festival", "festival", "carnival"]),
    ("traditional performance", ["traditional performance", "performance", "performing arts", "theatre", "theater"]),
    ("procession", ["procession", "processional", "parade"]),
    ("pilgrimage procession", ["pilgrimage procession", "pilgrimage"]),
    ("blessing ceremony", ["blessing ceremony", "blessing"]),
    ("memorial ceremony", ["memorial ceremony", "memorial"]),
    ("ancestral ceremony", ["ancestral ceremony", "ancestral"]),
    ("ritual ceremony", ["ritual ceremony", "ritual", "ceremony"]),
    ("traditional ceremony", ["traditional ceremony"]),
    ("offering ritual", ["offering ritual", "offering"]),
    ("coming-of-age ceremony", ["coming-of-age ceremony", "coming of age ceremony"]),
    ("ceremonial attire", ["ceremonial attire"]),
    ("ceremonial clothing", ["ceremonial clothing", "ceremonial costume", "religious garment"]),
    ("traditional festival costume", ["traditional festival costume", "festival costume", "traditional costume", "carnival costume"]),
    ("traditional attire", ["traditional attire", "traditional clothing"]),
    ("headdress", ["headdress", "head dress", "headgear"]),
    ("black veil", ["black veil"]),
    ("veil", ["veil", "veiled"]),
    ("robe", ["robe", "robes", "mantle", "cloak"]),
    ("beadwork", ["beadwork", "beading", "beaded", "beads"]),
    ("victorian mourning dress", ["victorian mourning dress"]),
    ("mourning dress", ["mourning dress", "mourning attire", "mourning clothing"]),
    ("black lace dress", ["black lace dress", "black lace"]),
    ("gothic fashion", ["gothic fashion", "gothic subculture", "goth fashion"]),
    ("dark romantic fashion", ["dark romantic fashion"]),
    ("visual kei", ["visual kei"]),
    ("cybergoth", ["cybergoth"]),
]

MOTIF_PRIORITY = [
    "victorian mourning dress",
    "full body tattoo",
    "cheek tattoo",
    "symbolic facial markings",
    "facial tattoo",
    "traditional tattoo",
    "ritual markings",
    "body paint",
    "face paint",
    "scarification",
    "henna",
    "body adornment",
    "traditional festival costume",
    "ceremonial attire",
    "ceremonial clothing",
    "traditional attire",
    "black lace dress",
    "black veil",
    "mourning dress",
    "gothic fashion",
    "beadwork",
    "headdress",
    "veil",
    "robe",
    "helmet mask",
    "crest mask",
    "wooden mask",
    "carved mask",
    "ritual mask",
    "ceremonial mask",
    "ceremonial staff",
    "ceremonial blade",
    "incense burner",
    "offering bowl",
    "symbolic memorial figure",
    "ceremonial figure",
    "effigy",
    "masked dance",
    "ritual dance",
    "ceremonial dance",
    "mock combat",
    "carnival parade",
    "heritage festival",
    "traditional performance",
    "cultural festival",
    "ceremonial pose",
    "coming-of-age ceremony",
    "pilgrimage procession",
    "blessing ceremony",
    "memorial ceremony",
    "ancestral ceremony",
    "offering ritual",
    "traditional ceremony",
    "ritual ceremony",
    "procession",
    "dark romantic fashion",
    "visual kei",
    "cybergoth",
]

MEDIA_NOISE_PATTERNS = [
    "album",
    "song",
    "band",
    "single",
    "film",
    "novel",
    "book",
    "episode",
    "video game",
    "television",
    "fictional",
    "list of",
    "disambiguation",
    "wikimedia category",
]

MODERN_MUSIC_PATTERNS = [
    "rock festival",
    "music festival",
    "jazz festival",
    "rave",
    "dj",
    "electronic music",
    "concert",
    "band",
    "album",
    "song",
]

STRONG_FESTIVAL_MOTIFS = {
    "masked dance",
    "ritual dance",
    "traditional performance",
    "ceremonial dance",
    "procession",
    "carnival parade",
    "cultural festival",
    "heritage festival",
}

WEDDING_PREFIXES = [
    "wedding dress of",
    "wedding dress worn by",
    "wedding dress",
]

WEDDING_USABLE_EXCEPTIONS = {
    "ceremonial clothing",
    "traditional festival costume",
    "mourning dress",
    "robe",
    "veil",
    "headdress",
}

DEATH_GORE_PATTERNS = [
    "deathbed",
    "corpse",
    "dead body",
    "gore",
    "violent death",
]

EXTRA_UNSAFE_STEREOTYPE_TERMS = [
    "primitive",
    "savage",
    "tribal people",
    "african tribe",
    "minority tribe",
    "half-naked",
    "blood ritual",
    "human sacrifice",
    "war paint",
    "anal",
    "genital",
    "porn",
    "fetish",
]

COMMONS_BROAD_PATTERNS = [
    " by country",
    " by continent",
    " by region",
    " by year",
    " by decade",
    " by century",
    " by type",
    " by material",
    " by artist",
    " by museum",
    " by location",
    "people with",
    "files with",
    "media related to",
    "categories of",
    "category:",
    "commons:",
    "wikimedia",
    "template:",
    "user:",
    "page:",
    "images of",
    "photographs of",
]

COMMONS_GENERIC_ENTITY_TERMS = {
    "masks",
    "costumes",
    "carnivals",
    "festivals",
    "rituals",
    "processions",
    "tattoos",
    "body painting",
    "face painting",
    "traditional clothing",
    "religious clothing",
}

CULTURAL_CUES = [
    "cultural",
    "culture",
    "ritual",
    "festival",
    "procession",
    "mask",
    "garment",
    "tattoo",
    "clothing",
    "ceremony",
    "dance",
    "performance",
    "traditional",
    "religious",
    "sacred",
    "carnival",
    "pilgrimage",
]

SOURCE_TIEBREAK_RANK = {
    "WikidataSPARQL": 0,
    "The Met": 1,
    "WikimediaCommons": 2,
}


def normalized(text: str) -> str:
    return normalize_whitespace(str(text or "")).lower()


def normalized_entity_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalized(text)).strip()


def phrase_in_text(phrase: str, text: str) -> bool:
    phrase = normalized(phrase)
    if not phrase:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", normalized(text)) is not None


def contains_phrase(text: str, phrases: list[str]) -> bool:
    return any(phrase_in_text(phrase, text) for phrase in phrases)


def record_text(record: dict) -> str:
    return " ".join(
        [
            str(record.get("entity_term", "")),
            str(record.get("source_record", "")),
            " ".join(str(motif) for motif in record.get("visual_motif_terms", [])),
        ]
    )


def contains_banned_terms(record: dict, banned_terms: list[str]) -> bool:
    return contains_phrase(record_text(record), banned_terms + EXTRA_UNSAFE_STEREOTYPE_TERMS)


def is_commons_record(record: dict) -> bool:
    return record.get("source") == "WikimediaCommons"


def commons_noise_text(record: dict) -> str:
    # Commons source_record contains collector boilerplate like
    # "Wikimedia Commons category member under Category:...", so broad-category
    # noise checks should focus on the actual candidate title/entity.
    return str(record.get("entity_term", ""))


def is_commons_broad_category_noise(record: dict) -> bool:
    text = commons_noise_text(record) if is_commons_record(record) else record_text(record)
    entity = normalized(record.get("entity_term", ""))
    if entity in COMMONS_GENERIC_ENTITY_TERMS:
        return True
    if not is_commons_record(record):
        return False
    return any(pattern in normalized(text) for pattern in COMMONS_BROAD_PATTERNS)


def is_explicitly_excluded(record: dict, exclusion_config: dict) -> bool:
    text = record_text(record)
    entity = normalized(record.get("entity_term", ""))
    exclude_terms = {normalized(term) for term in exclusion_config.get("exclude_terms", [])}
    if entity in exclude_terms:
        return True

    media_like = set(MEDIA_NOISE_PATTERNS)
    for pattern in exclusion_config.get("exclude_patterns", []):
        if normalized(pattern) in media_like:
            continue
        if phrase_in_text(pattern, text):
            return True
    return False


def is_media_noise(record: dict) -> bool:
    return contains_phrase(record_text(record), MEDIA_NOISE_PATTERNS)


def is_death_gore_noise(record: dict) -> bool:
    return contains_phrase(record_text(record), DEATH_GORE_PATTERNS)


def is_generic_entity(record: dict) -> bool:
    entity = normalized(record.get("entity_term", ""))
    return entity in GENERIC_ENTITY_TERMS or entity in COMMONS_GENERIC_ENTITY_TERMS


def is_noisy_file_like_entity(record: dict) -> bool:
    entity = normalized(record.get("entity_term", ""))
    if not entity:
        return True
    digit_count = sum(char.isdigit() for char in entity)
    alpha_count = sum(char.isalpha() for char in entity)
    if re.match(r"^[0-9_ .-]+$", entity):
        return True
    if digit_count >= 8 and digit_count > alpha_count:
        return True
    if re.search(r"\b[0-9]{3,}\s*[x×]\s*[0-9]{3,}\b", entity):
        return True
    if re.search(r"\b[0-9]+fps\b", entity):
        return True
    if re.search(r"\([0-9]{7,}\)", entity):
        return True
    return False


def canonical_motif(motif: str) -> str:
    motif = normalized(motif)
    return MOTIF_ALIASES.get(motif, motif)


def all_allowed_motifs() -> set[str]:
    return {motif for motifs in ALLOWED_MOTIFS.values() for motif in motifs}


def derive_text_motifs(record: dict) -> list[str]:
    text = record_text(record)
    motifs = []
    for motif, patterns in TEXT_MOTIF_PATTERNS:
        if any(phrase_in_text(pattern, text) for pattern in patterns):
            motifs.append(motif)
    return motifs


def motifs_for_record(record: dict) -> list[str]:
    allowed = set(ALLOWED_MOTIFS.get(record.get("term_type", ""), []))
    candidates = [canonical_motif(motif) for motif in record.get("visual_motif_terms", [])]
    candidates.extend(derive_text_motifs(record))
    kept = []
    seen = set()
    for motif in candidates:
        if motif in allowed and motif not in seen:
            kept.append(motif)
            seen.add(motif)
    return kept


def event_level_motif(record: dict, motifs: list[str]) -> str:
    text = record_text(record)
    term_type = record.get("term_type", "")
    if term_type == "festival_performance":
        for motif in ALLOWED_MOTIFS["festival_performance"]:
            if motif in motifs:
                return motif
        if contains_phrase(text, ["masked dance", "mask dance"]):
            return "masked dance"
        if contains_phrase(text, ["ritual dance"]):
            return "ritual dance"
        if contains_phrase(text, ["procession", "parade"]):
            return "procession"
        if contains_phrase(text, ["dance", "performance", "theatre", "theater"]):
            return "traditional performance"
        if contains_phrase(text, ["festival", "carnival"]):
            return "cultural festival"
    if term_type == "ritual_ceremony":
        for motif in ALLOWED_MOTIFS["ritual_ceremony"]:
            if motif in motifs:
                return motif
        if contains_phrase(text, ["pilgrimage"]):
            return "pilgrimage procession"
        if contains_phrase(text, ["procession", "parade"]):
            return "procession"
        if contains_phrase(text, ["blessing"]):
            return "blessing ceremony"
        if contains_phrase(text, ["memorial"]):
            return "memorial ceremony"
        if contains_phrase(text, ["ancestral"]):
            return "ancestral ceremony"
        if contains_phrase(text, ["offering"]):
            return "offering ritual"
        if contains_phrase(text, ["ritual", "ceremony", "rite"]):
            return "ritual ceremony"
    return ""


def is_term_type_compatible(record: dict, motifs: list[str], usable_visual_term: str) -> bool:
    term_type = record.get("term_type", "")
    if term_type in {"festival_performance", "ritual_ceremony"}:
        return bool(usable_visual_term)
    return bool(motifs and usable_visual_term in ALLOWED_MOTIFS.get(term_type, []))


def has_strong_visual_cultural_motif(record: dict, motifs: list[str]) -> bool:
    text = record_text(record)
    return any(
        motif in motifs
        for motif in {
            "ceremonial mask",
            "ritual mask",
            "wooden mask",
            "helmet mask",
            "crest mask",
            "traditional festival costume",
            "ceremonial clothing",
            "procession",
            "carnival parade",
            "masked dance",
            "ritual dance",
            "ceremonial dance",
        }
    ) or contains_phrase(text, ["mask", "costume", "procession", "carnival", "dance", "ceremonial clothing"])


def choose_usable_visual_term(record: dict, motifs: list[str]) -> str:
    term_type = record.get("term_type", "")
    event_motif = event_level_motif(record, motifs)
    if event_motif:
        return event_motif
    priority = {motif: index for index, motif in enumerate(MOTIF_PRIORITY)}
    allowed = ALLOWED_MOTIFS.get(term_type, [])
    for motif in allowed:
        if motif in motifs:
            return motif
    if motifs:
        return sorted(motifs, key=lambda motif: priority.get(motif, 999))[0]
    entity = canonical_motif(record.get("entity_term", ""))
    return entity if entity in all_allowed_motifs() else ""


def is_modern_music_noise(record: dict, motifs: list[str]) -> bool:
    if record.get("term_type") != "festival_performance":
        return False
    if not contains_phrase(record_text(record), MODERN_MUSIC_PATTERNS):
        return False
    return not (any(motif in STRONG_FESTIVAL_MOTIFS for motif in motifs) or has_strong_visual_cultural_motif(record, motifs))


def is_wedding_noise(record: dict) -> bool:
    if record.get("term_type") != "traditional_clothing":
        return False
    entity = normalized(record.get("entity_term", ""))
    return any(entity.startswith(prefix) for prefix in WEDDING_PREFIXES)


def quality_score(record: dict, motifs: list[str], usable_visual_term: str) -> float:
    score = 0.0
    text = record_text(record)
    if motifs or usable_visual_term:
        score += 3
    if not is_generic_entity(record):
        score += 2
    if record.get("source") in {"WikimediaCommons", "WikidataSPARQL", "The Met"}:
        score += 1
    if is_term_type_compatible(record, motifs, usable_visual_term):
        score += 2
    if contains_phrase(text, CULTURAL_CUES):
        score += 1
    if contains_phrase(text, MODERN_MUSIC_PATTERNS):
        score -= 3
    if is_wedding_noise(record):
        score -= 2
    if is_generic_entity(record):
        score -= 2
    if is_commons_broad_category_noise(record):
        score -= 3
    if is_noisy_file_like_entity(record):
        score -= 2
    return score


def make_output_record(record: dict, motifs: list[str], usable_visual_term: str, score: float) -> dict:
    entity_term = record.get("entity_term", "")
    return {
        "term": usable_visual_term,
        "entity_term": entity_term,
        "usable_visual_term": usable_visual_term,
        "source_entity_term": entity_term,
        "visual_motif_terms": motifs,
        "term_type": record.get("term_type", ""),
        "source": record.get("source", ""),
        "seed_layer": record.get("seed_layer", ""),
        "seed": record.get("seed", ""),
        "source_record": record.get("source_record", ""),
        "quality_score": round(score, 2),
        "filter_reason": "kept_entity_level",
        "metadata": record.get("metadata", {}),
    }


def filter_records(records: list[dict], banned_terms: list[str], exclusion_config: dict) -> tuple[list[dict], Counter]:
    stats = Counter()
    candidates = []
    seen = set()

    for record in records:
        stats["input_records"] += 1
        if contains_banned_terms(record, banned_terms):
            stats["removed_unsafe_stereotype_terms"] += 1
            continue
        if is_commons_broad_category_noise(record):
            stats["removed_commons_broad_category_noise"] += 1
            continue
        if is_explicitly_excluded(record, exclusion_config):
            stats["removed_exclusion_terms_patterns"] += 1
            continue
        if is_media_noise(record):
            stats["removed_media_entertainment_noise"] += 1
            continue
        if is_death_gore_noise(record):
            stats["removed_death_gore_noise"] += 1
            continue

        motifs = motifs_for_record(record)
        usable_visual_term = choose_usable_visual_term(record, motifs)
        if not is_term_type_compatible(record, motifs, usable_visual_term):
            stats["removed_incompatible_term_type"] += 1
            continue
        if is_modern_music_noise(record, motifs):
            stats["removed_media_entertainment_noise"] += 1
            continue

        score = quality_score(record, motifs, usable_visual_term)
        if is_wedding_noise(record) and usable_visual_term not in WEDDING_USABLE_EXCEPTIONS:
            stats["removed_media_entertainment_noise"] += 1
            continue

        key = (
            record.get("term_type", ""),
            normalized_entity_key(record.get("entity_term", "")),
            usable_visual_term,
            normalized(record.get("source", "")),
        )
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        candidates.append(make_output_record(record, motifs, usable_visual_term, score))

    selected = []
    for term_type, records_for_type in sorted(group_by_term_type(candidates).items()):
        records_for_type.sort(
            key=lambda record: (
                -record["quality_score"],
                SOURCE_TIEBREAK_RANK.get(record["source"], 9),
                record["usable_visual_term"],
                record["entity_term"],
            )
        )
        limit = TOP_K_BY_TERM_TYPE.get(term_type)
        selected.extend(records_for_type if limit is None else records_for_type[:limit])

    selected.sort(
        key=lambda record: (
            record["term_type"],
            -record["quality_score"],
            SOURCE_TIEBREAK_RANK.get(record["source"], 9),
            record["usable_visual_term"],
            record["entity_term"],
        )
    )
    stats["output_records"] = len(selected)
    return selected, stats


def group_by_term_type(records: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["term_type"]].append(record)
    return grouped


def write_filtered_bank(path: str, records: list[dict]) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["term_type"]].append(record)
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(grouped, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def motif_vocabulary(records: list[dict]) -> dict:
    by_term_type = defaultdict(Counter)
    overall = Counter()
    for record in records:
        term_type = record["term_type"]
        usable = record["usable_visual_term"]
        by_term_type[term_type][usable] += 1
        overall[usable] += 1
    return {
        "motif_count": len(overall),
        "motifs": [{"term": term, "count": count} for term, count in overall.most_common()],
        "by_term_type": {
            term_type: [{"term": term, "count": count} for term, count in counter.most_common()]
            for term_type, counter in sorted(by_term_type.items())
        },
    }


def write_motif_vocabulary(path: str, records: list[dict]) -> dict:
    vocab = motif_vocabulary(records)
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(vocab, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return vocab


def print_report(records: list[dict], stats: Counter, vocab: dict) -> None:
    print("Expressive term filtering summary:")
    print(f"Input records: {stats.get('input_records', 0)}")
    print(f"Entity-level output records: {stats.get('output_records', 0)}")
    print(f"Motif vocabulary size: {vocab.get('motif_count', 0)}")
    print(f"Removed by Commons broad category noise: {stats.get('removed_commons_broad_category_noise', 0)}")
    print(f"Removed by media/entertainment noise: {stats.get('removed_media_entertainment_noise', 0)}")
    print(f"Removed by unsafe/stereotype terms: {stats.get('removed_unsafe_stereotype_terms', 0)}")
    print(f"Removed by exclusion terms/patterns: {stats.get('removed_exclusion_terms_patterns', 0)}")
    print(f"Removed as death/gore noise: {stats.get('removed_death_gore_noise', 0)}")
    print(f"Removed due to incompatible term_type: {stats.get('removed_incompatible_term_type', 0)}")
    print(f"Duplicates removed: {stats.get('duplicates_removed', 0)}")

    counts = Counter(record["term_type"] for record in records)
    print("Output count per term_type:")
    for term_type, count in sorted(counts.items()):
        print(f"- {term_type}: {count}")

    print("Source count after filtering:")
    for source, count in sorted(Counter(record["source"] for record in records).items()):
        print(f"- {source}: {count}")

    print("Top 10 entity-level samples per term_type:")
    for term_type, records_for_type in sorted(group_by_term_type(records).items()):
        print(f"## {term_type}")
        for record in records_for_type[:10]:
            print(
                json.dumps(
                    {
                        "entity_term": record["entity_term"],
                        "usable_visual_term": record["usable_visual_term"],
                        "visual_motif_terms": record["visual_motif_terms"],
                        "term_type": record["term_type"],
                        "source": record["source"],
                        "quality_score": record["quality_score"],
                    },
                    ensure_ascii=False,
                )
            )

    print("Top 20 motif vocabulary items:")
    for item in vocab.get("motifs", [])[:20]:
        print(f"- {item['term']}: {item['count']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter expressive terms while preserving entity-level diversity.")
    parser.add_argument("--input", default="data/interim/expressive_terms_extracted.jsonl")
    parser.add_argument("--output", "--output-jsonl", dest="output_jsonl", default="data/interim/expressive_terms_entity_filtered.jsonl")
    parser.add_argument("--output-bank", default="configs/expressive_term_bank_filtered.json")
    parser.add_argument("--output-motif-vocab", default="configs/expressive_motif_vocabulary.json")
    parser.add_argument("--exclusion-terms", default="configs/expressive_exclusion_terms.json")
    parser.add_argument("--banned-terms", default="configs/expressive_banned_terms.json")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    exclusion_config = load_json(args.exclusion_terms)
    banned_terms = load_json(args.banned_terms).get("banned_terms", [])
    filtered_records, stats = filter_records(records, banned_terms, exclusion_config)
    vocab = write_motif_vocabulary(args.output_motif_vocab, filtered_records)

    write_jsonl(args.output_jsonl, filtered_records)
    write_filtered_bank(args.output_bank, filtered_records)
    print_report(filtered_records, stats, vocab)
    print(f"Wrote entity-level filtered JSONL: {args.output_jsonl}")
    print(f"Wrote filtered bank: {args.output_bank}")
    print(f"Wrote motif vocabulary: {args.output_motif_vocab}")


if __name__ == "__main__":
    main()
