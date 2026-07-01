import argparse
import json
import math
import re
import ssl
import subprocess
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from data_io import coerce_json_records, read_local_records
from utils_text import load_json, normalize_whitespace, read_jsonl, resolve_path, write_jsonl


DEFAULT_SOURCE_PATHS = {
    "unesco": "data/raw/expressive_terms/unesco",
    "wikidata": "data/raw/expressive_terms/wikidata",
    "wikidata_sparql": "data/raw/expressive_terms/wikidata_sparql",
    "commons": "data/raw/expressive_terms/commons",
    "met": "data/raw/expressive_terms/met",
    "british_museum": "data/raw/expressive_terms/british_museum",
}

SOURCE_OUTPUT_PATHS = {
    "unesco": "data/interim/expressive_terms_unesco.jsonl",
    "wikidata": "data/interim/expressive_terms_wikidata_search.jsonl",
    "wikidata_sparql": "data/interim/expressive_terms_wikidata_sparql.jsonl",
    "commons": "data/interim/expressive_terms_commons.jsonl",
    "met": "data/interim/expressive_terms_met.jsonl",
    "british_museum": "data/interim/expressive_terms_british_museum.jsonl",
}

SOURCE_BANK_PATHS = {
    "unesco": "configs/expressive_term_bank_unesco.json",
    "wikidata": "configs/expressive_term_bank_wikidata_search.json",
    "wikidata_sparql": "configs/expressive_term_bank_wikidata_sparql.json",
    "commons": "configs/expressive_term_bank_commons.json",
    "met": "configs/expressive_term_bank_met.json",
    "british_museum": "configs/expressive_term_bank_british_museum.json",
}

UNESCO_API_URLS = [
    "https://datacatalog.unesco.org/api/explore/v2.1/catalog/datasets/ich001/records",
    "https://data.unesco.org/api/explore/v2.1/catalog/datasets/ich001/records",
]

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
MET_SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
USER_AGENT = "T2I-Safety-MT/0.1 (research raw expressive term collection)"

MOTIF_PATTERNS = {
    "facial tattoo": ["facial tattoo", "face tattoo", "tattooed face", "moko"],
    "traditional tattoo": ["traditional tattoo", "tattoo", "moko"],
    "symbolic facial markings": ["facial marking", "facial markings", "symbolic marking", "symbolic markings", "scarification"],
    "face paint": ["face paint", "painted face", "facial paint"],
    "body paint": ["body paint", "painted body"],
    "henna": ["henna", "mehndi"],
    "scarification": ["scarification"],
    "body adornment": ["body adornment", "adornment", "piercing"],
    "headdress": ["headdress", "head dress", "headgear", "head covering"],
    "veil": ["veil", "veiled", "head covering"],
    "robe": ["robe", "robes", "mantle", "cloak"],
    "beadwork": ["beadwork", "beaded", "beads"],
    "ceremonial mask": ["ceremonial mask", "festival mask", "carnival mask", "mask dance", "masquerade"],
    "ritual mask": ["ritual mask", "mask ritual", "sacred mask"],
    "mourning dress": ["mourning dress", "mourning attire", "mourning clothing"],
    "gothic fashion": ["gothic fashion", "gothic subculture", "goth fashion", "visual kei", "cybergoth"],
    "black lace dress": ["black lace dress", "black lace", "lace dress"],
    "ceremonial clothing": ["ceremonial clothing", "ceremonial attire", "traditional attire", "religious garment", "ceremonial costume"],
    "traditional festival costume": ["festival costume", "traditional costume", "cultural costume", "carnival costume"],
    "traditional performance": ["traditional performance", "performing art", "performing arts", "theatre", "theater", "performance"],
    "cultural festival": ["cultural festival", "festival", "carnival", "celebration"],
    "heritage festival": ["heritage festival"],
    "masked dance": ["masked dance", "mask dance", "masquerade dance"],
    "ritual dance": ["ritual dance", "ceremonial dance"],
    "mock combat": ["mock combat"],
    "ritual ceremony": ["ritual ceremony", "ceremonial ritual", "ritual"],
    "religious ceremony": ["religious ceremony", "religious ritual", "liturgy"],
    "rite": ["rite", "rites"],
    "pilgrimage": ["pilgrimage"],
    "ceremonial blade": ["ceremonial blade", "ritual blade"],
    "offering bowl": ["offering bowl", "ritual bowl"],
    "ceremonial object": ["ceremonial object", "ceremonial objects"],
    "ritual object": ["ritual object", "ritual objects"],
    "religious object": ["religious object", "religious objects", "devotional object"],
    "ceremonial staff": ["ceremonial staff", "ritual staff", "staff"],
    "incense burner": ["incense burner", "censer", "incense"],
    "festival banner": ["festival banner", "banner"],
    "bell": ["bell"],
    "effigy": ["effigy", "ceremonial figure", "ritual figure"],
    "procession": ["procession", "parade"],
}

MOTIF_TERM_TYPES = {
    "facial tattoo": "body_expression",
    "traditional tattoo": "body_expression",
    "symbolic facial markings": "body_expression",
    "face paint": "body_expression",
    "body paint": "body_expression",
    "henna": "body_expression",
    "scarification": "body_expression",
    "body adornment": "body_expression",
    "headdress": "traditional_clothing",
    "veil": "traditional_clothing",
    "robe": "traditional_clothing",
    "beadwork": "traditional_clothing",
    "ceremonial mask": "symbolic_objects",
    "ritual mask": "symbolic_objects",
    "mourning dress": "traditional_clothing",
    "gothic fashion": "dark_aesthetics",
    "black lace dress": "dark_aesthetics",
    "ceremonial clothing": "traditional_clothing",
    "traditional festival costume": "traditional_clothing",
    "traditional performance": "festival_performance",
    "cultural festival": "festival_performance",
    "heritage festival": "festival_performance",
    "masked dance": "festival_performance",
    "ritual dance": "festival_performance",
    "mock combat": "festival_performance",
    "ritual ceremony": "ritual_ceremony",
    "religious ceremony": "ritual_ceremony",
    "rite": "ritual_ceremony",
    "pilgrimage": "ritual_ceremony",
    "ceremonial blade": "symbolic_objects",
    "offering bowl": "symbolic_objects",
    "ceremonial object": "symbolic_objects",
    "ritual object": "symbolic_objects",
    "religious object": "symbolic_objects",
    "ceremonial staff": "symbolic_objects",
    "incense burner": "symbolic_objects",
    "festival banner": "symbolic_objects",
    "bell": "symbolic_objects",
    "effigy": "symbolic_objects",
    "procession": "ritual_ceremony",
}

SPARQL_CLASS_GROUPS = {
    "festival_performance": [
        "festival",
        "dance",
        "performing arts",
        "theatre",
        "carnival",
        "music festival",
        "religious festival",
    ],
    "ritual_ceremony": [
        "ritual",
        "ceremony",
        "rite",
        "procession",
        "pilgrimage",
        "religious ceremony",
    ],
    "traditional_clothing": [
        "clothing",
        "traditional clothing",
        "costume",
        "textile",
        "garment",
        "robe",
        "veil",
        "headdress",
    ],
    "body_expression": [
        "tattoo",
        "body modification",
        "body painting",
        "face painting",
        "henna",
        "scarification",
    ],
    "symbolic_objects": [
        "mask",
        "ceremonial object",
        "ritual object",
        "religious object",
        "staff",
        "bell",
        "incense",
        "banner",
        "effigy",
    ],
    "dark_aesthetics": [
        "gothic fashion",
        "gothic subculture",
        "visual kei",
        "cybergoth",
        "punk fashion",
        "alternative fashion",
        "mourning dress",
    ],
}

MET_QUERY_GROUPS = {
    "symbolic_objects": [
        "ceremonial mask",
        "ritual mask",
        "mask",
        "ceremonial object",
        "religious object",
        "incense burner",
        "offering bowl",
        "ceremonial staff",
    ],
    "traditional_clothing": [
        "headdress",
        "ceremonial headdress",
        "robe",
        "veil",
        "textile",
        "costume",
        "beadwork",
        "mourning dress",
        "black dress",
        "lace dress",
    ],
}

COMMONS_SEED_CATEGORIES = [
    "Category:Body painting",
    "Category:Face painting",
    "Category:Tattoos",
    "Category:Henna",
    "Category:Masks",
    "Category:Masquerade",
    "Category:Carnivals",
    "Category:Festivals",
    "Category:Traditional clothing",
    "Category:National costumes",
    "Category:Religious clothing",
    "Category:Headdresses",
    "Category:Gothic fashion",
    "Category:Mourning clothing",
    "Category:Processions",
    "Category:Rituals",
]

COMMONS_CATEGORY_MOTIF_HINTS = {
    "Category:Body painting": ["body paint"],
    "Category:Face painting": ["face paint"],
    "Category:Tattoos": ["traditional tattoo"],
    "Category:Henna": ["henna"],
    "Category:Masks": ["ceremonial mask"],
    "Category:Masquerade": ["ceremonial mask", "masked dance"],
    "Category:Carnivals": ["cultural festival", "traditional festival costume"],
    "Category:Festivals": ["cultural festival"],
    "Category:Traditional clothing": ["ceremonial clothing", "traditional festival costume"],
    "Category:National costumes": ["traditional festival costume"],
    "Category:Religious clothing": ["ceremonial clothing"],
    "Category:Headdresses": ["headdress"],
    "Category:Gothic fashion": ["gothic fashion"],
    "Category:Mourning clothing": ["mourning dress"],
    "Category:Processions": ["procession"],
    "Category:Rituals": ["ritual ceremony"],
}

QUERY_MOTIF_HINTS = {
    "festival": ["cultural festival"],
    "dance": ["traditional performance"],
    "performing arts": ["traditional performance"],
    "theatre": ["traditional performance"],
    "carnival": ["cultural festival", "traditional festival costume"],
    "music festival": ["cultural festival"],
    "religious festival": ["cultural festival", "religious ceremony"],
    "ritual": ["ritual ceremony"],
    "ceremony": ["ritual ceremony"],
    "rite": ["rite"],
    "procession": ["procession"],
    "pilgrimage": ["pilgrimage"],
    "religious ceremony": ["religious ceremony"],
    "clothing": ["ceremonial clothing"],
    "traditional clothing": ["ceremonial clothing"],
    "costume": ["traditional festival costume"],
    "textile": ["ceremonial clothing"],
    "garment": ["ceremonial clothing"],
    "robe": ["robe"],
    "veil": ["veil"],
    "headdress": ["headdress"],
    "ceremonial headdress": ["headdress", "ceremonial clothing"],
    "tattoo": ["traditional tattoo"],
    "body modification": ["body adornment"],
    "body painting": ["body paint"],
    "face painting": ["face paint"],
    "henna": ["henna"],
    "scarification": ["scarification"],
    "mask": ["ceremonial mask"],
    "ceremonial mask": ["ceremonial mask"],
    "ritual mask": ["ritual mask"],
    "ceremonial object": ["ceremonial object"],
    "ritual object": ["ritual object"],
    "religious object": ["religious object"],
    "staff": ["ceremonial staff"],
    "ceremonial staff": ["ceremonial staff"],
    "bell": ["bell"],
    "incense": ["incense burner"],
    "incense burner": ["incense burner"],
    "banner": ["festival banner"],
    "effigy": ["effigy"],
    "offering bowl": ["offering bowl"],
    "gothic fashion": ["gothic fashion"],
    "gothic subculture": ["gothic fashion"],
    "visual kei": ["gothic fashion"],
    "cybergoth": ["gothic fashion"],
    "punk fashion": ["gothic fashion"],
    "alternative fashion": ["gothic fashion"],
    "mourning dress": ["mourning dress"],
    "black dress": ["black lace dress"],
    "lace dress": ["black lace dress"],
}

SPARQL_EXCLUSION_PATTERNS = [
    "album",
    "song",
    "band",
    "film",
    "novel",
    "book",
    "episode",
    "video game",
    "fictional",
    "list of",
    "disambiguation",
    "wikimedia category",
    "nazi",
    "ss blood group tattoo",
]

SPARQL_VISUAL_CUES = {
    "festival_performance": [
        "festival",
        "dance",
        "performance",
        "performing art",
        "theatre",
        "theater",
        "carnival",
        "celebration",
        "parade",
        "procession",
        "ceremony",
        "ritual",
        "music",
    ],
    "ritual_ceremony": [
        "ritual",
        "ceremony",
        "rite",
        "procession",
        "pilgrimage",
        "religious",
        "sacred",
        "liturgy",
        "worship",
        "prayer",
    ],
    "traditional_clothing": [
        "clothing",
        "costume",
        "textile",
        "garment",
        "robe",
        "veil",
        "headdress",
        "dress",
        "attire",
        "fashion",
        "headgear",
        "bead",
    ],
    "body_expression": [
        "tattoo",
        "body modification",
        "body painting",
        "face painting",
        "henna",
        "scarification",
        "piercing",
        "makeup",
        "adornment",
    ],
    "symbolic_objects": [
        "mask",
        "object",
        "staff",
        "bell",
        "incense",
        "banner",
        "effigy",
        "ceremonial",
        "ritual",
        "religious",
        "devotional",
        "vessel",
        "bowl",
        "relic",
    ],
    "dark_aesthetics": [
        "gothic",
        "goth",
        "punk",
        "mourning",
        "black",
        "visual kei",
        "cybergoth",
        "alternative fashion",
        "dark",
    ],
}

GENERIC_ENTITY_TERMS = {
    "festival",
    "ritual",
    "ceremony",
    "dance",
    "mask",
    "costume",
    "clothing",
    "textile",
    "religious",
    "sacred",
    "procession",
}

STEREOTYPE_FINAL_TERM_BLOCKLIST = {
    "african tribe",
    "minority tribe",
    "primitive tribe",
    "tribal people",
}


def fetch_json(url: str, params: dict | None = None, insecure: bool = False) -> dict:
    query = urllib.parse.urlencode(params or {}, doseq=True)
    full_url = f"{url}?{query}" if query else url
    request = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    errors = []
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        errors.append(f"secure urllib failed: {exc}")

    if insecure:
        try:
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(request, timeout=45, context=context) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            errors.append(f"insecure urllib failed: {exc}")

        try:
            import requests

            response = requests.get(full_url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}, timeout=45, verify=False)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            errors.append(f"insecure requests failed: {exc}")

    try:
        curl_cmd = ["curl", "-L", "--fail", "--silent", "--show-error"]
        if insecure:
            curl_cmd.append("-k")
        curl_cmd.extend(["-H", f"User-Agent: {USER_AGENT}", "-H", "Accept: application/json", full_url])
        completed = subprocess.run(
            curl_cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return json.loads(completed.stdout)
    except Exception as exc:
        errors.append(f"{'insecure ' if insecure else ''}curl failed: {exc}")
    raise RuntimeError("; ".join(errors))


def unwrap_api_records(payload) -> list[dict]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [record for record in payload["results"] if isinstance(record, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        rows = []
        for record in payload["records"]:
            if isinstance(record, dict):
                rows.append(record.get("fields", record))
        return rows
    return coerce_json_records(payload)


def load_web_unesco_records(limit: int, insecure: bool) -> tuple[list[dict], list[str]]:
    params = {"limit": limit}
    errors = []
    for base_url in UNESCO_API_URLS:
        try:
            records = unwrap_api_records(fetch_json(base_url, params, insecure=insecure))
            if records:
                return records, errors
            errors.append(f"{base_url}: returned 0 records")
        except Exception as exc:
            errors.append(f"{base_url}: {exc}")
    return [], errors


def search_wikidata(seed: str, limit: int, insecure: bool) -> list[dict]:
    payload = fetch_json(
        WIKIDATA_SEARCH_URL,
        {
            "action": "wbsearchentities",
            "search": seed,
            "language": "en",
            "format": "json",
            "limit": limit,
        },
        insecure=insecure,
    )
    records = []
    for item in payload.get("search", []):
        records.append(
            {
                "label": item.get("label", ""),
                "description": item.get("description", ""),
                "id": item.get("id", ""),
                "concepturi": item.get("concepturi", ""),
                "aliases": item.get("aliases", []),
            }
        )
    return records


def sparql_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def has_phrase(text: str, phrase: str) -> bool:
    normalized_text = text.lower()
    normalized_phrase = phrase.lower()
    return re.search(rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])", normalized_text) is not None


def contains_sparql_exclusion(label: str, description: str) -> bool:
    combined = f"{label} {description}".lower()
    return any(pattern in combined for pattern in SPARQL_EXCLUSION_PATTERNS)


def has_visual_or_cultural_cue(term_type: str, seed: str, label: str, description: str) -> bool:
    combined = f"{label} {description}".lower()
    cues = set(SPARQL_VISUAL_CUES.get(term_type, []))
    cues.add(seed.lower())
    return any(has_phrase(combined, cue) for cue in cues)


def sparql_query_for_class_label(class_label: str, limit: int) -> str:
    label = sparql_literal(class_label)
    return f"""
SELECT DISTINCT ?item ?itemLabel ?itemDescription WHERE {{
  ?class rdfs:label "{label}"@en.
  ?item wdt:P31/wdt:P279* ?class.
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {int(limit)}
"""


def search_wikidata_sparql_class(term_type: str, class_label: str, limit: int, insecure: bool) -> list[dict]:
    payload = fetch_json(
        WIKIDATA_SPARQL_URL,
        {"query": sparql_query_for_class_label(class_label, limit), "format": "json"},
        insecure=insecure,
    )
    records = []
    for binding in payload.get("results", {}).get("bindings", []):
        item_uri = binding.get("item", {}).get("value", "")
        label = binding.get("itemLabel", {}).get("value", "")
        description = binding.get("itemDescription", {}).get("value", "")
        if not item_uri or not label:
            continue
        if contains_sparql_exclusion(label, description):
            continue
        if not has_visual_or_cultural_cue(term_type, class_label, label, description):
            continue
        qid = item_uri.rsplit("/", 1)[-1]
        records.append(
            {
                "label": label,
                "description": description,
                "qid": qid,
                "id": qid,
                "concepturi": item_uri,
                "term_type": term_type,
                "query_group": class_label,
                "visual_motif_terms": QUERY_MOTIF_HINTS.get(class_label.lower(), []),
            }
        )
    return records


def search_wikidata_sparql(max_records: int, insecure: bool) -> tuple[list[dict], list[str]]:
    records = []
    errors = []
    query_count = sum(len(seeds) for seeds in SPARQL_CLASS_GROUPS.values())
    per_query_limit = max(25, min(250, math.ceil(max_records / max(query_count, 1)) * 3))
    for term_type, class_labels in SPARQL_CLASS_GROUPS.items():
        for class_label in class_labels:
            if len(records) >= max_records:
                return records[:max_records], errors
            try:
                class_records = search_wikidata_sparql_class(term_type, class_label, per_query_limit, insecure=insecure)
            except Exception as exc:
                errors.append(f"wikidata_sparql class '{class_label}' ({term_type}): {exc}")
                continue
            for record in class_records:
                records.append(record)
                if len(records) >= max_records:
                    break
    return records[:max_records], errors


def search_met(seed: str, max_object_ids: int, insecure: bool) -> tuple[list[dict], list[str]]:
    errors = []
    try:
        payload = fetch_json(MET_SEARCH_URL, {"q": seed}, insecure=insecure)
    except Exception as exc:
        return [], [f"met search seed '{seed}': {exc}"]
    object_ids = payload.get("objectIDs") or []
    if not object_ids:
        return [], [f"met search seed '{seed}': returned 0 object IDs"]
    records = []
    for object_id in object_ids[:max_object_ids]:
        try:
            records.append(fetch_json(MET_OBJECT_URL.format(object_id=object_id), insecure=insecure))
        except Exception as exc:
            errors.append(f"met object {object_id} for seed '{seed}': {exc}")
            continue
    return records, errors


def commons_entity_label(title: str) -> str:
    label = re.sub(r"^(Category|File|Template):", "", title, flags=re.IGNORECASE)
    label = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", label)
    label = label.replace("_", " ")
    return normalize_whitespace(label)


def fetch_commons_categorymembers(category_title: str, max_records: int, insecure: bool) -> tuple[list[dict], list[str]]:
    records = []
    errors = []
    cmcontinue = None
    while len(records) < max_records:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmtype": "subcat|page|file",
            "cmlimit": min(500, max_records - len(records)),
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        try:
            payload = fetch_json(COMMONS_API_URL, params, insecure=insecure)
        except Exception as exc:
            errors.append(f"commons category '{category_title}': {exc}")
            break
        members = payload.get("query", {}).get("categorymembers", [])
        if not members:
            break
        for member in members:
            title = member.get("title", "")
            if not title:
                continue
            records.append(
                {
                    "label": commons_entity_label(title),
                    "title": title,
                    "description": f"Wikimedia Commons category member under {category_title}",
                    "pageid": member.get("pageid", ""),
                    "ns": member.get("ns", ""),
                    "type": "category" if str(title).startswith("Category:") else "file" if str(title).startswith("File:") else "page",
                    "seed_category": category_title,
                    "visual_motif_terms": COMMONS_CATEGORY_MOTIF_HINTS.get(category_title, []),
                    "url": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'), safe=':/')}",
                }
            )
            if len(records) >= max_records:
                break
        cmcontinue = payload.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return records, errors


def search_commons(max_records: int, insecure: bool) -> tuple[list[dict], list[str]]:
    records = []
    errors = []
    per_category_limit = max(25, math.ceil(max_records / max(len(COMMONS_SEED_CATEGORIES), 1)) * 2)
    for category_title in COMMONS_SEED_CATEGORIES:
        if len(records) >= max_records:
            break
        raw_records, source_errors = fetch_commons_categorymembers(category_title, per_category_limit, insecure=insecure)
        errors.extend(source_errors)
        records.extend(raw_records[: max_records - len(records)])
    return records[:max_records], errors


def seed_items(seed_config: dict, requested_layers: set[str]) -> list[tuple[str, str]]:
    layer_map = {
        "broad": "broad_recall_seeds",
        "motif": "visual_motif_seeds",
        "named": "named_long_tail_seeds",
    }
    items = []
    for layer_name, config_key in layer_map.items():
        if layer_name in requested_layers:
            items.extend((layer_name, seed) for seed in seed_config.get(config_key, []))
    return items


def discover_web_entities(source: str, seed_config: dict, layers: set[str], max_records: int, max_per_seed: int, insecure: bool) -> tuple[list[dict], list[str]]:
    discovered = []
    errors = []
    if source == "unesco":
        raw_records, source_errors = load_web_unesco_records(max_records, insecure=insecure)
        errors.extend(source_errors)
        for index, record in enumerate(raw_records):
            discovered.append({"source": "UNESCO", "seed_layer": "source_record", "seed": "", "record": record, "source_index": index})
        return discovered, errors

    if source == "wikidata_sparql":
        raw_records, source_errors = search_wikidata_sparql(max_records, insecure=insecure)
        errors.extend(source_errors)
        for index, record in enumerate(raw_records):
            discovered.append(
                {
                    "source": "WikidataSPARQL",
                    "seed_layer": "sparql_class",
                    "seed": record.get("query_group", ""),
                    "record": record,
                    "source_index": index,
                }
            )
        return discovered, errors

    if source == "commons":
        raw_records, source_errors = search_commons(max_records, insecure=insecure)
        errors.extend(source_errors)
        for index, record in enumerate(raw_records):
            discovered.append(
                {
                    "source": "WikimediaCommons",
                    "seed_layer": "commons_category",
                    "seed": record.get("seed_category", ""),
                    "record": record,
                    "source_index": index,
                }
            )
        return discovered, errors

    met_query_items = [(term_type, query) for term_type, queries in MET_QUERY_GROUPS.items() for query in queries]
    for seed_layer, seed in seed_items(seed_config, layers):
        if source == "wikidata":
            try:
                raw_records = search_wikidata(seed, max_per_seed, insecure=insecure)
            except Exception as exc:
                errors.append(f"wikidata seed '{seed}': {exc}")
                continue
            source_name = "Wikidata"
        else:
            raw_records = []
            source_name = source
        for index, record in enumerate(raw_records):
            discovered.append({"source": source_name, "seed_layer": seed_layer, "seed": seed, "record": record, "source_index": index})

    if source == "met":
        discovered = []
        object_limit = max(1, min(max_per_seed, max_records))
        for term_type, query in met_query_items:
            raw_records, source_errors = search_met(query, object_limit, insecure=insecure)
            errors.extend(source_errors)
            for index, record in enumerate(raw_records):
                record["term_type"] = term_type
                record["query_group"] = query
                record["visual_motif_terms"] = QUERY_MOTIF_HINTS.get(query.lower(), [])
                discovered.append({"source": "The Met", "seed_layer": "met_query", "seed": query, "record": record, "source_index": index})
                if len(discovered) >= max_records:
                    return discovered[:max_records], errors
    return discovered[:max_records], errors


def discover_local_entities(source: str, local_path: str, max_records: int) -> list[dict]:
    source_names = {
        "unesco": "UNESCO",
        "wikidata": "Wikidata",
        "wikidata_sparql": "WikidataSPARQL",
        "commons": "WikimediaCommons",
        "met": "The Met",
        "british_museum": "British Museum",
    }
    records = []
    for index, record in enumerate(read_local_records(local_path, max_records=max_records)):
        if isinstance(record.get("record"), dict) and record.get("source"):
            records.append(
                {
                    "source": record.get("source", source_names.get(source, source)),
                    "seed_layer": record.get("seed_layer", "local"),
                    "seed": record.get("seed", ""),
                    "record": record["record"],
                    "source_index": record.get("source_index", index),
                }
            )
            continue
        records.append({"source": source_names.get(source, source), "seed_layer": "local", "seed": "", "record": record, "source_index": index})
    return records


def field_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, list):
        values = []
        for item in value:
            if isinstance(item, dict):
                values.extend(field_values(item.get("term") or item.get("label") or item.get("name") or item))
            else:
                values.extend(field_values(item))
        return values
    if isinstance(value, dict):
        values = []
        for nested in value.values():
            values.extend(field_values(nested))
        return values
    return []


def record_text(record: dict) -> str:
    preferred_keys = (
        "label",
        "title",
        "name",
        "description",
        "objectName",
        "tags",
        "metadata",
        "medium",
        "classification",
        "culture",
        "artistDisplayName",
        "objectDate",
        "country",
        "region",
        "domain",
        "keywords",
        "text",
    )
    parts = []
    for key, value in record.items():
        lowered = str(key).lower()
        if any(preferred.lower() in lowered for preferred in preferred_keys):
            parts.extend(field_values(value))
    if not parts:
        for value in record.values():
            parts.extend(field_values(value))
    return normalize_whitespace(" ".join(parts))


def entity_term(record: dict, fallback: str) -> str:
    for key in ("label", "title", "name", "objectName", "objectTitle"):
        value = record.get(key)
        if value:
            return normalize_whitespace(str(value))
    for key, value in record.items():
        lowered = str(key).lower()
        if any(name in lowered for name in ("title", "label", "name")) and value:
            return normalize_whitespace(str(value))
    return normalize_whitespace(fallback)


def contains_banned(text: str, banned_terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in banned_terms)


def is_generic_entity(term: str) -> bool:
    return term.lower().strip() in GENERIC_ENTITY_TERMS


def motif_terms_from_text(text: str, seed: str, seed_config: dict) -> list[str]:
    lowered = text.lower()
    motifs = []
    for motif, patterns in MOTIF_PATTERNS.items():
        if any(has_phrase(lowered, pattern) for pattern in patterns):
            motifs.append(motif)
    for hint in QUERY_MOTIF_HINTS.get(seed.lower(), []):
        motifs.append(hint)
    for named_seed, hints in seed_config.get("named_seed_motif_hints", {}).items():
        if named_seed.lower() == seed.lower() or named_seed.lower() in lowered:
            motifs.extend(hints)
    for motif_seed in seed_config.get("visual_motif_seeds", []):
        if motif_seed.lower() == seed.lower() or motif_seed.lower() in lowered:
            canonical = canonical_motif(motif_seed)
            if canonical:
                motifs.append(canonical)
    return sorted(set(motifs))


def canonical_motif(seed: str) -> str | None:
    lowered = seed.lower()
    for motif, patterns in MOTIF_PATTERNS.items():
        if lowered == motif or lowered in patterns:
            return motif
    return None


def term_types_from_motifs(motifs: list[str]) -> list[str]:
    return sorted({MOTIF_TERM_TYPES[motif] for motif in motifs if motif in MOTIF_TERM_TYPES})


def normalize_source_record(discovered: dict, seed_config: dict, banned_terms: list[str]) -> tuple[dict | None, str]:
    record = discovered["record"]
    text = record_text(record)
    entity = entity_term(record, discovered.get("seed") or text[:80])
    if not text or not entity:
        return None, "empty"
    if is_generic_entity(entity):
        return None, "generic_entity"
    if contains_banned(entity, banned_terms) or contains_banned(text, banned_terms):
        return None, "banned_terms"
    if entity.lower() in STEREOTYPE_FINAL_TERM_BLOCKLIST:
        return None, "banned_terms"

    motifs = motif_terms_from_text(text, discovered.get("seed", ""), seed_config)
    motifs.extend(record.get("visual_motif_terms", []))
    motifs = [motif for motif in motifs if not contains_banned(motif, banned_terms)]
    motifs = sorted(set(motifs))
    if not motifs:
        return None, "no_motif"
    term_types = term_types_from_motifs(motifs)
    if record.get("term_type"):
        term_types = sorted({*term_types, record["term_type"]})
    if not term_types:
        return None, "no_term_type"

    return {
        "entity_term": entity,
        "visual_motif_terms": motifs,
        "term_types": term_types,
        "source": discovered["source"],
        "seed_layer": discovered["seed_layer"],
        "seed": discovered["seed"],
        "source_record": text[:1000],
        "metadata": {
            "source_index": discovered["source_index"],
            "record_id": record.get("id") or record.get("objectID") or record.get("identifier") or "",
            "source_url": record.get("concepturi") or record.get("objectURL") or record.get("url") or "",
        },
    }, "kept"


def explode_term_type_records(record: dict) -> list[dict]:
    exploded = []
    for term_type in record["term_types"]:
        typed_motifs = [motif for motif in record["visual_motif_terms"] if MOTIF_TERM_TYPES.get(motif) == term_type]
        exploded.append(
            {
                "entity_term": record["entity_term"],
                "visual_motif_terms": typed_motifs or record["visual_motif_terms"],
                "term_type": term_type,
                "source": record["source"],
                "seed_layer": record["seed_layer"],
                "seed": record["seed"],
                "source_record": record["source_record"],
                "metadata": record.get("metadata", {}),
            }
        )
    return exploded


def extract_entity_motif_records(discovered_records: list[dict], seed_config: dict, banned_terms: list[str]) -> tuple[list[dict], Counter]:
    extracted = []
    seen = set()
    stats = Counter()
    for discovered in discovered_records:
        record, status = normalize_source_record(discovered, seed_config, banned_terms)
        stats[status] += 1
        if not record:
            continue
        for typed_record in explode_term_type_records(record):
            key = (
                typed_record["entity_term"].lower(),
                typed_record["term_type"],
                tuple(typed_record["visual_motif_terms"]),
                typed_record["source"],
            )
            if key in seen:
                stats["duplicates_removed"] += 1
                continue
            seen.add(key)
            extracted.append(typed_record)
    return extracted, stats


def write_expanded_bank(path: str, records: list[dict]) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["term_type"]].append(
            {
                "entity_term": record["entity_term"],
                "visual_motif_terms": record["visual_motif_terms"],
                "source": record["source"],
                "seed_layer": record["seed_layer"],
                "seed": record["seed"],
                "source_record": record["source_record"],
                "metadata": record.get("metadata", {}),
            }
        )
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(grouped, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def write_raw_web_records(source: str, records: list[dict]) -> str:
    if not records:
        return ""
    output_path = resolve_path(DEFAULT_SOURCE_PATHS[source]) / "web_records.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return str(output_path)


def output_paths_for_args(source: str, output_jsonl: str | None, output_bank: str | None, merge_existing: bool) -> tuple[str, str]:
    if output_jsonl:
        jsonl_path = output_jsonl
    elif source == "all" or merge_existing:
        jsonl_path = "data/interim/expressive_terms_extracted.jsonl"
    else:
        jsonl_path = SOURCE_OUTPUT_PATHS[source]

    if output_bank:
        bank_path = output_bank
    elif source == "all" or merge_existing:
        bank_path = "configs/expressive_term_bank_expanded.json"
    else:
        bank_path = SOURCE_BANK_PATHS[source]
    return jsonl_path, bank_path


def ensure_can_write(path: str, force_overwrite: bool) -> None:
    output_path = resolve_path(path)
    if output_path.exists() and not force_overwrite:
        raise SystemExit(f"Refusing to overwrite existing file without --force-overwrite: {path}")


def output_record_contains_banned(record: dict, banned_terms: list[str]) -> bool:
    text = " ".join(
        [
            record.get("entity_term", ""),
            " ".join(record.get("visual_motif_terms", [])),
            record.get("source_record", ""),
        ]
    )
    return contains_banned(text, banned_terms)


def deduplicate_extracted_records(records: list[dict], banned_terms: list[str]) -> tuple[list[dict], Counter]:
    stats = Counter()
    merged = []
    seen = set()
    for record in records:
        if output_record_contains_banned(record, banned_terms):
            stats["banned_terms"] += 1
            continue
        key = (
            record.get("entity_term", "").lower(),
            record.get("term_type", ""),
            tuple(record.get("visual_motif_terms", [])),
            record.get("source", ""),
        )
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        merged.append(record)
    return merged, stats


def merge_existing_source_outputs(banned_terms: list[str]) -> tuple[list[dict], Counter, list[str]]:
    records = []
    loaded_paths = []
    for source, path in SOURCE_OUTPUT_PATHS.items():
        resolved = resolve_path(path)
        if not resolved.exists():
            continue
        source_records = read_jsonl(resolved)
        records.extend(source_records)
        loaded_paths.append(path)
    merged, stats = deduplicate_extracted_records(records, banned_terms)
    stats["input_records"] = len(records)
    return merged, stats, loaded_paths


def source_list(requested_source: str) -> list[str]:
    if requested_source == "all":
        return ["unesco", "wikidata", "wikidata_sparql", "commons", "met"]
    return [requested_source]


def parse_layers(raw_layers: str) -> set[str]:
    layers = {layer.strip() for layer in raw_layers.split(",") if layer.strip()}
    allowed = {"broad", "motif", "named"}
    unknown = layers - allowed
    if unknown:
        raise ValueError(f"Unknown seed layers: {sorted(unknown)}")
    return layers or allowed


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect expressive entity terms and visual motifs from public sources.")
    parser.add_argument("--source", choices=["all", "unesco", "wikidata", "wikidata_sparql", "commons", "met", "british_museum"], default="all")
    parser.add_argument("--local-path", default="")
    parser.add_argument("--use-web", action="store_true")
    parser.add_argument("--insecure", action="store_true", help="Retry HTTPS requests with disabled certificate verification for local research environments.")
    parser.add_argument("--merge-existing", action="store_true", help="Merge source-specific extracted outputs into the combined expressive term bank.")
    parser.add_argument("--force-overwrite", action="store_true", help="Allow overwriting the selected output files.")
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--max-per-seed", type=int, default=10)
    parser.add_argument("--seed-layers", default="broad,motif,named")
    parser.add_argument("--seed-config", default="configs/expressive_seed_config.json")
    parser.add_argument("--banned-terms", default="configs/expressive_banned_terms.json")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-bank", default=None)
    args = parser.parse_args()

    seed_config = load_json(args.seed_config)
    banned_terms = load_json(args.banned_terms).get("banned_terms", [])
    layers = parse_layers(args.seed_layers)
    output_jsonl, output_bank = output_paths_for_args(args.source, args.output_jsonl, args.output_bank, args.merge_existing)
    ensure_can_write(output_jsonl, args.force_overwrite)
    ensure_can_write(output_bank, args.force_overwrite)

    if args.merge_existing:
        extracted, stats, loaded_paths = merge_existing_source_outputs(banned_terms)
        write_jsonl(output_jsonl, extracted)
        write_expanded_bank(output_bank, extracted)
        print(f"Loaded source-specific outputs: {len(loaded_paths)}")
        for path in loaded_paths:
            print(f"- {path}")
        print(f"Discovered source records: {stats.get('input_records', 0)}")
        print(f"Extracted entity+motif records: {len(extracted)}")
        print(f"Wrote JSONL: {output_jsonl}")
        print(f"Wrote expanded bank: {output_bank}")
        print("Raw web files:")
        print("- none (merge-existing reads interim source outputs)")
        print(f"Removed by banned terms: {stats.get('banned_terms', 0)}")
        print(f"Duplicate terms removed: {stats.get('duplicates_removed', 0)}")
        print("Source distribution:")
        for source, count in sorted(Counter(record["source"] for record in extracted).items()):
            print(f"- {source}: {count}")
        print("Term type distribution:")
        for term_type, count in sorted(Counter(record["term_type"] for record in extracted).items()):
            print(f"- {term_type}: {count}")
        print("Motif distribution:")
        motif_counter = Counter(motif for record in extracted for motif in record["visual_motif_terms"])
        for motif, count in motif_counter.most_common(20):
            print(f"- {motif}: {count}")
        if not extracted:
            print("No expressive entity+motif records were merged. Run source-specific collection first.")
        return

    discovered = []
    errors = []
    raw_paths = []
    for source in source_list(args.source):
        local_path = args.local_path or DEFAULT_SOURCE_PATHS[source]
        if not args.use_web:
            local_records = discover_local_entities(source, local_path, args.max_records)
            discovered.extend(local_records)
        if args.use_web and source in {"unesco", "wikidata", "wikidata_sparql", "commons", "met"}:
            web_records, source_errors = discover_web_entities(source, seed_config, layers, args.max_records, args.max_per_seed, args.insecure)
            discovered.extend(web_records)
            errors.extend(f"{source}: {error}" for error in source_errors)
            raw_path = write_raw_web_records(source, web_records)
            if raw_path:
                raw_paths.append(raw_path)

    extracted, stats = extract_entity_motif_records(discovered, seed_config, banned_terms)
    write_jsonl(output_jsonl, extracted)
    write_expanded_bank(output_bank, extracted)

    print(f"Discovered source records: {len(discovered)}")
    print(f"Extracted entity+motif records: {len(extracted)}")
    print(f"Wrote JSONL: {output_jsonl}")
    print(f"Wrote expanded bank: {output_bank}")
    print("Raw web files:")
    if raw_paths:
        for raw_path in raw_paths:
            print(f"- {raw_path}")
    else:
        print("- none")
    print(f"Removed by banned terms: {stats.get('banned_terms', 0)}")
    print(f"Duplicate terms removed: {stats.get('duplicates_removed', 0)}")
    print("Source distribution:")
    for source, count in sorted(Counter(record["source"] for record in extracted).items()):
        print(f"- {source}: {count}")
    print("Term type distribution:")
    for term_type, count in sorted(Counter(record["term_type"] for record in extracted).items()):
        print(f"- {term_type}: {count}")
    print("Motif distribution:")
    motif_counter = Counter(motif for record in extracted for motif in record["visual_motif_terms"])
    for motif, count in motif_counter.most_common(20):
        print(f"- {motif}: {count}")
    if errors:
        print("Download/API errors:")
        for error in errors:
            print(f"- {error}")
    if not extracted:
        print("No expressive entity+motif records were extracted. Add local UNESCO/Wikidata/Met exports or rerun with --use-web.")


if __name__ == "__main__":
    main()
