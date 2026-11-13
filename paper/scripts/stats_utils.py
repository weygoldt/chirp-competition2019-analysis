import re
import json
import numpy as np


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, path):
    def array_to_list(array):
        if isinstance(array, np.ndarray):
            return array_to_list(array.tolist())
        elif isinstance(array, list):
            return [array_to_list(item) for item in array]
        elif isinstance(array, tuple):
            return tuple(array_to_list(item) for item in array)
        else:
            return array

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = array_to_list(value)

    with open(path, "w") as f:
        json.dump(data, f)


def log_odds_to_prob(log_odds):
    if isinstance(log_odds, np.ndarray):
        return 1 / (1 + np.exp(-log_odds))
    elif isinstance(log_odds, list):
        return [log_odds_to_prob(item) for item in log_odds]
    else:
        return 1 / (1 + np.exp(-log_odds))


def parse_bambi_model_output(model):
    """
    Parses the text representation of a Bambi model into a dictionary.
    Lines that match a top-level pattern 'Key: Value' are placed in top-level fields.
    The block(s) under 'Priors:' are accumulated separately.
    Within 'Priors', we attempt to parse lines describing the distribution
    (e.g. 'Intercept ~ Normal(mu: 0.0, sigma: 4.3301)') with another regex.

    Returns a dict with:
      {
        "Formula": "...",
        "Family": "...",
        "Link": "...",
        "Observations": "...",
        "Priors": {
           "Common-level effects": [
              { "term": "Intercept",
                "distribution": "Normal",
                "parameters": { "mu": "0.0", "sigma": "4.3301" }
              },
              ...],
           "Group-level effects": [...],
           "Auxiliary parameters": [...]
        }
      }
    """
    text = str(model)
    lines = text.strip().split("\n")

    # Regex to match top-level lines of form "Key: value"
    top_level_regex = re.compile(r"^([A-Za-z][\w\s]*):\s*(.*)$")

    # We'll keep track of what section we're in (e.g. "Priors", "Common-level effects", etc.)
    # and store top-level keys in a dict.
    parsed = {}
    current_section = None
    # For storing lines under "Priors"
    priors_dict = {}
    section_name = None  # e.g. "Common-level effects"

    for line in lines:
        line = line.strip()

        # 1) Check if it's a top-level line like "Formula: chirps ~ ..."
        m = top_level_regex.match(line)
        if m:
            key, val = m.groups()
            key = key.strip()
            val = val.strip()
            # If this is "Priors:", we'll handle that differently
            if key.lower() == "priors":
                parsed["Priors"] = {}  # Initialize
                current_section = "Priors"
            else:
                parsed[key] = val
            continue

        # 2) If we're in the "Priors" section, look for lines like:
        #    "Common-level effects", "Group-level effects", "Auxiliary parameters"
        #    Or actual lines describing each effect
        if current_section == "Priors":
            # If line is empty, skip
            if not line:
                continue

            # If line matches e.g. "Common-level effects"
            # We'll start a sub-list in the priors dict
            if re.match(r"^[A-Za-z\- ]+ effects$", line, re.IGNORECASE):
                section_name = line.strip()
                parsed["Priors"][section_name] = []
                continue

            # If line matches "Auxiliary parameters"
            if re.match(r"^Auxiliary parameters$", line, re.IGNORECASE):
                section_name = line.strip()
                parsed["Priors"][section_name] = []
                continue

            # Otherwise, it might be something like:
            #   "Intercept ~ Normal(mu: 0.0, sigma: 4.3301)"
            #   "1|fish_id ~ Normal(mu: 0.0, sigma: HalfNormal(sigma: 4.3301))"
            # We'll parse that with a separate regex:
            #   group(1) = "Intercept" or "1|fish_id"
            #   group(2) = "Normal"
            #   group(3) = "mu: 0.0, sigma: 4.3301"
            #   group(4) = entire parentheses content, for more advanced parsing if needed

            dist_regex = re.compile(r"^\s*([^~]+)\s*~\s*([A-Za-z0-9]+)\((.*)\)$")
            dist_match = dist_regex.match(line)
            if dist_match and section_name is not None:
                term = dist_match.group(1).strip()
                dist_name = dist_match.group(2).strip()
                param_str = dist_match.group(3).strip()  # e.g. "mu: 0.0, sigma: 4.3301"

                # Now we can parse "mu: 0.0, sigma: 4.3301" further
                # We'll do a simple approach: split by commas
                params = {}
                for chunk in param_str.split(","):
                    chunk = chunk.strip()
                    # e.g. "mu: 0.0"
                    if ":" in chunk:
                        pkey, pval = chunk.split(":", maxsplit=1)
                        pkey = pkey.strip()
                        pval = pval.strip()
                        params[pkey] = pval
                    else:
                        # If there's no ":", just store as is
                        # e.g. "HalfNormal(sigma: 4.3301)" inside
                        # might appear. We'll keep it as a single param
                        params["param"] = chunk

                # Add to the current prior section
                parsed["Priors"][section_name].append(
                    {
                        "term": term,
                        "distribution": dist_name,
                        "parameters": params,
                    }
                )
            else:
                # If it doesn't match the distribution pattern, we can store the raw line
                # or ignore it.
                if section_name is not None:
                    parsed["Priors"][section_name].append({"raw_line": line})
                else:
                    # If we haven't encountered a section name yet, store it in some fallback
                    parsed["Priors"].setdefault("unparsed", []).append(line)
        else:
            # Lines outside "Priors" we can ignore or store
            # Usually everything is captured by top_level_regex or "Priors"
            pass

    return parsed


def format_p_value(p):
    if p < 0.001:
        return "<0.001"
    else:
        return format(p, ".2g")
