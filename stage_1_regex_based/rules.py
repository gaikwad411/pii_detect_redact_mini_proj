"""
rules.py
--------
Rule-based PII detection for Hindi NER.

Strategy:
  1. Gazetteer lookup  — known person names, org names, locations
  2. Context triggers  — words like "श्री", "डॉ" before a name
  3. Regex patterns    — phone numbers, Aadhaar, PAN, email, dates

All lists here are seeds. In a real project you would load these
from external files (CSV / JSON) so they can grow without touching code.
"""

import re

# ---------------------------------------------------------------------------
# 1. GAZETTEERS  (extend these lists with your domain data)
# ---------------------------------------------------------------------------

# Common Hindi / Indian first names (a representative seed)
PERSON_FIRST_NAMES = {
    "अहमद", "राहुल", "प्रियंका", "नरेंद्र", "सोनिया", "अमित", "योगी",
    "ममता", "अरविंद", "मनमोहन", "अटल", "इंदिरा", "राजीव", "सुभाष",
    "भगत", "चंद्रशेखर", "सरदार", "वल्लभभाई", "महात्मा", "जवाहरलाल",
    "रामनाथ", "द्रौपदी", "वेंकैया", "हामिद", "प्रतिभा", "शंकर",
    "रवि", "सुरेश", "महेश", "रमेश", "विनोद", "अनिल", "सुनील",
    "विजय", "अजय", "संजय", "मनोज", "राकेश", "दिनेश", "मुकेश",
    "रमा", "सीता", "गीता", "लता", "आशा", "उषा", "सुमन", "कमला",
    "शीला", "मीरा", "पूजा", "नीता", "रीता", "स्मिता", "अनीता",
    "फारुख", "इमरान", "सलमान", "शाहरुख", "आमिर", "अब्दुल", "मोहम्मद",
    "यूसुफ", "हसन", "हुसैन", "अली", "खान", "शेख", "मलिक",
}

# Common Indian surnames
PERSON_LAST_NAMES = {
    "पटेल", "शर्मा", "वर्मा", "गुप्ता", "सिंह", "यादव", "तिवारी",
    "मिश्रा", "पाण्डेय", "चौधरी", "जोशी", "राव", "नायर", "मेनन",
    "रेड्डी", "नायडू", "मुखर्जी", "बनर्जी", "चटर्जी", "बोस",
    "खान", "अंसारी", "सिद्दीकी", "कुरैशी", "शेख", "मलिक",
    "गांधी", "नेहरू", "वाजपेयी", "आडवाणी", "मोदी", "शाह",
    "ठाकुर", "राठौड़", "चौहान", "राजपूत", "भाटी",
    "देसाई", "मेहता", "शाह", "पारेख", "मोदी", "दवे", "भट्ट",
    "नाइक", "कामत", "हेगड़े", "शेट्टी", "कुमार", "त्रिपाठी",
}

# Known political / institutional organizations (seed)
ORG_NAMES = {
    "कांग्रेस", "भाजपा", "बसपा", "सपा", "आप", "तृणमूल",
    "राज्यसभा", "लोकसभा", "विधानसभा",
    "हाईकोर्ट", "सुप्रीमकोर्ट", "न्यायालय",
    "सीबीआई", "ईडी", "एनआईए", "इंटेलिजेंस",
    "संसद", "सरकार", "मंत्रालय", "आयोग",
    "विश्वविद्यालय", "कॉलेज", "संस्थान", "अकादमी",
    "बैंक", "रिजर्व", "स्टेट", "नेशनल",
    "रेलवे", "एयरपोर्ट", "पुलिस", "सेना",
    "इसरो", "डीआरडीओ", "नीति", "आयोग",
}

# Known Indian locations (seed — cities, states, regions)
LOCATION_NAMES = {
    "गुजरात", "महाराष्ट्र", "दिल्ली", "मुंबई", "कोलकाता", "चेन्नई",
    "बेंगलुरु", "हैदराबाद", "पुणे", "अहमदाबाद", "सूरत", "जयपुर",
    "लखनऊ", "कानपुर", "नागपुर", "इंदौर", "भोपाल", "पटना",
    "रांची", "भुवनेश्वर", "गुवाहाटी", "शिलांग", "इम्फाल",
    "श्रीनगर", "जम्मू", "लेह", "देहरादून", "शिमला",
    "उत्तर प्रदेश", "बिहार", "राजस्थान", "मध्यप्रदेश", "छत्तीसगढ़",
    "झारखंड", "ओडिशा", "पश्चिम बंगाल", "असम", "केरल",
    "तमिलनाडु", "कर्नाटक", "आंध्र प्रदेश", "तेलंगाना",
    "पंजाब", "हरियाणा", "हिमाचल प्रदेश", "उत्तराखंड",
    "भारत", "पाकिस्तान", "चीन", "अमेरिका", "रूस", "इंग्लैंड",
    "नई दिल्ली", "नोएडा", "गाजियाबाद", "फरीदाबाद", "गुरुग्राम",
}

# ---------------------------------------------------------------------------
# 2. CONTEXT TRIGGERS  (honorifics / role words that precede a person name)
# ---------------------------------------------------------------------------

PERSON_HONORIFICS = {
    "श्री", "श्रीमती", "श्रीमान", "डॉ", "डॉ.", "प्रो", "प्रो.",
    "अध्यक्ष", "मुख्यमंत्री", "प्रधानमंत्री", "राष्ट्रपति",
    "नेता", "सांसद", "विधायक", "मंत्री", "न्यायाधीश",
    "जज", "वकील", "अभियुक्त", "आरोपी", "गवाह",
}

# ---------------------------------------------------------------------------
# 3. REGEX PATTERNS  (structured PII — these are language-agnostic)
# ---------------------------------------------------------------------------

REGEX_PATTERNS = {
    # Indian mobile: starts with 6-9, 10 digits
    "PHONE": re.compile(r"\b[6-9]\d{9}\b"),

    # Aadhaar: 12-digit number (sometimes written with spaces: XXXX XXXX XXXX)
    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),

    # PAN card: 5 letters, 4 digits, 1 letter  e.g. ABCDE1234F
    "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),

    # Email
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),

    # Pincode: 6-digit Indian postal code
    "PINCODE": re.compile(r"\b[1-9][0-9]{5}\b"),

    # Date patterns:  DD/MM/YYYY  or  DD-MM-YYYY
    "DATE": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),

    # Vehicle registration (rough): MH 01 AB 1234
    "VEHICLE_REG": re.compile(r"\b[A-Z]{2}\s?\d{2}\s?[A-Z]{1,2}\s?\d{4}\b"),
}

# ---------------------------------------------------------------------------
# 4. HELPER LOOKUPS  (token → entity type)
# ---------------------------------------------------------------------------

def get_gazetteer_tag(token: str) -> str | None:
    """
    Returns entity type string if token is in any gazetteer, else None.
    Priority: PER > LOC > ORG
    (person names are more sensitive PII than org names)
    """
    if token in PERSON_FIRST_NAMES or token in PERSON_LAST_NAMES:
        return "PER"
    if token in LOCATION_NAMES:
        return "LOC"
    if token in ORG_NAMES:
        return "ORG"
    return None


def get_regex_tag(token: str) -> str | None:
    """
    Returns entity type string if token matches any regex pattern, else None.
    Regex PII maps to a generic 'PII' tag (phone, aadhaar, PAN etc.)
    """
    for label, pattern in REGEX_PATTERNS.items():
        if pattern.fullmatch(token):
            return label
    return None
