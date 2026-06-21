from __future__ import annotations

ORIGINAL_LABELS = [
    "AI services",
    "About Japeto",
    "Apps",
    "Billing",
    "Compliance",
    "Contact",
    "Feedback",
    "General conversation",
    "Japeto Chat",
    "Managed hosting",
    "Other",
    "Paige",
    "Partnerships",
    "Pat",
    "Project management",
    "Recruitment",
    "Services",
    "Support",
    "Technical stack",
    "Websites",
]

CATEGORY_ALIASES = {
    "artificial intelligence services": "AI services",
    "ai services": "AI services",
    "general conversation": "General conversation",
    "japeto chat": "Japeto Chat",
}

CATEGORY_MAPPING = {
    "General conversation": "General Inquiries",
    "Other": "General Inquiries",
    "Feedback": "General Inquiries",
    "Services": "Technical Services",
    "Apps": "Technical Services",
    "AI services": "Technical Services",
    "Managed hosting": "Technical Services",
    "Websites": "Technical Services",
    "Technical stack": "Technical Services",
    "Support": "Support & Contact",
    "Contact": "Support & Contact",
    "About Japeto": "Company Information",
    "Japeto Chat": "Company Information",
    "Paige": "Company Information",
    "Pat": "Company Information",
    "Project management": "Project & Compliance",
    "Compliance": "Project & Compliance",
    "Recruitment": "Business & Partnerships",
    "Partnerships": "Business & Partnerships",
    "Billing": "Billing & Finance",
}

GROUPED_LABELS = [
    "Billing & Finance",
    "Business & Partnerships",
    "Company Information",
    "General Inquiries",
    "Project & Compliance",
    "Support & Contact",
    "Technical Services",
]

INPUT_MODES = ("response_only", "context_enhanced")
LABEL_SCHEMES = ("original", "grouped")
FEATURE_TYPES = ("tfidf", "openai")
ALGORITHMS = ("svm", "random_forest", "naive_bayes")


def expected_model_ids() -> list[str]:
    ids: list[str] = []
    for mode in INPUT_MODES:
        for labels in LABEL_SCHEMES:
            for feature in FEATURE_TYPES:
                for algorithm in ALGORITHMS:
                    if feature == "openai" and algorithm == "naive_bayes":
                        continue
                    ids.append(f"{algorithm}__{feature}__{labels}__{mode}")
    return ids

