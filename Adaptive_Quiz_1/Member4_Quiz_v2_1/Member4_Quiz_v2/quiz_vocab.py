"""
Controlled vocabulary for the Member 4 quiz.

Why this file exists
--------------------
The museum CSV is messy: "Wood", "Wood, paint", "Sycomore wood", "Painted wood (Ficus
sycomorus)" are all the same material; "Bronze", "Cupreous metal", "Copper alloy" are the
same material family; "New Kingdom" and "New Kingdom, Ramesside" are nested periods.
If those raw strings were used as answer options, a question could have 2 correct options.

So every askable fact is mapped to ONE canonical label with a *conflict key*.
Two labels may appear in the same question only if their conflict keys are disjoint.
"""
import re

# ---------------------------------------------------------------- periods
# (label, start_year, end_year, arabic)   negative = B.C.   (disjoint spans, standard chronology)
PERIODS = [
    ("Predynastic Period",         -5300, -3000, "عصر ما قبل الأسرات"),
    ("Early Dynastic Period",      -3000, -2686, "العصر العتيق (بداية الأسرات)"),
    ("Old Kingdom",                -2686, -2181, "الدولة القديمة"),
    ("First Intermediate Period",  -2181, -2055, "عصر الانتقال الأول"),
    ("Middle Kingdom",             -2055, -1650, "الدولة الوسطى"),
    ("Second Intermediate Period", -1650, -1550, "عصر الانتقال الثاني"),
    ("New Kingdom",                -1550, -1069, "الدولة الحديثة"),
    ("Third Intermediate Period",  -1069,  -664, "عصر الانتقال الثالث"),
    ("Late Period",                 -664,  -332, "العصر المتأخر"),
    ("Ptolemaic Period",            -332,   -30, "العصر البطلمي"),
    ("Roman Period",                 -30,   395, "العصر الروماني"),
]
PERIOD_BY_NAME = {p[0].casefold(): p for p in PERIODS}
PERIOD_ALIASES = {"predynastic": "predynastic period"}

# ---------------------------------------------------------------- classes
CLASS_LABELS = {  # raw 'Project Class' -> (english label, arabic label, title-leak regex)
    "Statues": ("Statue", "تمثال", r"statue|statuette|figure|sphinx"),
    "Sarcophagi_Coffins": ("Sarcophagus / coffin", "تابوت", r"sarcophag|coffin|cartonnage|mummy case"),
    "Stelae": ("Stele (carved slab)", "لوحة حجرية منقوشة (ستيلا)", r"stela|stele|stelae"),
    "Papyrus": ("Papyrus", "بردية", r"papyr|book of the dead|scroll"),
}

# ---------------------------------------------------------------- materials
# Words that describe decoration / technique, not the base material.
_DECORATION = re.compile(
    r"\b(paint\w*|pigment\w*|gesso\w*|stucco\w*|plaster\w*|paste|varnish\w*|resin|ink|linen|cloth|"
    r"whitewash\w*|mortar|paraffin|bitumen|glue|fiber|fibre|gilded|gilding|gilt|inlaid|inlays?|"
    r"precious[- ]metals?|leaf|foil|sheet|traces?|glazed|glassy|solid|hollow|cast|"
    r"red|yellow|black|white|blue|green|pink|brown|gr[ae]y|pale|light|dark|faded|originally)\b",
    re.I)

# (regex, canonical label, conflict group, arabic, title-leak regex)   -- first match wins
# Materials in the same conflict group can never appear together in one question
# (e.g. granite/granodiorite/diorite, bronze/cupreous metal, limestone/alabaster).
MATERIAL_RULES = [
    (r"wood|cedar|sycomore|sycamore|acacia|ebony|tamarisk|boxwood|conifer\w*|ziziphus|ficus",
     "Wood", "wood", "خشب", r"wood|cedar|ebony|acacia"),
    (r"bronze|cupreous|cuperous|copper|brass",
     "Copper alloy (bronze)", "copper", "سبيكة نحاسية (برونز)", r"bronze|copper|cupreous"),
    (r"travertine|alabaster|calcite",
     "Travertine (Egyptian alabaster)", "carbonate", "المرمر المصري (ترافرتين)", r"alabaster|travertine|calcite"),
    (r"limestone", "Limestone", "carbonate", "حجر جيري", r"limestone"),
    (r"marble", "Marble", "carbonate", "رخام", r"marble"),
    (r"sandstone", "Sandstone", "clastic", "حجر رملي", r"sandstone"),
    (r"quartzite", "Quartzite", "clastic", "كوارتزيت", r"quartzite"),
    (r"graywacke|greywacke", "Greywacke", "clastic", "جريواكي", r"graywacke|greywacke"),
    (r"siltstone", "Siltstone", "clastic", "حجر غريني", r"siltstone"),
    (r"schist", "Schist", "clastic", "شيست", r"schist"),
    (r"slate", "Slate", "clastic", "أردواز", r"slate"),
    (r"granitodiorite|granodior\w*|granodriorite", "Granodiorite", "igneous", "جرانوديوريت", r"granodiorite"),
    (r"granite", "Granite", "igneous", "جرانيت", r"granite"),
    (r"diorite", "Diorite", "igneous", "ديوريت", r"diorite"),
    (r"gabbro", "Gabbro", "igneous", "جابرو", r"gabbro"),
    (r"dolerite|diabase", "Dolerite", "igneous", "دوليريت", r"dolerite|diabase"),
    (r"basalt", "Basalt", "igneous", "بازلت", r"basalt"),
    (r"porphyr\w*", "Porphyry", "igneous", "بورفير", r"porphyr"),
    (r"gneiss", "Gneiss", "igneous", "نيس", r"gneiss"),
    (r"steatite", "Steatite", "soft_green_stone", "ستياتيت (حجر صابوني)", r"steatite"),
    (r"serpentinite", "Serpentinite", "soft_green_stone", "سربنتينيت", r"serpentinite"),
    (r"faience", "Faience", "faience", "فاينس", r"faience"),
    (r"ivory", "Ivory", "ivory_bone", "عاج", r"ivory"),
    (r"bone", "Bone", "ivory_bone", "عظم", r"bone"),
    (r"papyrus", "Papyrus", "paper", "بردي", r"papyr"),
    (r"paper", "Paper", "paper", "ورق", r"paper"),
    (r"pottery|terracotta|clay|mud|earth", "Clay / pottery", "clay", "طين / فخار", r"pottery|terracotta|clay"),
    (r"gold", "Gold", "gold", "ذهب", r"gold"),
    (r"silver", "Silver", "silver", "فضة", r"silver"),
]
_MATERIAL_RULES = [(re.compile(r"\b(" + p + r")\b", re.I), *rest) for p, *rest in MATERIAL_RULES]
