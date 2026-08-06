# ============================================================
#  BIRD SPECIES IDENTIFIER — Complete Local Version
#  Includes:
#    ✅ Bird identification app (Gradio)
#    ✅ Feature 1: Full Evaluation Metrics
#    ✅ Feature 2: Model Comparison
#    ✅ Feature 3: Grad-CAM Visualisation
#    ✅ Feature 4: Confidence Calibration
#
#  HOW TO RUN:
#    pip3 install torch torchvision pillow gradio grad-cam scikit-learn seaborn
#    python3 bird_complete_local.py
#
#  Then open: http://127.0.0.1:7860
# ============================================================

import os
# Must be set before torch is imported: a few ops still have no MPS
# kernel and need to fall back to CPU instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr

try:
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer
    BIRDNET_ANALYZER = Analyzer()
    BIRDNET_AVAILABLE = True
    print("✅ BirdNET loaded successfully")
except ImportError:
    BIRDNET_AVAILABLE = False
    BIRDNET_ANALYZER = None
    print("⚠️  birdnetlib not installed. Run: pip3 install birdnetlib")

# ============================================================
# PATHS — resolved relative to this file, overridable by env var
# so the project runs from any checkout without editing source.
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.environ.get("BIRD_TEST_DIR", os.path.join(PROJECT_DIR, "birds_split", "test"))
SAVE_DIR = os.environ.get("BIRD_SAVE_DIR", os.path.join(PROJECT_DIR, "evaluation"))

# Prefer the slim inference checkpoint (79 MB); fall back to the full
# training checkpoint (235 MB) which additionally carries optimizer state.
_MODEL_CANDIDATES = [
    os.environ.get("BIRD_MODEL_PATH"),
    os.path.join(PROJECT_DIR, "best_bird_model_inference.pth"),
    os.path.join(PROJECT_DIR, "best_bird_model.pth"),
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if p and os.path.exists(p)), None)
if MODEL_PATH is None:
    raise FileNotFoundError(
        "No model checkpoint found. Expected best_bird_model_inference.pth or "
        f"best_bird_model.pth in {PROJECT_DIR}, or set BIRD_MODEL_PATH."
    )
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)


def pick_device():
    """CUDA if present, else Apple-Silicon MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = pick_device()
DEVICE_LABEL = {"cuda": "CUDA", "mps": "Apple MPS", "cpu": "CPU"}[device.type]
print(f"\n🚀 Running on: {device}")

# ── Load model ───────────────────────────────────────────────
print(f"Loading model from {os.path.basename(MODEL_PATH)} "
      f"({os.path.getsize(MODEL_PATH) / 2**20:.0f} MB)...")
checkpoint  = torch.load(MODEL_PATH, map_location=device)
CLASS_NAMES = checkpoint["class_names"]
NUM_SPECIES = len(CLASS_NAMES)

model = models.efficientnet_v2_s(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_SPECIES)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()
print(f"✅ Model loaded — {NUM_SPECIES} species")

# ── Transforms ───────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

raw_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
])

norm_transform = transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)

# ── Species naming ───────────────────────────────────────────
# Every lookup table below is keyed on the raw CUB folder name
# ("112.Artic_Tern"), never on the display name. CUB ships several
# misspelled folders — keying on the display name is how the old
# MIGRATION_MAP["Arctic Tern"] entry became permanently unreachable.
def display_name(folder_name):
    """'112.Artic_Tern' -> 'Artic Tern'. Presentation only, never a key."""
    return folder_name.split(".", 1)[-1].replace("_", " ")


# ── Habitat (keyed on CUB folder name from the model's class_names) ──
# Coverage is partial: 15 CUB species have no entry yet. Keys were
# previously written against a DIFFERENT 200-species list, so only
# 27/200 lookups ever succeeded. Now matched by species name.
HABITAT_MAP = {
    "001.Black_footed_Albatross":          "USA, Japan, Hawaii (North Pacific Ocean — nests on Hawaiian Islands)",
    "002.Laysan_Albatross":                "USA (Hawaii), Japan — breeds on Midway Atoll and Hawaiian Islands",
    "003.Sooty_Albatross":                 "South Atlantic & Indian Ocean — Tristan da Cunha, South Georgia",
    "004.Groove_billed_Ani":               "Mexico, Costa Rica, Panama, Colombia, Venezuela — tropical lowlands",
    "005.Crested_Auklet":                  "USA (Alaska), Russia — Bering Sea islands, Aleutian Islands",
    "006.Least_Auklet":                    "USA (Alaska), Russia — St. Lawrence Island, Pribilof Islands",
    "007.Parakeet_Auklet":                 "USA (Alaska), Russia — Bering Sea, Aleutian Islands",
    "008.Rhinoceros_Auklet":               "USA (California, Oregon, Washington), Canada (British Columbia), Japan",
    "009.Brewer_Blackbird":                "USA (western states), Canada — open farmlands of California, Oregon, Montana",
    "010.Red_winged_Blackbird":            "USA, Canada, Mexico — widespread across North America, wetlands",
    "011.Rusty_Blackbird":                 "Canada (boreal forest), USA (eastern states in winter)",
    "012.Yellow_headed_Blackbird":         "USA (western states), Canada (prairies), Mexico — Great Plains marshes",
    "013.Bobolink":                        "USA, Canada (breeds), Argentina, Bolivia (winters)",
    "014.Indigo_Bunting":                  "USA (eastern & central states), Mexico, Central America — common in Midwest",
    "015.Lazuli_Bunting":                  "USA (western states), Mexico — California, Arizona, Colorado shrublands",
    "016.Painted_Bunting":                 "USA (southern states — Texas, Florida, Louisiana), Mexico, Central America",
    "017.Cardinal":                        "USA (eastern & southern states), Canada (Ontario), Mexico — very common",
    "018.Spotted_Catbird":                 "Australia (Queensland, New South Wales) — rainforests of northeast Australia",
    "019.Gray_Catbird":                    "USA, Canada (breeds), Central America, Caribbean (winters)",
    "020.Yellow_breasted_Chat":            "USA, Canada (breeds), Mexico, Central America (winters)",
    "021.Eastern_Towhee":                  "USA (eastern states), Canada (Ontario) — common from Maine to Florida",
    "022.Chuck_will_Widow":                "USA (southeastern states — Florida, Georgia, Texas), Caribbean (winters)",
    "023.Brandt_Cormorant":                "USA (California, Oregon, Washington) — Pacific coast from Alaska to Baja",
    "024.Red_faced_Cormorant":             "USA (Alaska), Russia — Aleutian Islands, Kodiak Island",
    "025.Pelagic_Cormorant":               "USA (Pacific coast), Canada (British Columbia), Russia",
    "026.Bronzed_Cowbird":                 "USA (Texas, Arizona), Mexico, Central America",
    "027.Shiny_Cowbird":                   "South America (widespread), Caribbean — Argentina, Brazil, Colombia",
    "028.Brown_Creeper":                   "USA, Canada — found in mature forests across North America year-round",
    "029.American_Crow":                   "USA, Canada — one of the most intelligent and common birds in North America",
    "030.Fish_Crow":                       "USA (eastern & southeastern coastal states — New York to Texas)",
    "031.Black_billed_Cuckoo":             "USA (eastern & central states), Canada (Ontario, Quebec), South America",
    "033.Yellow_billed_Cuckoo":            "USA (widespread), Canada (Ontario), Mexico, Central America",
    "037.Acadian_Flycatcher":              "USA (eastern states — Ohio, Virginia, Georgia, Texas), Central America",
    "038.Great_Crested_Flycatcher":        "USA (eastern states), Canada (Ontario), Mexico, Central America",
    "039.Least_Flycatcher":                "USA (northeastern states), Canada (widespread), Mexico, Central America",
    "040.Olive_sided_Flycatcher":          "USA (western states, Alaska), Canada (boreal), South America (winters)",
    "043.Yellow_bellied_Flycatcher":       "Canada (boreal — Ontario to Newfoundland), USA (northeastern states)",
    "046.Gadwall":                         "USA, Canada (breeds), UK, Europe, Asia",
    "050.Eared_Grebe":                     "USA (western states), Canada (prairies), Mexico, Spain, Africa",
    "051.Horned_Grebe":                    "USA, Canada (breeds), UK, northern Europe",
    "052.Pied_billed_Grebe":               "USA, Canada — most common grebe in North America",
    "053.Western_Grebe":                   "USA (western states), Canada (British Columbia, prairies)",
    "058.Pigeon_Guillemot":                "USA (Alaska, California), Canada, Russia — North Pacific coastal cliffs",
    "059.California_Gull":                 "USA (California, Great Basin states), Canada",
    "060.Glaucous_winged_Gull":            "USA (Alaska, Washington, Oregon), Canada (British Columbia)",
    "061.Heermann_Gull":                   "USA (California coast), Mexico (Baja California)",
    "062.Herring_Gull":                    "USA, Canada, UK, northern Europe — widespread across North Atlantic",
    "063.Ivory_Gull":                      "Canada (Arctic), Russia (Arctic), Norway (Svalbard)",
    "064.Ring_billed_Gull":                "USA, Canada — one of the most common gulls across North America",
    "065.Slaty_backed_Gull":               "Russia (eastern Siberia), Japan, South Korea, China",
    "066.Western_Gull":                    "USA (California, Oregon, Washington) — Pacific coast only",
    "067.Anna_Hummingbird":                "USA (California, Arizona, Oregon), Mexico (Baja) — Pacific coast year-round",
    "068.Ruby_throated_Hummingbird":       "USA (eastern states), Canada (Ontario), Mexico, Central America",
    "069.Rufous_Hummingbird":              "USA (western states), Canada (British Columbia), Mexico",
    "070.Green_Violetear":                 "Mexico, Guatemala, Costa Rica, Venezuela, Colombia — mountain forests",
    "071.Long_tailed_Jaeger":              "Canada (Arctic), USA (Alaska), winters in South Atlantic",
    "072.Pomarine_Jaeger":                 "USA (Alaska), Canada (Arctic tundra)",
    "073.Blue_Jay":                        "USA (eastern & central states), Canada (Ontario, Quebec)",
    "074.Florida_Jay":                     "USA (Florida only) — found exclusively in Florida scrub habitat",
    "075.Green_Jay":                       "USA (southern Texas only), Mexico, Central America",
    "076.Dark_eyed_Junco":                 "USA, Canada — one of the most common birds in North America",
    "077.Tropical_Kingbird":               "USA (southern Arizona, Texas), Mexico, Central America, South America",
    "078.Gray_Kingbird":                   "USA (Florida), Caribbean islands — Cuba, Jamaica, Puerto Rico",
    "079.Belted_Kingfisher":               "USA, Canada — found near rivers and lakes across all of North America",
    "080.Green_Kingfisher":                "USA (southern Texas, Arizona), Mexico, Central America, South America",
    "081.Pied_Kingfisher":                 "Africa (sub-Saharan), India, Southeast Asia",
    "082.Ringed_Kingfisher":               "USA (southern Texas), Mexico, Central America, South America",
    "083.White_breasted_Kingfisher":       "India, Sri Lanka, Southeast Asia (Thailand, Vietnam, Philippines)",
    "084.Red_legged_Kittiwake":            "USA (Alaska — Pribilof Islands) — very restricted range",
    "085.Horned_Lark":                     "USA, Canada — most widespread lark in North America",
    "086.Pacific_Loon":                    "USA (Alaska), Canada (Arctic) — winters along Pacific coast",
    "087.Mallard":                         "USA, Canada, UK, Europe, Asia — most widespread duck in the world",
    "088.Western_Meadowlark":              "USA (western & central states), Canada (prairies), Mexico",
    "089.Hooded_Merganser":                "USA, Canada — breeds in forested lakes from Alaska to Florida",
    "090.Red_breasted_Merganser":          "USA, Canada, UK, northern Europe — circumpolar breeding range",
    "091.Mockingbird":                     "USA (all states), Canada (southern Ontario), Mexico, Caribbean",
    "092.Nighthawk":                       "USA, Canada (breeds), South America (winters)",
    "093.Clark_Nutcracker":                "USA (western mountain states — Colorado, Wyoming, Montana, California)",
    "094.White_breasted_Nuthatch":         "USA, Canada — common in deciduous forests across North America",
    "095.Baltimore_Oriole":                "USA (eastern states), Canada (Ontario), Mexico, Central America (winters)",
    "096.Hooded_Oriole":                   "USA (California, Arizona, Texas), Mexico",
    "097.Orchard_Oriole":                  "USA (eastern & central states), Canada, Mexico, Central America (winters)",
    "098.Scott_Oriole":                    "USA (Texas, Arizona, California), Mexico",
    "099.Ovenbird":                        "USA (eastern states), Canada — breeds from Georgia to Manitoba",
    "100.Brown_Pelican":                   "USA (coastal states — California, Florida, Texas), Mexico, Caribbean",
    "101.White_Pelican":                   "USA, Canada — breeds on lakes in Great Plains, winters on Gulf Coast",
    "102.Western_Wood_Pewee":              "USA (western states), Canada (British Columbia), Mexico, South America",
    "103.Sayornis":                        "USA, Canada, Mexico — widespread flycatcher across North America",
    "104.American_Pipit":                  "USA (Alaska, mountain states), Canada (Arctic)",
    "105.Whip_poor_Will":                  "USA (eastern states), Canada (Ontario, Quebec), Mexico, Central America",
    "106.Horned_Puffin":                   "USA (Alaska), Russia, Canada — North Pacific",
    "107.Common_Raven":                    "USA (western states, Alaska), Canada, UK, northern Europe, Asia",
    "108.White_necked_Raven":              "USA (Texas, Arizona), Mexico — Chihuahuan and Sonoran Desert",
    "109.American_Redstart":               "USA (eastern states), Canada — breeds from Georgia to Nova Scotia",
    "110.Geococcyx":                       "USA (Texas, New Mexico, Arizona, California), Mexico",
    "111.Loggerhead_Shrike":               "USA, Canada (southern Ontario) — widespread but declining",
    "112.Great_Grey_Shrike":               "Canada (Arctic), Russia, northern Europe (Scandinavia, Finland)",
    "113.Baird_Sparrow":                   "USA (Montana, North Dakota), Canada (Manitoba, Saskatchewan)",
    "114.Black_throated_Sparrow":          "USA (Texas, New Mexico, Arizona, Nevada, California), Mexico",
    "115.Brewer_Sparrow":                  "USA (western states), Canada (British Columbia, Alberta), Mexico",
    "116.Chipping_Sparrow":                "USA, Canada — one of the most common sparrows across North America",
    "117.Clay_colored_Sparrow":            "USA (Great Plains states), Canada (prairies)",
    "118.House_Sparrow":                   "USA, Canada, UK, Europe, Australia, India — introduced worldwide",
    "119.Field_Sparrow":                   "USA (eastern & central states) — common from Texas to New England",
    "120.Fox_Sparrow":                     "USA (western states, Alaska), Canada",
    "121.Grasshopper_Sparrow":             "USA (central & eastern states), Canada (Ontario), Mexico",
    "122.Harris_Sparrow":                  "USA (Great Plains — Kansas, Oklahoma, Texas in winter), Canada",
    "123.Henslow_Sparrow":                 "USA (Midwest — Ohio, Indiana, Illinois, New York) — rare and declining",
    "124.Le_Conte_Sparrow":                "USA (Great Plains, Southeast), Canada (prairies)",
    "125.Lincoln_Sparrow":                 "USA, Canada (breeds), Mexico, Central America (winters)",
    "126.Nelson_Sharp_tailed_Sparrow":     "USA (Atlantic coast — Maine to Virginia), Canada (Maritime provinces)",
    "127.Savannah_Sparrow":                "USA, Canada — one of the most widespread sparrows in North America",
    "128.Seaside_Sparrow":                 "USA (Atlantic & Gulf coast — Virginia to Texas)",
    "129.Song_Sparrow":                    "USA, Canada — extremely common and widespread across North America",
    "130.Tree_Sparrow":                    "Canada (Arctic, boreal), USA (northern states in winter)",
    "131.Vesper_Sparrow":                  "USA (western & central states), Canada (prairies), Mexico (winters)",
    "132.White_crowned_Sparrow":           "USA, Canada — widespread migrant, breeds in Canada and mountain west",
    "133.White_throated_Sparrow":          "USA (eastern states in winter), Canada (breeds)",
    "134.Cape_Glossy_Starling":            "Kenya, Tanzania, Uganda, Ethiopia, Somalia — East Africa savanna",
    "135.Bank_Swallow":                    "USA, Canada (breeds), South America (winters)",
    "136.Barn_Swallow":                    "USA, Canada (breeds), South America, Africa (winters)",
    "137.Cliff_Swallow":                   "USA, Canada (breeds), South America (winters)",
    "138.Tree_Swallow":                    "USA, Canada (breeds), Mexico, Central America (winters)",
    "139.Scarlet_Tanager":                 "USA (eastern states), Canada (Ontario), South America (winters)",
    "140.Summer_Tanager":                  "USA (southern states — Georgia, Texas, Carolina), Mexico, South America",
    "141.Artic_Tern":                      "Canada, USA (Alaska), UK, Norway — longest migration on Earth",
    "142.Black_Tern":                      "USA (Midwest lakes), Canada, Europe",
    "143.Caspian_Tern":                    "USA (Great Lakes, Gulf Coast), Canada, Europe, Africa, Australia",
    "144.Common_Tern":                     "USA, Canada, UK, Europe — widespread across North Atlantic coasts",
    "145.Elegant_Tern":                    "USA (California coast), Mexico (Baja California)",
    "146.Forsters_Tern":                   "USA, Canada — breeds in Great Plains marshes, winters on both coasts",
    "147.Least_Tern":                      "USA (Atlantic & Gulf coast, river valleys), Mexico, Caribbean",
    "148.Green_tailed_Towhee":             "USA (western mountain states — Colorado, Utah, California), Mexico",
    "149.Brown_Thrasher":                  "USA (eastern & central states), Canada (Ontario)",
    "150.Sage_Thrasher":                   "USA (Great Basin — Nevada, Idaho, Wyoming, Oregon), Mexico (winters)",
    "151.Black_capped_Vireo":              "USA (Texas, Oklahoma only), Mexico (winters) — endangered species",
    "152.Blue_headed_Vireo":               "USA (eastern states), Canada (Ontario, Quebec), Central America",
    "153.Philadelphia_Vireo":              "USA (northeastern states), Canada (Ontario to Alberta), Central America",
    "154.Red_eyed_Vireo":                  "USA, Canada — one of the most common breeding birds in eastern North America",
    "155.Warbling_Vireo":                  "USA, Canada (breeds), Mexico, Central America (winters)",
    "156.White_eyed_Vireo":                "USA (southeastern & eastern states), Mexico, Central America (winters)",
    "157.Yellow_throated_Vireo":           "USA (eastern states), Canada (Ontario), South America (winters)",
    "158.Bay_breasted_Warbler":            "Canada (boreal — Ontario to Newfoundland), USA (eastern states)",
    "159.Black_and_white_Warbler":         "USA (eastern states), Canada",
    "160.Black_throated_Blue_Warbler":     "USA (eastern mountain states — Appalachians), Canada, Caribbean",
    "161.Blue_winged_Warbler":             "USA (eastern states — Ohio, Indiana, New York), Mexico, Central America",
    "162.Canada_Warbler":                  "USA (northeastern states), Canada (Ontario to Nova Scotia), South America",
    "163.Cape_May_Warbler":                "Canada (boreal — Ontario, Quebec, Manitoba), USA (eastern states)",
    "164.Cerulean_Warbler":                "USA (Appalachians, Midwest), Canada (Ontario), South America (winters)",
    "165.Chestnut_sided_Warbler":          "USA (northeastern states), Canada (Ontario, Quebec), Central America",
    "166.Golden_winged_Warbler":           "USA (Appalachians, Great Lakes region), Canada, Central America",
    "167.Hooded_Warbler":                  "USA (eastern states — Georgia, Virginia, Tennessee), Mexico",
    "168.Kentucky_Warbler":                "USA (southeastern states — Kentucky, Tennessee, Arkansas), Central America",
    "169.Magnolia_Warbler":                "Canada (boreal — Ontario to Nova Scotia), USA (eastern states)",
    "170.Mourning_Warbler":                "Canada (Ontario to Manitoba), USA (northeastern states), Central America",
    "171.Myrtle_Warbler":                  "USA, Canada — one of the most common warblers, breeds across Canada",
    "172.Nashville_Warbler":               "USA (northeastern & western states), Canada, Mexico, Central America",
    "173.Orange_crowned_Warbler":          "USA (western states), Canada (British Columbia, Alberta), Mexico",
    "174.Palm_Warbler":                    "Canada (boreal bogs — Ontario to Newfoundland), USA (Florida in winter)",
    "175.Pine_Warbler":                    "USA (southeastern states — Florida, Georgia, Texas, Carolina)",
    "176.Prairie_Warbler":                 "USA (eastern states — Michigan, Ohio, New Jersey, Florida), Caribbean",
    "177.Prothonotary_Warbler":            "USA (southeastern states — Louisiana, Mississippi, Tennessee, Virginia)",
    "178.Swainson_Warbler":                "USA (southeastern states — Arkansas, Louisiana, South Carolina)",
    "179.Tennessee_Warbler":               "Canada (boreal — Ontario to Alberta), USA (eastern states)",
    "180.Wilson_Warbler":                  "USA (western states, Alaska), Canada (widespread), Mexico",
    "181.Worm_eating_Warbler":             "USA (eastern states — Pennsylvania, Maryland, Ohio, Georgia)",
    "182.Yellow_Warbler":                  "USA, Canada — most widespread warbler in North America",
    "183.Northern_Waterthrush":            "USA (northeastern states), Canada (widespread boreal), Central America",
    "184.Louisiana_Waterthrush":           "USA (eastern states — Pennsylvania to Georgia, west to Kansas)",
    "185.Bohemian_Waxwing":                "Canada (boreal — British Columbia to Manitoba), USA (northern states)",
    "186.Cedar_Waxwing":                   "USA, Canada — very common, found across all of North America year-round",
    "187.American_Three_toed_Woodpecker":  "USA (Alaska, mountain west), Canada (boreal)",
    "188.Pileated_Woodpecker":             "USA (eastern states, Pacific Northwest), Canada",
    "189.Red_bellied_Woodpecker":          "USA (eastern states) — very common from Florida to New York to Texas",
    "190.Red_cockaded_Woodpecker":         "USA (southeastern states — North Carolina to Texas) — endangered",
    "191.Red_headed_Woodpecker":           "USA (eastern & central states), Canada (Ontario, Manitoba)",
    "192.Downy_Woodpecker":                "USA, Canada — smallest and most common woodpecker in North America",
    "193.Bewick_Wren":                     "USA (western & southern states — California, Texas, Oregon), Mexico",
    "194.Cactus_Wren":                     "USA (Arizona, California, New Mexico, Texas), Mexico — Sonoran Desert",
    "195.Carolina_Wren":                   "USA (eastern & southeastern states)",
    "196.House_Wren":                      "USA, Canada (breeds), South America (winters)",
    "197.Marsh_Wren":                      "USA, Canada — freshwater marshes across North America",
    "198.Rock_Wren":                       "USA (western states — Colorado, Utah, California, Arizona), Mexico",
    "199.Winter_Wren":                     "USA (Pacific Northwest, Appalachians), Canada, UK, Europe, Asia",
    "200.Common_Yellowthroat":             "USA, Canada — one of the most widespread warblers in North America",
}

# ── Look-alike species ───────────────────────────────────────
# Partial by design: only visually confusable pairs.
SIMILAR_SPECIES = {
    "001.Black_footed_Albatross":          ("Laysan Albatross", "check the face — Black-footed has a dark face, Laysan has a white face"),
    "002.Laysan_Albatross":                ("Black footed Albatross", "check the face — Laysan has a white face, Black-footed has a dark face"),
    "014.Indigo_Bunting":                  ("Lazuli Bunting", "check the breast — Indigo is all blue, Lazuli has a rusty-orange breast"),
    "029.American_Crow":                   ("Common Raven", "check the size & tail — Crow is smaller with fan tail, Raven has wedge tail"),
    "062.Herring_Gull":                    ("Ring billed Gull", "check the bill — Herring Gull has a red spot, Ring-billed has a black ring"),
    "064.Ring_billed_Gull":                ("Herring Gull", "check the bill — Ring-billed has a black ring, Herring has a red spot"),
    "068.Ruby_throated_Hummingbird":       ("Rufous Hummingbird", "check the back — Ruby-throated is green-backed, Rufous has orange-brown back"),
    "069.Rufous_Hummingbird":              ("Ruby throated Hummingbird", "check the back — Rufous is orange-brown, Ruby-throated is metallic green"),
    "073.Blue_Jay":                        ("Florida Jay", "check the crest — Blue Jay has a crest, Florida Jay has no crest"),
    "095.Baltimore_Oriole":                ("Orchard Oriole", "check the color — Baltimore is bright orange, Orchard is darker chestnut"),
    "129.Song_Sparrow":                    ("Savannah Sparrow", "check the breast — Song Sparrow has a central spot, Savannah has streaks"),
    "136.Barn_Swallow":                    ("Cliff Swallow", "check the tail — Barn Swallow has a deep forked tail, Cliff has a square tail"),
    "137.Cliff_Swallow":                   ("Barn Swallow", "check the tail — Cliff has a square tail, Barn Swallow has a deep fork"),
    "139.Scarlet_Tanager":                 ("Summer Tanager", "check the wings — Scarlet has black wings, Summer Tanager is all red"),
    "186.Cedar_Waxwing":                   ("Bohemian Waxwing", "check the belly — Cedar has a yellow belly, Bohemian has rusty undertail"),
    "192.Downy_Woodpecker":                ("Pileated Woodpecker", "check the size — Downy is sparrow-sized, Pileated is crow-sized"),
}

# ── Migration ────────────────────────────────────────────────
MIGRATION_MAP = {
    "001.Black_footed_Albatross":          "Year-round in North Pacific Ocean. Breeds on Hawaiian Islands (Dec–Jul), roams Pacific rest of year",
    "002.Laysan_Albatross":                "Year-round in North Pacific. Breeds on Midway Atoll & Hawaii (Nov–Jul), roams North Pacific rest of year",
    "003.Sooty_Albatross":                 "Year-round in South Atlantic & Indian Ocean. Breeds on Tristan da Cunha & South Georgia islands",
    "004.Groove_billed_Ani":               "Year-round resident in Mexico & Central America. Some move to southern Texas in summer (Apr–Sep)",
    "005.Crested_Auklet":                  "Breeds on Aleutian Islands & Bering Sea (May–Aug) → winters in open North Pacific Ocean",
    "006.Least_Auklet":                    "Breeds on St. Lawrence & Pribilof Islands Alaska (Jun–Aug) → winters in North Pacific",
    "007.Parakeet_Auklet":                 "Breeds on Alaskan & Russian islands (May–Aug) → winters in open North Pacific Ocean",
    "008.Rhinoceros_Auklet":               "Breeds on Pacific coast islands (Apr–Aug) → winters offshore in North Pacific",
    "009.Brewer_Blackbird":                "Year-round in western USA. Northern Canada birds migrate south to California & Mexico (Oct–Mar)",
    "010.Red_winged_Blackbird":            "Year-round across most of USA. Northern Canada birds migrate south to southern USA (Oct–Apr)",
    "011.Rusty_Blackbird":                 "Breeds in Alaska & boreal Canada (May–Aug) → winters in southeastern USA (Sep–Apr)",
    "012.Yellow_headed_Blackbird":         "Breeds in Great Plains marshes USA & Canada (May–Aug) → winters in Mexico & southwestern USA",
    "013.Bobolink":                        "Breeds in USA & Canada prairies (May–Aug) → migrates 12,000 miles to Argentina & Bolivia (Sep–Apr). One of the longest migrations of any songbird",
    "014.Indigo_Bunting":                  "Breeds in eastern & central USA (May–Aug) → winters in Mexico, Cuba & Central America (Sep–Apr). Navigates by stars at night",
    "015.Lazuli_Bunting":                  "Breeds in western USA & Canada (May–Aug) → winters in western Mexico (Sep–Apr)",
    "016.Painted_Bunting":                 "Breeds in southern USA — Texas, Louisiana, Florida (Apr–Aug) → winters in Florida, Caribbean & Central America",
    "017.Cardinal":                        "Year-round resident across eastern & southern USA. Does not migrate",
    "018.Spotted_Catbird":                 "Year-round resident in Queensland & New South Wales, Australia. Short local movements only",
    "019.Gray_Catbird":                    "Breeds in USA & Canada (May–Aug) → winters in Florida, Caribbean & Central America (Sep–Apr)",
    "020.Yellow_breasted_Chat":            "Breeds across USA & Canada (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "021.Eastern_Towhee":                  "Year-round in southeastern USA. Northern birds migrate south in winter (Oct–Mar)",
    "022.Chuck_will_Widow":                "Breeds in southeastern USA (Apr–Aug) → winters in Caribbean & Central America (Sep–Mar)",
    "023.Brandt_Cormorant":                "Year-round on Pacific coast from Alaska to Baja California. Short local movements",
    "024.Red_faced_Cormorant":             "Year-round resident on Aleutian Islands & Kodiak, Alaska. Does not migrate",
    "025.Pelagic_Cormorant":               "Year-round on Pacific coast. Some northernmost birds move south slightly in winter",
    "026.Bronzed_Cowbird":                 "Year-round in Mexico & Central America. Moves into southern Texas & Arizona (Mar–Sep)",
    "027.Shiny_Cowbird":                   "Year-round across South America & Caribbean. Expanding northward into USA",
    "028.Brown_Creeper":                   "Year-round in mature forests across North America. Mountain birds move to lower elevations in winter",
    "029.American_Crow":                   "Year-round across most of USA. Northern birds may move south in harsh winters. Highly intelligent",
    "030.Fish_Crow":                       "Year-round on Atlantic & Gulf coasts eastern USA. Short local movements only",
    "031.Black_billed_Cuckoo":             "Breeds in eastern USA & Canada (May–Aug) → winters in South America — Colombia to Bolivia (Sep–Apr)",
    "033.Yellow_billed_Cuckoo":            "Breeds across USA & Canada (May–Aug) → winters in South America (Aug–Apr). Famous for calling just before rainstorms",
    "037.Acadian_Flycatcher":              "Breeds in eastern USA (May–Aug) → winters in Colombia, Ecuador & Central America (Sep–Apr)",
    "038.Great_Crested_Flycatcher":        "Breeds in eastern USA & Canada (May–Aug) → winters in Florida, Caribbean & South America (Sep–Apr)",
    "039.Least_Flycatcher":                "Breeds in northeastern USA & Canada (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "040.Olive_sided_Flycatcher":          "Breeds in western USA, Alaska & boreal Canada (Jun–Aug) → winters in Andes of South America — Peru, Bolivia (Sep–May). One of the longest Flycatcher migrations",
    "043.Yellow_bellied_Flycatcher":       "Breeds in boreal Canada & northeastern USA (Jun–Aug) → winters in Mexico & Central America (Sep–May)",
    "046.Gadwall":                         "Breeds in Great Plains USA & Canada (May–Aug) → winters across southern USA, Mexico & Caribbean (Oct–Mar)",
    "050.Eared_Grebe":                     "Breeds on western USA & Canada lakes (May–Aug) → winters on Pacific coast & Gulf of Mexico (Sep–Apr)",
    "051.Horned_Grebe":                    "Breeds in Alaska, Canada & northern Europe (May–Aug) → winters on both USA coasts (Sep–Apr)",
    "052.Pied_billed_Grebe":               "Year-round across most of USA. Northern birds migrate south in winter (Oct–Mar)",
    "053.Western_Grebe":                   "Breeds on inland lakes western USA & Canada (Apr–Aug) → winters on Pacific coast (Sep–Mar)",
    "058.Pigeon_Guillemot":                "Year-round on North Pacific coast. Short offshore movements in winter",
    "059.California_Gull":                 "Breeds inland at Great Basin lakes (Apr–Aug) → winters on California & Pacific coast (Sep–Mar)",
    "060.Glaucous_winged_Gull":            "Year-round on Pacific Northwest coast. Some move south to California in winter",
    "061.Heermann_Gull":                   "Breeds on Isla Raza Mexico (Jan–Jun) → moves north to California & Oregon coast (Jul–Nov) — reverse migration",
    "062.Herring_Gull":                    "Breeds in Canada & northern USA (Apr–Aug) → winters across all USA coasts (Sep–Mar)",
    "063.Ivory_Gull":                      "Year-round in high Arctic. Moves south only when sea ice forces it — rarely seen in USA",
    "064.Ring_billed_Gull":                "Breeds in Canada & northern USA (Apr–Aug) → winters across all of USA (Sep–Mar). Very common in parking lots!",
    "065.Slaty_backed_Gull":               "Year-round in eastern Russia & Japan. Rare winter visitor to Alaska & Pacific coast",
    "066.Western_Gull":                    "Year-round on California, Oregon & Washington coast. Does not migrate far",
    "067.Anna_Hummingbird":                "Year-round on Pacific coast — one of very few hummingbirds that does not migrate south. Stays in California & Oregon all winter",
    "068.Ruby_throated_Hummingbird":       "Breeds in eastern USA & Canada (Apr–Aug) → crosses Gulf of Mexico non-stop to winter in Mexico & Central America (Sep–Apr). Incredible 500-mile non-stop ocean crossing",
    "069.Rufous_Hummingbird":              "Breeds in Pacific Northwest & Alaska (Apr–Jul) → migrates south through Rocky Mountains to winter in Mexico (Aug–Mar). Longest migration of any hummingbird — 3,900 miles",
    "070.Green_Violetear":                 "Year-round resident in mountain forests of Mexico & Central America. Short altitudinal movements only",
    "071.Long_tailed_Jaeger":              "Breeds on Arctic tundra Canada & Alaska (Jun–Aug) → migrates over ocean to winter in South Atlantic (Sep–May)",
    "072.Pomarine_Jaeger":                 "Breeds on Arctic tundra (Jun–Aug) → winters off coasts of South America & West Africa (Sep–May). Rarely seen inland",
    "073.Blue_Jay":                        "Year-round across eastern USA. Some northern birds migrate south in large flocks in autumn — not all individuals migrate",
    "074.Florida_Jay":                     "Year-round resident in Florida scrub only. Does not migrate at all — one of the most sedentary birds in North America",
    "075.Green_Jay":                       "Year-round resident in southern Texas & Central America. Does not migrate",
    "076.Dark_eyed_Junco":                 "Breeds in Canada & mountain USA (May–Aug) → winters across all of USA (Oct–Apr). Called the snowbird — their arrival signals winter coming",
    "077.Tropical_Kingbird":               "Year-round in Mexico & Central America. Moves into southern Arizona & Texas (Apr–Sep)",
    "078.Gray_Kingbird":                   "Breeds in Florida & Caribbean (Apr–Aug) → winters in northern South America (Sep–Mar)",
    "079.Belted_Kingfisher":               "Year-round across most of USA near water. Northern Canada birds move south in winter",
    "080.Green_Kingfisher":                "Year-round resident in southern Texas, Mexico & Central America. Does not migrate",
    "081.Pied_Kingfisher":                 "Year-round resident across Africa & South Asia. Does not migrate",
    "082.Ringed_Kingfisher":               "Year-round in southern Texas, Mexico & South America. Does not migrate",
    "083.White_breasted_Kingfisher":       "Year-round resident across India & Southeast Asia. Does not migrate",
    "084.Red_legged_Kittiwake":            "Breeds on Pribilof Islands Alaska (May–Aug) → winters in North Pacific Ocean. Rarely seen on land outside breeding season",
    "085.Horned_Lark":                     "Year-round across open areas of USA & Canada. Northern birds move south in winter (Oct–Mar)",
    "086.Pacific_Loon":                    "Breeds on Arctic lakes Canada & Alaska (Jun–Aug) → winters along Pacific coast from Alaska to California (Sep–May)",
    "087.Mallard":                         "Year-round in most of USA. Northern Canada & Alaska birds migrate south to USA & Mexico in winter (Oct–Mar)",
    "088.Western_Meadowlark":              "Year-round across western & central USA. Northern Canada birds migrate south in winter (Oct–Mar)",
    "089.Hooded_Merganser":                "Breeds in forested lakes USA & Canada (Apr–Aug) → winters in southern USA & Mexico (Oct–Mar)",
    "090.Red_breasted_Merganser":          "Breeds in Arctic & boreal Canada (May–Aug) → winters on both USA coasts (Sep–Apr)",
    "091.Mockingbird":                     "Year-round across USA & Mexico. Northern birds may move slightly south in harsh winters",
    "092.Nighthawk":                       "Breeds across USA & Canada (May–Aug) → migrates to South America (Bolivia, Argentina) for winter (Sep–Apr). One of the longest migrations among North American birds",
    "093.Clark_Nutcracker":                "Year-round in western mountain USA. Short altitudinal movements — moves to lower elevations in winter",
    "094.White_breasted_Nuthatch":         "Year-round resident across North America. Does not migrate",
    "095.Baltimore_Oriole":                "Breeds in eastern USA & Canada (May–Aug) → winters in Central America & northern South America (Sep–Apr)",
    "096.Hooded_Oriole":                   "Breeds in southwestern USA (Apr–Aug) → winters in Mexico (Sep–Mar)",
    "097.Orchard_Oriole":                  "Breeds in eastern & central USA (May–Aug) → winters in Central America & northern South America (Aug–Apr). Leaves very early — one of first migrants to depart",
    "098.Scott_Oriole":                    "Breeds in desert southwest USA (Apr–Aug) → winters in Mexico (Sep–Mar)",
    "099.Ovenbird":                        "Breeds in eastern USA & Canada (May–Aug) → winters in Caribbean, Mexico & Central America (Sep–Apr)",
    "100.Brown_Pelican":                   "Year-round on USA coasts. Some northern birds move south in winter. Florida birds stay year-round",
    "101.White_Pelican":                   "Breeds on inland lakes Great Plains (Apr–Aug) → winters on Gulf Coast & Pacific coast (Sep–Mar)",
    "102.Western_Wood_Pewee":              "Breeds in western USA & Canada (May–Aug) → winters in South America (Bolivia, Peru, Ecuador) (Sep–Apr)",
    "103.Sayornis":                        "Year-round in southwestern USA & Mexico. Northern birds migrate south in winter (Oct–Mar)",
    "104.American_Pipit":                  "Breeds on Arctic tundra & mountain tops (Jun–Aug) → winters across southern USA & Mexico (Sep–May)",
    "105.Whip_poor_Will":                  "Breeds in eastern USA & Canada (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "106.Horned_Puffin":                   "Breeds on Alaskan sea cliffs (May–Aug) → winters in open North Pacific Ocean far from shore (Sep–Apr)",
    "107.Common_Raven":                    "Year-round resident. Does not migrate. Stays in same territory year-round",
    "108.White_necked_Raven":              "Year-round resident in desert southwest USA & Mexico. Does not migrate",
    "109.American_Redstart":               "Breeds in eastern USA & Canada (May–Aug) → winters in Caribbean, Mexico & South America (Sep–Apr)",
    "110.Geococcyx":                       "Year-round resident in desert southwest USA & Mexico. Does not migrate",
    "111.Loggerhead_Shrike":               "Year-round in southern USA. Northern birds migrate south in winter (Oct–Mar)",
    "112.Great_Grey_Shrike":               "Year-round in northern Europe & Russia. Irruptive — moves south into Europe in some winters",
    "113.Baird_Sparrow":                   "Breeds in Great Plains USA & Canada (May–Aug) → winters in Texas, New Mexico & Mexico (Sep–Apr)",
    "114.Black_throated_Sparrow":          "Year-round in desert southwest USA & Mexico. Some move to lower elevations in winter",
    "115.Brewer_Sparrow":                  "Breeds in Great Basin USA & Canada (May–Aug) → winters in Mexico & southwestern USA (Sep–Apr)",
    "116.Chipping_Sparrow":                "Breeds across USA & Canada (Apr–Aug) → winters in southern USA & Mexico (Sep–Apr)",
    "117.Clay_colored_Sparrow":            "Breeds in Great Plains Canada & USA (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "118.House_Sparrow":                   "Year-round resident worldwide. Does not migrate — introduced species stays put all year",
    "119.Field_Sparrow":                   "Year-round in eastern USA. Northern birds move south slightly in winter (Oct–Mar)",
    "120.Fox_Sparrow":                     "Breeds in Alaska & Canada (May–Aug) → winters in western & southern USA (Oct–Apr)",
    "121.Grasshopper_Sparrow":             "Breeds in eastern & central USA (May–Aug) → winters in southern USA, Caribbean & Central America",
    "122.Harris_Sparrow":                  "Breeds in boreal Canada (Jun–Aug) → winters in Great Plains USA — Kansas, Oklahoma, Texas (Sep–Apr)",
    "123.Henslow_Sparrow":                 "Breeds in Midwest USA (May–Aug) → winters in southeastern USA — Florida, Georgia, Carolina (Sep–Apr)",
    "124.Le_Conte_Sparrow":                "Breeds in northern Great Plains Canada (Jun–Aug) → winters in southeastern USA (Sep–Apr)",
    "125.Lincoln_Sparrow":                 "Breeds in Canada & mountain USA (May–Aug) → winters in southern USA & Mexico (Sep–Apr)",
    "126.Nelson_Sharp_tailed_Sparrow":     "Breeds in Canadian prairies & Atlantic coast marshes (Jun–Aug) → winters on Atlantic & Gulf coast (Sep–Apr)",
    "127.Savannah_Sparrow":                "Breeds across USA & Canada (Apr–Aug) → winters in southern USA, Mexico & Caribbean (Sep–Apr)",
    "128.Seaside_Sparrow":                 "Year-round resident in Atlantic & Gulf coast salt marshes. Does not migrate far",
    "129.Song_Sparrow":                    "Year-round across most of USA. Northern Canada birds migrate south in winter (Oct–Mar)",
    "130.Tree_Sparrow":                    "Breeds in Arctic Canada & Alaska (Jun–Aug) → winters across northern USA (Oct–Apr)",
    "131.Vesper_Sparrow":                  "Breeds in western & central USA & Canada (May–Aug) → winters in southern USA & Mexico (Sep–Apr)",
    "132.White_crowned_Sparrow":           "Breeds in Arctic Canada & Alaska (Jun–Aug) → winters across southern USA & Mexico (Oct–Apr)",
    "133.White_throated_Sparrow":          "Breeds in boreal Canada (Jun–Aug) → winters in eastern & southern USA (Oct–Apr). Very common winter bird feeder visitor",
    "134.Cape_Glossy_Starling":            "Year-round resident in East Africa. Short local movements following rainfall & food",
    "135.Bank_Swallow":                    "Breeds across USA & Canada (May–Aug) → migrates to South America — Peru, Bolivia, Brazil (Sep–Apr)",
    "136.Barn_Swallow":                    "Breeds across USA & Canada (Apr–Aug) → migrates to Argentina & southern Brazil (Sep–Mar). One of the most widespread migrants in the world",
    "137.Cliff_Swallow":                   "Breeds across USA & Canada (Apr–Aug) → winters in Argentina (Sep–Mar)",
    "138.Tree_Swallow":                    "Breeds in USA & Canada (Apr–Aug) → winters in Florida, Gulf Coast & Central America (Sep–Apr)",
    "139.Scarlet_Tanager":                 "Breeds in eastern USA & Canada (May–Aug) → winters in Colombia, Ecuador & Peru (Sep–Apr)",
    "140.Summer_Tanager":                  "Breeds in southern USA (Apr–Aug) → winters in Mexico, Central & South America (Sep–Apr)",
    "141.Artic_Tern":                      "Breeds in Arctic Canada, Alaska & UK (Jun–Aug) → migrates to Antarctic (Sep–May). Longest migration on Earth — up to 70,000 km per year, seeing more daylight than any other creature",
    "142.Black_Tern":                      "Breeds in Midwest freshwater marshes USA & Canada (May–Aug) → winters off West Africa & northern South America (Sep–Apr)",
    "143.Caspian_Tern":                    "Breeds on Great Lakes & Gulf Coast (Apr–Aug) → winters on Gulf Coast, Caribbean & northern South America (Sep–Mar)",
    "144.Common_Tern":                     "Breeds on North Atlantic coasts USA & Canada (May–Aug) → winters off West Africa & South America (Sep–Apr)",
    "145.Elegant_Tern":                    "Breeds on Isla Raza Mexico (Apr–Jul) → moves north to California coast (Jul–Oct) then winters off Peru & Chile",
    "146.Forsters_Tern":                   "Breeds in Great Plains marshes USA & Canada (May–Aug) → winters on both USA coasts & Caribbean (Sep–Apr)",
    "147.Least_Tern":                      "Breeds on USA beaches & river sandbars (May–Aug) → winters off northern South America (Sep–Apr)",
    "148.Green_tailed_Towhee":             "Breeds in western mountain USA (May–Aug) → winters in Mexico & southwestern USA desert (Sep–Apr)",
    "149.Brown_Thrasher":                  "Year-round in southeastern USA. Northern birds migrate south in winter (Oct–Mar)",
    "150.Sage_Thrasher":                   "Breeds in Great Basin USA (Apr–Aug) → winters in Mexico & Chihuahuan Desert (Sep–Mar)",
    "151.Black_capped_Vireo":              "Breeds in Texas & Oklahoma (Apr–Aug) → winters in western Mexico (Sep–Mar). Endangered species",
    "152.Blue_headed_Vireo":               "Breeds in eastern USA & Canada (May–Aug) → winters in Florida, Caribbean & Central America (Sep–Apr)",
    "153.Philadelphia_Vireo":              "Breeds in Canada & northeastern USA (Jun–Aug) → winters in Central America (Sep–May)",
    "154.Red_eyed_Vireo":                  "Breeds across USA & Canada (May–Aug) → winters in Amazon basin South America (Sep–Apr). Sings more than almost any other bird — up to 20,000 songs per day",
    "155.Warbling_Vireo":                  "Breeds across USA & Canada (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "156.White_eyed_Vireo":                "Breeds in eastern USA (Apr–Aug) → winters in Florida, Caribbean & Central America (Sep–Apr)",
    "157.Yellow_throated_Vireo":           "Breeds in eastern USA & Canada (May–Aug) → winters in Colombia, Venezuela & Central America (Sep–Apr)",
    "158.Bay_breasted_Warbler":            "Breeds in boreal Canada (Jun–Aug) → winters in Panama & northern South America (Sep–May)",
    "159.Black_and_white_Warbler":         "Breeds in eastern USA & Canada (Apr–Aug) → winters in Florida, Caribbean & South America (Sep–Apr)",
    "160.Black_throated_Blue_Warbler":     "Breeds in Appalachian mountains & Canada (May–Aug) → winters in Caribbean — Cuba, Jamaica, Haiti (Sep–Apr)",
    "161.Blue_winged_Warbler":             "Breeds in eastern USA (May–Aug) → winters in Central America (Sep–Apr)",
    "162.Canada_Warbler":                  "Breeds in northeastern USA & Canada (Jun–Aug) → winters in Colombia, Ecuador & Peru (Sep–May)",
    "163.Cape_May_Warbler":                "Breeds in boreal Canada (Jun–Aug) → winters in Caribbean islands (Sep–May)",
    "164.Cerulean_Warbler":                "Breeds in Appalachians & Midwest USA (May–Aug) → winters in Andes of Colombia, Ecuador & Peru (Sep–Apr)",
    "165.Chestnut_sided_Warbler":          "Breeds in northeastern USA & Canada (May–Aug) → winters in Central America (Sep–Apr)",
    "166.Golden_winged_Warbler":           "Breeds in Appalachians & Great Lakes (May–Aug) → winters in Central America & Venezuela (Sep–Apr)",
    "167.Hooded_Warbler":                  "Breeds in eastern USA (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "168.Kentucky_Warbler":                "Breeds in southeastern USA (May–Aug) → winters in Central America & northern South America (Sep–Apr)",
    "169.Magnolia_Warbler":                "Breeds in boreal Canada (Jun–Aug) → winters in Caribbean & Central America (Sep–May)",
    "170.Mourning_Warbler":                "Breeds in Canada & northeastern USA (Jun–Aug) → winters in Costa Rica, Colombia & Venezuela (Sep–May)",
    "171.Myrtle_Warbler":                  "Breeds across Canada (May–Aug) → winters across all of USA — one of the most widespread warblers in winter (Sep–Apr)",
    "172.Nashville_Warbler":               "Breeds in northeastern & western USA & Canada (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "173.Orange_crowned_Warbler":          "Breeds in western USA & Canada (Apr–Aug) → winters in southern USA & Mexico (Sep–Mar)",
    "174.Palm_Warbler":                    "Breeds in boreal bogs Canada (Jun–Aug) → winters in Florida & Caribbean (Sep–Apr). Often seen walking on ground wagging its tail",
    "175.Pine_Warbler":                    "Year-round in southeastern USA pine forests. Northern birds move to southern USA in winter (Oct–Mar)",
    "176.Prairie_Warbler":                 "Breeds in eastern USA (May–Aug) → winters in Florida, Caribbean & Central America (Sep–Apr)",
    "177.Prothonotary_Warbler":            "Breeds in southeastern USA swamps (Apr–Aug) → winters in Colombia, Venezuela & Central America (Sep–Apr)",
    "178.Swainson_Warbler":                "Breeds in southeastern USA (May–Aug) → winters in Caribbean & Yucatan Mexico (Sep–Apr)",
    "179.Tennessee_Warbler":               "Breeds in boreal Canada (Jun–Aug) → winters in Costa Rica, Colombia & Venezuela (Sep–May)",
    "180.Wilson_Warbler":                  "Breeds in western USA, Alaska & Canada (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "181.Worm_eating_Warbler":             "Breeds in eastern USA (May–Aug) → winters in Caribbean & Central America (Sep–Apr)",
    "182.Yellow_Warbler":                  "Breeds across all of USA & Canada (May–Aug) → winters in Mexico, Central & South America (Sep–Apr). Most widespread warbler in North America",
    "183.Northern_Waterthrush":            "Breeds in boreal Canada & northeastern USA (May–Aug) → winters in Caribbean & northern South America (Sep–Apr)",
    "184.Louisiana_Waterthrush":           "Breeds in eastern USA (Apr–Aug) → winters in Caribbean & Central America (Aug–Apr). One of earliest spring migrants to arrive",
    "185.Bohemian_Waxwing":                "Breeds in boreal Canada & Alaska (Jun–Aug) → irruptive winter visitor to northern USA (Oct–Mar). Appears in large flocks unpredictably following berry crops",
    "186.Cedar_Waxwing":                   "Year-round across USA but nomadic — follows fruit & berry crops. Northern birds move south in winter. Travels in flocks",
    "187.American_Three_toed_Woodpecker":  "Year-round in Rocky Mountains, Cascades & boreal Canada. Short movements to lower elevations in winter",
    "188.Pileated_Woodpecker":             "Year-round resident across eastern USA & Pacific Northwest. Does not migrate",
    "189.Red_bellied_Woodpecker":          "Year-round resident in eastern USA. Does not migrate",
    "190.Red_cockaded_Woodpecker":         "Year-round resident in southeastern USA pine forests. Does not migrate. Endangered species",
    "191.Red_headed_Woodpecker":           "Year-round in eastern USA. Northern birds may move south in winter following acorn crops",
    "192.Downy_Woodpecker":                "Year-round resident across North America. Does not migrate — stays in same territory all year",
    "193.Bewick_Wren":                     "Year-round in western & southern USA & Mexico. Does not migrate",
    "194.Cactus_Wren":                     "Year-round resident in Sonoran Desert USA & Mexico. Does not migrate",
    "195.Carolina_Wren":                   "Year-round resident in eastern USA. Does not migrate — very sensitive to cold winters",
    "196.House_Wren":                      "Breeds across USA & Canada (Apr–Aug) → winters in southern USA, Mexico & Central America (Sep–Apr)",
    "197.Marsh_Wren":                      "Year-round on both coasts USA. Interior birds migrate south in winter (Oct–Mar)",
    "198.Rock_Wren":                       "Year-round in western USA rocky areas. Mountain birds move to lower elevations in winter",
    "199.Winter_Wren":                     "Breeds in Pacific Northwest, Appalachians & boreal Canada (May–Aug) → winters across eastern USA (Oct–Apr)",
    "200.Common_Yellowthroat":             "Breeds across USA & Canada (May–Aug) → winters in southern USA, Caribbean & Central America (Sep–Apr)",
}

# ── Coverage guard ───────────────────────────────────────────
# A key that is not one of the model's classes is always a bug: it can
# never be looked up, so it is dead data. Fail hard on that.
#
# A *missing* key is a known, tracked gap — 15 CUB species have no
# hand-written notes — so it warns rather than crashing. Historically
# this went unnoticed because the tables were keyed against a different
# 200-species list, leaving 173/200 lookups silently falling through to
# "Location data not available".
_unknown_keys = (set(HABITAT_MAP) | set(MIGRATION_MAP)
                 | set(SIMILAR_SPECIES)) - set(CLASS_NAMES)
assert not _unknown_keys, (
    f"{len(_unknown_keys)} lookup key(s) match no model class and can never "
    f"be reached: {sorted(_unknown_keys)}"
)

SPECIES_WITHOUT_NOTES = sorted(set(CLASS_NAMES) - set(HABITAT_MAP))
print(f"✅ Species data — habitat {len(HABITAT_MAP)}/{NUM_SPECIES}, "
      f"migration {len(MIGRATION_MAP)}/{NUM_SPECIES}, "
      f"look-alikes {len(SIMILAR_SPECIES)}/{NUM_SPECIES}")
if SPECIES_WITHOUT_NOTES:
    print(f"⚠️  {len(SPECIES_WITHOUT_NOTES)} species have no habitat/migration "
          f"notes yet (e.g. {display_name(SPECIES_WITHOUT_NOTES[0])})")


# ════════════════════════════════════════════════════════════
# BIRD IDENTIFICATION FUNCTION
# ════════════════════════════════════════════════════════════
def predict_bird(image):
    img = Image.fromarray(image).convert("RGB")
    x   = val_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out   = model(x)
        probs = F.softmax(out, dim=1)
        top5  = probs.topk(5)

    results = []
    for prob, idx in zip(top5.values[0], top5.indices[0]):
        folder_name  = CLASS_NAMES[idx.item()]
        species_name = display_name(folder_name)
        habitat      = HABITAT_MAP.get(folder_name, "Location data not available")
        confidence   = prob.item() * 100
        results.append((folder_name, species_name, confidence, habitat))

    top_folder  = results[0][0]
    top_name    = results[0][1]
    top_conf    = results[0][2]
    top_habitat = results[0][3]

    if top_conf < 60:
        conf_msg = f"⚠️  LOW CONFIDENCE ({top_conf:.1f}%) — Try a clearer photo"
    elif top_conf < 80:
        conf_msg = f"🟡  MODERATE CONFIDENCE ({top_conf:.1f}%) — Likely correct"
    else:
        conf_msg = f"✅  HIGH CONFIDENCE ({top_conf:.1f}%) — Very sure"

    similar  = SIMILAR_SPECIES.get(top_folder)
    migration = MIGRATION_MAP.get(top_folder, "Migration data not available for this species")

    output = f"""
🐦  SPECIES     : {top_name}
{conf_msg}

📍  FOUND IN    : {top_habitat}

✈️   MIGRATION   : {migration}
"""
    if similar:
        output += f"\n⚠️  LOOKS SIMILAR TO : {similar[0]}\n    How to tell apart  : {similar[1]}\n"

    output += f"""
{'─'*55}
TOP 5 PREDICTIONS:
"""
    for i, (_folder, name, conf, habitat) in enumerate(results):
        marker = " ◀ TOP PICK" if i == 0 else ""
        output += f"\n#{i+1}  {name}  ({conf:.1f}%){marker}\n    📍 {habitat}\n"

    return output


# ════════════════════════════════════════════════════════════
# GRAD-CAM FUNCTION
# ════════════════════════════════════════════════════════════
def generate_gradcam(image):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        img        = Image.fromarray(image).convert("RGB")
        raw_tensor = raw_transform(img)
        inp_tensor = norm_transform(raw_tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            out      = model(inp_tensor)
            probs    = F.softmax(out, dim=1)
            pred_idx = probs.argmax().item()
            pred_conf = probs[0][pred_idx].item() * 100
            pred_name = CLASS_NAMES[pred_idx].split(".")[-1].replace("_", " ")

        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=inp_tensor, targets=targets)[0]

        rgb_img = raw_tensor.permute(1, 2, 0).numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(rgb_img)
        ax1.set_title("Original Image", fontsize=13)
        ax1.axis("off")

        ax2.imshow(cam_image)
        ax2.set_title(
            f"Grad-CAM — Model focuses on red/yellow areas\nPredicted: {pred_name} ({pred_conf:.1f}%)",
            fontsize=11
        )
        ax2.axis("off")

        plt.suptitle("Grad-CAM Explainability — What the model looks at", fontsize=13)
        plt.tight_layout()

        cam_path = os.path.join(SAVE_DIR, "gradcam_single.png")
        plt.savefig(cam_path, dpi=150, bbox_inches="tight")
        plt.close()

        return cam_path, f"✅ Grad-CAM generated!\nPredicted: {pred_name} ({pred_conf:.1f}%)\nRed/Yellow areas = what the model focused on\nSaved to: {cam_path}"

    except ImportError:
        return None, "⚠️ grad-cam not installed. Run: pip3 install grad-cam"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ════════════════════════════════════════════════════════════
# EVALUATION METRICS FUNCTION
# ════════════════════════════════════════════════════════════
def run_evaluation():
    if not os.path.exists(TEST_DIR):
        return (
            "❌ Test folder not found!\n\n"
            f"Expected at: {TEST_DIR}\n\n"
            "Please download the test images from Kaggle or Colab first.\n"
            "See instructions in the README."
        )
    try:
        from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
        import seaborn as sns

        print("Loading test dataset...")
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)
        test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        all_preds, all_labels, all_probs, all_top5 = [], [], [], []

        print("Running evaluation...")
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images  = images.to(device)
                outputs = model(images)
                probs   = F.softmax(outputs, dim=1)
                top5    = probs.topk(5, dim=1).indices.cpu().numpy()
                preds   = probs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                all_top5.extend(top5)
                if (i+1) % 5 == 0:
                    print(f"   Batch {i+1}/{len(test_loader)}...")

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)

        top1_acc = (all_preds == all_labels).mean() * 100
        top5_acc = np.mean([all_labels[i] in all_top5[i] for i in range(len(all_labels))]) * 100

        short_names = [n.split(".")[-1].replace("_", " ") for n in CLASS_NAMES]
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

        best10  = np.argsort(f1)[-10:][::-1]
        worst10 = np.argsort(f1)[:10]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.barh([short_names[i][:25] for i in best10],  [f1[i]*100 for i in best10],  color="#1D9E75")
        ax1.set_title("Top 10 Best Recognised Species", fontsize=12)
        ax1.set_xlabel("F1 Score (%)")
        for i, v in enumerate([f1[j]*100 for j in best10]):
            ax1.text(v+0.5, i, f"{v:.1f}%", va="center", fontsize=9)

        ax2.barh([short_names[i][:25] for i in worst10], [f1[i]*100 for i in worst10], color="#A32D2D")
        ax2.set_title("Top 10 Hardest Species", fontsize=12)
        ax2.set_xlabel("F1 Score (%)")
        for i, v in enumerate([f1[j]*100 for j in worst10]):
            ax2.text(v+0.5, i, f"{v:.1f}%", va="center", fontsize=9)

        plt.suptitle("Per-Species F1 Score Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "f1_per_species.png"), dpi=150, bbox_inches="tight")
        plt.close()

        report = classification_report(all_labels, all_preds, target_names=short_names, digits=3)
        report_path = os.path.join(SAVE_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\nTop-5 Accuracy: {top5_acc:.2f}%\n\n{report}")

        return (
            f"✅ EVALUATION COMPLETE!\n\n"
            f"Top-1 Accuracy : {top1_acc:.2f}%\n"
            f"Top-5 Accuracy : {top5_acc:.2f}%\n\n"
            f"Best species   : {short_names[best10[0]]} (F1: {f1[best10[0]]*100:.1f}%)\n"
            f"Hardest species: {short_names[worst10[0]]} (F1: {f1[worst10[0]]*100:.1f}%)\n\n"
            f"Files saved to: {SAVE_DIR}\n"
            f"  - classification_report.txt\n"
            f"  - f1_per_species.png"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ════════════════════════════════════════════════════════════
# CONFIDENCE CALIBRATION FUNCTION
# ════════════════════════════════════════════════════════════
def run_calibration():
    if not os.path.exists(TEST_DIR):
        return "❌ Test folder not found! See instructions above."
    try:
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transform)
        test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images  = images.to(device)
                outputs = model(images)
                probs   = F.softmax(outputs, dim=1)
                all_preds.extend(probs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)
        confidences = np.max(all_probs, axis=1)
        correctness = (all_preds == all_labels).astype(float)

        n_bins = 10
        bins   = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs, bin_sizes = [], [], []

        for i in range(n_bins):
            mask = (confidences > bins[i]) & (confidences <= bins[i+1])
            if mask.sum() > 0:
                bin_accs.append(correctness[mask].mean())
                bin_confs.append(confidences[mask].mean())
                bin_sizes.append(int(mask.sum()))
            else:
                bin_accs.append(0)
                bin_confs.append((bins[i]+bins[i+1])/2)
                bin_sizes.append(0)

        bin_accs  = np.array(bin_accs)
        ece = float(np.sum((np.array(bin_sizes)/len(confidences)) * np.abs(bin_accs - np.array(bin_confs))) * 100)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        bin_centers = [(bins[i]+bins[i+1])/2 for i in range(n_bins)]
        ax1.bar(bin_centers, bin_accs, width=0.09, alpha=0.75, color="#534AB7", label="Actual accuracy")
        ax1.plot([0,1],[0,1],"k--", linewidth=1.5, label="Perfect calibration")
        ax1.set_xlabel("Confidence Score"); ax1.set_ylabel("Actual Accuracy")
        ax1.set_title(f"Reliability Diagram\nECE = {ece:.2f}%")
        ax1.set_xlim(0,1); ax1.set_ylim(0,1)
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.hist(confidences, bins=20, color="#1D9E75", edgecolor="white")
        ax2.axvline(x=0.6, color="#A32D2D", linestyle="--", label="60% threshold")
        ax2.axvline(x=confidences.mean(), color="#BA7517", linestyle="--",
                    label=f"Mean ({confidences.mean()*100:.1f}%)")
        ax2.set_xlabel("Confidence Score"); ax2.set_ylabel("Number of Predictions")
        ax2.set_title("Distribution of Confidence Scores")
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.suptitle("Confidence Calibration Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "confidence_calibration.png"), dpi=150, bbox_inches="tight")
        plt.close()

        return (
            f"✅ CALIBRATION COMPLETE!\n\n"
            f"ECE (Expected Calibration Error) : {ece:.2f}%\n"
            f"Mean confidence                  : {confidences.mean()*100:.2f}%\n"
            f"Mean accuracy                    : {(all_preds==all_labels).mean()*100:.2f}%\n\n"
            f"ECE closer to 0% = better calibrated model\n\n"
            f"Saved to: {SAVE_DIR}/confidence_calibration.png"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ════════════════════════════════════════════════════════════
# BIRDNET AUDIO IDENTIFICATION
# ════════════════════════════════════════════════════════════
def predict_bird_from_audio(audio_path):
    if not BIRDNET_AVAILABLE:
        return (
            "⚠️  BirdNET not installed.\n\n"
            "Fix: pip3 install birdnetlib\n"
            "Then restart the app."
        )

    if audio_path is None:
        return "❌ Please upload an audio file (.mp3 or .wav)"

    try:
        from datetime import date
        recording = Recording(
            BIRDNET_ANALYZER,
            audio_path,
            lat=20.5937,
            lon=78.9629,
            date=date.today(),
            min_conf=0.01,
        )
        recording.analyze()
        detections = recording.detections

        if not detections:
            return (
                "🔇 No bird detected in this recording.\n\n"
                "Tips for better results:\n"
                "  • Use a quiet recording with minimal background noise\n"
                "  • Recording should be at least 3 seconds long\n"
                "  • Try lowering min_conf threshold\n"
                "  • Download test audio from xeno-canto.org"
            )

        # Sort by confidence descending
        detections = sorted(
            detections, key=lambda x: x['confidence'], reverse=True
        )

        top = detections[0]
        top_name      = top['common_name']
        top_sci       = top['scientific_name']
        top_conf      = top['confidence'] * 100

        if top_conf < 40:
            conf_msg = f"⚠️  LOW CONFIDENCE ({top_conf:.1f}%)"
        elif top_conf < 70:
            conf_msg = f"🟡  MODERATE CONFIDENCE ({top_conf:.1f}%)"
        else:
            conf_msg = f"✅  HIGH CONFIDENCE ({top_conf:.1f}%)"

        # Try to match BirdNET name to our HABITAT_MAP
        habitat   = "Location data not available"
        migration = "Migration data not available"
        similar   = None

        for folder_name in HABITAT_MAP:
            species_key = display_name(folder_name).lower()
            if species_key in top_name.lower() or top_name.lower() in species_key:
                habitat   = HABITAT_MAP[folder_name]
                migration = MIGRATION_MAP.get(folder_name, "Migration data not available")
                similar   = SIMILAR_SPECIES.get(folder_name)
                break

        output = f"""
🎵  IDENTIFIED FROM CALL : {top_name}
🔬  Scientific name      : {top_sci}
{conf_msg}

📍  FOUND IN   : {habitat}

✈️   MIGRATION  : {migration}

🤖  Powered by : BirdNET (Cornell Lab of Ornithology)
"""
        if similar:
            output += (
                f"\n⚠️  SOUNDS SIMILAR TO : {similar[0]}\n"
                f"    How to tell apart  : {similar[1]}\n"
            )

        if len(detections) > 1:
            output += f"\n{'─'*55}\nALL DETECTIONS IN RECORDING:\n"
            for i, det in enumerate(detections[:8]):
                bar = "█" * int(det['confidence'] * 20)
                output += (
                    f"\n#{i+1}  {det['common_name']}"
                    f"  ({det['confidence']*100:.1f}%)\n"
                    f"    {bar}\n"
                )

        return output

    except Exception as e:
        return (
            f"❌ Error: {str(e)}\n\n"
            "Make sure the file is a valid .mp3 or .wav\n"
            "Minimum 3 seconds of audio recommended"
        )


# ════════════════════════════════════════════════════════════
# GRADIO APP — Native Gradio + DotField Background
# ════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

body {
    font-family: 'Inter', sans-serif !important;
    background: #120F17 !important;
    color: #e6edf3 !important;
}
.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: transparent !important;
    color: #e6edf3 !important;
    position: relative;
    z-index: 5;
}
footer { display: none !important; }

/* DotField canvas */
#dotfield-bg {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0; pointer-events: none;
}

/* Hero */
.hero-wrap {
    text-align: center;
    padding: 40px 20px 28px;
    position: relative; z-index: 5;
}
.hero-wrap h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #a855f7, #c084fc, #e879f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px !important;
}
.hero-wrap p {
    color: #9ca3af !important;
    font-size: 0.92rem !important;
    margin: 0 !important;
}
.pill-row {
    display: flex; justify-content: center; gap: 10px;
    flex-wrap: wrap; margin-top: 18px;
}
.pill-item {
    background: rgba(168,85,247,0.08);
    border: 1px solid rgba(168,85,247,0.2);
    border-radius: 50px; padding: 5px 14px;
    font-size: 0.75rem; color: #c4b5fd;
}
.pill-item strong { color: #a855f7; }

/* Tabs */
.tabs > .tab-nav {
    background: rgba(168,85,247,0.06) !important;
    border: 1px solid rgba(168,85,247,0.15) !important;
    border-bottom: none !important;
    border-radius: 14px 14px 0 0 !important;
    padding: 6px 6px 0 !important;
    backdrop-filter: blur(10px);
}
.tabs > .tab-nav button {
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 22px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s !important;
    font-family: 'Inter', sans-serif !important;
}
.tabs > .tab-nav button:hover {
    color: #c084fc !important;
    background: rgba(168,85,247,0.08) !important;
}
.tabs > .tab-nav button.selected {
    color: #c084fc !important;
    background: rgba(168,85,247,0.13) !important;
    border-bottom: 2px solid #a855f7 !important;
}

/* Tab panels */
.tabitem {
    background: rgba(18,15,23,0.7) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(168,85,247,0.12) !important;
    border-top: none !important;
    border-radius: 0 0 16px 16px !important;
    padding: 28px !important;
}

/* Inputs */
textarea, input[type=text] {
    background: rgba(18,15,23,0.85) !important;
    color: #d1d5db !important;
    border: 1px solid rgba(168,85,247,0.15) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    line-height: 1.7 !important;
}
textarea:focus, input[type=text]:focus {
    border-color: rgba(168,85,247,0.4) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(168,85,247,0.15) !important;
}

/* Blocks (image upload, etc) */
.gr-box, .gr-panel, .block {
    background: rgba(18,15,23,0.5) !important;
    border: 1px solid rgba(168,85,247,0.1) !important;
    border-radius: 12px !important;
}

/* Primary button */
button.primary {
    background: linear-gradient(135deg, #7c3aed, #a855f7, #c026d3) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(168,85,247,0.35) !important;
    font-family: 'Inter', sans-serif !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(168,85,247,0.5) !important;
}

/* Labels */
label span {
    color: #9ca3af !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Section desc */
.sec-desc {
    font-size: 0.82rem !important;
    color: #6b7280 !important;
    margin-bottom: 16px !important;
}

/* Footer */
.app-footer {
    text-align: center; margin-top: 28px; padding-top: 18px;
    border-top: 1px solid rgba(168,85,247,0.08);
    color: #374151; font-size: 0.74rem;
}
.app-footer a { color: #7c3aed; text-decoration: none; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(168,85,247,0.25); border-radius: 3px; }
"""

DOTFIELD_JS = """
<canvas id="dotfield-bg"></canvas>
<script>
(function(){
  var cv=document.getElementById('dotfield-bg');
  if(!cv){setTimeout(arguments.callee,200);return;}
  var ctx=cv.getContext('2d');
  var R=1.5,SP=14,CR=600,CF=0.1,BS=67,GR=160;
  var mx=-9999,my=-9999,dots=[];
  function resize(){
    cv.width=innerWidth;cv.height=innerHeight;dots=[];
    var cols=Math.ceil(cv.width/SP)+2,rows=Math.ceil(cv.height/SP)+2;
    for(var r=0;r<=rows;r++)for(var c=0;c<=cols;c++)dots.push({ox:c*SP,oy:r*SP});
  }
  function clr(d){
    if(d>=CR)return'rgba(180,151,207,0.07)';
    var t=d/CR,
      rv=Math.round(168+(180-168)*t),
      gv=Math.round(85+(151-85)*t),
      bv=Math.round(247+(207-247)*t),
      glow=Math.max(0,1-d/GR),
      a=Math.min(1,(0.35+(0.25-0.35)*t)+glow*0.65);
    return'rgba('+rv+','+gv+','+bv+','+a.toFixed(3)+')';
  }
  function draw(){
    ctx.clearRect(0,0,cv.width,cv.height);
    for(var i=0;i<dots.length;i++){
      var dot=dots[i],dx=dot.ox-mx,dy=dot.oy-my,
        dist=Math.sqrt(dx*dx+dy*dy),x=dot.ox,y=dot.oy;
      if(dist<CR&&dist>0.01){
        var ratio=1-dist/CR,disp=ratio*ratio*BS*CF,
          ang=Math.atan2(dy,dx);
        x=dot.ox+Math.cos(ang)*disp;
        y=dot.oy+Math.sin(ang)*disp;
      }
      ctx.beginPath();ctx.arc(x,y,R,0,Math.PI*2);
      ctx.fillStyle=clr(dist);ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  addEventListener('mousemove',function(e){mx=e.clientX;my=e.clientY;});
  addEventListener('mouseleave',function(){mx=-9999;my=-9999;});
  addEventListener('resize',resize);
  resize();draw();
})();
</script>
"""

BIRD_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#120F17",
    body_background_fill_dark="#120F17",
    block_background_fill="rgba(18,15,23,0.5)",
    block_background_fill_dark="rgba(18,15,23,0.5)",
    block_border_color="rgba(168,85,247,0.1)",
    block_border_color_dark="rgba(168,85,247,0.1)",
    block_label_text_color="#9ca3af",
    block_label_text_color_dark="#9ca3af",
    input_background_fill="rgba(18,15,23,0.85)",
    input_background_fill_dark="rgba(18,15,23,0.85)",
    button_primary_background_fill="linear-gradient(135deg,#7c3aed,#a855f7)",
    button_primary_background_fill_dark="linear-gradient(135deg,#7c3aed,#a855f7)",
    button_primary_text_color="#ffffff",
)

with gr.Blocks(title="🐦 Bird Species Identifier") as app:

    # ── DotField background ──
    gr.HTML(DOTFIELD_JS)

    # ── Hero ──
    gr.HTML(f"""
    <div class="hero-wrap">
      <h1>🐦 Bird Species Identifier</h1>
      <p>AI-powered recognition · EfficientNetV2-S · {NUM_SPECIES} species</p>
      <div class="pill-row">
        <span class="pill-item">🧠 Model <strong>EfficientNetV2-S</strong></span>
        <span class="pill-item">🦜 Species <strong>{NUM_SPECIES}</strong></span>
        <span class="pill-item">📷 Input <strong>380×380 px</strong></span>
        <span class="pill-item">⚡ Device <strong>{DEVICE_LABEL}</strong></span>
      </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Identify Bird ──
        with gr.Tab("🔍  Identify Bird"):
            gr.HTML('<p class="sec-desc">Upload a bird photo — get species, habitat, migration info and top 5 predictions</p>')
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Bird Photo", type="numpy", height=340)
                    identify_btn = gr.Button("🔍  Identify Species", variant="primary")
                with gr.Column(scale=1):
                    txt_output = gr.Textbox(label="Identification Results", lines=22,
                                            placeholder="Results will appear here…")
            identify_btn.click(fn=predict_bird, inputs=img_input, outputs=txt_output)

        # ── Tab 2: Grad-CAM ──
        with gr.Tab("🔥  Grad-CAM"):
            gr.HTML("""<p class="sec-desc">See what part of the image the model focuses on ·
              <span style="color:#f97316;">Red/Yellow</span> = high attention ·
              <span style="color:#60a5fa;">Blue</span> = low attention</p>""")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    cam_input = gr.Image(label="Upload Bird Photo", type="numpy", height=340)
                    cam_btn = gr.Button("🔥  Generate Grad-CAM Heatmap", variant="primary")
                with gr.Column(scale=1):
                    cam_output = gr.Image(label="Grad-CAM Heatmap", height=300)
                    cam_text = gr.Textbox(label="Analysis Result", lines=5,
                                          placeholder="Heatmap result will appear here…")
            cam_btn.click(fn=generate_gradcam, inputs=cam_input, outputs=[cam_output, cam_text])

        # ── Tab 3: BirdNET Audio ID ───────────────────────────
        with gr.Tab("🎵 Audio ID"):
            gr.Markdown("### Identify a bird from its call using BirdNET")
            gr.Markdown(
                "Upload a **.mp3 or .wav** bird call recording — "
                "BirdNET by Cornell Lab will identify the species.\n\n"
                "**Best results:** quiet recording, 3+ seconds, "
                "single bird calling clearly.\n\n"
                "**Get test audio:** [xeno-canto.org](https://xeno-canto.org)"
            )
            audio_input  = gr.Audio(
                label="Upload Bird Call (.mp3 / .wav)",
                type="filepath"
            )
            audio_result = gr.Textbox(
                label="BirdNET Identification Result",
                lines=25
            )
            audio_btn = gr.Button(
                "🎵 Identify from Audio",
                variant="primary"
            )
            audio_btn.click(
                fn=predict_bird_from_audio,
                inputs=audio_input,
                outputs=audio_result
            )

    # ── Footer ──
    gr.HTML("""
    <div class="app-footer">
      🐦 Bird Species Identifier · EfficientNetV2-S · CUB-200-2011 ·
      Built with <a href="https://gradio.app" target="_blank">Gradio</a>
    </div>
    """)

print("\n✅ Starting Bird Identifier App...")
print("   Open in browser: http://127.0.0.1:7860\n")
app.launch(theme=BIRD_THEME, css=CUSTOM_CSS, share=False)
