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
import os
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
# ✅ SET YOUR PATHS HERE
# ============================================================
MODEL_PATH = "/Users/swayam/Desktop/birdsproject/best_bird_model.pth"
TEST_DIR   = "/Users/swayam/Desktop/birdsproject/birds_split/test"
SAVE_DIR   = "/Users/swayam/Desktop/birdsproject/evaluation"
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Running on: {device}")

# ── Load model ───────────────────────────────────────────────
print("Loading model...")
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

# ── Habitat map ──────────────────────────────────────────────
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
    "028.Brown_headed_Cowbird":            "USA, Canada — widespread across North America",
    "029.Pigeon_Guillemot":                "USA (Alaska, California), Canada, Russia — North Pacific coastal cliffs",
    "030.California_Gull":                 "USA (California, Great Basin states), Canada",
    "031.Glaucous_winged_Gull":            "USA (Alaska, Washington, Oregon), Canada (British Columbia)",
    "032.Heermann_Gull":                   "USA (California coast), Mexico (Baja California)",
    "033.Herring_Gull":                    "USA, Canada, UK, northern Europe — widespread across North Atlantic",
    "034.Ivory_Gull":                      "Canada (Arctic), Russia (Arctic), Norway (Svalbard)",
    "035.Ring_billed_Gull":                "USA, Canada — one of the most common gulls across North America",
    "036.Slaty_backed_Gull":               "Russia (eastern Siberia), Japan, South Korea, China",
    "037.Western_Gull":                    "USA (California, Oregon, Washington) — Pacific coast only",
    "038.Anna_Hummingbird":                "USA (California, Arizona, Oregon), Mexico (Baja) — Pacific coast year-round",
    "039.Ruby_throated_Hummingbird":       "USA (eastern states), Canada (Ontario), Mexico, Central America",
    "040.Rufous_Hummingbird":              "USA (western states), Canada (British Columbia), Mexico",
    "041.Green_Violetear":                 "Mexico, Guatemala, Costa Rica, Venezuela, Colombia — mountain forests",
    "042.Long_tailed_Jaeger":              "Canada (Arctic), USA (Alaska), winters in South Atlantic",
    "043.Pomarine_Jaeger":                 "USA (Alaska), Canada (Arctic tundra)",
    "044.Blue_Jay":                        "USA (eastern & central states), Canada (Ontario, Quebec)",
    "045.Florida_Jay":                     "USA (Florida only) — found exclusively in Florida scrub habitat",
    "046.Green_Jay":                       "USA (southern Texas only), Mexico, Central America",
    "047.Dark_eyed_Junco":                 "USA, Canada — one of the most common birds in North America",
    "048.Tropical_Kingbird":               "USA (southern Arizona, Texas), Mexico, Central America, South America",
    "049.Gray_Kingbird":                   "USA (Florida), Caribbean islands — Cuba, Jamaica, Puerto Rico",
    "050.Belted_Kingfisher":               "USA, Canada — found near rivers and lakes across all of North America",
    "051.Green_Kingfisher":                "USA (southern Texas, Arizona), Mexico, Central America, South America",
    "052.Pied_Kingfisher":                 "Africa (sub-Saharan), India, Southeast Asia",
    "053.Ringed_Kingfisher":               "USA (southern Texas), Mexico, Central America, South America",
    "054.White_breasted_Kingfisher":       "India, Sri Lanka, Southeast Asia (Thailand, Vietnam, Philippines)",
    "055.Red_legged_Kittiwake":            "USA (Alaska — Pribilof Islands) — very restricted range",
    "056.Horned_Lark":                     "USA, Canada — most widespread lark in North America",
    "057.Pacific_Loon":                    "USA (Alaska), Canada (Arctic) — winters along Pacific coast",
    "058.Mallard":                         "USA, Canada, UK, Europe, Asia — most widespread duck in the world",
    "059.Western_Meadowlark":              "USA (western & central states), Canada (prairies), Mexico",
    "060.Hooded_Merganser":                "USA, Canada — breeds in forested lakes from Alaska to Florida",
    "061.Red_breasted_Merganser":          "USA, Canada, UK, northern Europe — circumpolar breeding range",
    "062.Mockingbird":                     "USA (all states), Canada (southern Ontario), Mexico, Caribbean",
    "063.Nighthawk":                       "USA, Canada (breeds), South America (winters)",
    "064.Clark_Nutcracker":                "USA (western mountain states — Colorado, Wyoming, Montana, California)",
    "065.White_breasted_Nuthatch":         "USA, Canada — common in deciduous forests across North America",
    "066.Baltimore_Oriole":                "USA (eastern states), Canada (Ontario), Mexico, Central America (winters)",
    "067.Hooded_Oriole":                   "USA (California, Arizona, Texas), Mexico",
    "068.Orchard_Oriole":                  "USA (eastern & central states), Canada, Mexico, Central America (winters)",
    "069.Scott_Oriole":                    "USA (Texas, Arizona, California), Mexico",
    "070.Ovenbird":                        "USA (eastern states), Canada — breeds from Georgia to Manitoba",
    "071.Brown_Pelican":                   "USA (coastal states — California, Florida, Texas), Mexico, Caribbean",
    "072.White_Pelican":                   "USA, Canada — breeds on lakes in Great Plains, winters on Gulf Coast",
    "073.Western_Wood_Pewee":              "USA (western states), Canada (British Columbia), Mexico, South America",
    "074.Sayornis":                        "USA, Canada, Mexico — widespread flycatcher across North America",
    "075.American_Pipit":                  "USA (Alaska, mountain states), Canada (Arctic)",
    "076.Whip_poor_Will":                  "USA (eastern states), Canada (Ontario, Quebec), Mexico, Central America",
    "077.Horned_Puffin":                   "USA (Alaska), Russia, Canada — North Pacific",
    "078.Common_Raven":                    "USA (western states, Alaska), Canada, UK, northern Europe, Asia",
    "079.White_necked_Raven":              "USA (Texas, Arizona), Mexico — Chihuahuan and Sonoran Desert",
    "080.American_Redstart":               "USA (eastern states), Canada — breeds from Georgia to Nova Scotia",
    "081.Geococcyx":                       "USA (Texas, New Mexico, Arizona, California), Mexico",
    "082.Loggerhead_Shrike":               "USA, Canada (southern Ontario) — widespread but declining",
    "083.Great_Grey_Shrike":               "Canada (Arctic), Russia, northern Europe (Scandinavia, Finland)",
    "084.Baird_Sparrow":                   "USA (Montana, North Dakota), Canada (Manitoba, Saskatchewan)",
    "085.Black_throated_Sparrow":          "USA (Texas, New Mexico, Arizona, Nevada, California), Mexico",
    "086.Brewer_Sparrow":                  "USA (western states), Canada (British Columbia, Alberta), Mexico",
    "087.Chipping_Sparrow":                "USA, Canada — one of the most common sparrows across North America",
    "088.Clay_colored_Sparrow":            "USA (Great Plains states), Canada (prairies)",
    "089.House_Sparrow":                   "USA, Canada, UK, Europe, Australia, India — introduced worldwide",
    "090.Field_Sparrow":                   "USA (eastern & central states) — common from Texas to New England",
    "091.Fox_Sparrow":                     "USA (western states, Alaska), Canada",
    "092.Grasshopper_Sparrow":             "USA (central & eastern states), Canada (Ontario), Mexico",
    "093.Harris_Sparrow":                  "USA (Great Plains — Kansas, Oklahoma, Texas in winter), Canada",
    "094.Henslow_Sparrow":                 "USA (Midwest — Ohio, Indiana, Illinois, New York) — rare and declining",
    "095.Le_Conte_Sparrow":                "USA (Great Plains, Southeast), Canada (prairies)",
    "096.Lincoln_Sparrow":                 "USA, Canada (breeds), Mexico, Central America (winters)",
    "097.Nelson_Sharp_tailed_Sparrow":     "USA (Atlantic coast — Maine to Virginia), Canada (Maritime provinces)",
    "098.Savannah_Sparrow":                "USA, Canada — one of the most widespread sparrows in North America",
    "099.Seaside_Sparrow":                 "USA (Atlantic & Gulf coast — Virginia to Texas)",
    "100.Song_Sparrow":                    "USA, Canada — extremely common and widespread across North America",
    "101.Tree_Sparrow":                    "Canada (Arctic, boreal), USA (northern states in winter)",
    "102.Vesper_Sparrow":                  "USA (western & central states), Canada (prairies), Mexico (winters)",
    "103.White_crowned_Sparrow":           "USA, Canada — widespread migrant, breeds in Canada and mountain west",
    "104.White_throated_Sparrow":          "USA (eastern states in winter), Canada (breeds)",
    "105.Cape_Glossy_Starling":            "Kenya, Tanzania, Uganda, Ethiopia, Somalia — East Africa savanna",
    "106.Bank_Swallow":                    "USA, Canada (breeds), South America (winters)",
    "107.Barn_Swallow":                    "USA, Canada (breeds), South America, Africa (winters)",
    "108.Cliff_Swallow":                   "USA, Canada (breeds), South America (winters)",
    "109.Tree_Swallow":                    "USA, Canada (breeds), Mexico, Central America (winters)",
    "110.Scarlet_Tanager":                 "USA (eastern states), Canada (Ontario), South America (winters)",
    "111.Summer_Tanager":                  "USA (southern states — Georgia, Texas, Carolina), Mexico, South America",
    "112.Artic_Tern":                      "Canada, USA (Alaska), UK, Norway — longest migration on Earth",
    "113.Black_Tern":                      "USA (Midwest lakes), Canada, Europe",
    "114.Caspian_Tern":                    "USA (Great Lakes, Gulf Coast), Canada, Europe, Africa, Australia",
    "115.Common_Tern":                     "USA, Canada, UK, Europe — widespread across North Atlantic coasts",
    "116.Elegant_Tern":                    "USA (California coast), Mexico (Baja California)",
    "117.Forsters_Tern":                   "USA, Canada — breeds in Great Plains marshes, winters on both coasts",
    "118.Least_Tern":                      "USA (Atlantic & Gulf coast, river valleys), Mexico, Caribbean",
    "119.Green_tailed_Towhee":             "USA (western mountain states — Colorado, Utah, California), Mexico",
    "120.Brown_Thrasher":                  "USA (eastern & central states), Canada (Ontario)",
    "121.Sage_Thrasher":                   "USA (Great Basin — Nevada, Idaho, Wyoming, Oregon), Mexico (winters)",
    "122.Black_capped_Vireo":              "USA (Texas, Oklahoma only), Mexico (winters) — endangered species",
    "123.Blue_headed_Vireo":               "USA (eastern states), Canada (Ontario, Quebec), Central America",
    "124.Philadelphia_Vireo":              "USA (northeastern states), Canada (Ontario to Alberta), Central America",
    "125.Red_eyed_Vireo":                  "USA, Canada — one of the most common breeding birds in eastern North America",
    "126.Warbling_Vireo":                  "USA, Canada (breeds), Mexico, Central America (winters)",
    "127.White_eyed_Vireo":                "USA (southeastern & eastern states), Mexico, Central America (winters)",
    "128.Yellow_throated_Vireo":           "USA (eastern states), Canada (Ontario), South America (winters)",
    "129.Bay_breasted_Warbler":            "Canada (boreal — Ontario to Newfoundland), USA (eastern states)",
    "130.Black_and_white_Warbler":         "USA (eastern states), Canada",
    "131.Black_throated_Blue_Warbler":     "USA (eastern mountain states — Appalachians), Canada, Caribbean",
    "132.Blue_winged_Warbler":             "USA (eastern states — Ohio, Indiana, New York), Mexico, Central America",
    "133.Canada_Warbler":                  "USA (northeastern states), Canada (Ontario to Nova Scotia), South America",
    "134.Cape_May_Warbler":                "Canada (boreal — Ontario, Quebec, Manitoba), USA (eastern states)",
    "135.Cerulean_Warbler":                "USA (Appalachians, Midwest), Canada (Ontario), South America (winters)",
    "136.Chestnut_sided_Warbler":          "USA (northeastern states), Canada (Ontario, Quebec), Central America",
    "137.Golden_winged_Warbler":           "USA (Appalachians, Great Lakes region), Canada, Central America",
    "138.Hooded_Warbler":                  "USA (eastern states — Georgia, Virginia, Tennessee), Mexico",
    "139.Kentucky_Warbler":                "USA (southeastern states — Kentucky, Tennessee, Arkansas), Central America",
    "140.Magnolia_Warbler":                "Canada (boreal — Ontario to Nova Scotia), USA (eastern states)",
    "141.Mourning_Warbler":                "Canada (Ontario to Manitoba), USA (northeastern states), Central America",
    "142.Myrtle_Warbler":                  "USA, Canada — one of the most common warblers, breeds across Canada",
    "143.Nashville_Warbler":               "USA (northeastern & western states), Canada, Mexico, Central America",
    "144.Orange_crowned_Warbler":          "USA (western states), Canada (British Columbia, Alberta), Mexico",
    "145.Palm_Warbler":                    "Canada (boreal bogs — Ontario to Newfoundland), USA (Florida in winter)",
    "146.Pine_Warbler":                    "USA (southeastern states — Florida, Georgia, Texas, Carolina)",
    "147.Prairie_Warbler":                 "USA (eastern states — Michigan, Ohio, New Jersey, Florida), Caribbean",
    "148.Prothonotary_Warbler":            "USA (southeastern states — Louisiana, Mississippi, Tennessee, Virginia)",
    "149.Swainson_Warbler":                "USA (southeastern states — Arkansas, Louisiana, South Carolina)",
    "150.Tennessee_Warbler":               "Canada (boreal — Ontario to Alberta), USA (eastern states)",
    "151.Wilson_Warbler":                  "USA (western states, Alaska), Canada (widespread), Mexico",
    "152.Worm_eating_Warbler":             "USA (eastern states — Pennsylvania, Maryland, Ohio, Georgia)",
    "153.Yellow_Warbler":                  "USA, Canada — most widespread warbler in North America",
    "154.Northern_Waterthrush":            "USA (northeastern states), Canada (widespread boreal), Central America",
    "155.Louisiana_Waterthrush":           "USA (eastern states — Pennsylvania to Georgia, west to Kansas)",
    "156.Bohemian_Waxwing":                "Canada (boreal — British Columbia to Manitoba), USA (northern states)",
    "157.Cedar_Waxwing":                   "USA, Canada — very common, found across all of North America year-round",
    "158.American_Three_toed_Woodpecker":  "USA (Alaska, mountain west), Canada (boreal)",
    "159.Pileated_Woodpecker":             "USA (eastern states, Pacific Northwest), Canada",
    "160.Red_bellied_Woodpecker":          "USA (eastern states) — very common from Florida to New York to Texas",
    "161.Red_cockaded_Woodpecker":         "USA (southeastern states — North Carolina to Texas) — endangered",
    "162.Red_headed_Woodpecker":           "USA (eastern & central states), Canada (Ontario, Manitoba)",
    "163.Downy_Woodpecker":                "USA, Canada — smallest and most common woodpecker in North America",
    "164.Bewick_Wren":                     "USA (western & southern states — California, Texas, Oregon), Mexico",
    "165.Cactus_Wren":                     "USA (Arizona, California, New Mexico, Texas), Mexico — Sonoran Desert",
    "166.Carolina_Wren":                   "USA (eastern & southeastern states)",
    "167.House_Wren":                      "USA, Canada (breeds), South America (winters)",
    "168.Marsh_Wren":                      "USA, Canada — freshwater marshes across North America",
    "169.Rock_Wren":                       "USA (western states — Colorado, Utah, California, Arizona), Mexico",
    "170.Winter_Wren":                     "USA (Pacific Northwest, Appalachians), Canada, UK, Europe, Asia",
    "171.Common_Yellowthroat":             "USA, Canada — one of the most widespread warblers in North America",
    "172.Wilson_Snipe":                    "USA, Canada (breeds), Central America, South America (winters)",
    "173.American_Woodcock":               "USA (eastern states), Canada (Ontario, Quebec)",
    "174.Great_Crested_Flycatcher":        "USA (eastern states), Canada (Ontario), Mexico, Central America",
    "175.Least_Flycatcher":                "USA (northeastern states), Canada (widespread), Mexico, Central America",
    "176.Olive_sided_Flycatcher":          "USA (western states, Alaska), Canada (boreal), South America (winters)",
    "177.Acadian_Flycatcher":              "USA (eastern states — Ohio, Virginia, Georgia, Texas), Central America",
    "178.Yellow_bellied_Flycatcher":       "Canada (boreal — Ontario to Newfoundland), USA (northeastern states)",
    "179.Pacific_slope_Flycatcher":        "USA (California, Oregon, Washington), Canada (British Columbia), Mexico",
    "180.Black_billed_Cuckoo":             "USA (eastern & central states), Canada (Ontario, Quebec), South America",
    "181.Yellow_billed_Cuckoo":            "USA (widespread), Canada (Ontario), Mexico, Central America",
    "182.American_Crow":                   "USA, Canada — one of the most intelligent and common birds in North America",
    "183.Fish_Crow":                       "USA (eastern & southeastern coastal states — New York to Texas)",
    "184.Brown_Creeper":                   "USA, Canada — found in mature forests across North America year-round",
    "185.Rock_Pigeon":                     "USA, Canada, UK, Europe, India, worldwide — introduced everywhere",
    "186.White_crowned_Pigeon":            "USA (Florida Keys only), Caribbean — Cuba, Bahamas, Haiti, Jamaica",
    "187.Band_tailed_Pigeon":              "USA (California, Oregon, Washington, Arizona), Canada, Mexico",
    "188.Eared_Grebe":                     "USA (western states), Canada (prairies), Mexico, Spain, Africa",
    "189.Horned_Grebe":                    "USA, Canada (breeds), UK, northern Europe",
    "190.Red_necked_Grebe":                "USA (Pacific & Atlantic coasts in winter), Canada (breeds)",
    "191.Pied_billed_Grebe":               "USA, Canada — most common grebe in North America",
    "192.Western_Grebe":                   "USA (western states), Canada (British Columbia, prairies)",
    "193.Gadwall":                         "USA, Canada (breeds), UK, Europe, Asia",
    "194.Canvasback":                      "USA (Great Plains, western states), Canada (prairies)",
    "195.Redhead":                         "USA (Great Plains, Great Lakes), Canada (prairies)",
    "196.Ring_necked_Duck":                "USA, Canada (boreal) — very common diving duck across North America",
    "197.Lesser_Scaup":                    "USA, Canada (breeds in Alaska and prairies)",
    "198.Surf_Scoter":                     "USA (Pacific & Atlantic coasts), Canada (breeds in Alaska and boreal)",
    "199.White_winged_Scoter":             "USA (both coasts in winter), Canada (breeds in boreal)",
    "200.Long_tailed_Duck":                "USA (Great Lakes, both coasts), Canada (Arctic breeds)",
}

SIMILAR_SPECIES = {
    "Black footed Albatross":    ("Laysan Albatross",         "check the face — Black-footed has a dark face, Laysan has a white face"),
    "Laysan Albatross":          ("Black footed Albatross",   "check the face — Laysan has a white face, Black-footed has a dark face"),
    "Herring Gull":              ("Ring billed Gull",         "check the bill — Herring Gull has a red spot, Ring-billed has a black ring"),
    "Ring billed Gull":          ("Herring Gull",             "check the bill — Ring-billed has a black ring, Herring has a red spot"),
    "Ruby throated Hummingbird": ("Rufous Hummingbird",       "check the back — Ruby-throated is green-backed, Rufous has orange-brown back"),
    "Rufous Hummingbird":        ("Ruby throated Hummingbird","check the back — Rufous is orange-brown, Ruby-throated is metallic green"),
    "Barn Swallow":              ("Cliff Swallow",            "check the tail — Barn Swallow has a deep forked tail, Cliff has a square tail"),
    "Cliff Swallow":             ("Barn Swallow",             "check the tail — Cliff has a square tail, Barn Swallow has a deep fork"),
    "Downy Woodpecker":          ("Pileated Woodpecker",      "check the size — Downy is sparrow-sized, Pileated is crow-sized"),
    "Baltimore Oriole":          ("Orchard Oriole",           "check the color — Baltimore is bright orange, Orchard is darker chestnut"),
    "American Crow":             ("Common Raven",             "check the size & tail — Crow is smaller with fan tail, Raven has wedge tail"),
    "Scarlet Tanager":           ("Summer Tanager",           "check the wings — Scarlet has black wings, Summer Tanager is all red"),
    "Cedar Waxwing":             ("Bohemian Waxwing",         "check the belly — Cedar has a yellow belly, Bohemian has rusty undertail"),
    "Indigo Bunting":            ("Lazuli Bunting",           "check the breast — Indigo is all blue, Lazuli has a rusty-orange breast"),
    "Song Sparrow":              ("Savannah Sparrow",         "check the breast — Song Sparrow has a central spot, Savannah has streaks"),
    "Blue Jay":                  ("Florida Jay",              "check the crest — Blue Jay has a crest, Florida Jay has no crest"),
}

MIGRATION_MAP = {
    "Indigo Bunting":                "Breeds eastern & central USA (May-Aug) → winters in Mexico, Cuba & Central America (Sep-Apr). Navigates by stars at night",
    "Barn Swallow":                  "Breeds USA & Canada (Apr-Aug) → migrates to Argentina & Brazil (Sep-Mar)",
    "Ruby throated Hummingbird":     "Breeds eastern USA & Canada (May-Aug) → crosses Gulf of Mexico to winter in Mexico & Central America",
    "Rufous Hummingbird":            "Breeds Pacific Northwest & Alaska (Apr-Jul) → winters in Mexico. Longest hummingbird migration — 3,900 miles",
    "Baltimore Oriole":              "Breeds eastern USA & Canada (May-Aug) → winters in Central & South America (Sep-Apr)",
    "Scarlet Tanager":               "Breeds eastern USA & Canada → winters in Colombia, Ecuador & Peru (Sep-Apr)",
    "Bobolink":                      "Breeds USA & Canada prairies → migrates 12,000 miles to Argentina — one of longest songbird migrations",
    "Arctic Tern":                   "Breeds Arctic Canada & Alaska → winters in Antarctic. Longest migration on Earth — 70,000 km/year",
    "Dark eyed Junco":               "Breeds Canada & mountain USA → winters across all of USA. Called the snowbird",
    "Yellow Warbler":                "Breeds all of USA & Canada → winters in Mexico, Central & South America",
    "Cedar Waxwing":                 "Nomadic — follows berry crops. Northern birds move south in winter",
    "White throated Sparrow":        "Breeds Canada → winters in eastern & southern USA. Common winter feeder bird",
    "White crowned Sparrow":         "Breeds Arctic Canada & Alaska → winters across southern USA & Mexico",
    "American Redstart":             "Breeds eastern USA & Canada → winters in Caribbean, Mexico & South America",
    "Common Yellowthroat":           "Northern birds migrate to Caribbean & Central America. Southern birds stay year-round",
    "Mallard":                       "Year-round in most of USA. Northern Canada birds migrate to southern USA in winter",
    "American Crow":                 "Year-round across most of North America. Northern birds may move south slightly",
    "Downy Woodpecker":              "Year-round resident across North America. Does not migrate",
    "Cardinal":                      "Year-round resident across eastern & southern USA. Does not migrate",
    "Blue Jay":                      "Mostly year-round but northern populations move south in winter in large flocks",
    "Song Sparrow":                  "Year-round across most of USA. Northern Canada birds migrate south in winter",
    "House Sparrow":                 "Year-round resident worldwide. Does not migrate — introduced species",
    "Mourning Warbler":              "Breeds Canada & northeastern USA → winters in Costa Rica, Colombia & Venezuela",
    "Magnolia Warbler":              "Breeds boreal Canada → winters in Caribbean & Central America",
    "Black and white Warbler":       "Breeds eastern USA & Canada → winters in Florida, Caribbean & South America",
    "Ovenbird":                      "Breeds eastern USA & Canada → winters in Caribbean, Mexico & Central America",
    "Nighthawk":                     "Breeds USA & Canada → migrates to South America (Bolivia, Argentina). One of longest migrations",
    "Cliff Swallow":                 "Breeds USA & Canada (Apr-Aug) → winters in Argentina (Sep-Mar)",
    "Tree Swallow":                  "Breeds USA & Canada (Apr-Aug) → winters in Florida, Gulf Coast & Central America",
    "Bank Swallow":                  "Breeds USA & Canada → migrates to Peru, Bolivia, Brazil (Sep-Apr)",
    "Horned Puffin":                 "Breeds Alaskan sea cliffs (May-Aug) → winters in open North Pacific Ocean",
    "Pacific Loon":                  "Breeds Arctic Canada & Alaska → winters along Pacific coast to California",
    "Bohemian Waxwing":              "Breeds boreal Canada & Alaska → irruptive winter visitor to northern USA. Appears in large flocks following berry crops",
}


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
        species_name = folder_name.split(".")[-1].replace("_", " ")
        habitat      = HABITAT_MAP.get(folder_name, "Location data not available")
        confidence   = prob.item() * 100
        results.append((species_name, confidence, habitat))

    top_name    = results[0][0]
    top_conf    = results[0][1]
    top_habitat = results[0][2]

    if top_conf < 60:
        conf_msg = f"⚠️  LOW CONFIDENCE ({top_conf:.1f}%) — Try a clearer photo"
    elif top_conf < 80:
        conf_msg = f"🟡  MODERATE CONFIDENCE ({top_conf:.1f}%) — Likely correct"
    else:
        conf_msg = f"✅  HIGH CONFIDENCE ({top_conf:.1f}%) — Very sure"

    similar  = SIMILAR_SPECIES.get(top_name)
    migration = MIGRATION_MAP.get(top_name, "Migration data not available for this species")

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
    for i, (name, conf, habitat) in enumerate(results):
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
            species_key = folder_name.split(".")[-1].replace("_", " ").lower()
            if species_key in top_name.lower() or top_name.lower() in species_key:
                habitat   = HABITAT_MAP[folder_name]
                migration = MIGRATION_MAP.get(
                    folder_name.split(".")[-1].replace("_", " "),
                    "Migration data not available"
                )
                similar = SIMILAR_SPECIES.get(
                    folder_name.split(".")[-1].replace("_", " ")
                )
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
    gr.HTML("""
    <div class="hero-wrap">
      <h1>🐦 Bird Species Identifier</h1>
      <p>AI-powered recognition · EfficientNetV2-S · 200 species</p>
      <div class="pill-row">
        <span class="pill-item">🧠 Model <strong>EfficientNetV2-S</strong></span>
        <span class="pill-item">🦜 Species <strong>200</strong></span>
        <span class="pill-item">📷 Input <strong>380×380 px</strong></span>
        <span class="pill-item">⚡ Device <strong>CPU / CUDA</strong></span>
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

_DEAD_CODE = """
.gradio-container { background: transparent !important; max-width: 100% !important; padding: 0 !important; }
footer, .footer, .built-with, #footer, .svelte-1gfkn6j { display: none !important; }
.gr-prose h2 { display: none !important; }

/* ── Canvas ── */
#dotfield-canvas { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; }

/* ── Offscreen hidden Gradio components ── */
#hidden-backend {
    position: absolute !important;
    left: -99999px !important;
    width: 1px !important; height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* ── Custom App Shell ── */
#bird-app {
    position: relative; z-index: 10;
    max-width: 1060px; margin: 0 auto;
    padding: 0 20px 60px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 52px 0 36px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 4.5vw, 3rem);
    font-weight: 700;
    background: linear-gradient(135deg, #a855f7 0%, #c084fc 50%, #e879f9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    line-height: 1.2; margin-bottom: 12px;
}
.hero-sub { font-size: 0.95rem; color: #9ca3af; letter-spacing: 0.02em; }
.stat-pills { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-top: 20px; }
.stat-pill {
    background: rgba(168,85,247,0.08);
    border: 1px solid rgba(168,85,247,0.22);
    border-radius: 50px; padding: 6px 16px;
    font-size: 0.77rem; color: #c4b5fd;
    backdrop-filter: blur(8px);
}
.stat-pill b { color: #a855f7; }

/* ── Tabs ── */
.tab-bar {
    display: flex; gap: 4px;
    background: rgba(168,85,247,0.06);
    border: 1px solid rgba(168,85,247,0.15);
    border-bottom: none;
    border-radius: 14px 14px 0 0;
    padding: 8px 8px 0;
    backdrop-filter: blur(12px);
}
.tab-btn {
    flex: 1; padding: 10px 20px;
    background: transparent; border: none;
    border-radius: 10px 10px 0 0;
    color: #6b7280; font-size: 0.88rem; font-weight: 500;
    cursor: pointer; transition: all 0.2s;
    font-family: 'Inter', sans-serif;
}
.tab-btn:hover { color: #c084fc; background: rgba(168,85,247,0.08); }
.tab-btn.active { color: #c084fc; background: rgba(168,85,247,0.13); border-bottom: 2px solid #a855f7; }

/* ── Tab Panel ── */
.tab-panel {
    display: none;
    background: rgba(18,15,23,0.72);
    backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
    border: 1px solid rgba(168,85,247,0.12);
    border-top: none; border-radius: 0 0 16px 16px;
    padding: 32px;
}
.tab-panel.active { display: block; }

/* ── Two-column layout ── */
.panel-row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
@media (max-width: 720px) { .panel-row { grid-template-columns: 1fr; } }

/* ── Upload Zone ── */
.upload-zone {
    border: 2px dashed rgba(168,85,247,0.35);
    border-radius: 14px;
    background: rgba(168,85,247,0.04);
    min-height: 240px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    cursor: pointer; transition: all 0.25s; position: relative;
    overflow: hidden;
}
.upload-zone:hover, .upload-zone.drag-over {
    border-color: #a855f7;
    background: rgba(168,85,247,0.1);
    box-shadow: 0 0 28px rgba(168,85,247,0.2);
}
.upload-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.upload-icon { font-size: 2.4rem; margin-bottom: 10px; }
.upload-label { font-size: 0.88rem; color: #9ca3af; text-align: center; }
.upload-label b { color: #c084fc; }
.preview-img { max-width: 100%; max-height: 220px; border-radius: 10px; object-fit: contain; display: none; }

/* ── Submit Button ── */
.submit-btn {
    width: 100%; margin-top: 14px; padding: 13px;
    background: linear-gradient(135deg, #7c3aed, #a855f7, #c026d3);
    border: none; border-radius: 10px;
    color: #fff; font-size: 0.92rem; font-weight: 600;
    cursor: pointer; transition: all 0.25s;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 4px 22px rgba(168,85,247,0.35);
    letter-spacing: 0.02em;
}
.submit-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(168,85,247,0.5); }
.submit-btn:active { transform: translateY(0); }
.submit-btn:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }

/* ── Results Card ── */
.results-card {
    background: rgba(18,15,23,0.6);
    border: 1px solid rgba(168,85,247,0.12);
    border-radius: 14px; padding: 20px;
    min-height: 240px;
}
.results-label { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px; }
.results-text { font-size: 0.84rem; color: #d1d5db; line-height: 1.8; white-space: pre-wrap; word-break: break-word; }
.results-text.placeholder { color: #4b5563; font-style: italic; }

/* ── Loading Spinner ── */
.spinner {
    display: none; width: 32px; height: 32px; margin: 60px auto;
    border: 3px solid rgba(168,85,247,0.15);
    border-top-color: #a855f7;
    border-radius: 50%; animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── CAM Output Image ── */
.cam-result-img { width: 100%; border-radius: 12px; display: none; margin-top: 12px; }

/* ── Section label ── */
.sec-label { font-size: 0.78rem; color: #9ca3af; margin-bottom: 8px; letter-spacing: 0.04em; text-transform: uppercase; }

/* ── Footer ── */
.custom-footer {
    text-align: center; margin-top: 32px;
    color: #374151; font-size: 0.76rem;
    border-top: 1px solid rgba(168,85,247,0.08); padding-top: 20px;
}
.custom-footer a { color: #7c3aed; text-decoration: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(168,85,247,0.3); border-radius: 3px; }
"""

BIRD_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#120F17",
    body_background_fill_dark="#120F17",
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    input_background_fill="rgba(18,15,23,0.8)",
    input_background_fill_dark="rgba(18,15,23,0.8)",
)

FRONTEND_HTML = """
<!-- DotField Canvas -->
<canvas id="dotfield-canvas"></canvas>

<!-- Custom App Shell -->
<div id="bird-app">

  <!-- Hero -->
  <div class="hero">
    <div class="hero-title">🐦 Bird Species Identifier</div>
    <div class="hero-sub">AI-powered recognition · EfficientNetV2-S · 200 species</div>
    <div class="stat-pills">
      <span class="stat-pill">🧠 Model <b>EfficientNetV2-S</b></span>
      <span class="stat-pill">🦜 Species <b>200</b></span>
      <span class="stat-pill">📷 Input <b>380×380px</b></span>
      <span class="stat-pill">⚡ Device <b>CPU/CUDA</b></span>
    </div>
  </div>

  <!-- Tab Bar -->
  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('identify',this)">🔍&nbsp; Identify Bird</button>
    <button class="tab-btn"       onclick="switchTab('gradcam',this)">🔥&nbsp; Grad-CAM</button>
  </div>

  <!-- Tab: Identify -->
  <div class="tab-panel active" id="tab-identify">
    <div class="panel-row">
      <div>
        <div class="sec-label">Upload Bird Photo</div>
        <div class="upload-zone" id="upload-zone-id"
             ondragover="ev.preventDefault();this.classList.add('drag-over')"
             ondragleave="this.classList.remove('drag-over')"
             ondrop="handleDrop(event,'id')">
          <input type="file" accept="image/*" id="file-id" onchange="handleFile(this,'id')">
          <div id="upload-ui-id">
            <div class="upload-icon">🖼️</div>
            <div class="upload-label">Drag &amp; drop or <b>click to browse</b><br><span style="font-size:0.72rem;color:#4b5563;">JPG · PNG · WEBP</span></div>
          </div>
          <img class="preview-img" id="preview-id" alt="preview">
        </div>
        <button class="submit-btn" id="btn-id" onclick="runIdentify()" disabled>🔍&nbsp; Identify Species</button>
      </div>
      <div>
        <div class="sec-label">Results</div>
        <div class="results-card">
          <div class="spinner" id="spinner-id"></div>
          <div class="results-text placeholder" id="result-id">Upload a photo and click Identify to see the species, habitat, migration info and top 5 predictions.</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Tab: Grad-CAM -->
  <div class="tab-panel" id="tab-gradcam">
    <div class="panel-row">
      <div>
        <div class="sec-label">Upload Bird Photo</div>
        <div class="upload-zone" id="upload-zone-cam"
             ondragover="event.preventDefault();this.classList.add('drag-over')"
             ondragleave="this.classList.remove('drag-over')"
             ondrop="handleDrop(event,'cam')">
          <input type="file" accept="image/*" id="file-cam" onchange="handleFile(this,'cam')">
          <div id="upload-ui-cam">
            <div class="upload-icon">🖼️</div>
            <div class="upload-label">Drag &amp; drop or <b>click to browse</b><br><span style="font-size:0.72rem;color:#4b5563;">JPG · PNG · WEBP</span></div>
          </div>
          <img class="preview-img" id="preview-cam" alt="preview">
        </div>
        <button class="submit-btn" id="btn-cam" onclick="runGradCam()" disabled>🔥&nbsp; Generate Grad-CAM</button>
        <div class="sec-label" style="margin-top:18px;font-size:0.72rem;">
          <span style="color:#f97316;">■</span> Red/Yellow = high attention &nbsp;
          <span style="color:#60a5fa;">■</span> Blue = low attention
        </div>
      </div>
      <div>
        <div class="sec-label">Heatmap</div>
        <div class="results-card">
          <div class="spinner" id="spinner-cam"></div>
          <img class="cam-result-img" id="cam-img" alt="Grad-CAM heatmap">
          <div class="results-text placeholder" id="result-cam">Upload a photo and click Generate to see where the model focuses.</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <div class="custom-footer">
    🐦 Bird Species Identifier &nbsp;·&nbsp; EfficientNetV2-S &nbsp;·&nbsp; CUB-200-2011 &nbsp;·&nbsp;
    Built with <a href="https://gradio.app" target="_blank">Gradio</a>
  </div>
</div>

<script>
/* ══════════════════════════════════════════════════
   DOTFIELD BACKGROUND
══════════════════════════════════════════════════ */
(function(){
  var cv = document.getElementById('dotfield-canvas');
  if(!cv){ setTimeout(arguments.callee,300); return; }
  var ctx = cv.getContext('2d');
  var R=1.5, SP=14, CR=600, CF=0.1, BS=67, GR=160;
  var mx=-9999, my=-9999, dots=[];
  function resize(){
    cv.width=innerWidth; cv.height=innerHeight; dots=[];
    var cols=Math.ceil(cv.width/SP)+2, rows=Math.ceil(cv.height/SP)+2;
    for(var r=0;r<=rows;r++) for(var c=0;c<=cols;c++) dots.push({ox:c*SP,oy:r*SP});
  }
  function color(d){
    if(d>=CR) return 'rgba(180,151,207,0.07)';
    var t=d/CR;
    var rv=Math.round(168+(180-168)*t), gv=Math.round(85+(151-85)*t), bv=Math.round(247+(207-247)*t);
    var glow=Math.max(0,1-d/GR);
    var a=Math.min(1,(0.35+(0.25-0.35)*t)+glow*0.65);
    return 'rgba('+rv+','+gv+','+bv+','+a.toFixed(3)+')';
  }
  function draw(){
    ctx.clearRect(0,0,cv.width,cv.height);
    for(var i=0;i<dots.length;i++){
      var d=dots[i], dx=d.ox-mx, dy=d.oy-my, dist=Math.sqrt(dx*dx+dy*dy);
      var x=d.ox, y=d.oy;
      if(dist<CR&&dist>0.01){
        var ratio=1-dist/CR, disp=ratio*ratio*BS*CF, ang=Math.atan2(dy,dx);
        x=d.ox+Math.cos(ang)*disp; y=d.oy+Math.sin(ang)*disp;
      }
      ctx.beginPath(); ctx.arc(x,y,R,0,Math.PI*2);
      ctx.fillStyle=color(dist); ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  addEventListener('mousemove',function(e){mx=e.clientX;my=e.clientY;});
  addEventListener('mouseleave',function(){mx=-9999;my=-9999;});
  addEventListener('resize',resize);
  resize(); draw();
  // Fix Gradio footer
  var footer=document.querySelector('footer,.built-with');
  if(footer) footer.style.display='none';
})();

/* ══════════════════════════════════════════════════
   TAB SWITCHING
══════════════════════════════════════════════════ */
function switchTab(name, btn){
  document.querySelectorAll('.tab-panel').forEach(function(p){p.classList.remove('active');});
  document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active');});
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}

/* ══════════════════════════════════════════════════
   FILE HANDLING
══════════════════════════════════════════════════ */
var fileStore = { id: null, cam: null };

function handleFile(input, key){
  var file = input.files[0];
  if(!file) return;
  fileStore[key] = file;
  showPreview(file, key);
  document.getElementById('btn-'+key).disabled = false;
}

function handleDrop(ev, key){
  ev.preventDefault();
  document.getElementById('upload-zone-'+key).classList.remove('drag-over');
  var file = ev.dataTransfer.files[0];
  if(!file || !file.type.startsWith('image/')) return;
  fileStore[key] = file;
  showPreview(file, key);
  document.getElementById('btn-'+key).disabled = false;
}

function showPreview(file, key){
  var reader = new FileReader();
  reader.onload = function(e){
    var img = document.getElementById('preview-'+key);
    img.src = e.target.result;
    img.style.display = 'block';
    document.getElementById('upload-ui-'+key).style.display = 'none';
  };
  reader.readAsDataURL(file);
}

/* ══════════════════════════════════════════════════
   GRADIO API CALLS
══════════════════════════════════════════════════ */
var SESSION = Math.random().toString(36).slice(2);

async function uploadToGradio(file){
  var fd = new FormData();
  fd.append('files', file);
  var r = await fetch('/upload?upload_id='+SESSION, {method:'POST', body:fd});
  var data = await r.json();
  return data[0];
}

async function runIdentify(){
  var file = fileStore['id'];
  if(!file) return;
  setLoading('id', true);
  try {
    var filePath = await uploadToGradio(file);
    var payload = {
      data: [{path: filePath, orig_name: file.name, size: file.size, mime_type: file.type, is_stream: false, meta:{_type:'gradio.FileData'}}],
      fn_index: 0, session_hash: SESSION
    };
    var r = await fetch('/api/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    var data = await r.json();
    var text = data.data ? data.data[0] : (data.error || 'Error: unexpected response');
    showResult('id', text, false);
  } catch(e) {
    showResult('id', '❌ Error: ' + e.message, false);
  }
  setLoading('id', false);
}

async function runGradCam(){
  var file = fileStore['cam'];
  if(!file) return;
  setLoading('cam', true);
  try {
    var filePath = await uploadToGradio(file);
    var payload = {
      data: [{path: filePath, orig_name: file.name, size: file.size, mime_type: file.type, is_stream: false, meta:{_type:'gradio.FileData'}}],
      fn_index: 1, session_hash: SESSION
    };
    var r = await fetch('/api/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    var data = await r.json();
    if(data.data){
      var imgData = data.data[0];
      var textData = data.data[1] || '';
      // imgData can be {url:...} or {path:...} or base64
      var camImg = document.getElementById('cam-img');
      if(imgData && imgData.url){
        camImg.src = imgData.url; camImg.style.display = 'block';
      } else if(imgData && imgData.path){
        camImg.src = '/file=' + imgData.path; camImg.style.display = 'block';
      }
      showResult('cam', textData || '✅ Grad-CAM generated!', false);
    } else {
      showResult('cam', '❌ ' + (data.error||'Error'), false);
    }
  } catch(e) {
    showResult('cam', '❌ Error: ' + e.message, false);
  }
  setLoading('cam', false);
}

function setLoading(key, on){
  document.getElementById('spinner-'+key).style.display = on ? 'block' : 'none';
  document.getElementById('result-'+key).style.display  = on ? 'none'  : 'block';
  document.getElementById('btn-'+key).disabled = on;
  if(key==='cam') document.getElementById('cam-img').style.display = 'none';
}

function showResult(key, text, isPlaceholder){
  var el = document.getElementById('result-'+key);
  el.textContent = text;
  el.className = 'results-text' + (isPlaceholder ? ' placeholder' : '');
  el.style.display = 'block';
}
</script>
"""

with gr.Blocks(title="🐦 Bird Species Identifier") as app:

    # ── Hidden backend components (offscreen, still functional) ──
    with gr.Group(elem_id="hidden-backend"):
        _img  = gr.Image(type="numpy")
        _txt  = gr.Textbox()
        _btn  = gr.Button()
        _cimg = gr.Image(type="numpy")
        _cout = gr.Image()
        _ctxt = gr.Textbox()
        _cbtn = gr.Button()

    _btn.click(fn=predict_bird,    inputs=_img,  outputs=_txt)
    _cbtn.click(fn=generate_gradcam, inputs=_cimg, outputs=[_cout, _ctxt])

    # ── Custom frontend ──
    gr.HTML(FRONTEND_HTML)

print("\n✅ Starting Bird Identifier — Custom UI Edition...")
print("   Open in browser: http://127.0.0.1:7860\n")
app.launch(theme=BIRD_THEME, css=CUSTOM_CSS)
"""
"""
