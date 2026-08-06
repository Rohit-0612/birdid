# ============================================================
#  BIRD IDENTIFIER — Run Locally (No Training Needed)
#  Requirements: pip install torch torchvision pillow gradio
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import os

# ============================================================
# ✅ STEP 1 — SET THIS PATH to where you saved your model file
# ============================================================
MODEL_PATH ="/Users/swayam/Desktop/birdsproject/best_bird_model.pth"
# On Mac/Linux use: MODEL_PATH = "/Users/yourname/Desktop/BirdProject/best_bird_model.pth"
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ── Load class names and model ───────────────────────────────
print("Loading model...")
checkpoint  = torch.load(MODEL_PATH, map_location=device)
CLASS_NAMES = checkpoint["class_names"]
NUM_SPECIES = len(CLASS_NAMES)

model = models.efficientnet_v2_s(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_SPECIES)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()
print(f"✅ Model loaded! Knows {NUM_SPECIES} bird species.")

# ── Image transform (same as training) ──────────────────────
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Habitat map ──────────────────────────────────────────────
HABITAT_MAP = {
    "001.Black_footed_Albatross":          "USA, Japan, Hawaii (North Pacific Ocean — nests on Hawaiian Islands)",
    "002.Laysan_Albatross":                "USA (Hawaii), Japan — breeds on Midway Atoll and Hawaiian Islands",
    "003.Sooty_Albatross":                 "South Atlantic & Indian Ocean — Tristan da Cunha, South Georgia (UK territory)",
    "004.Groove_billed_Ani":               "Mexico, Costa Rica, Panama, Colombia, Venezuela — tropical lowlands",
    "005.Crested_Auklet":                  "USA (Alaska), Russia — Bering Sea islands, Aleutian Islands",
    "006.Least_Auklet":                    "USA (Alaska), Russia — St. Lawrence Island, Pribilof Islands",
    "007.Parakeet_Auklet":                 "USA (Alaska), Russia — Bering Sea, Aleutian Islands",
    "008.Rhinoceros_Auklet":               "USA (California, Oregon, Washington), Canada (British Columbia), Japan",
    "009.Brewer_Blackbird":                "USA (western states), Canada — open farmlands of California, Oregon, Montana",
    "010.Red_winged_Blackbird":            "USA, Canada, Mexico — widespread across North America, wetlands",
    "011.Rusty_Blackbird":                 "Canada (boreal forest), USA (eastern states in winter) — Alaska to Newfoundland",
    "012.Yellow_headed_Blackbird":         "USA (western states), Canada (prairies), Mexico — Great Plains marshes",
    "013.Bobolink":                        "USA, Canada (breeds), Argentina, Bolivia (winters) — long-distance migrant",
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
    "025.Pelagic_Cormorant":               "USA (Pacific coast), Canada (British Columbia), Russia — rocky Pacific coasts",
    "026.Bronzed_Cowbird":                 "USA (Texas, Arizona), Mexico, Central America — southern border states",
    "027.Shiny_Cowbird":                   "South America (widespread), Caribbean — Argentina, Brazil, Colombia",
    "028.Brown_headed_Cowbird":            "USA, Canada — widespread across North America, most common in Midwest",
    "029.Pigeon_Guillemot":                "USA (Alaska, California), Canada, Russia — North Pacific coastal cliffs",
    "030.California_Gull":                 "USA (California, Great Basin states), Canada — breeds inland, winters on coast",
    "031.Glaucous_winged_Gull":            "USA (Alaska, Washington, Oregon), Canada (British Columbia) — Pacific Northwest",
    "032.Heermann_Gull":                   "USA (California coast), Mexico (Baja California) — breeds on Isla Raza, Mexico",
    "033.Herring_Gull":                    "USA, Canada, UK, northern Europe — extremely widespread across North Atlantic",
    "034.Ivory_Gull":                      "Canada (Arctic), Russia (Arctic), Norway (Svalbard) — high Arctic only",
    "035.Ring_billed_Gull":                "USA, Canada — one of the most common gulls across North America",
    "036.Slaty_backed_Gull":               "Russia (eastern Siberia), Japan, South Korea, China — East Asian coasts",
    "037.Western_Gull":                    "USA (California, Oregon, Washington) — Pacific coast only",
    "038.Anna_Hummingbird":                "USA (California, Arizona, Oregon), Mexico (Baja) — Pacific coast year-round",
    "039.Ruby_throated_Hummingbird":       "USA (eastern states), Canada (Ontario), Mexico, Central America — very common",
    "040.Rufous_Hummingbird":              "USA (western states), Canada (British Columbia), Mexico — long-distance migrant",
    "041.Green_Violetear":                 "Mexico, Guatemala, Costa Rica, Venezuela, Colombia — mountain forests",
    "042.Long_tailed_Jaeger":              "Canada (Arctic), USA (Alaska), winters in South Atlantic — rare migrant",
    "043.Pomarine_Jaeger":                 "USA (Alaska), Canada (Arctic tundra) — winters off South America & Africa",
    "044.Blue_Jay":                        "USA (eastern & central states), Canada (Ontario, Quebec) — extremely common",
    "045.Florida_Jay":                     "USA (Florida only) — found exclusively in Florida scrub habitat",
    "046.Green_Jay":                       "USA (southern Texas only), Mexico, Central America, Colombia, Venezuela",
    "047.Dark_eyed_Junco":                 "USA, Canada — one of the most common birds in North America, all states",
    "048.Tropical_Kingbird":               "USA (southern Arizona, Texas), Mexico, Central America, South America",
    "049.Gray_Kingbird":                   "USA (Florida), Caribbean islands — Cuba, Jamaica, Puerto Rico, Bahamas",
    "050.Belted_Kingfisher":               "USA, Canada — found near rivers and lakes across all of North America",
    "051.Green_Kingfisher":                "USA (southern Texas, Arizona), Mexico, Central America, South America",
    "052.Pied_Kingfisher":                 "Africa (sub-Saharan), India, Southeast Asia — very widespread in Africa",
    "053.Ringed_Kingfisher":               "USA (southern Texas), Mexico, Central America, South America to Argentina",
    "054.White_breasted_Kingfisher":       "India, Sri Lanka, Southeast Asia (Thailand, Vietnam, Philippines)",
    "055.Red_legged_Kittiwake":            "USA (Alaska — Pribilof Islands, St. George Island) — very restricted range",
    "056.Horned_Lark":                     "USA, Canada — most widespread lark in North America, open plains",
    "057.Pacific_Loon":                    "USA (Alaska), Canada (Arctic) — winters along Pacific coast to California",
    "058.Mallard":                         "USA, Canada, UK, Europe, Asia — most widespread duck in the world",
    "059.Western_Meadowlark":              "USA (western & central states), Canada (prairies), Mexico — Great Plains",
    "060.Hooded_Merganser":                "USA, Canada — breeds in forested lakes from Alaska to Florida",
    "061.Red_breasted_Merganser":          "USA, Canada, UK, northern Europe — circumpolar breeding range",
    "062.Mockingbird":                     "USA (all states), Canada (southern Ontario), Mexico, Caribbean — very common",
    "063.Nighthawk":                       "USA, Canada (breeds), South America (winters) — widespread migrant",
    "064.Clark_Nutcracker":                "USA (western mountain states — Colorado, Wyoming, Montana, California)",
    "065.White_breasted_Nuthatch":         "USA, Canada — common in deciduous forests across North America",
    "066.Baltimore_Oriole":                "USA (eastern states), Canada (Ontario), Mexico, Central America (winters)",
    "067.Hooded_Oriole":                   "USA (California, Arizona, Texas), Mexico — southwestern USA and Mexico",
    "068.Orchard_Oriole":                  "USA (eastern & central states), Canada, Mexico, Central America (winters)",
    "069.Scott_Oriole":                    "USA (Texas, Arizona, California), Mexico — Chihuahuan Desert region",
    "070.Ovenbird":                        "USA (eastern states), Canada — breeds from Georgia to Manitoba",
    "071.Brown_Pelican":                   "USA (coastal states — California, Florida, Texas), Mexico, Caribbean",
    "072.White_Pelican":                   "USA, Canada — breeds on lakes in Great Plains, winters on Gulf Coast",
    "073.Western_Wood_Pewee":              "USA (western states), Canada (British Columbia), Mexico, South America",
    "074.Sayornis":                        "USA, Canada, Mexico — widespread flycatcher across North America",
    "075.American_Pipit":                  "USA (Alaska, mountain states), Canada (Arctic) — winters across southern USA",
    "076.Whip_poor_Will":                  "USA (eastern states), Canada (Ontario, Quebec), Mexico, Central America",
    "077.Horned_Puffin":                   "USA (Alaska), Russia, Canada — North Pacific, breeds on Alaskan sea cliffs",
    "078.Common_Raven":                    "USA (western states, Alaska), Canada, UK, northern Europe, Asia — widespread",
    "079.White_necked_Raven":              "USA (Texas, Arizona), Mexico — Chihuahuan and Sonoran Desert regions",
    "080.American_Redstart":               "USA (eastern states), Canada — breeds from Georgia to Nova Scotia",
    "081.Geococcyx":                       "USA (Texas, New Mexico, Arizona, California), Mexico — Greater Roadrunner",
    "082.Loggerhead_Shrike":               "USA, Canada (southern Ontario) — widespread but declining across North America",
    "083.Great_Grey_Shrike":               "Canada (Arctic), Russia, northern Europe (Scandinavia, Finland) — boreal",
    "084.Baird_Sparrow":                   "USA (Montana, North Dakota), Canada (Manitoba, Saskatchewan) — Great Plains",
    "085.Black_throated_Sparrow":          "USA (Texas, New Mexico, Arizona, Nevada, California), Mexico — desert southwest",
    "086.Brewer_Sparrow":                  "USA (western states), Canada (British Columbia, Alberta), Mexico — Great Basin",
    "087.Chipping_Sparrow":                "USA, Canada — one of the most common sparrows across North America",
    "088.Clay_colored_Sparrow":            "USA (Great Plains states), Canada (prairies — Saskatchewan, Manitoba)",
    "089.House_Sparrow":                   "USA, Canada, UK, Europe, Australia, India — introduced worldwide, very common",
    "090.Field_Sparrow":                   "USA (eastern & central states) — common from Texas to New England",
    "091.Fox_Sparrow":                     "USA (western states, Alaska), Canada — breeds in Canada, winters in USA",
    "092.Grasshopper_Sparrow":             "USA (central & eastern states), Canada (Ontario), Mexico, Central America",
    "093.Harris_Sparrow":                  "USA (Great Plains — Kansas, Oklahoma, Texas in winter), Canada (breeds)",
    "094.Henslow_Sparrow":                 "USA (Midwest — Ohio, Indiana, Illinois, New York) — rare and declining",
    "095.Le_Conte_Sparrow":                "USA (Great Plains, Southeast), Canada (prairies) — secretive marsh bird",
    "096.Lincoln_Sparrow":                 "USA, Canada (breeds), Mexico, Central America (winters) — widespread",
    "097.Nelson_Sharp_tailed_Sparrow":     "USA (Atlantic coast — Maine to Virginia), Canada (Maritime provinces)",
    "098.Savannah_Sparrow":                "USA, Canada — one of the most widespread sparrows in North America",
    "099.Seaside_Sparrow":                 "USA (Atlantic & Gulf coast — Virginia to Texas) — coastal salt marshes only",
    "100.Song_Sparrow":                    "USA, Canada — extremely common and widespread across North America",
    "101.Tree_Sparrow":                    "Canada (Arctic, boreal), USA (northern states in winter) — winters in Midwest",
    "102.Vesper_Sparrow":                  "USA (western & central states), Canada (prairies), Mexico (winters)",
    "103.White_crowned_Sparrow":           "USA, Canada — widespread migrant, breeds in Canada and mountain west",
    "104.White_throated_Sparrow":          "USA (eastern states in winter), Canada (breeds) — very common migrant",
    "105.Cape_Glossy_Starling":            "Kenya, Tanzania, Uganda, Ethiopia, Somalia — East Africa savanna",
    "106.Bank_Swallow":                    "USA, Canada (breeds), South America (winters) — circumpolar breeding range",
    "107.Barn_Swallow":                    "USA, Canada (breeds), South America, Africa (winters) — worldwide migrant",
    "108.Cliff_Swallow":                   "USA, Canada (breeds), South America (winters) — common across North America",
    "109.Tree_Swallow":                    "USA, Canada (breeds), Mexico, Central America (winters) — very common",
    "110.Scarlet_Tanager":                 "USA (eastern states), Canada (Ontario), South America (winters) — striking",
    "111.Summer_Tanager":                  "USA (southern states — Georgia, Texas, Carolina), Mexico, South America",
    "112.Artic_Tern":                      "Canada, USA (Alaska), UK, Norway — longest migration on Earth to Antarctica",
    "113.Black_Tern":                      "USA (Midwest lakes), Canada, Europe — breeds in freshwater marshes",
    "114.Caspian_Tern":                    "USA (Great Lakes, Gulf Coast), Canada, Europe, Africa, Australia — worldwide",
    "115.Common_Tern":                     "USA, Canada, UK, Europe — widespread across North Atlantic coasts",
    "116.Elegant_Tern":                    "USA (California coast), Mexico (Baja California) — breeds on Mexican islands",
    "117.Forsters_Tern":                   "USA, Canada — breeds in Great Plains marshes, winters on both coasts",
    "118.Least_Tern":                      "USA (Atlantic & Gulf coast, river valleys), Mexico, Caribbean",
    "119.Green_tailed_Towhee":             "USA (western mountain states — Colorado, Utah, California), Mexico (winters)",
    "120.Brown_Thrasher":                  "USA (eastern & central states), Canada (Ontario) — common in Midwest",
    "121.Sage_Thrasher":                   "USA (Great Basin — Nevada, Idaho, Wyoming, Oregon), Mexico (winters)",
    "122.Black_capped_Vireo":              "USA (Texas, Oklahoma only), Mexico (winters) — endangered species",
    "123.Blue_headed_Vireo":               "USA (eastern states), Canada (Ontario, Quebec), Central America (winters)",
    "124.Philadelphia_Vireo":              "USA (northeastern states), Canada (Ontario to Alberta), Central America",
    "125.Red_eyed_Vireo":                  "USA, Canada — one of the most common breeding birds in eastern North America",
    "126.Warbling_Vireo":                  "USA, Canada (breeds), Mexico, Central America (winters) — widespread",
    "127.White_eyed_Vireo":                "USA (southeastern & eastern states), Mexico, Central America (winters)",
    "128.Yellow_throated_Vireo":           "USA (eastern states), Canada (Ontario), South America (winters)",
    "129.Bay_breasted_Warbler":            "Canada (boreal — Ontario to Newfoundland), USA (eastern states in migration)",
    "130.Black_and_white_Warbler":         "USA (eastern states), Canada — breeds from Florida to Northwest Territories",
    "131.Black_throated_Blue_Warbler":     "USA (eastern mountain states — Appalachians), Canada, Caribbean (winters)",
    "132.Blue_winged_Warbler":             "USA (eastern states — Ohio, Indiana, New York), Mexico, Central America",
    "133.Canada_Warbler":                  "USA (northeastern states), Canada (Ontario to Nova Scotia), South America",
    "134.Cape_May_Warbler":                "Canada (boreal — Ontario, Quebec, Manitoba), USA (eastern states, migration)",
    "135.Cerulean_Warbler":                "USA (Appalachians, Midwest), Canada (Ontario), South America (winters)",
    "136.Chestnut_sided_Warbler":          "USA (northeastern states), Canada (Ontario, Quebec), Central America",
    "137.Golden_winged_Warbler":           "USA (Appalachians, Great Lakes region), Canada, Central America (winters)",
    "138.Hooded_Warbler":                  "USA (eastern states — Georgia, Virginia, Tennessee), Mexico, Central America",
    "139.Kentucky_Warbler":                "USA (southeastern states — Kentucky, Tennessee, Arkansas), Central America",
    "140.Magnolia_Warbler":                "Canada (boreal — Ontario to Nova Scotia), USA (eastern states in migration)",
    "141.Mourning_Warbler":                "Canada (Ontario to Manitoba), USA (northeastern states), Central America",
    "142.Myrtle_Warbler":                  "USA, Canada — one of the most common warblers, breeds across Canada",
    "143.Nashville_Warbler":               "USA (northeastern & western states), Canada, Mexico, Central America",
    "144.Orange_crowned_Warbler":          "USA (western states), Canada (British Columbia, Alberta), Mexico",
    "145.Palm_Warbler":                    "Canada (boreal bogs — Ontario to Newfoundland), USA (Florida in winter)",
    "146.Pine_Warbler":                    "USA (southeastern states — Florida, Georgia, Texas, Carolina) — year-round",
    "147.Prairie_Warbler":                 "USA (eastern states — Michigan, Ohio, New Jersey, Florida), Caribbean",
    "148.Prothonotary_Warbler":            "USA (southeastern states — Louisiana, Mississippi, Tennessee, Virginia)",
    "149.Swainson_Warbler":                "USA (southeastern states — Arkansas, Louisiana, South Carolina) — rare",
    "150.Tennessee_Warbler":               "Canada (boreal — Ontario to Alberta), USA (eastern states in migration)",
    "151.Wilson_Warbler":                  "USA (western states, Alaska), Canada (widespread), Mexico, Central America",
    "152.Worm_eating_Warbler":             "USA (eastern states — Pennsylvania, Maryland, Ohio, Georgia), Central America",
    "153.Yellow_Warbler":                  "USA, Canada — most widespread warbler, found across all of North America",
    "154.Northern_Waterthrush":            "USA (northeastern states), Canada (widespread boreal), Central America",
    "155.Louisiana_Waterthrush":           "USA (eastern states — Pennsylvania to Georgia, west to Kansas), South America",
    "156.Bohemian_Waxwing":                "Canada (boreal — British Columbia to Manitoba), USA (northern states in winter)",
    "157.Cedar_Waxwing":                   "USA, Canada — very common, found across all of North America year-round",
    "158.American_Three_toed_Woodpecker":  "USA (Alaska, mountain west), Canada (boreal) — Rocky Mountains, Cascades",
    "159.Pileated_Woodpecker":             "USA (eastern states, Pacific Northwest), Canada — largest woodpecker in N. America",
    "160.Red_bellied_Woodpecker":          "USA (eastern states) — very common from Florida to New York to Texas",
    "161.Red_cockaded_Woodpecker":         "USA (southeastern states — North Carolina to Texas) — endangered species",
    "162.Red_headed_Woodpecker":           "USA (eastern & central states), Canada (Ontario, Manitoba) — declining",
    "163.Downy_Woodpecker":                "USA, Canada — smallest and most common woodpecker in North America",
    "164.Bewick_Wren":                     "USA (western & southern states — California, Texas, Oregon), Mexico",
    "165.Cactus_Wren":                     "USA (Arizona, California, New Mexico, Texas), Mexico — Sonoran Desert",
    "166.Carolina_Wren":                   "USA (eastern & southeastern states) — very common from Texas to New England",
    "167.House_Wren":                      "USA, Canada (breeds), South America (winters) — most common wren in Americas",
    "168.Marsh_Wren":                      "USA, Canada — freshwater marshes across North America, both coasts",
    "169.Rock_Wren":                       "USA (western states — Colorado, Utah, California, Arizona), Mexico",
    "170.Winter_Wren":                     "USA (Pacific Northwest, Appalachians), Canada, UK, Europe, Asia — widespread",
    "171.Common_Yellowthroat":             "USA, Canada — one of the most widespread warblers in North America",
    "172.Wilson_Snipe":                    "USA, Canada (breeds), Central America, South America (winters)",
    "173.American_Woodcock":               "USA (eastern states), Canada (Ontario, Quebec) — secretive forest bird",
    "174.Great_Crested_Flycatcher":        "USA (eastern states), Canada (Ontario), Mexico, Central America (winters)",
    "175.Least_Flycatcher":                "USA (northeastern states), Canada (widespread), Mexico, Central America",
    "176.Olive_sided_Flycatcher":          "USA (western states, Alaska), Canada (boreal), South America (winters)",
    "177.Acadian_Flycatcher":              "USA (eastern states — Ohio, Virginia, Georgia, Texas), Central America",
    "178.Yellow_bellied_Flycatcher":       "Canada (boreal — Ontario to Newfoundland), USA (northeastern states)",
    "179.Pacific_slope_Flycatcher":        "USA (California, Oregon, Washington), Canada (British Columbia), Mexico",
    "180.Black_billed_Cuckoo":             "USA (eastern & central states), Canada (Ontario, Quebec), South America",
    "181.Yellow_billed_Cuckoo":            "USA (widespread), Canada (Ontario), Mexico, Central America, South America",
    "182.American_Crow":                   "USA, Canada — one of the most intelligent and common birds in North America",
    "183.Fish_Crow":                       "USA (eastern & southeastern coastal states — New York to Texas) — near water",
    "184.Brown_Creeper":                   "USA, Canada — found in mature forests across North America year-round",
    "185.Rock_Pigeon":                     "USA, Canada, UK, Europe, India, worldwide — introduced everywhere globally",
    "186.White_crowned_Pigeon":            "USA (Florida Keys only), Caribbean — Cuba, Bahamas, Haiti, Jamaica",
    "187.Band_tailed_Pigeon":              "USA (California, Oregon, Washington, Arizona), Canada (British Columbia), Mexico",
    "188.Eared_Grebe":                     "USA (western states), Canada (prairies), Mexico, Spain, Africa — widespread",
    "189.Horned_Grebe":                    "USA, Canada (breeds), UK, northern Europe — circumpolar breeding range",
    "190.Red_necked_Grebe":                "USA (Pacific & Atlantic coasts in winter), Canada (breeds), northern Europe",
    "191.Pied_billed_Grebe":               "USA, Canada — most common grebe in North America, found on any lake",
    "192.Western_Grebe":                   "USA (western states), Canada (British Columbia, prairies) — large elegant grebe",
    "193.Gadwall":                         "USA, Canada (breeds), UK, Europe, Asia — widespread dabbling duck",
    "194.Canvasback":                      "USA (Great Plains, western states), Canada (prairies) — prized diving duck",
    "195.Redhead":                         "USA (Great Plains, Great Lakes), Canada (prairies) — common diving duck",
    "196.Ring_necked_Duck":                "USA, Canada (boreal) — very common diving duck across North America",
    "197.Lesser_Scaup":                    "USA, Canada (breeds in Alaska and prairies) — most abundant diving duck",
    "198.Surf_Scoter":                     "USA (Pacific & Atlantic coasts), Canada (breeds in Alaska and boreal)",
    "199.White_winged_Scoter":             "USA (both coasts in winter), Canada (breeds in boreal and prairie lakes)",
    "200.Long_tailed_Duck":                "USA (Great Lakes, both coasts), Canada (Arctic breeds) — formerly Oldsquaw",
}

SIMILAR_SPECIES = {
    "Black footed Albatross":    ("Laysan Albatross",         "check the face — Black-footed has a dark face, Laysan has a white face"),
    "Laysan Albatross":          ("Black footed Albatross",   "check the face — Laysan has a white face, Black-footed has a dark face"),
    "Herring Gull":              ("Ring billed Gull",         "check the bill — Herring Gull has a red spot, Ring-billed has a black ring"),
    "Ring billed Gull":          ("Herring Gull",             "check the bill — Ring-billed has a black ring, Herring has a red spot"),
    "Ruby throated Hummingbird": ("Rufous Hummingbird",       "check the throat — Ruby-throated is green-backed, Rufous has orange-brown back"),
    "Barn Swallow":              ("Cliff Swallow",            "check the tail — Barn Swallow has a deep forked tail, Cliff has a square tail"),
    "Downy Woodpecker":          ("Pileated Woodpecker",      "check the size — Downy is sparrow-sized, Pileated is crow-sized"),
    "Baltimore Oriole":          ("Orchard Oriole",           "check the color — Baltimore is bright orange, Orchard is darker chestnut"),
    "American Crow":             ("Common Raven",             "check the size & tail — Crow is smaller with a fan tail, Raven is larger with wedge tail"),
    "Scarlet Tanager":           ("Summer Tanager",           "check the wings — Scarlet has black wings, Summer Tanager is all red"),
    "Cedar Waxwing":             ("Bohemian Waxwing",         "check the belly — Cedar has a yellow belly, Bohemian has rusty undertail"),
    "Indigo Bunting":            ("Lazuli Bunting",           "check the breast — Indigo is all blue, Lazuli has a rusty-orange breast"),
}

MIGRATION_MAP = {
    "Black footed Albatross":        "Year-round in North Pacific Ocean. Breeds on Hawaiian Islands (Dec–Jul), roams Pacific rest of year",
    "Laysan Albatross":              "Year-round in North Pacific. Breeds on Midway Atoll & Hawaii (Nov–Jul), roams North Pacific rest of year",
    "Sooty Albatross":               "Year-round in South Atlantic & Indian Ocean. Breeds on Tristan da Cunha & South Georgia islands",
    "Groove billed Ani":             "Year-round resident in Mexico & Central America. Some move to southern Texas in summer (Apr–Sep)",
    "Crested Auklet":                "Breeds on Aleutian Islands & Bering Sea (May–Aug) → winters in open North Pacific Ocean",
    "Least Auklet":                  "Breeds on St. Lawrence & Pribilof Islands Alaska (Jun–Aug) → winters in North Pacific",
    "Parakeet Auklet":               "Breeds on Alaskan & Russian islands (May–Aug) → winters in open North Pacific Ocean",
    "Rhinoceros Auklet":             "Breeds on Pacific coast islands (Apr–Aug) → winters offshore in North Pacific",
    "Brewer Blackbird":              "Year-round in western USA. Northern Canada birds migrate south to California & Mexico (Oct–Mar)",
    "Red winged Blackbird":          "Year-round across most of USA. Northern Canada birds migrate south to southern USA (Oct–Apr)",
    "Rusty Blackbird":               "Breeds in Alaska & boreal Canada (May–Aug) → winters in southeastern USA (Sep–Apr)",
    "Yellow headed Blackbird":       "Breeds in Great Plains marshes USA & Canada (May–Aug) → winters in Mexico & southwestern USA",
    "Bobolink":                      "Breeds in USA & Canada prairies (May–Aug) → migrates 12,000 miles to Argentina & Bolivia (Sep–Apr). One of the longest migrations of any songbird",
    "Indigo Bunting":                "Breeds in eastern & central USA (May–Aug) → winters in Mexico, Cuba & Central America (Sep–Apr). Navigates by stars at night",
    "Lazuli Bunting":                "Breeds in western USA & Canada (May–Aug) → winters in western Mexico (Sep–Apr)",
    "Painted Bunting":               "Breeds in southern USA — Texas, Louisiana, Florida (Apr–Aug) → winters in Florida, Caribbean & Central America",
    "Cardinal":                      "Year-round resident across eastern & southern USA. Does not migrate",
    "Spotted Catbird":               "Year-round resident in Queensland & New South Wales, Australia. Short local movements only",
    "Gray Catbird":                  "Breeds in USA & Canada (May–Aug) → winters in Florida, Caribbean & Central America (Sep–Apr)",
    "Yellow breasted Chat":          "Breeds across USA & Canada (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "Eastern Towhee":                "Year-round in southeastern USA. Northern birds migrate south in winter (Oct–Mar)",
    "Chuck will Widow":              "Breeds in southeastern USA (Apr–Aug) → winters in Caribbean & Central America (Sep–Mar)",
    "Brandt Cormorant":              "Year-round on Pacific coast from Alaska to Baja California. Short local movements",
    "Red faced Cormorant":           "Year-round resident on Aleutian Islands & Kodiak, Alaska. Does not migrate",
    "Pelagic Cormorant":             "Year-round on Pacific coast. Some northernmost birds move south slightly in winter",
    "Bronzed Cowbird":               "Year-round in Mexico & Central America. Moves into southern Texas & Arizona (Mar–Sep)",
    "Shiny Cowbird":                 "Year-round across South America & Caribbean. Expanding northward into USA",
    "Brown headed Cowbird":          "Year-round across most of USA. Northern Canada birds migrate south in winter (Oct–Mar)",
    "Pigeon Guillemot":              "Year-round on North Pacific coast. Short offshore movements in winter",
    "California Gull":               "Breeds inland at Great Basin lakes (Apr–Aug) → winters on California & Pacific coast (Sep–Mar)",
    "Glaucous winged Gull":          "Year-round on Pacific Northwest coast. Some move south to California in winter",
    "Heermann Gull":                 "Breeds on Isla Raza Mexico (Jan–Jun) → moves north to California & Oregon coast (Jul–Nov) — reverse migration",
    "Herring Gull":                  "Breeds in Canada & northern USA (Apr–Aug) → winters across all USA coasts (Sep–Mar)",
    "Ivory Gull":                    "Year-round in high Arctic. Moves south only when sea ice forces it — rarely seen in USA",
    "Ring billed Gull":              "Breeds in Canada & northern USA (Apr–Aug) → winters across all of USA (Sep–Mar). Very common in parking lots!",
    "Slaty backed Gull":             "Year-round in eastern Russia & Japan. Rare winter visitor to Alaska & Pacific coast",
    "Western Gull":                  "Year-round on California, Oregon & Washington coast. Does not migrate far",
    "Anna Hummingbird":              "Year-round on Pacific coast — one of very few hummingbirds that does not migrate south. Stays in California & Oregon all winter",
    "Ruby throated Hummingbird":     "Breeds in eastern USA & Canada (Apr–Aug) → crosses Gulf of Mexico non-stop to winter in Mexico & Central America (Sep–Apr). Incredible 500-mile non-stop ocean crossing",
    "Rufous Hummingbird":            "Breeds in Pacific Northwest & Alaska (Apr–Jul) → migrates south through Rocky Mountains to winter in Mexico (Aug–Mar). Longest migration of any hummingbird — 3,900 miles",
    "Green Violetear":               "Year-round resident in mountain forests of Mexico & Central America. Short altitudinal movements only",
    "Long tailed Jaeger":            "Breeds on Arctic tundra Canada & Alaska (Jun–Aug) → migrates over ocean to winter in South Atlantic (Sep–May)",
    "Pomarine Jaeger":               "Breeds on Arctic tundra (Jun–Aug) → winters off coasts of South America & West Africa (Sep–May). Rarely seen inland",
    "Blue Jay":                      "Year-round across eastern USA. Some northern birds migrate south in large flocks in autumn — not all individuals migrate",
    "Florida Jay":                   "Year-round resident in Florida scrub only. Does not migrate at all — one of the most sedentary birds in North America",
    "Green Jay":                     "Year-round resident in southern Texas & Central America. Does not migrate",
    "Dark eyed Junco":               "Breeds in Canada & mountain USA (May–Aug) → winters across all of USA (Oct–Apr). Called the snowbird — their arrival signals winter coming",
    "Tropical Kingbird":             "Year-round in Mexico & Central America. Moves into southern Arizona & Texas (Apr–Sep)",
    "Gray Kingbird":                 "Breeds in Florida & Caribbean (Apr–Aug) → winters in northern South America (Sep–Mar)",
    "Belted Kingfisher":             "Year-round across most of USA near water. Northern Canada birds move south in winter",
    "Green Kingfisher":              "Year-round resident in southern Texas, Mexico & Central America. Does not migrate",
    "Pied Kingfisher":               "Year-round resident across Africa & South Asia. Does not migrate",
    "Ringed Kingfisher":             "Year-round in southern Texas, Mexico & South America. Does not migrate",
    "White breasted Kingfisher":     "Year-round resident across India & Southeast Asia. Does not migrate",
    "Red legged Kittiwake":          "Breeds on Pribilof Islands Alaska (May–Aug) → winters in North Pacific Ocean. Rarely seen on land outside breeding season",
    "Horned Lark":                   "Year-round across open areas of USA & Canada. Northern birds move south in winter (Oct–Mar)",
    "Pacific Loon":                  "Breeds on Arctic lakes Canada & Alaska (Jun–Aug) → winters along Pacific coast from Alaska to California (Sep–May)",
    "Mallard":                       "Year-round in most of USA. Northern Canada & Alaska birds migrate south to USA & Mexico in winter (Oct–Mar)",
    "Western Meadowlark":            "Year-round across western & central USA. Northern Canada birds migrate south in winter (Oct–Mar)",
    "Hooded Merganser":              "Breeds in forested lakes USA & Canada (Apr–Aug) → winters in southern USA & Mexico (Oct–Mar)",
    "Red breasted Merganser":        "Breeds in Arctic & boreal Canada (May–Aug) → winters on both USA coasts (Sep–Apr)",
    "Mockingbird":                   "Year-round across USA & Mexico. Northern birds may move slightly south in harsh winters",
    "Nighthawk":                     "Breeds across USA & Canada (May–Aug) → migrates to South America (Bolivia, Argentina) for winter (Sep–Apr). One of the longest migrations among North American birds",
    "Clark Nutcracker":              "Year-round in western mountain USA. Short altitudinal movements — moves to lower elevations in winter",
    "White breasted Nuthatch":       "Year-round resident across North America. Does not migrate",
    "Baltimore Oriole":              "Breeds in eastern USA & Canada (May–Aug) → winters in Central America & northern South America (Sep–Apr)",
    "Hooded Oriole":                 "Breeds in southwestern USA (Apr–Aug) → winters in Mexico (Sep–Mar)",
    "Orchard Oriole":                "Breeds in eastern & central USA (May–Aug) → winters in Central America & northern South America (Aug–Apr). Leaves very early — one of first migrants to depart",
    "Scott Oriole":                  "Breeds in desert southwest USA (Apr–Aug) → winters in Mexico (Sep–Mar)",
    "Ovenbird":                      "Breeds in eastern USA & Canada (May–Aug) → winters in Caribbean, Mexico & Central America (Sep–Apr)",
    "Brown Pelican":                 "Year-round on USA coasts. Some northern birds move south in winter. Florida birds stay year-round",
    "White Pelican":                 "Breeds on inland lakes Great Plains (Apr–Aug) → winters on Gulf Coast & Pacific coast (Sep–Mar)",
    "Western Wood Pewee":            "Breeds in western USA & Canada (May–Aug) → winters in South America (Bolivia, Peru, Ecuador) (Sep–Apr)",
    "Sayornis":                      "Year-round in southwestern USA & Mexico. Northern birds migrate south in winter (Oct–Mar)",
    "American Pipit":                "Breeds on Arctic tundra & mountain tops (Jun–Aug) → winters across southern USA & Mexico (Sep–May)",
    "Whip poor Will":                "Breeds in eastern USA & Canada (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "Horned Puffin":                 "Breeds on Alaskan sea cliffs (May–Aug) → winters in open North Pacific Ocean far from shore (Sep–Apr)",
    "Common Raven":                  "Year-round resident. Does not migrate. Stays in same territory year-round",
    "White necked Raven":            "Year-round resident in desert southwest USA & Mexico. Does not migrate",
    "American Redstart":             "Breeds in eastern USA & Canada (May–Aug) → winters in Caribbean, Mexico & South America (Sep–Apr)",
    "Geococcyx":                     "Year-round resident in desert southwest USA & Mexico. Does not migrate",
    "Loggerhead Shrike":             "Year-round in southern USA. Northern birds migrate south in winter (Oct–Mar)",
    "Great Grey Shrike":             "Year-round in northern Europe & Russia. Irruptive — moves south into Europe in some winters",
    "Baird Sparrow":                 "Breeds in Great Plains USA & Canada (May–Aug) → winters in Texas, New Mexico & Mexico (Sep–Apr)",
    "Black throated Sparrow":        "Year-round in desert southwest USA & Mexico. Some move to lower elevations in winter",
    "Brewer Sparrow":                "Breeds in Great Basin USA & Canada (May–Aug) → winters in Mexico & southwestern USA (Sep–Apr)",
    "Chipping Sparrow":              "Breeds across USA & Canada (Apr–Aug) → winters in southern USA & Mexico (Sep–Apr)",
    "Clay colored Sparrow":          "Breeds in Great Plains Canada & USA (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "House Sparrow":                 "Year-round resident worldwide. Does not migrate — introduced species stays put all year",
    "Field Sparrow":                 "Year-round in eastern USA. Northern birds move south slightly in winter (Oct–Mar)",
    "Fox Sparrow":                   "Breeds in Alaska & Canada (May–Aug) → winters in western & southern USA (Oct–Apr)",
    "Grasshopper Sparrow":           "Breeds in eastern & central USA (May–Aug) → winters in southern USA, Caribbean & Central America",
    "Harris Sparrow":                "Breeds in boreal Canada (Jun–Aug) → winters in Great Plains USA — Kansas, Oklahoma, Texas (Sep–Apr)",
    "Henslow Sparrow":               "Breeds in Midwest USA (May–Aug) → winters in southeastern USA — Florida, Georgia, Carolina (Sep–Apr)",
    "Le Conte Sparrow":              "Breeds in northern Great Plains Canada (Jun–Aug) → winters in southeastern USA (Sep–Apr)",
    "Lincoln Sparrow":               "Breeds in Canada & mountain USA (May–Aug) → winters in southern USA & Mexico (Sep–Apr)",
    "Nelson Sharp tailed Sparrow":   "Breeds in Canadian prairies & Atlantic coast marshes (Jun–Aug) → winters on Atlantic & Gulf coast (Sep–Apr)",
    "Savannah Sparrow":              "Breeds across USA & Canada (Apr–Aug) → winters in southern USA, Mexico & Caribbean (Sep–Apr)",
    "Seaside Sparrow":               "Year-round resident in Atlantic & Gulf coast salt marshes. Does not migrate far",
    "Song Sparrow":                  "Year-round across most of USA. Northern Canada birds migrate south in winter (Oct–Mar)",
    "Tree Sparrow":                  "Breeds in Arctic Canada & Alaska (Jun–Aug) → winters across northern USA (Oct–Apr)",
    "Vesper Sparrow":                "Breeds in western & central USA & Canada (May–Aug) → winters in southern USA & Mexico (Sep–Apr)",
    "White crowned Sparrow":         "Breeds in Arctic Canada & Alaska (Jun–Aug) → winters across southern USA & Mexico (Oct–Apr)",
    "White throated Sparrow":        "Breeds in boreal Canada (Jun–Aug) → winters in eastern & southern USA (Oct–Apr). Very common winter bird feeder visitor",
    "Cape Glossy Starling":          "Year-round resident in East Africa. Short local movements following rainfall & food",
    "Bank Swallow":                  "Breeds across USA & Canada (May–Aug) → migrates to South America — Peru, Bolivia, Brazil (Sep–Apr)",
    "Barn Swallow":                  "Breeds across USA & Canada (Apr–Aug) → migrates to Argentina & southern Brazil (Sep–Mar). One of the most widespread migrants in the world",
    "Cliff Swallow":                 "Breeds across USA & Canada (Apr–Aug) → winters in Argentina (Sep–Mar)",
    "Tree Swallow":                  "Breeds in USA & Canada (Apr–Aug) → winters in Florida, Gulf Coast & Central America (Sep–Apr)",
    "Scarlet Tanager":               "Breeds in eastern USA & Canada (May–Aug) → winters in Colombia, Ecuador & Peru (Sep–Apr)",
    "Summer Tanager":                "Breeds in southern USA (Apr–Aug) → winters in Mexico, Central & South America (Sep–Apr)",
    "Arctic Tern":                   "Breeds in Arctic Canada, Alaska & UK (Jun–Aug) → migrates to Antarctic (Sep–May). Longest migration on Earth — up to 70,000 km per year, seeing more daylight than any other creature",
    "Black Tern":                    "Breeds in Midwest freshwater marshes USA & Canada (May–Aug) → winters off West Africa & northern South America (Sep–Apr)",
    "Caspian Tern":                  "Breeds on Great Lakes & Gulf Coast (Apr–Aug) → winters on Gulf Coast, Caribbean & northern South America (Sep–Mar)",
    "Common Tern":                   "Breeds on North Atlantic coasts USA & Canada (May–Aug) → winters off West Africa & South America (Sep–Apr)",
    "Elegant Tern":                  "Breeds on Isla Raza Mexico (Apr–Jul) → moves north to California coast (Jul–Oct) then winters off Peru & Chile",
    "Forsters Tern":                 "Breeds in Great Plains marshes USA & Canada (May–Aug) → winters on both USA coasts & Caribbean (Sep–Apr)",
    "Least Tern":                    "Breeds on USA beaches & river sandbars (May–Aug) → winters off northern South America (Sep–Apr)",
    "Green tailed Towhee":           "Breeds in western mountain USA (May–Aug) → winters in Mexico & southwestern USA desert (Sep–Apr)",
    "Brown Thrasher":                "Year-round in southeastern USA. Northern birds migrate south in winter (Oct–Mar)",
    "Sage Thrasher":                 "Breeds in Great Basin USA (Apr–Aug) → winters in Mexico & Chihuahuan Desert (Sep–Mar)",
    "Black capped Vireo":            "Breeds in Texas & Oklahoma (Apr–Aug) → winters in western Mexico (Sep–Mar). Endangered species",
    "Blue headed Vireo":             "Breeds in eastern USA & Canada (May–Aug) → winters in Florida, Caribbean & Central America (Sep–Apr)",
    "Philadelphia Vireo":            "Breeds in Canada & northeastern USA (Jun–Aug) → winters in Central America (Sep–May)",
    "Red eyed Vireo":                "Breeds across USA & Canada (May–Aug) → winters in Amazon basin South America (Sep–Apr). Sings more than almost any other bird — up to 20,000 songs per day",
    "Warbling Vireo":                "Breeds across USA & Canada (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "White eyed Vireo":              "Breeds in eastern USA (Apr–Aug) → winters in Florida, Caribbean & Central America (Sep–Apr)",
    "Yellow throated Vireo":         "Breeds in eastern USA & Canada (May–Aug) → winters in Colombia, Venezuela & Central America (Sep–Apr)",
    "Bay breasted Warbler":          "Breeds in boreal Canada (Jun–Aug) → winters in Panama & northern South America (Sep–May)",
    "Black and white Warbler":       "Breeds in eastern USA & Canada (Apr–Aug) → winters in Florida, Caribbean & South America (Sep–Apr)",
    "Black throated Blue Warbler":   "Breeds in Appalachian mountains & Canada (May–Aug) → winters in Caribbean — Cuba, Jamaica, Haiti (Sep–Apr)",
    "Blue winged Warbler":           "Breeds in eastern USA (May–Aug) → winters in Central America (Sep–Apr)",
    "Canada Warbler":                "Breeds in northeastern USA & Canada (Jun–Aug) → winters in Colombia, Ecuador & Peru (Sep–May)",
    "Cape May Warbler":              "Breeds in boreal Canada (Jun–Aug) → winters in Caribbean islands (Sep–May)",
    "Cerulean Warbler":              "Breeds in Appalachians & Midwest USA (May–Aug) → winters in Andes of Colombia, Ecuador & Peru (Sep–Apr)",
    "Chestnut sided Warbler":        "Breeds in northeastern USA & Canada (May–Aug) → winters in Central America (Sep–Apr)",
    "Golden winged Warbler":         "Breeds in Appalachians & Great Lakes (May–Aug) → winters in Central America & Venezuela (Sep–Apr)",
    "Hooded Warbler":                "Breeds in eastern USA (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "Kentucky Warbler":              "Breeds in southeastern USA (May–Aug) → winters in Central America & northern South America (Sep–Apr)",
    "Magnolia Warbler":              "Breeds in boreal Canada (Jun–Aug) → winters in Caribbean & Central America (Sep–May)",
    "Mourning Warbler":              "Breeds in Canada & northeastern USA (Jun–Aug) → winters in Costa Rica, Colombia & Venezuela (Sep–May)",
    "Myrtle Warbler":                "Breeds across Canada (May–Aug) → winters across all of USA — one of the most widespread warblers in winter (Sep–Apr)",
    "Nashville Warbler":             "Breeds in northeastern & western USA & Canada (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "Orange crowned Warbler":        "Breeds in western USA & Canada (Apr–Aug) → winters in southern USA & Mexico (Sep–Mar)",
    "Palm Warbler":                  "Breeds in boreal bogs Canada (Jun–Aug) → winters in Florida & Caribbean (Sep–Apr). Often seen walking on ground wagging its tail",
    "Pine Warbler":                  "Year-round in southeastern USA pine forests. Northern birds move to southern USA in winter (Oct–Mar)",
    "Prairie Warbler":               "Breeds in eastern USA (May–Aug) → winters in Florida, Caribbean & Central America (Sep–Apr)",
    "Prothonotary Warbler":          "Breeds in southeastern USA swamps (Apr–Aug) → winters in Colombia, Venezuela & Central America (Sep–Apr)",
    "Swainson Warbler":              "Breeds in southeastern USA (May–Aug) → winters in Caribbean & Yucatan Mexico (Sep–Apr)",
    "Tennessee Warbler":             "Breeds in boreal Canada (Jun–Aug) → winters in Costa Rica, Colombia & Venezuela (Sep–May)",
    "Wilson Warbler":                "Breeds in western USA, Alaska & Canada (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "Worm eating Warbler":           "Breeds in eastern USA (May–Aug) → winters in Caribbean & Central America (Sep–Apr)",
    "Yellow Warbler":                "Breeds across all of USA & Canada (May–Aug) → winters in Mexico, Central & South America (Sep–Apr). Most widespread warbler in North America",
    "Northern Waterthrush":          "Breeds in boreal Canada & northeastern USA (May–Aug) → winters in Caribbean & northern South America (Sep–Apr)",
    "Louisiana Waterthrush":         "Breeds in eastern USA (Apr–Aug) → winters in Caribbean & Central America (Aug–Apr). One of earliest spring migrants to arrive",
    "Bohemian Waxwing":              "Breeds in boreal Canada & Alaska (Jun–Aug) → irruptive winter visitor to northern USA (Oct–Mar). Appears in large flocks unpredictably following berry crops",
    "Cedar Waxwing":                 "Year-round across USA but nomadic — follows fruit & berry crops. Northern birds move south in winter. Travels in flocks",
    "American Three toed Woodpecker":"Year-round in Rocky Mountains, Cascades & boreal Canada. Short movements to lower elevations in winter",
    "Pileated Woodpecker":           "Year-round resident across eastern USA & Pacific Northwest. Does not migrate",
    "Red bellied Woodpecker":        "Year-round resident in eastern USA. Does not migrate",
    "Red cockaded Woodpecker":       "Year-round resident in southeastern USA pine forests. Does not migrate. Endangered species",
    "Red headed Woodpecker":         "Year-round in eastern USA. Northern birds may move south in winter following acorn crops",
    "Downy Woodpecker":              "Year-round resident across North America. Does not migrate — stays in same territory all year",
    "Bewick Wren":                   "Year-round in western & southern USA & Mexico. Does not migrate",
    "Cactus Wren":                   "Year-round resident in Sonoran Desert USA & Mexico. Does not migrate",
    "Carolina Wren":                 "Year-round resident in eastern USA. Does not migrate — very sensitive to cold winters",
    "House Wren":                    "Breeds across USA & Canada (Apr–Aug) → winters in southern USA, Mexico & Central America (Sep–Apr)",
    "Marsh Wren":                    "Year-round on both coasts USA. Interior birds migrate south in winter (Oct–Mar)",
    "Rock Wren":                     "Year-round in western USA rocky areas. Mountain birds move to lower elevations in winter",
    "Winter Wren":                   "Breeds in Pacific Northwest, Appalachians & boreal Canada (May–Aug) → winters across eastern USA (Oct–Apr)",
    "Common Yellowthroat":           "Breeds across USA & Canada (May–Aug) → winters in southern USA, Caribbean & Central America (Sep–Apr)",
    "Wilson Snipe":                  "Breeds in northern USA & Canada (May–Aug) → winters across southern USA & Central America (Sep–Apr)",
    "American Woodcock":             "Breeds in eastern USA & Canada (Mar–Aug) → winters in southeastern USA (Oct–Mar). Famous for spring sky dance display",
    "Great Crested Flycatcher":      "Breeds in eastern USA & Canada (May–Aug) → winters in Florida, Caribbean & South America (Sep–Apr)",
    "Least Flycatcher":              "Breeds in northeastern USA & Canada (May–Aug) → winters in Mexico & Central America (Sep–Apr)",
    "Olive sided Flycatcher":        "Breeds in western USA, Alaska & boreal Canada (Jun–Aug) → winters in Andes of South America — Peru, Bolivia (Sep–May). One of the longest Flycatcher migrations",
    "Acadian Flycatcher":            "Breeds in eastern USA (May–Aug) → winters in Colombia, Ecuador & Central America (Sep–Apr)",
    "Yellow bellied Flycatcher":     "Breeds in boreal Canada & northeastern USA (Jun–Aug) → winters in Mexico & Central America (Sep–May)",
    "Pacific slope Flycatcher":      "Breeds in Pacific coast USA & Canada (Apr–Aug) → winters in Mexico (Sep–Mar)",
    "Black billed Cuckoo":           "Breeds in eastern USA & Canada (May–Aug) → winters in South America — Colombia to Bolivia (Sep–Apr)",
    "Yellow billed Cuckoo":          "Breeds across USA & Canada (May–Aug) → winters in South America (Aug–Apr). Famous for calling just before rainstorms",
    "American Crow":                 "Year-round across most of USA. Northern birds may move south in harsh winters. Highly intelligent",
    "Fish Crow":                     "Year-round on Atlantic & Gulf coasts eastern USA. Short local movements only",
    "Brown Creeper":                 "Year-round in mature forests across North America. Mountain birds move to lower elevations in winter",
    "Rock Pigeon":                   "Year-round resident worldwide. Does not migrate — introduced species",
    "White crowned Pigeon":          "Breeds in Florida Keys & Caribbean (Apr–Sep) → winters in Caribbean islands (Oct–Mar)",
    "Band tailed Pigeon":            "Year-round on Pacific coast. Interior birds move to coast or Mexico in winter (Oct–Mar)",
    "Eared Grebe":                   "Breeds on western USA & Canada lakes (May–Aug) → winters on Pacific coast & Gulf of Mexico (Sep–Apr)",
    "Horned Grebe":                  "Breeds in Alaska, Canada & northern Europe (May–Aug) → winters on both USA coasts (Sep–Apr)",
    "Red necked Grebe":              "Breeds in Alaska & Canada (May–Aug) → winters on Pacific & Atlantic coasts (Sep–Apr)",
    "Pied billed Grebe":             "Year-round across most of USA. Northern birds migrate south in winter (Oct–Mar)",
    "Western Grebe":                 "Breeds on inland lakes western USA & Canada (Apr–Aug) → winters on Pacific coast (Sep–Mar)",
    "Gadwall":                       "Breeds in Great Plains USA & Canada (May–Aug) → winters across southern USA, Mexico & Caribbean (Oct–Mar)",
    "Canvasback":                    "Breeds in Great Plains Canada (May–Aug) → winters on Great Lakes, Atlantic & Gulf coasts (Oct–Mar)",
    "Redhead":                       "Breeds in Great Plains USA & Canada (May–Aug) → winters on Gulf Coast & Atlantic coast (Oct–Mar)",
    "Ring necked Duck":              "Breeds in boreal Canada (Jun–Aug) → winters across southern USA & Caribbean (Oct–Mar)",
    "Lesser Scaup":                  "Breeds in Alaska & boreal Canada (Jun–Aug) → winters across both USA coasts & inland waters (Oct–Apr)",
    "Surf Scoter":                   "Breeds in boreal Canada & Alaska (Jun–Aug) → winters on both Pacific & Atlantic coasts (Oct–Apr)",
    "White winged Scoter":           "Breeds in boreal & Arctic Canada (Jun–Aug) → winters on both USA coasts (Oct–Apr)",
    "Long tailed Duck":              "Breeds on Arctic tundra Canada & Alaska (Jun–Aug) → winters on Great Lakes & both coasts (Oct–May)",
}


# ── Prediction function ──────────────────────────────────────
def predict_bird(image):
    img = Image.fromarray(image).convert("RGB")
    x   = transform(img).unsqueeze(0).to(device)

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

    # Confidence message
    if top_conf < 60:
        conf_msg = f"⚠️ LOW CONFIDENCE ({top_conf:.1f}%) — Try a clearer photo with the bird more visible."
    elif top_conf < 80:
        conf_msg = f"🟡 MODERATE CONFIDENCE ({top_conf:.1f}%) — Likely correct but not certain."
    else:
        conf_msg = f"✅ HIGH CONFIDENCE ({top_conf:.1f}%) — I am quite sure about this."

    # Similar species
    similar = SIMILAR_SPECIES.get(top_name)
    similar_msg = f"⚠️ Looks similar to: {similar[0]}\nHow to tell apart: {similar[1]}" if similar else ""

    # Migration
    migration = MIGRATION_MAP.get(top_name, "Migration data not available for this species")

    output = f"""
🐦  SPECIES     : {top_name}
{conf_msg}

📍  FOUND IN    : {top_habitat}

✈️   MIGRATION   : {migration}

{similar_msg}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP 5 PREDICTIONS:
"""
    for i, (name, conf, habitat) in enumerate(results):
        marker = " ◀ TOP PICK" if i == 0 else ""
        output += f"\n#{i+1}  {name}  ({conf:.1f}%){marker}\n    📍 {habitat}\n"

    return output

# ── Launch Gradio app ────────────────────────────────────────
app = gr.Interface(
    fn=predict_bird,
    inputs=gr.Image(label="Upload a Bird Photo"),
    outputs=gr.Textbox(label="Results", lines=20),
    title="🐦 Bird Species Identifier",
    description="Upload any bird photo — I'll tell you the species, where it's found, and migration info!",
    theme=gr.themes.Soft()
)

print("\n✅ Starting Bird Identifier App...")
print("   Once loaded, open the link shown below in your browser.\n")
app.launch()
