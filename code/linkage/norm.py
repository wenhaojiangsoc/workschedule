"""Shared normalisation for Orbis <-> Glassdoor entity resolution."""
import re, functools
from unidecode import unidecode
import tldextract
_tld = tldextract.TLDExtract(suffix_list_urls=None)   # offline snapshot

# ---- legal form suffixes (multi-country). Order matters: longest first.
LEGAL = [
 "incorporated","corporation","company limited","limited liability company","limited partnership",
 "public limited company","proprietary limited","private limited","societe anonyme","aktiengesellschaft",
 "kabushiki kaisha","naamloze vennootschap","besloten vennootschap","aktiebolag","osakeyhtio",
 "sociedad anonima","sociedade anonima","societa per azioni","gesellschaft mit beschrankter haftung",
 "s a de c v","sab de cv","sa de cv","pte ltd","sdn bhd","pt tbk","co ltd","co inc","and co",
 "inc","corp","co","ltd","limited","llc","llp","lllp","lp","plc","gmbh","mbh","ag","kgaa","kg","ohg",
 "sa","sas","sarl","sl","srl","spa","spz oo","sp z oo","nv","bv","cv","ab","asa","as","a s","aps","oy","oyj",
 "kk","kft","zrt","nyrt","doo","dd","ad","se","ug","ev","gbr","scs","sca","sca sicav","sicav","sicaf",
 "pty","pty ltd","pjsc","jsc","ojsc","cjsc","pao","oao","zao","ooo","tbk","bhd","berhad","psc","qsc",
 "aktieselskab","aksjeselskap","anpartsselskab","kommanditselskab","handelsbolag","stiftung","ek",
 "trust","reit","nv sa","sa nv","sp zoo","spolka akcyjna","gk","yk","llp lp","lc","plc ltd",
]
LEGAL_RE = re.compile(r"\s+(?:" + "|".join(sorted((re.escape(x) for x in LEGAL), key=len, reverse=True)) + r")$")
# descriptive tails that are frequently dropped in brand names
SOFT = ["holdings","holding","group","groupe","gruppe","international","worldwide","global","company",
        "corporation","enterprises","enterprise","industries","ventures","partners","technologies",
        "systems","solutions","services","the"]
SOFT_RE = re.compile(r"\s+(?:" + "|".join(SOFT) + r")$")
PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
WS_RE = re.compile(r"\s+")
NULLS = {"", "nan", "none", "n.a.", "na", "n/a", "-", "null", "<na>"}

def _clean(x):
    """None / NaN / placeholder -> '' ; else stripped str."""
    if x is None: return ""
    if isinstance(x, float) and x != x: return ""
    s = str(x).strip()
    return "" if s.lower() in NULLS else s

@functools.lru_cache(maxsize=1_000_000)
def norm_name(s):
    """Casefold, de-accent, strip punctuation and trailing legal forms."""
    s = _clean(s)
    if not s: return ""
    s = unidecode(s).lower()
    s = s.replace("&", " and ").replace("+", " plus ")
    s = re.sub(r"\b(u\.?s\.?a\.?)\b", "usa", s)
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    s = re.sub(r"^the\s+", "", s)
    prev = None
    while prev != s:                      # peel repeated legal tails: "x inc ltd"
        prev = s
        s = LEGAL_RE.sub("", s).strip()
    return s

@functools.lru_cache(maxsize=1_000_000)
def core_name(s):
    """norm_name minus soft descriptive tails -> aggressive blocking key."""
    s = norm_name(s)
    prev = None
    while prev != s and len(s.split()) > 1:
        prev = s
        s = SOFT_RE.sub("", s).strip()
    return s

def name_key(s):
    return norm_name(s).replace(" ", "")

def core_key(s):
    return core_name(s).replace(" ", "")

GENERIC_HOSTS = {"wixsite.com","weebly.com","wordpress.com","blogspot.com","facebook.com",
                 "linkedin.com","sites.google.com","godaddysites.com","squarespace.com",
                 "business.site","myshopify.com","webs.com","tripod.com","angelfire.com"}

@functools.lru_cache(maxsize=1_000_000)
def domain_of(url):
    """Registrable domain (eTLD+1), lowercased; '' if unusable."""
    u = _clean(url).lower()
    if not u: return ""
    u = re.sub(r"^[a-z]+://", "", u)
    u = u.split("/")[0].split("?")[0].split("#")[0].split("@")[-1].split(":")[0]
    if not u or "." not in u: return ""
    try: ex = _tld(u)
    except Exception: return ""
    d = ex.registered_domain
    if not d or d in GENERIC_HOSTS: return ""
    return d

def domain_stem(d):
    """Registrable domain minus its public suffix: walmart.com -> walmart."""
    d = _clean(d)
    if not d: return ""
    try: return _tld(d).domain
    except Exception: return d.split(".")[0]

TICKER_RE = re.compile(r"[^A-Z0-9.\-]")
def norm_ticker(t):
    t = _clean(t).upper()
    if not t: return ""
    t = t.split(":")[-1]                 # "NYSE:WMT" -> "WMT"
    t = TICKER_RE.sub("", t)
    t = re.sub(r"[.\-][A-Z]{1,2}$", "", t)   # drop exchange suffix: 7203.T, BHP.AX
    return t if 1 <= len(t) <= 6 else ""

US_STATES = {
 'alabama':'AL','alaska':'AK','arizona':'AZ','arkansas':'AR','california':'CA','colorado':'CO',
 'connecticut':'CT','delaware':'DE','florida':'FL','georgia':'GA','hawaii':'HI','idaho':'ID',
 'illinois':'IL','indiana':'IN','iowa':'IA','kansas':'KS','kentucky':'KY','louisiana':'LA',
 'maine':'ME','maryland':'MD','massachusetts':'MA','michigan':'MI','minnesota':'MN',
 'mississippi':'MS','missouri':'MO','montana':'MT','nebraska':'NE','nevada':'NV',
 'new hampshire':'NH','new jersey':'NJ','new mexico':'NM','new york':'NY','north carolina':'NC',
 'north dakota':'ND','ohio':'OH','oklahoma':'OK','oregon':'OR','pennsylvania':'PA',
 'rhode island':'RI','south carolina':'SC','south dakota':'SD','tennessee':'TN','texas':'TX',
 'utah':'UT','vermont':'VT','virginia':'VA','washington':'WA','west virginia':'WV',
 'wisconsin':'WI','wyoming':'WY','district of columbia':'DC','puerto rico':'PR'}

def norm_city(c):
    c = _clean(c)
    if not c: return ""
    c = unidecode(c).lower()
    c = re.sub(r"^(st|saint)\b\.?", "saint", c)
    c = PUNCT_RE.sub(" ", c)
    return WS_RE.sub(" ", c).strip()
