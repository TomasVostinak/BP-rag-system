##################################################
### Soubor pro chunkování textu z ingestion.py ###
##################################################

from transformers import AutoTokenizer
import nltk
nltk.download("punkt_tab")

TEXT_FILE = "data/ingested-text.jsonl"
CHUNK_FILE = "data/chunked-text.jsonl"

CHUNK_SIZE = 400
OVERLAP = 100
MIN_CHUNK_SIZE = 50

tokenization_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(tokenization_model)

def split_sentences(text):
    return nltk.sent_tokenize(text, language="czech")

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

text = "Našli jste pod stromečkem nový mobil? Zbavte se toho starého s knihovnou! Nepotřebné mobily a nabíječky v šuplíku nemají valný význam. Dějte jim druhou šanci a přidejte se k jejich sběru pro neziskový projekt Remobil. Zajistíte tak jeho další efektivní využití či recyklaci a zároveň podpoříte dobrou věc. Upozornění – 2. ledna ZAVŘENO Vážení čtenáři a návštěvníci knihovny, upozorňujeme vás, že letos je provoz knihovny na konci roku trochu jiný, než obvykle. Poslední den, kdy nás můžete navštívit, je úterý 30. prosince. Na Silvestra, na Nový rok a v pátek 2. ledna bude knihovna uzavřena. během svátků a uzavření knihovny si náš bibliobox také dává malou pauzu. Prosíme vás proto, abyste do něj v tomto období knihy . Nebude možné ho pravidelně vybírat a mohl by se rychle zaplnit. Milí čtenáři a návštěvníci naší knihovny, přejeme vám klidné a radostné Vánoce plné pohody, inspirace a krásných příběhů – ať už knižních, nebo těch prožitých s vašimi blízkými. Do nového roku pak přejeme hlavně zdraví, optimismus, chuť objevovat nové světy a mnoho dobrých důvodů se k nám znovu vracet. Těšíme se na vás v roce 2026. Ve středu 17. prosince se naše pobočka v Kokoníně proměnila v malé vánoční království. Dopoledne zde prožily děti ze druhé třídy ZŠ Kokonín, pro které byl připraven pestrý program plný adventní nálady, tradic a tvořivosti. Poděkování za zapojení do sbírky Děkujeme všem dárcům teplého oblečení, kteří se zapojili do naší sbírky Dotykový online katalog: další krok k pohodlnějším službám knihovny Naše knihovna rozšiřuje své služby o Po zavedení selfchecků, které čtenářům umožňují samostatně si půjčovat a vracet dokumenty, přichází další novinka usnadňující orientaci v knihovně. Děkujeme všem, kteří se zapojili do jubilejního 25. ročníku naší literární soutěže. Každý odevzdaný příspěvek v sobě nesl kus odvahy psát, přemýšlet a sdílet vlastní pohled na svět. Velmi si vážíme času a energie, které jste svým textům věnovali. Gratulujeme všem autorům bez ohledu na výsledné umístění a věříme, že účast v soutěži pro mnohé z vás znamená důležitý krok na další tvůrčí cestě. Nejen v naší zemi (a knihovně) straší Zavítali k nám čtvrťáci ze ZŠ Kokonín, aby u nás zažili lehce strašidelnou cestu za poznáním. Vydali jsme se společně za strašidly, pověstmi a legendami po naší vlasti, ale i za hranice. Navštívili jsme hrady, zámky a další místa opředená tajemstvím a záhadami. Naše skřítka Knihomůrka nezahálí a těší se na Vánoce. A protože jí je jasné, že na Vánoce  se těší i děti (a vlastně i rodiče), připravila si pro vás adventní kalendář. Na těchto webových stránkách se používají soubory cookies a další síťové identifikátory, které jsou nezbytné pro provoz některých funkcionalit webu. Souhlasím s použitím statistických cookies, které umožňují měřit návštěvnost webu Souhlasím s použitím analytických cookies, které umožňují měřit výkon webu"

split_text = split_sentences(text)
print(split_text)

for sentence in split_text:
    print(count_tokens(sentence))