"""Generate baseline labeled dataset for Week 1."""
import csv
from pathlib import Path

DATA = [
    # food
    ("canteen samosa and chai", "food"),
    ("mess lunch token", "food"),
    ("maggi at night canteen", "food"),
    ("zomato biryani delivery", "food"),
    ("swiggy burger order", "food"),
    ("tea stall biscuits", "food"),
    ("shawarma roll dinner", "food"),
    ("dosa at udipi restaurant", "food"),
    ("dominos pizza treat", "food"),
    ("groceries milk bread eggs", "food"),
    ("coffee at campus cafe", "food"),
    # travel
    ("metro smart card recharge", "travel"),
    ("auto fare to college", "travel"),
    ("rapido bike ride to hostel", "travel"),
    ("uber cab to railway station", "travel"),
    ("bus ticket to home town", "travel"),
    ("petrol for motorcycle", "travel"),
    ("ola auto ride", "travel"),
    ("local train monthly pass", "travel"),
    ("porter tempo luggage shifting", "travel"),
    ("train ticket booking irctc", "travel"),
    ("airport taxi ride", "travel"),
    # education
    ("semester exam fee portal", "education"),
    ("photocopy lecture notes", "education"),
    ("stationery notebooks pens", "education"),
    ("dbms textbook purchase", "education"),
    ("python machine learning course", "education"),
    ("gate coaching monthly fee", "education"),
    ("library overdue fine", "education"),
    ("lab record files and prints", "education"),
    ("scientific calculator fx991", "education"),
    ("ieee student branch membership", "education"),
    ("coursera certificate course", "education"),
    # entertainment
    ("netflix monthly subscription", "entertainment"),
    ("pvr movie ticket weekend", "entertainment"),
    ("spotify premium student plan", "entertainment"),
    ("youtube premium membership", "entertainment"),
    ("gaming arcade center", "entertainment"),
    ("standup comedy show ticket", "entertainment"),
    ("board games cafe entry", "entertainment"),
    ("hotstar subscription renewal", "entertainment"),
    ("cinema popcorn snack combo", "entertainment"),
    ("steam sale pc game", "entertainment"),
    ("music concert pass", "entertainment"),
    # bills
    ("hostel room electricity bill", "bills"),
    ("jio mobile prepaid recharge", "bills"),
    ("wifi broadband bill payment", "bills"),
    ("hostel monthly mess bill", "bills"),
    ("laundry service payment", "bills"),
    ("room rent split transfer", "bills"),
    ("drinking water can delivery", "bills"),
    ("lpg gas cylinder refill", "bills"),
    ("apartment maintenance share", "bills"),
    ("airtel postpaid mobile bill", "bills"),
    # health
    ("pharmacy fever tablets strip", "health"),
    ("doctor clinic consultation fee", "health"),
    ("cough syrup and paracetamol", "health"),
    ("bandaid and antiseptic cream", "health"),
    ("eye clinic vision test", "health"),
    ("dental scaling and cleaning", "health"),
    ("multivitamin capsules bottle", "health"),
    ("blood test diagnostic lab", "health"),
    ("fever clinic consultation", "health"),
    ("pain relief spray volini", "health"),
    ("gym monthly fitness pass", "health"),
    # shopping
    ("college backpack bag", "shopping"),
    ("myntra cotton casual shirt", "shopping"),
    ("flipkart phone back cover", "shopping"),
    ("running sports shoes", "shopping"),
    ("winter hoodie jacket", "shopping"),
    ("bluetooth wireless earphones", "shopping"),
    ("study desk led lamp", "shopping"),
    ("formal trousers interview", "shopping"),
    ("stainless steel water bottle", "shopping"),
    ("folding umbrella for rain", "shopping"),
]

def generate_csv(output_path=None):
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "expenses_labeled.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category"])
        writer.writerows(DATA)
    print(f"Wrote {len(DATA)} rows to {output_path}")
    return output_path

if __name__ == "__main__":
    generate_csv()
