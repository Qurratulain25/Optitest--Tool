# config.py
# Contains the linguistic scale, the alternative mapping, and an optional criteria list.

LINGUISTIC_SCALE = {
    "1": (1, 1, 1),
    "2": (2, 3, 4),
    "3": (4, 5, 6),
    "4": (6, 7, 8),
    "5": (9, 9, 9),
    "intermittent_2": (1, 2, 3),
    "intermittent_4": (3, 4, 5),
    "intermittent_6": (5, 6, 7),
    "intermittent_8": (7, 8, 9),
    "equally_important": (1, 1, 1),
    "weakly_more_important": (2, 3, 4),
    "fairly_more_important": (4, 5, 6),
    "strongly_more_important": (6, 7, 8),
    "absolutely_more_important": (9, 9, 9),
}

ALTERNATIVE_MAPPING = {
    "Homepage": ["home", "go to home page", "landing page", "main page"],
    "Login": ["login", "sign in", "authenticate", "user login", "account access"],
    "Add to Cart": ["add to cart", "cart", "basket", "add item to cart", "shopping cart"],
    "View Details": ["view details", "product info", "item details", "check blog", "select food for employees", "check ubl careers page"],
    "Checkout": ["checkout", "payment", "purchase", "confirm order", "place order"],
    "Upload File": ["upload", "file upload", "attach", "submit document", "send file"],
    "Update Profile": ["update profile", "edit profile", "modify account", "profile settings"],
    "Select Item": ["select item", "choose product", "pick item", "pick category"],
    "View Transactions": ["view transactions", "transaction history", "order history", "payment history"],
    "Payments": ["submit payment", "confirm payment", "process payment", "withdraw", "transfer", "loan", "apply for loan", "bank account", "open bank account", "deposit", "send money", "receive money", "withdraw money"],
    "Booking Process": ["select any hotel", "check-in", "room booking", "room selection", "hotel reservation"],
    "Booking Modification": ["edit check-in", "edit check-out", "change reservation", "modify booking"],
    "Search & Selection": ["search item", "select category", "filter products", "check all pictures of room"],
    "Order Management": ["place order", "track order", "modify order", "cancel order", "reorder"],
    "Profile Management": ["open profile", "go to profile", "user profile", "account settings"],
    "Information & Navigation": ["check blog", "view help page", "search help", "faq"],
    "Financial Products": ["loan options", "credit card info", "services-check cards products"]
}

CRITERIA_LIST = [
    "Throughput",
    "Latency",
    "Response_Time",
    "Network_Load"
]
