brand_mapping = {
    "budget": [
        "Aixam", "Baic", "Casalini", "Chatenet", "DFSK", "DKW", "Ligier",
        "Microcar", "Moskwicz", "Nysa", "Polonez", "Tarpan", "Trabant",
        "Zaporożec", "Żuk", "Syrena", "Tavria", "Warszawa"
    ],
    "normal": [
        "Abarth", "Acura", "Austin", "Autobianchi", "Chevrolet", "Chrysler",
        "Citroën", "Cupra", "Dacia", "Daewoo", "Daihatsu", "Fiat", "Ford",
        "Gaz", "GMC", "Honda", "Hyundai", "Isuzu", "Iveco", "Jeep", "Kia",
        "Lada", "Lancia", "Mazda", "Mercury", "MG", "MINI", "Mitsubishi",
        "Nissan", "Oldsmobile", "Opel", "Peugeot", "Renault", "Seat", "Škoda",
        "Smart", "SsangYong", "Suzuki", "Subaru", "Tata", "Toyota", "Vauxhall",
        "Volkswagen", "Wartburg", "Zastava", "Inny"
    ],
    "premium": [
        "Alfa Romeo", "Alpine", "Audi", "BMW", "Buick", "Cadillac",
        "DS Automobiles", "Infiniti", "Jaguar", "Land Rover", "Lexus",
        "Mercedes-Benz", "Porsche", "Volvo", "Scion", "Rover", "RAM", "Pontiac",
        "Saturn", "Triumph"
    ],
    "luxury": [
        "Aston Martin", "Bentley", "Ferrari", "Hummer", "Lamborghini",
        "Lotus", "Maserati", "Maybach", "McLaren", "Rolls-Royce", "Tesla",
        "Vanderhall"
    ]
}

brand_to_category = {brand: category for category, brands in brand_mapping.items() for brand in brands}

brand_category_weights = {
    "Luxury": 2.0,
    "Premium": 1.7,
    "Mid-Range": 1.3,
    "Economy": 1.0,
    "Budget": 0.6
}
