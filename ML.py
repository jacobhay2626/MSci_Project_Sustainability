from classification import classification
descs = ["HTP(ingestion)log10", "HTP(Inhalation)log10", "XVP", "Flash Point (℃)",
         "Peroxide formation", "AIT", "CGPlog10", "CLP", "Aquatic Toxicity (mg/L)log10",
         "Vapour Pressure (mmHg)", "OH radical RC (cm/molecule s)"]

preds, real, matrix, report = classification(descs, "RF", 10)

